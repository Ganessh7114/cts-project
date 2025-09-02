import os
import json
import ssl
import time
import logging
import threading
from typing import Any, Dict, Optional, Callable

import paho.mqtt.client as mqtt

logger = logging.getLogger("services.mqtt_service")

class MQTTService:
    """
    Handles one-shot MQTT publish and one-shot receive.
    Works with HiveMQ Cloud (TLS on 8883).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 8883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls: bool = True,
        client_id_prefix: str = "client",
    ):
        self.host = host or os.getenv("HIVEMQ_HOST")
        self.port = port or int(os.getenv("HIVEMQ_PORT", 8883))
        self.username = username or os.getenv("HIVEMQ_USERNAME")
        self.password = password or os.getenv("HIVEMQ_PASSWORD")
        self.tls = tls
        self.client_id_prefix = client_id_prefix

    def _mk_client(self, suffix: str) -> mqtt.Client:
        client = mqtt.Client(client_id=f"{self.client_id_prefix}-{suffix}", protocol=mqtt.MQTTv311, transport="tcp")
        # Use latest callback API to avoid deprecation warnings
        client._callback_api_version = mqtt.CallbackAPIVersion.VERSION2  # internal switch used by paho
        if self.username:
            client.username_pw_set(self.username, self.password or "")
        if self.tls:
            client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        return client

    def publish_once(self, topic: str, message: Dict[str, Any], qos: int = 1, retain: bool = False, timeout: float = 15.0) -> bool:
        """Connect -> publish -> disconnect."""
        payload = json.dumps(message, separators=(",", ":"), default=str)
        client = self._mk_client("pub")

        pub_ok = threading.Event()

        def on_connect(c, u, f, rc, props=None):
            if rc == 0:
                logger.info("MQTT publisher connected")
                result = c.publish(topic, payload=payload, qos=qos, retain=retain)
                result.wait_for_publish(timeout=timeout)
                if result.is_published():
                    logger.info("Published to %s: %s", topic, payload)
                    pub_ok.set()
                else:
                    logger.error("Publish timed out")
                c.disconnect()
            else:
                logger.error("Publisher connect failed rc=%s", rc)
                c.disconnect()

        client.on_connect = on_connect
        client.connect(self.host, self.port, keepalive=30)
        client.loop_forever(timeout=10.0)
        return pub_ok.is_set()

    def receive_once(
        self,
        topic: str,
        timeout: float = 20.0,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Connect, subscribe, wait for the FIRST message on the topic, then disconnect.
        If on_ready is provided, it is called right after SUBACK (ensures subscriber is ready).
        """
        msg_event = threading.Event()
        received_payload: Optional[Dict[str, Any]] = None
        client = self._mk_client("sub")

        def on_connect(c, u, f, rc, props=None):
            if rc == 0:
                logger.info("MQTT subscriber connected")
                c.subscribe(topic, qos=1)
            else:
                logger.error("Subscriber connect failed rc=%s", rc)

        def on_subscribe(c, u, mid, granted_qos, props=None):
            logger.info("Subscribed to %s (qos=%s)", topic, granted_qos)
            # ←— the only change you asked for: a tiny pause before publishing
            time.sleep(1.0)
            if on_ready:
                try:
                    on_ready()
                except Exception as e:
                    logger.exception("on_ready callback failed: %s", e)

        def on_message(c, u, msg):
            nonlocal received_payload
            try:
                payload = msg.payload.decode("utf-8", errors="replace")
                received_payload = json.loads(payload)
                logger.info("Received from %s: %s", msg.topic, payload)
            except Exception as e:
                logger.exception("Failed to parse message: %s", e)
                received_payload = None
            finally:
                msg_event.set()
                c.disconnect()

        client.on_connect = on_connect
        client.on_subscribe = on_subscribe
        client.on_message = on_message

        client.connect(self.host, self.port, keepalive=30)
        # run network loop in this thread until we get a message or timeout
        start = time.time()
        while not msg_event.is_set():
            client.loop(timeout=10.0)
            if time.time() - start > timeout:
                logger.error("Timed out waiting for MQTT message on %s", topic)
                try:
                    client.disconnect()
                except Exception:
                    pass
                break

        return received_payload
