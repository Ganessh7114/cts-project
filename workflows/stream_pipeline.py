# workflows/stream_pipeline.py
import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from itertools import cycle

# Allow "services/..." and "presentation/..." imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Services
from services.mqtt_service import MQTTService
from services.azure_ml_client import AzureMLClient
from services.local_model import LocalModel
from services.rule_fallback import RuleBasedAssessor
from services.blob_storage import BlobStorage
from services.random_generator import generate_random_payload, device_mapping  # updated

# Presentation helpers
from presentation.logging_config import setup_logging
from presentation.presenter import present_step, present_final


# ---------------------------------------------------------------------
# Setup logging & environment
# ---------------------------------------------------------------------
load_dotenv()
log_mode = setup_logging()
logger = logging.getLogger(__name__)

# Quiet noisy libs in presentation mode
if (os.getenv("LOG_MODE", "presentation").strip().lower() == "presentation"):
    for name in [
        "azure",
        "azure.core.pipeline.policies.http_logging_policy",
        "urllib3",
        "paho",
        "services.mqtt_service",
        "services.azure_ml_client",
        "services.local_model",
        "services.rule_fallback",
        "services.blob_storage",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MQTT_TLS = os.getenv("MQTT_TLS", "true").strip().lower() != "false"
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "iot/devices")

PUBLISH_INTERVAL_SEC = float(os.getenv("STREAM_PUBLISH_INTERVAL_SEC", "2.0"))
RECEIVE_TIMEOUT_SEC = float(os.getenv("STREAM_RECEIVE_TIMEOUT_SEC", "30.0"))

# New: limit stream by cycles (1 cycle = all devices once)
MAX_CYCLES = os.getenv("STREAM_MAX_CYCLES")
MAX_CYCLES = int(MAX_CYCLES) if (MAX_CYCLES and MAX_CYCLES.isdigit()) else None


# ---------------------------------------------------------------------
# Instantiate services
# ---------------------------------------------------------------------
azure_client = AzureMLClient()
local_model = LocalModel()
rules = RuleBasedAssessor()
blob = BlobStorage()
mqtt = MQTTService(
    tls=MQTT_TLS,
    client_id_prefix="stream"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def choose_final(telemetry, aml_result, local_result):
    base_label = aml_result.get("label") or local_result.get("label")
    base_conf = aml_result.get("confidence") or local_result.get("confidence")

    refined = rules.refine_prediction(telemetry, base_label, base_conf)
    return {
        "label": refined.get("label") or base_label,
        "confidence": refined.get("confidence") if refined.get("confidence") is not None else base_conf,
        "factors": refined.get("factors", []),
    }


def archive_to_blob(record):
    fn = "prediction"
    blob.upload_json(fn, record)
    return fn


def publisher_loop(stop_flag: list, devices):
    """Background publisher: round-robin over devices."""
    device_cycle = cycle(devices)
    while not stop_flag[0]:
        try:
            device_name, device_type = next(device_cycle)
            payload_str, payload = generate_random_payload(device_name, device_type)
            mqtt.publish_once(MQTT_TOPIC, payload, qos=1, retain=False, timeout=8.0)
        except Exception as e:
            logger.error("Publisher error: %s", e)
        time.sleep(PUBLISH_INTERVAL_SEC)


def receive_one():
    return mqtt.receive_once(MQTT_TOPIC, timeout=RECEIVE_TIMEOUT_SEC)


# ---------------------------------------------------------------------
# Main stream runner
# ---------------------------------------------------------------------
def run_stream(devices):
    present_step("Starting streaming pipeline")

    stop_flag = [False]
    t = None
    try:
        import threading
        t = threading.Thread(target=publisher_loop, args=(stop_flag, devices), daemon=True)
        t.start()

        cycles_done = 0
        while True:
            if MAX_CYCLES is not None and cycles_done >= MAX_CYCLES:
                present_step(f"Reached max cycles ({MAX_CYCLES}). Stopping stream.")
                break

            # one cycle = process each device once
            for _ in devices:
                present_step("Waiting for telemetry message")
                telemetry = receive_one()
                if telemetry is None:
                    present_step("No telemetry received in time (will keep listening)", level="warn")
                    continue

                present_step("Telemetry received")

                try:
                    present_step("Invoking cloud AI model")
                    aml_result = azure_client.predict(telemetry)
                except Exception as e:
                    logger.error("Cloud inference failed: %s", e)
                    aml_result = {"ok": False, "label": None, "confidence": None, "error": str(e)}

                try:
                    present_step("Evaluating with local model")
                    local_result = local_model.predict(telemetry)
                except Exception as e:
                    logger.error("Local model failed: %s", e)
                    local_result = {"ok": False, "label": None, "confidence": None, "error": str(e)}

                present_step("Applying fallback rules")
                
                device_name = telemetry.get("DeviceName")
                device_type = telemetry.get("DeviceType")

                final = choose_final(telemetry, aml_result, local_result)

                # Add identity into final prediction
                final["device_name"] = device_name
                final["device_type"] = device_type

                present_step("AI decision computed")

                record = {
                    "telemetry": telemetry,
                    "azure_ml": aml_result,
                    "local_model": local_result,
                    "final": final,
                    "device_name": device_name,   # top-level identity
                    "device_type": device_type,   # top-level identity
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pipeline": "mqtt_stream_roundrobin"
                }


                archive_to_blob(record)
                present_step("Uploaded prediction to blob storage")
                present_final(record, log_mode=log_mode)

            cycles_done += 1

    except KeyboardInterrupt:
        present_step("Keyboard interrupt received. Stopping stream.")
    finally:
        stop_flag[0] = True
        if t and t.is_alive():
            t.join(timeout=2.0)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    present_step("Starting pipeline")

    missing = []
    if not os.getenv("AZURE_ML_ENDPOINT_URL"):
        missing.append("AZURE_ML_ENDPOINT_URL")
    if not os.getenv("AZURE_ML_API_KEY"):
        missing.append("AZURE_ML_API_KEY")
    if not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        missing.append("AZURE_STORAGE_CONNECTION_STRING")
    if missing:
        msg = "Missing required environment variables: " + ", ".join(missing)
        logger.error(msg)
        raise SystemExit(msg)

    devices = list(device_mapping.items())  # [(device_name, type), ...]
    run_stream(devices)
