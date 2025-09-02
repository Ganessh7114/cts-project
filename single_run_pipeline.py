# file: single_run_pipeline.py
import os, json, ssl, sys, time, threading
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
import requests
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# -------------------- Load .env --------------------
load_dotenv()

BROKER   = os.getenv("BROKER")                     # e.g. 27cccf...s1.eu.hivemq.cloud
PORT     = int(os.getenv("PORT", "8883"))          # 8883 for TLS
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
TOPIC = os.getenv("TOPIC")

AZURE_ML_ENDPOINT_URL = os.getenv("AZURE_ML_ENDPOINT_URL")
AZURE_ML_API_KEY      = os.getenv("AZURE_ML_API_KEY")
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME          = os.getenv("CONTAINER_NAME")

# -------------------- Helpers --------------------
def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()

def fail_if_missing(name, value):
    if not value:
        print(f"❌ Missing required env: {name}")
        sys.exit(1)

# Validate minimum required envs
for k in ["BROKER","USERNAME","PASSWORD","AZURE_ML_ENDPOINT_URL","AZURE_CONNECTION_STRING","CONTAINER_NAME"]:
    fail_if_missing(k, globals()[k])

# -------------------- Azure ML call --------------------
def call_azure_ml(payload: dict):
    headers = {"Content-Type": "application/json"}
    if AZURE_ML_API_KEY:
        headers["Authorization"] = f"Bearer {AZURE_ML_API_KEY}"
    try:
        resp = requests.post(AZURE_ML_ENDPOINT_URL, headers=headers, json={"data":[payload]}, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Azure ML call failed: {e}")
        return None

# -------------------- Fallback (rule-based) --------------------
def rule_based_prediction(data: dict):
    factors = []
    risk_score = 0.0
    pred = "Low"

    if data.get("TemperatureC", 0) > 37:
        pred = "High"
        risk_score += 0.2
        factors.append("High temperature")
    if data.get("VibrationMM_S", 0) > 0.05:
        pred = "High"
        risk_score += 0.2
        factors.append("High vibration")

    return {
        "success": True,
        "model_loaded": False,
        "model_version": "rules-fallback",
        "predictions": [{
            "device_name": data.get("DeviceName","Unknown"),
            "prediction": pred,
            "confidence": 0.7,
            "risk_score": risk_score,
            "factors": factors or ["All parameters within normal ranges"],
            "model_version": "rules-fallback",
            "timestamp": utcnow_iso(),
            "debug": {"used_model": False, "probabilities": {}},
        }]
    }

# -------------------- Blob storage --------------------
def store_to_blob(obj: dict):
    try:
        svc = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container = svc.get_container_client(CONTAINER_NAME)
        name = f"prediction_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        container.upload_blob(name=name, data=json.dumps(obj, ensure_ascii=False), overwrite=True)
        print(f"✅ Stored result in blob: {name}")
    except Exception as e:
        print(f"❌ Blob upload failed: {e}")

# -------------------- MQTT (Paho v2 callbacks) --------------------
# We use two clients: one SUB (receive once) and one PUB (send once)

connect_ok_event  = threading.Event()
message_got_event = threading.Event()
received_payload  = {}

def on_connect_sub(client, userdata, flags, reason_code, properties):
    if reason_code != mqtt.ReasonCodes.SUCCESS:
        print(f"❌ SUB connect failed: {reason_code} ({reason_code.getName()})")
        # Fail fast: stop the loop and exit
        client.disconnect()
        os._exit(1)
    print("✅ SUB connected")
    client.subscribe(TOPIC, qos=1)

def on_message_sub(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        print(f"📨 SUB received on {msg.topic}: {payload}")
        received_payload["data"] = payload
        message_got_event.set()
    except Exception as e:
        print(f"❌ Failed to parse message: {e}")

def on_connect_pub(client, userdata, flags, reason_code, properties):
    if reason_code != mqtt.ReasonCodes.SUCCESS:
        print(f"❌ PUB connect failed: {reason_code} ({reason_code.getName()})")
        client.disconnect()
        os._exit(1)
    print("✅ PUB connected")
    connect_ok_event.set()

def make_client(client_id_suffix, on_connect_cb=None, on_message_cb=None):
    # Unique client ID per connection
    cid = f"med-iot-{client_id_suffix}-{int(time.time()*1000)}"
    c = mqtt.Client(client_id=cid, protocol=mqtt.MQTTv5, transport="tcp", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    c.username_pw_set(USERNAME, PASSWORD)
    ctx = ssl.create_default_context()
    c.tls_set_context(ctx)
    c.on_connect = on_connect_cb
    if on_message_cb:
        c.on_message = on_message_cb
    return c

def publish_once(payload: dict):
    pub = make_client("pub", on_connect_cb=on_connect_pub)
    print(f"🚀 PUB connecting to {BROKER}:{PORT} ...")
    pub.connect(BROKER, PORT, keepalive=60)
    pub.loop_start()
    # Wait until connected or fail-fast in on_connect
    if not connect_ok_event.wait(timeout=10):
        print("❌ PUB connect timeout")
        os._exit(1)

    # Publish exactly once
    j = json.dumps(payload, ensure_ascii=False)
    info = pub.publish(TOPIC, j, qos=1)
    info.wait_for_publish()
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"❌ Publish failed: rc={info.rc}")
        os._exit(1)
    print(f"✅ Published once to {TOPIC}: {j}")
    pub.disconnect()
    pub.loop_stop()

def subscribe_receive_once():
    sub = make_client("sub", on_connect_cb=on_connect_sub, on_message_cb=on_message_sub)
    print(f"👂 SUB connecting to {BROKER}:{PORT} and listening on '{TOPIC}' ...")
    sub.connect(BROKER, PORT, keepalive=60)
    sub.loop_start()
    # Wait for exactly one message (timeout to avoid hanging)
    got = message_got_event.wait(timeout=20)
    sub.disconnect()
    sub.loop_stop()
    if not got:
        print("❌ No message received within timeout")
        sys.exit(1)

def main():
    # 1) Start subscriber first (in a thread) to ensure it is ready
    t = threading.Thread(target=subscribe_receive_once, daemon=True)
    t.start()

    # 2) Publish one telemetry message
    sample_telemetry = {
        "DeviceType": "CT Scanner",
        "DeviceName": "GE Revolution",
        "RuntimeHours": 1200,
        "TemperatureC": 38.5,
        "PressureKPa": 101.3,
        "VibrationMM_S": 0.02,
        "CurrentDrawA": 3.4,
        "SignalNoiseLevel": 0.1,
        "ClimateControl": "Yes",
        "HumidityPercent": 45,
        "Location": "Hospital A - Central Region",
        "OperationalCycles": 340,
        "UserInteractionsPerDay": 15,
        "ApproxDeviceAgeYears": 3,
        "NumRepairs": 1,
        "ErrorLogsCount": 2,
        "SentTimestamp": utcnow_iso(),
    }
    publish_once(sample_telemetry)

    # 3) Wait for the subscriber to finish (one message)
    t.join()

    # 4) Process received payload → Azure ML → fallback if needed
    data = received_payload.get("data", {})
    print("🔧 Processing received telemetry via Azure ML ...")
    ml = call_azure_ml(data)
    if ml and ml.get("success") and ml.get("model_loaded", True):
        final_result = ml
        print("✅ Azure ML result accepted")
    else:
        print("⚠️ Azure ML unavailable or returned fallback flag → using rule-based")
        final_result = rule_based_prediction(data)

    # 5) Save to Blob
    store_to_blob(final_result)

    # 6) Optionally publish result once, then exit
    try:
        pub = make_client("pub2", on_connect_cb=on_connect_pub)
        connect_ok_event.clear()
        pub.connect(BROKER, PORT, keepalive=60)
        pub.loop_start()
        if connect_ok_event.wait(timeout=10):
            result_json = json.dumps(final_result, ensure_ascii=False)
            info = pub.publish(TOPIC, result_json, qos=1)
            info.wait_for_publish()
            print(f"📤 Published result once to {TOPIC}")
        else:
            print("⚠️ Skipped result publish (could not reconnect PUB)")
        pub.disconnect()
        pub.loop_stop()
    except Exception as e:
        print(f"⚠️ Result publish error: {e}")

    print("🏁 Done (sent once, received once, processed, stored).")

if __name__ == "__main__":
    main()
