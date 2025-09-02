# flowtest.py
import os
import json
import joblib
import requests
import paho.mqtt.client as mqtt
from azure.storage.blob import BlobServiceClient
from datetime import datetime, UTC
from dotenv import load_dotenv
import pandas as pd

from rule_fallback import local_or_rules   # our new fallback logic

# ---------------- Load environment variables ----------------
load_dotenv()

ENDPOINT_URL = os.getenv("AZURE_ML_ENDPOINT_URL")
API_KEY = os.getenv("AZURE_ML_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_ML_DEPLOYMENT_NAME")

BLOB_CONN_STR = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

BROKER = os.getenv("BROKER")
PORT = int(os.getenv("PORT", 8883))
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
TOPIC = os.getenv("TOPIC", "iot/devices")

MODEL_PATH = os.getenv("MODEL_PATH", "xgboost_pipeline.pkl")

# ---------------- Sample Input ----------------
sample_input = {
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
    "ErrorLogsCount": 2
}

# ---------------- Azure ML + Fallback Logic ----------------
def predict_with_fallback(data: dict):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        resp = requests.post(ENDPOINT_URL, headers=headers, json={"data": [data]})
        resp.raise_for_status()
        result = resp.json()

        if result.get("success", False):
            print("✅ ML Studio prediction:", result)
            return result
        else:
            print("⚠️ ML Studio failed, falling back...")
    except Exception as e:
        print("⚠️ Azure ML error:", e)

    # fallback to local model or rules
    fallback_result = local_or_rules(data)
    print("✅ Fallback prediction:", fallback_result)
    return {"success": True, "predictions": [fallback_result]}

# ---------------- Store in Blob ----------------
def store_in_blob(result: dict):
    try:
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
        container = blob_service.get_container_client(CONTAINER_NAME)

        filename = f"prediction_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        container.upload_blob(name=filename, data=json.dumps(result), overwrite=True)
        print(f"✅ Stored prediction in blob: {filename}")
    except Exception as e:
        print("❌ Blob storage error:", e)

# ---------------- MQTT Producer ----------------
def publish_mqtt(data: dict):
    print("\n📡 Publishing telemetry to MQTT...")
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()

    def on_connect(client, userdata, flags, rc, properties=None):
        print("✅ MQTT Connected with result code", rc)
        client.publish(TOPIC, json.dumps(data))
        print(f"✅ Published to MQTT: {json.dumps(data)}")
        client.disconnect()

    client.on_connect = on_connect
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

# ---------------- MQTT Consumer ----------------
def consume_mqtt():
    print("\n📡 Listening for MQTT message...")
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()

    def on_connect(client, userdata, flags, rc, properties=None):
        print("✅ MQTT Receiver connected with result code", rc)
        client.subscribe(TOPIC)

    def on_message(client, userdata, msg):
        print("📨 Received MQTT message")
        data = json.loads(msg.payload.decode())

        # run prediction pipeline (ML → Local → Rules)
        result = predict_with_fallback(data)

        # store prediction
        store_in_blob(result)

        client.disconnect()

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

# ---------------- Run Pipeline ----------------
if __name__ == "__main__":
    try:
        # Step 1: Publish input
        publish_mqtt(sample_input)

        # Step 2: Consume + Predict + Store
        consume_mqtt()

    except KeyboardInterrupt:
        print("🛑 Workflow stopped by user")

    except Exception as e:
        print("Error: ",e)