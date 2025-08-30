import os
import json
import joblib
import pickle
import requests
import paho.mqtt.client as mqtt
from azure.storage.blob import BlobServiceClient
from datetime import datetime, UTC
from dotenv import load_dotenv
import pandas as pd

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
TOPIC = os.getenv("TOPIC")

MODEL_PATH = os.getenv("MODEL_PATH", "xgboost_pipeline.pkl")

# ---------------- Sample Input (16 features) ----------------
sample_input = {
    "data": [
        {
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
    ]
}

# ---------------- Test Azure ML + Blob ----------------
def test_ml_and_store():
    print("\n🚀 Testing Azure ML Endpoint...")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        resp = requests.post(ENDPOINT_URL, headers=headers, json=sample_input)
        resp.raise_for_status()
        result = resp.json()
        print("✅ ML Response:", result)

        # Save result to blob
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
        container = blob_service.get_container_client(CONTAINER_NAME)

        filename = f"prediction_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        container.upload_blob(name=filename, data=json.dumps(result), overwrite=True)
        print(f"✅ Stored prediction in blob: {filename}")

    except Exception as e:
        print("❌ Error calling Azure ML or storing blob:", e)


# ---------------- Test Blob Storage ----------------
def test_blob():
    print("\n📦 Testing Azure Blob Storage...")
    try:
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
        container = blob_service.get_container_client(CONTAINER_NAME)
        blobs = list(container.list_blobs())
        print(f"✅ Found {len(blobs)} blobs in '{CONTAINER_NAME}'")
        for b in blobs[:5]:
            print(" -", b.name)
    except Exception as e:
        print("❌ Blob storage error:", e)


# ---------------- Test Local Model ----------------
def test_local_model():
    print("\n🧠 Testing local model...")
    try:
        with open(MODEL_PATH, "rb") as f:
            model = joblib.load(f)  # safer for sklearn/xgboost pipelines
        df = pd.DataFrame(sample_input["data"])
        preds = model.predict(df)
        print("✅ Local model prediction:", preds.tolist())
    except Exception as e:
        print("❌ Error loading local model:", e)


# ---------------- Test MQTT ----------------
def test_mqtt():
    print("\n📡 Testing MQTT...")

    def on_connect(client, userdata, flags, rc, properties=None):
        print("✅ MQTT Connected with result code", rc)
        client.subscribe(TOPIC)
        client.publish(TOPIC, json.dumps({"test": "message", "ts": str(datetime.utcnow())}))

    def on_message(client, userdata, msg):
        print("📨 Received message:", msg.topic, msg.payload.decode())
        client.disconnect()

    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()  # HiveMQ Cloud requires TLS
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()


# ---------------- Run All ----------------
if __name__ == "__main__":
    test_ml_and_store()
    test_blob()
    test_local_model()
    test_mqtt()
