import os
import json
import joblib
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


# ---------------- Publish JSON to MQTT ----------------
def publish_to_mqtt(data):
    print("\n📡 Publishing telemetry to MQTT...")

    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()

    try:
        client.connect(BROKER, PORT, 60)
        payload = json.dumps(data)
        client.publish(TOPIC, payload)
        print("✅ Published to MQTT:", payload)
        client.disconnect()
    except Exception as e:
        print("❌ MQTT Publish Error:", e)


# ---------------- Receive → ML Studio → Blob ----------------
def receive_and_process():
    print("\n📡 Listening for MQTT message...")

    def on_connect(client, userdata, flags, rc, properties=None):
        print("✅ MQTT Receiver connected with result code", rc)
        client.subscribe(TOPIC)

    def on_message(client, userdata, msg):
        try:
            print("📨 Received MQTT message:", msg.payload.decode())
            incoming_data = json.loads(msg.payload.decode())
            ml_input = {"data": [incoming_data]}

            # ---- Call Azure ML ----
            headers = {"Content-Type": "application/json"}
            if API_KEY:
                headers["Authorization"] = f"Bearer {API_KEY}"
            resp = requests.post(ENDPOINT_URL, headers=headers, json=ml_input)
            resp.raise_for_status()
            result = resp.json()
            print("✅ ML Studio Prediction:", result)

            # ---- Save to Blob ----
            blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
            container = blob_service.get_container_client(CONTAINER_NAME)
            filename = f"prediction_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            container.upload_blob(name=filename, data=json.dumps(result), overwrite=True)
            print(f"✅ Stored prediction in blob: {filename}")

        except Exception as e:
            print("❌ Error in Receiver Pipeline:", e)
        finally:
            client.disconnect()  # stop after 1 message

    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("🛑 Receiver stopped by user")
        client.disconnect()


# ---------------- Run All ----------------
if __name__ == "__main__":
    test_ml_and_store()
    test_blob()
    test_local_model()

    # --- Step 1: Publish telemetry
    publish_to_mqtt(sample_input["data"][0])

    # --- Step 2: Receiver listens once → ML → Blob
    receive_and_process()
