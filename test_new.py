import os
import json
import time
import joblib
import pickle
import requests
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()

# Azure ML
ENDPOINT_URL = os.getenv("AZURE_ML_ENDPOINT_URL")
API_KEY = os.getenv("AZURE_ML_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_ML_DEPLOYMENT_NAME", "medical-device-deploy")

# Blob Storage
BLOB_CONN_STR = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "medicaldevicestorage")

# MQTT
BROKER = os.getenv("BROKER")
PORT = int(os.getenv("PORT", 8883))
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
TOPIC = os.getenv("TOPIC", "iot/failure")

# Local model
MODEL_PATH = os.getenv("MODEL_PATH", "xgboost_pipeline.pkl")

# --------------------------
# Test 1: Azure ML Endpoint
# --------------------------
def test_ml_and_store():
    print("\n🚀 Testing Azure ML Endpoint...")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Example input (adjust according to your training features)
    sample_input = {
        "input_data": [
            {
                "feature1": 0.5,
                "feature2": 1.2,
                "feature3": 0.7,
                "feature4": 0.3,
                "feature5": 0.9,
                "feature6": 1.5,
                "feature7": 0.2,
                "feature8": 0.1
            }
        ]
    }

    try:
        resp = requests.post(ENDPOINT_URL, headers=headers, json=sample_input, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        print("✅ ML Response:", result)

        # Save prediction to Blob Storage
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
        container_client = blob_service.get_container_client(CONTAINER_NAME)

        filename = f"prediction_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        container_client.upload_blob(name=filename, data=json.dumps(result), overwrite=True)
        print(f"✅ Stored prediction in blob: {filename}")

    except Exception as e:
        print("❌ Error calling ML endpoint:", e)


# --------------------------
# Test 2: Blob Storage
# --------------------------
def test_blob():
    print("\n📦 Testing Azure Blob Storage...")
    try:
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
        container_client = blob_service.get_container_client(CONTAINER_NAME)

        blobs = list(container_client.list_blobs())
        print(f"✅ Found {len(blobs)} blobs in '{CONTAINER_NAME}'")
        for blob in blobs[-5:]:  # Show last 5
            print(" -", blob.name)

    except Exception as e:
        print("❌ Blob storage error:", e)


# --------------------------
# Test 3: Local Model
# --------------------------
def test_local_model():
    print("\n🧠 Testing local model...")
    try:
        model = joblib.load(MODEL_PATH)  # safer than pickle.load
        sample_input = [[0.5, 1.2, 0.7, 0.3, 0.9, 1.5, 0.2, 0.1]]
        prediction = model.predict(sample_input)
        print("✅ Local Model Prediction:", prediction)

    except Exception as e:
        print("❌ Error loading local model:", e)


# --------------------------
# Test 4: MQTT
# --------------------------
def test_mqtt():
    print("\n📡 Testing MQTT...")

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("✅ MQTT Connected successfully")
        else:
            print(f"❌ MQTT Connection failed with code {rc}")

    try:
        client = mqtt.Client()
        client.username_pw_set(USERNAME, PASSWORD)
        client.on_connect = on_connect
        client.tls_set()  # enable TLS
        client.connect(BROKER, PORT, 60)

        payload = {"test": "message", "ts": str(datetime.now(timezone.utc))}
        client.publish(TOPIC, json.dumps(payload))
        client.loop_start()
        time.sleep(2)
        client.loop_stop()

    except Exception as e:
        print("❌ MQTT error:", e)


# --------------------------
# Run all tests
# --------------------------
if __name__ == "__main__":
    test_ml_and_store()
    test_blob()
    test_local_model()
    test_mqtt()
