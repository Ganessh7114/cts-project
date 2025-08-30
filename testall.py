import os
import json
import time
import pickle
from datetime import datetime
from dotenv import load_dotenv
import requests
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import paho.mqtt.client as mqtt
import joblib

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()

# Azure ML
ENDPOINT_URL = os.getenv("AZURE_ML_ENDPOINT_URL")
API_KEY = os.getenv("AZURE_ML_API_KEY")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

# Blob Storage
BLOB_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

# MQTT
BROKER = os.getenv("BROKER")
PORT = int(os.getenv("PORT", "8883"))
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
TOPIC = os.getenv("TOPIC")

# Local Model
MODEL_PATH = os.getenv("MODEL_PATH")

# --------------------------
# Auth for ML
# --------------------------
def get_auth_headers():
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    else:
        cred = ClientSecretCredential(tenant_id=TENANT_ID, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        token = cred.get_token("https://ml.azure.com/.default")
        return {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}

# --------------------------
# Test Azure ML
# --------------------------
def test_ml_and_store():
    print("\n🚀 Testing Azure ML Endpoint...")
    headers = get_auth_headers()

    sample_input = {
        "data": [
            {
                "temperature": 37,
                "vibration": 0.2,
                "error_logs": 1,
                "runtime_hours": 1200,
                "device_age": 2,
                "repairs": 0,
                "pressure": 101.5,
                "current_draw": 1.1
            }
        ]
    }

    resp = requests.post(ENDPOINT_URL, headers=headers, json=sample_input)
    result = resp.json()
    print("✅ ML Response:", result)

    # Upload to blob
    blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service.get_container_client(CONTAINER_NAME)
    filename = f"prediction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    container_client.upload_blob(name=filename, data=json.dumps(result, indent=2), overwrite=True)
    print(f"✅ Stored prediction in blob: {filename}")

# --------------------------
# Test Blob Storage
# --------------------------
def test_blob_storage():
    print("\n📦 Testing Azure Blob Storage...")
    blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service.get_container_client(CONTAINER_NAME)

    blobs = list(container_client.list_blobs())
    print(f"✅ Found {len(blobs)} blobs in '{CONTAINER_NAME}'")
    for b in blobs[:3]:
        print(" -", b.name)

# --------------------------
# Test MQTT
# --------------------------
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

# --------------------------
# Test Local Model


def test_local_model():
    model_file = "xgboost_pipeline.pkl"  # directly reference the file
    if not os.path.exists(model_file):
        print(f"\n⚠️ Local model file not found: {model_file}")
        return
    
    print("\n🧠 Testing local model...")
    try:
        model = joblib.load(model_file)
        # Example input — must match feature order used in training
        sample = [[37, 0.2, 1, 1200, 2, 0, 101.5, 1.1]]
        pred = model.predict(sample)
        print("✅ Local model prediction:", pred)
    except Exception as e:
        print("❌ Error loading model:", e)

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    test_ml_and_store()
    test_blob_storage()
    test_local_model()
    test_mqtt()
