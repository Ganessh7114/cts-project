import os
import json
import requests
import paho.mqtt.client as mqtt
from datetime import datetime, UTC
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# ---------------- Load environment variables ----------------
load_dotenv()

ENDPOINT_URL = os.getenv("AZURE_ML_ENDPOINT_URL")
API_KEY = os.getenv("AZURE_ML_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_ML_DEPLOYMENT_NAME")

BLOB_CONN_STR = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

BROKER = "27cccf2df0a94906a26cbbabbd9afe51.s1.eu.hivemq.cloud"   # replace
PORT = 8883
USERNAME = "Ganessh7114"
PASSWORD = "Ganessh@2004"
TOPIC = "iot/devices"


# ---------------- Rule-based fallback ----------------
def rule_based_prediction(data: dict):
    """Simple fallback if ML model fails"""
    factors = []
    risk_score = 0.0
    prediction = "Low"

    if data.get("TemperatureC", 0) > 37:
        prediction = "High"
        risk_score += 0.2
        factors.append("High temperature")

    if data.get("VibrationMM_S", 0) > 0.05:
        prediction = "High"
        risk_score += 0.2
        factors.append("High vibration")

    return {
        "device_name": data.get("DeviceName", "Unknown"),
        "prediction": prediction,
        "confidence": 0.7,
        "risk_score": risk_score,
        "factors": factors or ["All parameters within normal ranges"],
        "model_version": "rule-fallback",
        "timestamp": str(datetime.now(UTC))
    }


# ---------------- Call Azure ML Endpoint ----------------
def call_azure_ml(payload: dict):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        resp = requests.post(ENDPOINT_URL, headers=headers, json={"data": [payload]}, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        return result
    except Exception as e:
        print("❌ Azure ML call failed:", e)
        return None


# ---------------- Store result in Blob ----------------
def store_result(result: dict):
    try:
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
        container = blob_service.get_container_client(CONTAINER_NAME)
        filename = f"prediction_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        container.upload_blob(name=filename, data=json.dumps(result), overwrite=True)
        print(f"✅ Stored result in blob: {filename}")
    except Exception as e:
        print("❌ Failed to store result in blob:", e)


# ---------------- MQTT Handlers ----------------
def on_connect(client, userdata, flags, rc, properties=None):
    print("✅ MQTT Connected with result code", rc)
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    print(f"📨 Received message on {msg.topic}")
    try:
        data = json.loads(msg.payload.decode())
        print("📦 Payload:", data)

        # Send to Azure ML
        ml_result = call_azure_ml(data)
        if ml_result and ml_result.get("success", False) and ml_result.get("model_loaded", False):
            print("✅ Azure ML prediction:", ml_result)
            store_result(ml_result)
        else:
            print("⚠️ Falling back to rule-based prediction")
            fallback = rule_based_prediction(data)
            store_result(fallback)

    except Exception as e:
        print("❌ Error processing message:", e)


# ---------------- Main ----------------
def main():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"🚀 Connecting to HiveMQ broker {BROKER}:{PORT} and listening on topic '{TOPIC}'...")
    client.connect(BROKER, PORT, 60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("🛑 Stopped by user")


if __name__ == "__main__":
    main()
