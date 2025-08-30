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

# ---------------- Rule-Based Fallback ----------------
# ---------------- Rule-Based Fallback ----------------
def rule_based_fallback(input_data):
    """Simple rule-based prediction for fallback."""
    try:
        device = input_data[0]  # take first record
        risk_factors = []

        # Example simple checks
        if device.get("TemperatureC", 0) > 37:
            risk_factors.append("High temperature")
        if device.get("VibrationMM_S", 0) > 0.05:
            risk_factors.append("High vibration")
        if device.get("ErrorLogsCount", 0) > 5:
            risk_factors.append("Frequent errors")

        prediction = "High" if risk_factors else "Low"
        return {
            "device_name": device.get("DeviceName", "Unknown"),
            "prediction": prediction,
            "confidence": 0.7 if prediction == "High" else 0.5,
            "risk_score": len(risk_factors) * 0.2,
            "factors": risk_factors or ["All parameters within typical ranges"],
            "model_version": "rule-fallback",
            "timestamp": str(datetime.now(UTC))
        }
    except Exception as e:
        return {"error": f"Fallback failed: {e}"}


# ---------------- Test Azure ML + Blob ----------------
def test_ml_and_store():
    print("\n🚀 Testing Azure ML Endpoint...")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        resp = requests.post(ENDPOINT_URL, headers=headers, json=sample_input)
        resp.raise_for_status()
        ml_result = resp.json()
        print("✅ ML Response:", ml_result)

        # Always run rule-based fallback as well
        fallback_result = rule_based_fallback(sample_input["data"])

        combined_result = {
            "azure_ml_result": ml_result,
            "rule_based_result": fallback_result
        }

        # Save result to blob
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
        container = blob_service.get_container_client(CONTAINER_NAME)

        filename = f"prediction_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        container.upload_blob(name=filename, data=json.dumps(combined_result), overwrite=True)
        print(f"✅ Stored combined prediction in blob: {filename}")

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
        ml_local_result = {
            "prediction": preds.tolist(),
            "model_version": "local-pkl"
        }
        print("✅ Local model prediction:", ml_local_result["prediction"])
    except Exception as e:
        print("❌ Error loading local model:", e)
        ml_local_result = {"error": str(e), "model_version": "local-pkl-failed"}

    # Always run rule-based fallback
    fallback_result = rule_based_fallback(sample_input["data"])

    combined_result = {
        "local_model_result": ml_local_result,
        "rule_based_result": fallback_result
    }

    # Print combined
    print("📊 Combined local results:", json.dumps(combined_result, indent=2))
    return combined_result


# ---------------- Test MQTT (with ML + Blob pipeline) ----------------
def test_mqtt():
    print("\n📡 Testing MQTT pipeline...")

    def on_connect(client, userdata, flags, rc, properties=None):
        print("✅ MQTT Connected with result code", rc)
        client.subscribe(TOPIC)
        # For testing, publish a sample input
        client.publish(TOPIC, json.dumps(sample_input["data"][0]))

    def on_message(client, userdata, msg):
        print("📨 Received MQTT message:", msg.topic, msg.payload.decode())
        try:
            # Parse incoming MQTT message as JSON
            incoming_data = json.loads(msg.payload.decode())
            ml_input = {"data": [incoming_data]}  # wrap to match schema

            # ---- Call Azure ML Endpoint ----
            headers = {"Content-Type": "application/json"}
            if API_KEY:
                headers["Authorization"] = f"Bearer {API_KEY}"
            resp = requests.post(ENDPOINT_URL, headers=headers, json=ml_input)
            resp.raise_for_status()
            result = resp.json()
            print("✅ ML Studio Prediction:", result)

            # ---- Save to Blob Storage ----
            blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
            container = blob_service.get_container_client(CONTAINER_NAME)
            filename = f"prediction_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            container.upload_blob(name=filename, data=json.dumps(result), overwrite=True)
            print(f"✅ Stored prediction in blob: {filename}")

        except Exception as e:
            print("❌ Error in MQTT→ML→Blob pipeline:", e)
        finally:
            client.disconnect()  # stop after one cycle for testing

    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()  # HiveMQ Cloud requires TLS
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("🛑 MQTT test ended by user")
        client.disconnect()
    except Exception as e:
        print("❌ Error in MQTT test:", e)

# ---------------- Run All ----------------
if __name__ == "__main__":
    test_ml_and_store()
    test_blob()
    test_local_model()
    try:
        test_mqtt()
    except KeyboardInterrupt:
        print("\n🛑 Program ended by user.")
