from services.mqtt_consumer import receive_message_once
from services.azure_ml_service import call_azure_ml
from services.blob_service import save_to_blob
from services.rule_fallback import rule_based_prediction

if __name__ == "__main__":
    print("📡 Waiting for device telemetry...")
    data = receive_message_once()
    print("✅ Received telemetry:", data)

    try:
        print("🚀 Sending telemetry to Azure ML...")
        prediction = call_azure_ml({"data": [data]})
        print("✅ Prediction from Azure ML:", prediction)
    except Exception as e:
        print(f"❌ Azure ML call failed: {e}")
        prediction = {"predictions": [rule_based_prediction(data)]}
        print("⚡ Using rule-based fallback:", prediction)

    print("💾 Saving result to Azure Blob Storage...")
    filename = save_to_blob(prediction)
    print(f"✅ Prediction saved as '{filename}' in Blob Storage")
