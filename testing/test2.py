import os
import json
from datetime import datetime
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Load .env
load_dotenv()
BLOB_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

def upload_prediction(result: dict):
    blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service.get_container_client(CONTAINER_NAME)

    # filename with timestamp
    filename = f"prediction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    blob_client = container_client.get_blob_client(filename)
    blob_client.upload_blob(json.dumps(result, indent=2), overwrite=True)

    print(f"✅ Uploaded prediction to blob: {filename}")

# Example usage after calling ML endpoint
if __name__ == "__main__":
    # Simulating your ML result (replace with real response JSON)
    fake_result = {
        "prediction": "Low",
        "confidence": 0.75,
        "timestamp": str(datetime.utcnow())
    }
    upload_prediction(fake_result)
