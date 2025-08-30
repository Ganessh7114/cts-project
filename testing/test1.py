import os
import json
import requests
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
WORKSPACE = os.getenv("AZURE_ML_WORKSPACE")

TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

ENDPOINT_URL = os.getenv("AZURE_ML_ENDPOINT_URL")
API_KEY = os.getenv("AZURE_ML_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_ML_DEPLOYMENT_NAME")

BLOB_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

# --------------------------
# Authentication
# --------------------------
def get_auth_headers():
    if API_KEY:
        print("🔑 Using API Key authentication")
        return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    else:
        print("🔑 Using Service Principal authentication")
        cred = ClientSecretCredential(tenant_id=TENANT_ID, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        token = cred.get_token("https://ml.azure.com/.default")
        return {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}

# --------------------------
# Test ML Endpoint
# --------------------------
def test_ml_endpoint():
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

    try:
        resp = requests.post(ENDPOINT_URL, headers=headers, json=sample_input)
        print("✅ Status:", resp.status_code)
        print("Response:", resp.json())
    except Exception as e:
        print("❌ Error calling endpoint:", e)

# --------------------------
# Test Blob Storage
# --------------------------
def test_blob_storage():
    print("\n📦 Testing Azure Blob Storage...")
    try:
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client = blob_service.get_container_client(CONTAINER_NAME)

        blobs = list(container_client.list_blobs())
        print(f"✅ Found {len(blobs)} blobs in container '{CONTAINER_NAME}'")
        if blobs:
            first_blob = blobs[0].name
            print("Downloading first blob:", first_blob)
            blob_client = container_client.get_blob_client(first_blob)
            data = blob_client.download_blob().readall()
            print("First 500 chars:\n", data.decode("utf-8")[:500])
    except Exception as e:
        print("❌ Error accessing blob storage:", e)

def list_containers():
    print("\n📂 Listing all containers in the storage account...")
    try:
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        containers = blob_service.list_containers()
        for c in containers:
            print("-", c["name"])
    except Exception as e:
        print("❌ Error listing containers:", e)


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    test_ml_endpoint()
    list_containers()
    test_blob_storage()
