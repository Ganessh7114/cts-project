import os
import json
import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError

logger = logging.getLogger("services.blob_storage")


class BlobStorage:
    """
    Minimal helper to upload JSON to Azure Blob.
    .env:
      - AZURE_STORAGE_CONNECTION_STRING
      - AZURE_BLOB_CONTAINER (default: medicaldevicestorage)
    """

    def __init__(self):
        conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not conn:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set")
        self.container = os.getenv("AZURE_BLOB_CONTAINER", "medicaldevicestorage")
        self.client = BlobServiceClient.from_connection_string(conn)
        self.container_client = self.client.get_container_client(self.container)
        try:
            self.container_client.create_container()
        except ResourceExistsError:
            pass
        logger.info("BlobStorage initialized (container: %s)", self.container)

    def upload_json(self, prefix: str, obj: Dict[str, Any]) -> str:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        name = f"{prefix}_{ts}.json"
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        self.container_client.upload_blob(
            name,
            data,
            overwrite=False,
            content_settings=ContentSettings(content_type="application/json"),
        )
        logger.info("Uploaded JSON to blob: %s", name)
        return name

    # -----------------------------------------------------------------
    # NEW METHODS (for retrieval)
    # -----------------------------------------------------------------
    def list_json_blobs(self, prefix: str = "prediction_", limit: int = 100) -> List:
        """
        List JSON blobs in the container (newest first).
        """
        blobs = []
        try:
            for blob in self.container_client.list_blobs(name_starts_with=prefix):
                if blob.name.endswith(".json"):
                    blobs.append(blob)
            blobs.sort(key=lambda b: b.last_modified, reverse=True)
            return blobs[:limit]
        except Exception as e:
            logger.error("Error listing blobs: %s", e)
            return []

    def download_json(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """
        Download and parse a JSON blob by name.
        Returns dict or None on error.
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            data = blob_client.download_blob().readall()
            return json.loads(data)
        except Exception as e:
            logger.error("Error downloading %s: %s", blob_name, e)
            return None
