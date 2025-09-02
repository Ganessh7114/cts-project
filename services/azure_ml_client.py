import os
import json
import logging
from typing import Any, Dict, Optional
import requests

logger = logging.getLogger("services.azure_ml_client")

class AzureMLClient:
    """
    Thin REST client for Azure ML online endpoint.
    .env (any of these aliases):
      - AZURE_ML_ENDPOINT_URL  | ENDPOINT_URL
      - AZURE_ML_API_KEY       | API_KEY
      - AZURE_ML_DEPLOYMENT_NAME | DEPLOYMENT_NAME  (adds 'azureml-model-deployment' header)
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        timeout_sec: float = 20.0,
    ):
        self.endpoint_url = endpoint_url or os.getenv("AZURE_ML_ENDPOINT_URL") or os.getenv("ENDPOINT_URL")
        self.api_key = api_key or os.getenv("AZURE_ML_API_KEY") or os.getenv("API_KEY")
        self.deployment_name = deployment_name or os.getenv("AZURE_ML_DEPLOYMENT_NAME") or os.getenv("DEPLOYMENT_NAME")
        self.timeout_sec = timeout_sec

        if not self.endpoint_url:
            raise ValueError("Azure ML endpoint URL is missing. Set AZURE_ML_ENDPOINT_URL in .env")

        logger.info("AzureMLClient initialized for endpoint: %s", self.endpoint_url)

    def _make_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.deployment_name:
            headers["azureml-model-deployment"] = self.deployment_name
        return headers

    def predict(self, telemetry_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a single telemetry row using a shape that your score.py accepts.
        We send under 'data' (score.py also accepts 'instances', 'records', 'input_data.data').
        Returns:
          {
            ok, raw, label, confidence, model_version
          }
        """
        payload = {"data": [telemetry_row]}
        headers = self._make_headers()

        try:
            resp = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            raw = resp.json()

            # Handle score.py "success": False cases (HTTP 200 with an error in body)
            if isinstance(raw, dict) and raw.get("success") is False:
                return {"ok": False, "error": raw.get("error"), "raw": raw,
                        "label": None, "confidence": None, "model_version": raw.get("model_version")}

            # Try to parse common shapes
            label = None
            conf = None
            model_ver = None

            if isinstance(raw, dict):
                preds = None
                if "predictions" in raw and isinstance(raw["predictions"], list) and raw["predictions"]:
                    preds = raw["predictions"]
                elif isinstance(raw.get("output"), list) and raw["output"]:
                    preds = raw["output"]
                elif "result" in raw and isinstance(raw["result"], dict):
                    preds = [raw["result"]]

                if preds:
                    p0 = preds[0]
                    label = p0.get("prediction") or p0.get("label") or p0.get("class")
                    conf = p0.get("confidence") or p0.get("score") or p0.get("probability")
                    model_ver = p0.get("model_version") or raw.get("model_version")

            return {"ok": True, "raw": raw, "label": label, "confidence": conf, "model_version": model_ver}

        except Exception as e:
            logger.exception("Azure ML prediction failed: %s", e)
            return {"ok": False, "error": str(e), "raw": None, "label": None, "confidence": None, "model_version": None}
