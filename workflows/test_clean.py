# workflows/test_clean.py
import os
import sys
import json
import asyncio
import logging
from dotenv import load_dotenv
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.mqtt_service import MQTTService
from services.azure_ml_client import AzureMLClient
from services.local_model import LocalModel
from services.blob_storage import BlobStorage
from services.rule_fallback import RuleBasedAssessor

from presentation.logging_config import setup_logging
from presentation.presenter import present_step, present_final

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()
log_mode = setup_logging()
logger = logging.getLogger(__name__)

USE_DUMMY_MQTT = os.getenv("USE_DUMMY_MQTT", "false").lower() == "true"

HIVEMQ_HOST = os.getenv("HIVEMQ_HOST")
HIVEMQ_PORT = int(os.getenv("HIVEMQ_PORT", "8883"))
HIVEMQ_USERNAME = os.getenv("HIVEMQ_USERNAME")
HIVEMQ_PASSWORD = os.getenv("HIVEMQ_PASSWORD")
TOPIC = os.getenv("TOPIC")

azure_client = AzureMLClient()
local_model = LocalModel()
blob = BlobStorage()
rule_assessor = RuleBasedAssessor()

mqtt = MQTTService(
    host=HIVEMQ_HOST,
    port=HIVEMQ_PORT,
    username=HIVEMQ_USERNAME,
    password=HIVEMQ_PASSWORD,
    tls=True,
    client_id_prefix="iot-one-shot"
)

# ---------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------
async def run_once():
    present_step("Starting pipeline", log_mode=log_mode)

    telemetry = None
    if USE_DUMMY_MQTT:
        # Dummy test payload
        dummy_payload = {
            "DeviceType": "CT Scanner",
            "DeviceName": "GE Revolution",
            "RuntimeHours": 9400,
            "TemperatureC": 45.5,
            "PressureKPa": 201.3,
            "VibrationMM_S": 1.02,
            "CurrentDrawA": 13.4,
            "SignalNoiseLevel": 0.1,
            "ClimateControl": "Yes",
            "HumidityPercent": 45,
            "Location": "Hospital A - Central Region",
            "OperationalCycles": 340,
            "UserInteractionsPerDay": 15,
            "ApproxDeviceAgeYears": 3,
            "NumRepairs": 1,
            "ErrorLogsCount": 2,
            "SentTimestamp": datetime.now(timezone.utc).isoformat(),
        }

        present_step("Waiting for telemetry message", log_mode=log_mode)
        payload = mqtt.receive_once(
            TOPIC,
            on_ready=lambda: mqtt.publish_once(TOPIC, dummy_payload)
        )
        telemetry = payload
    else:
        present_step("Waiting for telemetry message", log_mode=log_mode)
        payload = mqtt.receive_once(TOPIC)
        telemetry = payload

    if not telemetry:
        logger.error("No telemetry received within timeout")
        return

    present_step("Telemetry received", log_mode=log_mode)

    # --- Azure ML ----------------------------------------------------
    present_step("Invoking cloud AI model", log_mode=log_mode)
    aml_result = azure_client.predict(telemetry)


    # --- Local model -------------------------------------------------
    present_step("Evaluating with local model", log_mode=log_mode)
    local_result = local_model.predict(telemetry)

    # --- Rules fallback ----------------------------------------------
    present_step("Applying fallback rules", log_mode=log_mode)
    refined = rule_assessor.refine_prediction(
        telemetry,
        aml_result.get("label"),
        aml_result.get("confidence"),
    )

    final = {
        "label": refined.get("label"),
        "confidence": refined.get("confidence"),
        "factors": refined.get("factors"),
    }
    present_step("AI decision computed", log_mode=log_mode)

    record = {
        "telemetry": telemetry,
        "azure_ml": aml_result,
        "local_model": local_result,
        "final": final,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": "mqtt_one_shot_aml_local_rules_blob",
    }

    filename = "prediction"
    blob.upload_json(filename, record)
    present_step("Uploaded prediction to blob storage", log_mode=log_mode)

    present_final(record, log_mode=log_mode)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(run_once())
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down.")
