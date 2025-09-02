"""
Manual, single-shot test:
1) Start MQTT subscriber (same topic).
2) After SUBACK, publish exactly one message on the same topic.
3) Receive that message once.
4) Send it to Azure ML Studio endpoint.
5) Apply rule-based fallback ONLY if AML result is missing/low confidence.
6) Store the final result JSON to Azure Blob Storage.
7) Exit cleanly (KeyboardInterrupt prints a friendly message).
"""

import os
import sys
import json
import logging
from datetime import datetime, UTC

from dotenv import load_dotenv

# Make package imports work when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.mqtt_service import MQTTService
from services.azure_ml_client import AzureMLClient
from services.blob_storage import BlobStorage
from services.rule_fallback import RuleBasedAssessor

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("workflows.test_services")

# ------------------ ENV ------------------
load_dotenv()

# MQTT / HiveMQ Cloud
HIVEMQ_HOST = "27cccf2df0a94906a26cbbabbd9afe51.s1.eu.hivemq.cloud" 
HIVEMQ_PORT = 8883
HIVEMQ_USERNAME = "Ganessh7114"
HIVEMQ_PASSWORD = "Ganessh@2004"
TOPIC = "iot/devices" # you said use same topic

if not HIVEMQ_HOST or not HIVEMQ_USERNAME:
    raise SystemExit("Missing MQTT config. Set HIVEMQ_HOST, HIVEMQ_USERNAME, HIVEMQ_PASSWORD in .env")

# Azure ML + Blob
# (AzureMLClient reads its own env; BlobStorage reads its own env)
aml_client = AzureMLClient()
blob = BlobStorage()
rules = RuleBasedAssessor()

mqtt = MQTTService(
    host=HIVEMQ_HOST,
    port=HIVEMQ_PORT,
    username=HIVEMQ_USERNAME,
    password=HIVEMQ_PASSWORD,
    tls=True,
    client_id_prefix="iot-one-shot"
)

# ------------------ Sample telemetry ------------------
telemetry = {
    "DeviceType": "CT Scanner",
    "DeviceName": "GE Revolution",
    "RuntimeHours": 1400,
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
    "ErrorLogsCount": 2,
    "SentTimestamp": datetime.now(UTC).isoformat(),
}

def main():
    log.info("Starting one-shot MQTT -> Azure ML -> rules fallback -> Blob flow")
    log.info("Broker: %s:%s  Topic: '%s'", HIVEMQ_HOST, HIVEMQ_PORT, TOPIC)

    # Step 1 & 2: set up subscriber; after SUBACK, publish exactly once
    def do_publish_once():
        mqtt.publish_once(TOPIC, telemetry, qos=1, retain=False)

    received = mqtt.receive_once(topic=TOPIC, timeout=30.0, on_ready=do_publish_once)
    if received is None:
        raise SystemExit("Did not receive any MQTT payload within timeout")

    # Step 3: got the message -> Step 4: Azure ML prediction
    log.info("Invoking Azure ML for prediction...")
    aml = aml_client.predict(received)
    log.info("Azure ML response parsed -> ok=%s label=%s conf=%s model=%s",
             aml.get("ok"), aml.get("label"), aml.get("confidence"), aml.get("model_version"))

    # Step 5: Rule fallback ONLY if AML is missing/low confidence
    final_pred = rules.refine_prediction(
        telemetry_row=received,
        aml_label=aml.get("label"),
        aml_conf=aml.get("confidence"),
    )

    # Prepare record to store
    record = {
        "telemetry": received,
        "azure_ml": {
            "ok": aml.get("ok"),
            "label": aml.get("label"),
            "confidence": aml.get("confidence"),
            "model_version": aml.get("model_version"),
            "raw": aml.get("raw"),     # store full raw for audit/debug
            "error": aml.get("error"),
        },
        "final": final_pred,
        "timestamp": datetime.now(UTC).isoformat(),
        "pipeline": "mqtt_one_shot_aml_rules_blob",
    }

    # Step 6: store to blob (after rule-fix)
    blob_name = blob.upload_json("prediction", record)
    log.info("Stored final result to blob: %s", blob_name)

    # Also show on console
    print("\n=== FINAL RESULT (stored) ===")
    print(json.dumps(record, indent=2, default=str))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("🛑 Ended by user")
    except Exception as e:
        log.exception("Flow failed: %s", e)
        raise
