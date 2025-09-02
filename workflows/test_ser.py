"""
Manual, single-shot test:
1) Start MQTT subscriber (same topic).
2) After SUBACK, publish exactly one message on the same topic (with a tiny sleep).
3) Receive that message once.
4) Send it to Azure ML endpoint.
5) Apply rule-based fallback ONLY if AML result is missing/low confidence.
6) Score with local .pkl model.
7) Pick the best prediction (highest confidence) WITHOUT revealing its source in 'final'.
8) Store the final result JSON to Azure Blob Storage.
9) Exit cleanly (KeyboardInterrupt prints a friendly message).
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, UTC

from dotenv import load_dotenv

# Make package imports work when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.mqtt_service import MQTTService
from services.azure_ml_client import AzureMLClient
from services.local_model import LocalModel
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
TOPIC ="iot/devices" # single topic for pub+sub

if not all([HIVEMQ_HOST, HIVEMQ_USERNAME, HIVEMQ_PASSWORD]):
    raise SystemExit("Missing MQTT config. Set HIVEMQ_HOST, HIVEMQ_USERNAME, HIVEMQ_PASSWORD in .env")

aml_client = AzureMLClient()   # reads its own env
local_model = LocalModel()     # uses LOCAL_MODEL_PATH or default xgboost_pipeline.pkl
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

def pick_best(aml_refined: dict, local_res: dict, rules_res: dict) -> dict:
    """
    Decide final output without exposing the source.
    Preference: higher confidence among valid candidates.
    If both missing confidence, prefer AML-refined, else local, else rules.
    """
    candidates = []

    if aml_refined and aml_refined.get("label") is not None:
        candidates.append(("aml", aml_refined.get("label"), aml_refined.get("confidence"), aml_refined.get("factors", [])))
    if local_res and local_res.get("ok") and local_res.get("label") is not None:
        candidates.append(("local", local_res.get("label"), local_res.get("confidence"), []))
    if not candidates and rules_res and rules_res.get("label") is not None:
        candidates.append(("rules", rules_res.get("label"), rules_res.get("confidence"), rules_res.get("factors", [])))

    if not candidates:
        # extreme safety fallback
        return {"label": None, "confidence": None, "factors": []}

    # choose by confidence (None is treated as -1)
    best = max(candidates, key=lambda t: (t[2] if isinstance(t[2], (int, float)) else -1.0))
    _, label, conf, factors = best
    return {"label": label, "confidence": conf, "factors": factors}

def main():
    log.info("Starting one-shot MQTT -> Azure ML -> local -> rules -> Blob flow")
    log.info("Broker: %s:%s  Topic: '%s'", HIVEMQ_HOST, HIVEMQ_PORT, TOPIC)

    # Step 1 & 2: set up subscriber; after SUBACK, publish exactly once (with small sleep)
    def do_publish_once():
        time.sleep(1.0)  # extra guard even though mqtt_service also sleeps after SUBACK
        mqtt.publish_once(TOPIC, telemetry, qos=1, retain=False, timeout=15.0)

    received = mqtt.receive_once(topic=TOPIC, timeout=30.0, on_ready=do_publish_once)
    if received is None:
        raise SystemExit("Did not receive any MQTT payload within timeout")

    # Step 3/4: Azure ML prediction
    log.info("Invoking Azure ML for prediction...")
    aml = aml_client.predict(received)
    log.info("Azure ML response parsed -> ok=%s label=%s conf=%s model=%s",
             aml.get("ok"), aml.get("label"), aml.get("confidence"), aml.get("model_version"))

    # Step 5: Apply rule fallback ONLY to AML results if missing/low confidence
    aml_refined = rules.refine_prediction(
        telemetry_row=received,
        aml_label=aml.get("label") if aml.get("ok") else None,
        aml_conf=aml.get("confidence") if aml.get("ok") else None,
    )

    # Step 6: Local model prediction
    local_res = local_model.predict(received)

    # Optional: direct rules if both AML+local fail hard
    rules_res = rules.direct(received)

    # Step 7: choose best WITHOUT revealing source
    final_pred = pick_best(aml_refined, local_res, rules_res)

    # Prepare record to store (keep detailed sections for audit; 'final' hides source)
    record = {
        "telemetry": received,
        "azure_ml": {
            "ok": aml.get("ok"),
            "label": aml.get("label"),
            "confidence": aml.get("confidence"),
            "model_version": aml.get("model_version"),
            "raw": aml.get("raw"),
            "error": aml.get("error"),
            "refined": aml_refined,  # shows whether rules were applied to AML
        },
        "local_model": {
            "ok": local_res.get("ok"),
            "label": local_res.get("label"),
            "confidence": local_res.get("confidence"),
            "model_version": local_res.get("model_version"),
            "probs": local_res.get("probs", {}),
            "error": local_res.get("error"),
        },
        "final": final_pred,  # no source field here
        "timestamp": datetime.now(UTC).isoformat(),
        "pipeline": "mqtt_one_shot_aml_local_rules_blob",
    }

    # Step 8: store to blob
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
