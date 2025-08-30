import os
import json
import csv
import time
import ssl
import threading
import logging
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

import paho.mqtt.client as mqtt
import requests
from azure.storage.blob import BlobServiceClient

# -----------------------------------------------------
# Configuration (env-driven; safe defaults preserved)
# -----------------------------------------------------


load_dotenv()

HIVEMQ_HOST = os.getenv("HIVEMQ_HOST", "localhost")
HIVEMQ_PORT = int(os.getenv("HIVEMQ_PORT", "8883"))
HIVEMQ_USERNAME = os.getenv("HIVEMQ_USERNAME")
HIVEMQ_PASSWORD = os.getenv("HIVEMQ_PASSWORD")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "iot/failure")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "medicaldevicestorage")

AML_ENDPOINT_URL = os.getenv("AML_ENDPOINT_URL")              # e.g., https://...azureml.net/score
AML_API_KEY = os.getenv("AML_API_KEY")

OUTPUT_CSV = "predictions_output.csv"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
PUBLISH_COUNT = int(os.getenv("PUBLISH_COUNT", "100"))
ONE_SHOT = os.getenv("ONE_SHOT", "0") == "1"

# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("MQTT_ML_Streaming")

# -----------------------------------------------------
# Azure Blob client
# -----------------------------------------------------
blob_service: Optional[BlobServiceClient] = None
if AZURE_STORAGE_CONNECTION_STRING:
    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        log.info("Azure Blob Storage client initialized successfully")
    except Exception as e:
        log.error("Failed to init Azure Blob client: %s", e)
else:
    log.warning("AZURE_STORAGE_CONNECTION_STRING not set; blob uploads will be skipped")

def upload_csv_to_blob(local_path: str) -> None:
    if not blob_service:
        log.warning("Blob client unavailable; skipping upload for %s", local_path)
        return
    try:
        container_client = blob_service.get_container_client(AZURE_STORAGE_CONTAINER)
        # Blob name follows your existing pattern
        blob_name = f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        with open(local_path, "rb") as f:
            container_client.upload_blob(name=blob_name, data=f, overwrite=False)
        log.info("Uploaded '%s' to Azure Blob Storage", blob_name)
    except Exception as e:
        log.error("Blob upload failed: %s", e)

# -----------------------------------------------------
# Azure ML Online Endpoint
# -----------------------------------------------------
def azure_ml_predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends one JSON example to your Azure ML online endpoint.
    Returns a normalized dict with keys:
       prediction_label, confidence, risk_score, model_version, raw
    Raises on network/HTTP failures so caller can fallback to rules.
    """
    if not AML_ENDPOINT_URL or not AML_API_KEY:
        raise RuntimeError("AML endpoint not configured (set AML_ENDPOINT_URL and AML_API_KEY)")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AML_API_KEY}",
    }

    # Many AML deployments accept {"input_data": [ { your fields } ]}
    # If your deployment expects another schema, adapt here.
    body = {"input_data": [payload]}

    resp = requests.post(AML_ENDPOINT_URL, headers=headers, json=body, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    # Try to be liberal in what we accept (your earlier endpoint returned this shape):
    # {"success": true, "model_version":"...", "predictions":[ { "prediction":"Low","confidence":0.75,... } ]}
    pred_label = None
    confidence = None
    risk_score = None
    model_version = None

    if isinstance(data, dict):
        model_version = data.get("model_version")
        # Case A: object list with rich prediction
        if "predictions" in data and isinstance(data["predictions"], list) and data["predictions"]:
            first = data["predictions"][0]
            if isinstance(first, dict):
                pred_label = first.get("prediction") or first.get("label") or first.get("risk")
                confidence = first.get("confidence")
                risk_score = first.get("risk_score")
                model_version = first.get("model_version", model_version)
            else:
                pred_label = first
        # Case B: simple "predictions": [0/1/2] or ["Low"]
        elif "predictions" in data:
            preds = data["predictions"]
            if isinstance(preds, list) and preds:
                pred_label = preds[0]
    elif isinstance(data, list) and data:
        pred_label = data[0]

    # As a last resort, try direct number to class mapping
    if isinstance(pred_label, (int, float)):
        # Map 0/1/2 -> Low/Medium/High (keeps your old semantics)
        pred_label = {0: "Low", 1: "Medium", 2: "High"}.get(int(pred_label), str(pred_label))

    if not pred_label:
        # If endpoint answered but didn’t include a usable prediction, force fallback by raising
        raise ValueError(f"AML response missing usable prediction: {data}")

    return {
        "prediction_label": pred_label,
        "confidence": confidence,
        "risk_score": risk_score,
        "model_version": model_version or "aml-endpoint",
        "raw": data,
    }

# -----------------------------------------------------
# Rule-based fallback (kept simple and interpretable)
# -----------------------------------------------------
def rule_based_predict(d: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    t = float(d.get("TemperatureC", 0))
    vib = float(d.get("VibrationMM_S", 0))
    errs = int(d.get("ErrorLogsCount", 0))
    hum = float(d.get("HumidityPercent", 0))
    age = float(d.get("ApproxDeviceAgeYears", 0))
    repairs = int(d.get("NumRepairs", 0))

    # Temperature
    if t > 40: score += 2
    elif t > 37.5: score += 1

    # Vibration
    if vib > 0.05: score += 2
    elif vib > 0.02: score += 1

    # Errors and maintenance
    if errs >= 5: score += 2
    elif errs >= 2: score += 1

    if repairs >= 3: score += 1
    if age > 7: score += 1
    if hum > 70: score += 1

    if score >= 4:
        label = "High"; cls = 2
    elif score >= 2:
        label = "Medium"; cls = 1
    else:
        label = "Low"; cls = 0

    conf = min(0.95, 0.55 + 0.1 * score)  # heuristic
    return {
        "prediction_label": label,
        "class": cls,
        "risk_score": round(score / 7.0, 3),
        "confidence": round(conf, 3),
        "model_version": "rules-fallback",
        "raw": {"score": score},
    }

# -----------------------------------------------------
# CSV buffering & flush
# -----------------------------------------------------
_buffer_rows: List[Dict[str, Any]] = []

def _ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

def flush_buffer() -> None:
    if not _buffer_rows:
        return
    fieldnames = sorted({k for row in _buffer_rows for k in row.keys()})
    _ensure_csv_header(OUTPUT_CSV, fieldnames)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for row in _buffer_rows:
            w.writerow(row)
    log.info("Saved %d records to %s", len(_buffer_rows), OUTPUT_CSV)
    upload_csv_to_blob(OUTPUT_CSV)
    _buffer_rows.clear()

# -----------------------------------------------------
# Message processing
# -----------------------------------------------------
def process_one_reading(reading: Dict[str, Any]) -> None:
    """Send to AML; fallback to rules; add to batch & flush in chunks."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        aml = azure_ml_predict(reading)
        source = "azure-ml"
        pred = aml
    except Exception as e:
        log.warning("AML call failed → using rule-based fallback: %s", e)
        source = "rules"
        pred = rule_based_predict(reading)

    record = dict(reading)  # include all original telemetry
    record.update({
        "PredictedRisk": pred["prediction_label"],
        "RiskScore": pred.get("risk_score"),
        "Confidence": pred.get("confidence"),
        "ModelVersion": pred.get("model_version"),
        "Source": source,
        "ProcessedTimestamp": ts,
    })

    _buffer_rows.append(record)
    if len(_buffer_rows) >= BATCH_SIZE:
        flush_buffer()

# -----------------------------------------------------
# MQTT subscriber
# -----------------------------------------------------
def mqtt_subscribe(stop_event: threading.Event) -> None:
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            log.info("Connected to MQTT broker")
            client.subscribe(MQTT_TOPIC)
        else:
            # rc=5 is "Not authorized" on HiveMQ — credentials/topic ACLs
            log.error("MQTT connect failed with rc=%s", rc)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            process_one_reading(payload)
        except Exception as e:
            log.error("Failed to handle MQTT message: %s", e)

    # Keep the (older) API style to avoid breaking anything in your env
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.username_pw_set(HIVEMQ_USERNAME, HIVEMQ_PASSWORD)
    # HiveMQ Cloud requires TLS
    try:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    except Exception:
        # If local/unauth broker, TLS may not be needed
        pass

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(HIVEMQ_HOST, HIVEMQ_PORT, keepalive=60)
    client.loop_start()

    while not stop_event.is_set():
        time.sleep(0.2)

    client.loop_stop()
    client.disconnect()

# -----------------------------------------------------
# MQTT publisher (simulated data)
# -----------------------------------------------------
DEVICE_TYPES = [
    "CT Scanner", "ECG Monitor", "Patient Ventilator", "Infusion Pump",
    "Defibrillator", "Ultrasound Machine", "Dialysis Machine", "Anesthesia Machine"
]
BRANDS = [
    "GE Revolution", "GE Logiq E9", "Philips EPIQ", "Philips HeartStrart",
    "Siemens S2000", "Drager V500", "Baxter Flo-Gard", "Smiths Medfusion",
    "Datex Ohmeda S5", "NxStage System One", "Hamilton G5", "Alaris GH", "Lifepak 20", "GE MAC 2000"
]

def gen_sample() -> Dict[str, Any]:
    dev = random.choice(DEVICE_TYPES)
    name = random.choice(BRANDS)
    return {
        "DeviceType": dev,
        "DeviceName": name,
        "RuntimeHours": random.randint(50, 5000),
        "TemperatureC": round(random.uniform(34, 44), 1),
        "PressureKPa": round(random.uniform(98, 105), 1),
        "VibrationMM_S": round(random.uniform(0.0, 0.07), 3),
        "CurrentDrawA": round(random.uniform(0.5, 6.5), 2),
        "SignalNoiseLevel": round(random.uniform(0.0, 0.5), 2),
        "ClimateControl": random.choice(["Yes", "No"]),
        "HumidityPercent": random.randint(20, 85),
        "Location": random.choice(["Hospital A - Central Region", "Hospital B - North", "Hospital C - South"]),
        "OperationalCycles": random.randint(10, 500),
        "UserInteractionsPerDay": random.randint(1, 40),
        "ApproxDeviceAgeYears": random.randint(0, 12),
        "NumRepairs": random.randint(0, 6),
        "ErrorLogsCount": random.randint(0, 7),
        "SentTimestamp": datetime.now(timezone.utc).isoformat(),
    }

def publish_simulated(total: int, stop_event: threading.Event) -> None:
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            log.info("MQTT publisher connected")
        else:
            log.error("MQTT publisher connect rc=%s", rc)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.username_pw_set(HIVEMQ_USERNAME, HIVEMQ_PASSWORD)
    try:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    except Exception:
        pass

    client.on_connect = on_connect
    client.connect(HIVEMQ_HOST, HIVEMQ_PORT, keepalive=60)
    client.loop_start()

    log.info("Starting IoT data publishing...")
    for i in range(total):
        if stop_event.is_set():
            break
        payload = gen_sample()
        client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
        # small pacing to avoid flood/quotas
        time.sleep(0.05)

    log.info("Finished publishing %d simulated records", total)
    client.loop_stop()
    client.disconnect()

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    log.info("Starting IoT ML Streaming Application (Azure ML + fallback rules)")

    stop_event = threading.Event()

    # Start subscriber first so it can receive what we publish
    sub_thr = threading.Thread(target=mqtt_subscribe, args=(stop_event,), daemon=True)
    sub_thr.start()
    log.info("MQTT subscriber started")

    # Optionally publish simulated data
    pub_thr = threading.Thread(target=publish_simulated, args=(1 if ONE_SHOT else PUBLISH_COUNT, stop_event), daemon=True)
    pub_thr.start()
    log.info("MQTT publisher started")

    try:
        if ONE_SHOT:
            # Wait briefly for a single message to arrive and be processed
            time.sleep(3.0)
            stop_event.set()
        else:
            # Run until publisher finishes; give subscriber time to drain
            pub_thr.join()
            time.sleep(2.0)
            stop_event.set()

        sub_thr.join()

        # Final flush (in case rows < BATCH_SIZE)
        flush_buffer()
        log.info("Stopped.")
    except KeyboardInterrupt:
        log.info("Stopping due to user interrupt...")
        stop_event.set()
        sub_thr.join(timeout=2)
        flush_buffer()

if __name__ == "__main__":
    main()
