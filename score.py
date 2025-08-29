# score.py
import os, json, logging, traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# joblib is in AzureML sklearn envs
from joblib import load as joblib_load

logger = logging.getLogger("azureml-inference")
logging.basicConfig(level=logging.INFO)

# -------- Globals set in init() --------
MODEL = None
MODEL_VERSION = None
MODEL_LABELS_GUESS = os.getenv("LABELS_ORDER", "High,Low,Medium").split(",")  # best guess for label encoder order
FEATURE_DEFAULTS = {
    "DeviceType": "Unknown",
    "DeviceName": "Unknown",
    "RuntimeHours": 0.0,
    "TemperatureC": 25.0,
    "PressureKPa": 100.0,
    "VibrationMM_S": 0.1,
    "CurrentDrawA": 1.0,
    "SignalNoiseLevel": 1.0,
    "ClimateControl": "Yes",
    "HumidityPercent": 50.0,
    "Location": "Unknown",
    "OperationalCycles": 1000,
    "UserInteractionsPerDay": 5.0,
    "LastServiceDate": datetime.utcnow().strftime("%d-%m-%Y"),
    "ApproxDeviceAgeYears": 1.0,
    "NumRepairs": 0,
    "ErrorLogsCount": 0,
}

NUMERIC_COLS = [
    "RuntimeHours","TemperatureC","PressureKPa","VibrationMM_S","CurrentDrawA",
    "SignalNoiseLevel","HumidityPercent","OperationalCycles","UserInteractionsPerDay",
    "ApproxDeviceAgeYears","NumRepairs","ErrorLogsCount"
]

CATEGORICAL_COLS = ["DeviceType","DeviceName","ClimateControl","Location"]

def _find_model_path() -> str:
    """Locate xgboost_pipeline.pkl in the mounted model dir or local working dir."""
    # AzureML sets AZUREML_MODEL_DIR to the mount point for the registered model
    candidates = []
    mdl_dir = os.getenv("AZUREML_MODEL_DIR")
    if mdl_dir and os.path.isdir(mdl_dir):
        for root, _, files in os.walk(mdl_dir):
            for f in files:
                if f.lower().endswith(".pkl"):
                    candidates.append(os.path.join(root, f))
    # local fallbacks
    for name in ["xgboost_pipeline.pkl", "model.pkl", "pipeline.pkl"]:
        if os.path.exists(name):
            candidates.append(os.path.abspath(name))
    # pick first matching pipeline file
    for path in candidates:
        if os.path.basename(path).lower() == "xgboost_pipeline.pkl":
            return path
    return candidates[0] if candidates else ""

def _json_safe(x: Any) -> Any:
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return [ _json_safe(v) for v in x.tolist() ]
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x

def _coerce_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing fields, cast types, and drop training-excluded columns."""
    clean = {}
    for k, default in FEATURE_DEFAULTS.items():
        v = rec.get(k, default)
        # allow mapping from camel/snake the UI might send
        if v is None:
            v = default
        if k in NUMERIC_COLS:
            try:
                # ints are acceptable for some fields
                clean[k] = float(v)
            except Exception:
                clean[k] = float(default)
        elif k == "OperationalCycles" or k == "NumRepairs" or k == "ErrorLogsCount":
            try:
                clean[k] = int(v)
            except Exception:
                clean[k] = int(default)
        else:
            clean[k] = str(v)
    # Normalize LastServiceDate like training (parsed then dropped)
    # Training converted to datetime and then removed from features.
    # We'll parse for robustness then drop before inference.
    try:
        _ = datetime.strptime(clean["LastServiceDate"], "%d-%m-%Y")
    except Exception:
        try:
            # try a few common formats
            _ = pd.to_datetime(clean["LastServiceDate"], errors="coerce")
        except Exception:
            pass
    return clean

def _build_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [_coerce_record(r) for r in items]
    df = pd.DataFrame(rows)
    # Drop LastServiceDate to mirror training pipeline (it was dropped in training)  # noqa
    if "LastServiceDate" in df.columns:
        df = df.drop(columns=["LastServiceDate"])
    return df

# --------- Fallback rule model (aligned with your streaming rules) ---------
def _fallback_score_one(r: Dict[str, Any]) -> Tuple[str, float, List[str], float]:
    """
    Returns (label, risk_score[0..1], factors, confidence[0..1])
    Heuristic mirrors Hive_streaming.py rules for consistent behavior.
    """
    risk = 0.0
    factors = []
    if float(r.get("RuntimeHours", 0)) > 8000:
        risk += 0.3; factors.append("High runtime hours")
    if float(r.get("TemperatureC", 0)) > 35:
        risk += 0.2; factors.append("High temperature")
    if float(r.get("VibrationMM_S", 0)) > 0.8:
        risk += 0.2; factors.append("Excessive vibration")
    if int(float(r.get("ErrorLogsCount", 0))) > 15:
        risk += 0.2; factors.append("Many error logs")
    if int(float(r.get("NumRepairs", 0))) > 10:
        risk += 0.3; factors.append("Frequent repairs")

    if risk >= 0.7:
        label = "High"
    elif risk >= 0.4:
        label = "Medium"
    else:
        label = "Low"

    conf = 0.75 if label != "Medium" else 0.65
    if not factors:
        factors = ["All parameters within typical ranges"]
    return label, min(max(risk, 0.0), 1.0), factors, conf

# ---------------- Azure ML entry points ----------------
def init():
    global MODEL, MODEL_VERSION
    try:
        model_path = _find_model_path()
        if not model_path:
            logger.warning("No model artifact (.pkl) found. Will use fallback rules.")
            MODEL = None
            MODEL_VERSION = os.getenv("MODEL_VERSION", "rules-fallback")
            return

        logger.info(f"Loading model from: {model_path}")
        try:
            MODEL = joblib_load(model_path)
            MODEL_VERSION = os.getenv("MODEL_VERSION", os.path.basename(os.path.dirname(model_path)) or "1")
            logger.info("Model loaded successfully.")
        except Exception as e:
            # Most common cause in managed endpoints: xgboost not installed in env
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            MODEL = None
            MODEL_VERSION = os.getenv("MODEL_VERSION", "rules-fallback")
    except Exception as e:
        logger.error(f"init() unexpected error: {e}")
        logger.error(traceback.format_exc())
        MODEL = None
        MODEL_VERSION = os.getenv("MODEL_VERSION", "rules-fallback")

def _predict_batch(df: pd.DataFrame, raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    used_model = MODEL is not None
    model_labels = MODEL_LABELS_GUESS

    if used_model:
        try:
            # Predict probabilities if available
            probs = None
            if hasattr(MODEL, "predict_proba"):
                probs = MODEL.predict_proba(df)
            preds = MODEL.predict(df)
        except Exception as e:
            logger.error(f"Model prediction failed, switching to fallback: {e}")
            logger.error(traceback.format_exc())
            used_model = False

    for i, rec in enumerate(raw_items):
        device_name = rec.get("DeviceName") or rec.get("device_name") or rec.get("DeviceName", "Unknown")
        fb_label, fb_risk, fb_factors, fb_conf = _fallback_score_one(rec)

        pred_label = fb_label
        confidence = fb_conf
        probabilities = {}

        if used_model:
            # Map numeric class → guessed label order, but verify against fallback if confidence is low
            try:
                pred_idx = int(preds[i])
            except Exception:
                pred_idx = 0
            # build prob mapping if available
            if 'probs' in locals() and probs is not None:
                row = probs[i]
                # zip with guess labels; pad/truncate defensively
                n = min(len(row), len(model_labels))
                probabilities = { model_labels[j]: float(row[j]) for j in range(n) }
                confidence = float(np.max(row)) if row is not None else confidence
                # choose label by max prob among guessed labels
                try:
                    pred_label_guess = model_labels[int(np.argmax(row[:n]))]
                except Exception:
                    pred_label_guess = fb_label
            else:
                pred_label_guess = model_labels[pred_idx] if pred_idx < len(model_labels) else fb_label

            # If model is confident, take it; otherwise prefer fallback that explains factors clearly
            if confidence >= 0.6:
                pred_label = pred_label_guess
            else:
                pred_label = fb_label

        results.append({
            "device_name": device_name,
            "prediction": pred_label,
            "confidence": float(confidence),
            "risk_score": float(fb_risk),            # interpretable score from rules
            "factors": fb_factors,                   # human-friendly reasons
            "model_version": MODEL_VERSION or "unknown",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "debug": {
                "used_model": bool(MODEL is not None),
                "probabilities": probabilities
            }
        })
    return results

def run(raw_data):
    """
    Expects JSON with {"data": [ { feature: value, ... }, ... ]}
    Returns {"predictions": [ {prediction, confidence, ...}, ... ], "success": true}
    """
    try:
        if isinstance(raw_data, (str, bytes, bytearray)):
            payload = json.loads(raw_data)
        elif isinstance(raw_data, dict):
            payload = raw_data
        else:
            payload = json.loads(str(raw_data))

        # Your gg.py sends under "data": [...]  (also used by batch paths)  # noqa
        items = payload.get("data")
        if items is None:
            raise ValueError("Request must contain a 'data' list with one or more records.")

        if not isinstance(items, list) or len(items) == 0:
            raise ValueError("'data' must be a non-empty list.")

        # Normalize records and build dataframe (drop LastServiceDate to mirror training)
        df = _build_dataframe(items)

        # Predict (or fallback)
        preds = _predict_batch(df, [_coerce_record(r) for r in items])

        return {
            "success": True,
            "model_loaded": bool(MODEL is not None),
            "model_version": MODEL_VERSION or "unknown",
            "predictions": preds
        }
    except Exception as e:
        logger.error(f"run() error: {e}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }
