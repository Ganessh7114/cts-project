# score.py
import os, json, logging, traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure joblib is available (used to load sklearn/xgboost pipelines)
from joblib import load as joblib_load

# Try importing xgboost so joblib can resolve XGB* classes on load if present.
# If not installed, we just log and continue (fallback rules will still work).
try:
    import xgboost  # noqa: F401
except Exception:
    xgboost = None

logger = logging.getLogger("azureml-inference")
logging.basicConfig(level=logging.INFO)

# ----------------- Globals set in init() -----------------
MODEL = None
MODEL_VERSION = None

# If your LabelEncoder / class order is known, set it via env LABELS_ORDER="High,Low,Medium"
MODEL_LABELS_GUESS = os.getenv("LABELS_ORDER", "High,Low,Medium").split(",")

# Defaults and schema used to coerce inbound JSON → DataFrame columns expected by your pipeline
FEATURE_DEFAULTS: Dict[str, Any] = {
    "DeviceType": "Unknown",
    "DeviceName": "Unknown",
    "RuntimeHours": 10.0,
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
    "LastServiceDate": datetime.utcnow().strftime("%d-%m-%Y"),  # drop before inference
    "ApproxDeviceAgeYears": 1.0,
    "NumRepairs": 0,
    "ErrorLogsCount": 0,
}

# Numeric / categorical splits (must match training)
NUMERIC_COLS = [
    "RuntimeHours", "TemperatureC", "PressureKPa", "VibrationMM_S", "CurrentDrawA",
    "SignalNoiseLevel", "HumidityPercent", "OperationalCycles", "UserInteractionsPerDay",
    "ApproxDeviceAgeYears", "NumRepairs", "ErrorLogsCount",
]
CATEGORICAL_COLS = ["DeviceType", "DeviceName", "ClimateControl", "Location"]

# ----------------- Utilities -----------------
def _find_model_path() -> str:
    """Locate a .pkl model artifact in AZUREML_MODEL_DIR or local working dir."""
    candidates: List[str] = []
    mdl_dir = os.getenv("AZUREML_MODEL_DIR")
    if mdl_dir and os.path.isdir(mdl_dir):
        for root, _, files in os.walk(mdl_dir):
            for f in files:
                if f.lower().endswith(".pkl"):
                    candidates.append(os.path.join(root, f))
    for name in ["xgboost_pipeline.pkl", "model.pkl", "pipeline.pkl"]:
        if os.path.exists(name):
            candidates.append(os.path.abspath(name))
    # Prefer the known name if present
    for p in candidates:
        if os.path.basename(p).lower() == "xgboost_pipeline.pkl":
            return p
    return candidates[0] if candidates else ""

def _json_safe(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x

def _coerce_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing fields, cast types; parse LastServiceDate but drop later."""
    clean: Dict[str, Any] = {}
    for k, default in FEATURE_DEFAULTS.items():
        v = rec.get(k, default)
        if v is None:
            v = default
        if k in {"OperationalCycles", "NumRepairs", "ErrorLogsCount"}:
            try:
                clean[k] = int(v)
            except Exception:
                clean[k] = int(default)
        elif k in NUMERIC_COLS:
            try:
                clean[k] = float(v)
            except Exception:
                clean[k] = float(default)
        else:
            clean[k] = str(v)

    # Attempt to parse LastServiceDate for robustness (training dropped it).
    try:
        _ = datetime.strptime(clean["LastServiceDate"], "%d-%m-%Y")
    except Exception:
        try:
            _ = pd.to_datetime(clean["LastServiceDate"], errors="coerce")
        except Exception:
            pass
    return clean

def _build_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [_coerce_record(r) for r in items]
    df = pd.DataFrame(rows)
    # Drop training-excluded column
    if "LastServiceDate" in df.columns:
        df = df.drop(columns=["LastServiceDate"])
    # Ensure all expected columns exist (order is not critical for ColumnTransformer by-name)
    for col in (CATEGORICAL_COLS + NUMERIC_COLS):
        if col not in df.columns:
            df[col] = FEATURE_DEFAULTS.get(col, np.nan)
    return df

# --------- Human-friendly fallback (mirrors your rules) ---------
def _fallback_score_one(r: Dict[str, Any]) -> Tuple[str, float, List[str], float]:
    """
    Returns (label, risk_score[0..1], factors, confidence[0..1])
    """
    risk = 0.0
    factors: List[str] = []
    try:
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
    except Exception:
        # If any cast fails, just keep whatever we already accumulated
        pass

    if risk >= 0.7:
        label = "High"
    elif risk >= 0.4:
        label = "Medium"
    else:
        label = "Low"

    conf = 0.75 if label != "Medium" else 0.65
    if not factors:
        factors = ["All parameters within typical ranges"]
    return label, max(0.0, min(1.0, risk)), factors, conf

# ----------------- Azure ML entry points -----------------
def init():
    """
    Called once on container start. Load the model if available; otherwise
    keep None and use human-readable fallback rules.
    """
    global MODEL, MODEL_VERSION
    try:
        model_path = _find_model_path()
        if not model_path:
            logger.warning("No .pkl model found. Using fallback rules.")
            MODEL = None
            MODEL_VERSION = os.getenv("MODEL_VERSION", "rules-fallback")
            return

        logger.info(f"Loading model from: {model_path}")
        try:
            MODEL = joblib_load(model_path)  # sklearn/xgboost pipelines
            # If AZUREML_MODEL_DIR has a versioned folder, use that; otherwise use env/model filename
            MODEL_VERSION = os.getenv("MODEL_VERSION", os.path.basename(os.path.dirname(model_path)) or "1")
            logger.info("Model loaded successfully.")
        except Exception as e:
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
    labels_guess = MODEL_LABELS_GUESS

    if used_model:
        try:
            probs = MODEL.predict_proba(df) if hasattr(MODEL, "predict_proba") else None
            preds = MODEL.predict(df)
        except Exception as e:
            # Typical failure: env mismatch, missing xgboost, or feature mismatch.
            logger.error(f"Model prediction failed, switching to fallback: {e}")
            logger.error(traceback.format_exc())
            used_model = False

    for i, rec in enumerate(raw_items):
        device_name = rec.get("DeviceName") or rec.get("device_name") or "Unknown"
        fb_label, fb_risk, fb_factors, fb_conf = _fallback_score_one(rec)

        pred_label = fb_label
        confidence = fb_conf
        probabilities: Dict[str, float] = {}

        if used_model:
            try:
                if "probs" in locals() and probs is not None:
                    row = probs[i]
                    n = min(len(row), len(labels_guess))
                    probabilities = {labels_guess[j]: float(row[j]) for j in range(n)}
                    confidence = float(np.max(row))
                    pred_label_guess = labels_guess[int(np.argmax(row[:n]))]
                else:
                    # Some pipelines return string labels directly; guard for both int/str
                    raw_pred = preds[i]
                    if isinstance(raw_pred, (np.integer, int)):
                        pred_label_guess = labels_guess[int(raw_pred)] if int(raw_pred) < len(labels_guess) else fb_label
                    else:
                        pred_label_guess = str(raw_pred)
                # prefer model when reasonably confident
                pred_label = pred_label_guess if confidence >= 0.6 else fb_label
            except Exception:
                pred_label = fb_label

        results.append({
            "device_name": device_name,
            "prediction": pred_label,
            "confidence": float(confidence),
            "risk_score": float(fb_risk),         # interpretable rules score
            "factors": fb_factors,                # human-friendly reasons
            "model_version": MODEL_VERSION or "unknown",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "debug": {
                "used_model": bool(MODEL is not None),
                "probabilities": probabilities
            }
        })
    return results

def _extract_items(payload: Any) -> List[Dict[str, Any]]:
    """
    Accepts several shapes commonly used with Azure endpoints:
      - {"data": [ {...}, {...} ]}
      - {"instances": [ {...}, ... ]}  (Vertex-style; just in case)
      - {"input_data": {"data": [ {...}, ... ]}}  (older AML samples)
      - {"records": [ {...}, ... ]}
    """
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("instances"), list):
            return payload["instances"]
        if isinstance(payload.get("records"), list):
            return payload["records"]
        if isinstance(payload.get("input_data"), dict) and isinstance(payload["input_data"].get("data"), list):
            return payload["input_data"]["data"]
    raise ValueError("Request must contain a list of records under 'data', 'instances', 'records' or 'input_data.data'.")

def run(raw_data):
    """
    Azure ML entry point. Returns:
      {
        "success": true/false,
        "model_loaded": bool,
        "model_version": "...",
        "predictions": [ { device_name, prediction, confidence, risk_score, factors[], ... }, ... ],
        "error"/"trace": "...optional..."
      }
    """
    try:
        if isinstance(raw_data, (str, bytes, bytearray)):
            payload = json.loads(raw_data)
        elif isinstance(raw_data, dict):
            payload = raw_data
        else:
            payload = json.loads(str(raw_data))

        items = _extract_items(payload)
        if not items:
            raise ValueError("No records provided.")

        # Normalize to training schema and build DF
        cleaned = [_coerce_record(r) for r in items]
        df = _build_dataframe(cleaned)

        # Predict (or fallback)
        preds = _predict_batch(df, cleaned)

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
