# score.py
# Azure ML Online Endpoint entrypoint for Medical Device Failure Risk
#
# - Loads an sklearn Pipeline (with ColumnTransformer + XGBClassifier) from xgboost_pipeline.pkl
# - Accepts multiple input payload shapes: {"data":[...]}, {"instances":[...]}, {"records":[...]},
#   {"input_data":{"data":[...]}} as seen in your previous runs.
# - Predicts and returns a normalized response. If model load/predict fails, runs rule-based fallback.
#
# Response shape (example):
# {
#   "success": true,
#   "model_loaded": true,
#   "model_version": "1.0.0",
#   "predictions": [
#     {
#       "device_name": "GE Revolution",
#       "prediction": "Low",
#       "confidence": 0.75,
#       "risk_score": 0.20,
#       "factors": ["High temperature"],
#       "model_version": "1.0.0",
#       "timestamp": "2025-08-30T19:03:56.177864Z",
#       "debug": {
#         "used_model": true,
#         "probabilities": {"Low": 0.75, "Medium": 0.20, "High": 0.05},
#         "missing_features": [],
#         "extra_features": []
#       }
#     }
#   ]
# }

import os
import io
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

# Import xgboost so deserialization succeeds even if not directly referenced
try:
    import xgboost  # noqa: F401
except Exception:
    xgboost = None  # model load may still work if xgboost is not required during unpickle

# ----------------------------
# Globals initialized in init()
# ----------------------------
MODEL: Optional[Any] = None
MODEL_VERSION: Optional[str] = None
CLASS_NAMES: Optional[List[str]] = None
EXPECTED_FEATURES: Optional[List[str]] = None  # best-effort extraction from ColumnTransformer


# ---------- Utilities ----------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_text_if_exists(path: str) -> Optional[str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass
    return None


def _find_model_file(base_dir: str) -> Optional[str]:
    """
    Try to find the model file under AZUREML_MODEL_DIR (or current working dir).
    We look for common names and .pkl files.
    """
    candidates = []
    for root, _, files in os.walk(base_dir):
        for fn in files:
            lower = fn.lower()
            if lower in ("xgboost_pipeline.pkl", "model.pkl", "pipeline.pkl") or lower.endswith(".pkl"):
                candidates.append(os.path.join(root, fn))
    # Prefer the exact expected name if present
    for preferred in candidates:
        if os.path.basename(preferred).lower() == "xgboost_pipeline.pkl":
            return preferred
    return candidates[0] if candidates else None


def _extract_items(payload: Any) -> List[Dict[str, Any]]:
    """
    Accept multiple wrapper shapes and always return a list[dict].
    Valid shapes we've seen:
      - {"data":[{...}, {...}]}
      - {"instances":[{...}]}
      - {"records":[{...}]}
      - {"input_data":{"data":[{...}]}}
      - A single dict {...}  -> wrap to list
    """
    if payload is None:
        raise ValueError("Empty payload")

    if isinstance(payload, dict):
        # direct keys
        for key in ("data", "instances", "records"):
            if key in payload:
                items = payload[key]
                if isinstance(items, list):
                    return items
                # allow one object
                if isinstance(items, dict):
                    return [items]

        # input_data wrapper (e.g., from your earlier fix)
        if "input_data" in payload:
            inner = payload["input_data"]
            if isinstance(inner, dict):
                # common case: {"input_data":{"data":[...]}}
                if "data" in inner:
                    items = inner["data"]
                    if isinstance(items, list):
                        return items
                    if isinstance(items, dict):
                        return [items]
                # Azure variations sometimes use 'columns' + 'data' 2D matrix
                if "columns" in inner and "data" in inner and isinstance(inner["data"], list):
                    cols = inner["columns"]
                    rows = inner["data"]
                    if not isinstance(cols, list):
                        raise ValueError("'input_data.columns' must be a list")
                    dict_rows = []
                    for row in rows:
                        if isinstance(row, list) and len(row) == len(cols):
                            dict_rows.append({c: v for c, v in zip(cols, row)})
                        else:
                            raise ValueError("Row length mismatch against 'input_data.columns'")
                    return dict_rows

        # Single-record direct dict (no wrapper)
        # Be careful not to treat a top-level dict with special keys as a record incorrectly.
        # We'll heuristically assume a telemetry-like dict has >3 fields and at least one known key.
        if any(k in payload for k in ("DeviceType", "DeviceName", "TemperatureC", "RuntimeHours")):
            return [payload]

        # If we get here, the dict didn't match expected shapes
        raise ValueError("Request must contain a list of records under 'data', 'instances', 'records' or 'input_data.data'.")

    if isinstance(payload, list):
        # already a list of objects (or list of lists)
        if len(payload) == 0:
            return []
        if isinstance(payload[0], dict):
            return payload
        raise ValueError("List payload must contain objects (dicts).")

    # String "ping" support / health probe
    if isinstance(payload, str) and payload.strip().lower() == "ping":
        return []

    # Fallback not supported
    raise ValueError("Unsupported payload type. Provide JSON with a list of records.")


def _coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dtypes but keep names. We don't enforce exact schema here because the
    ColumnTransformer in the pipeline will select its columns by name.
    - Numeric-like columns will be converted with errors='ignore' to avoid breaking strings.
    - Common numeric telemetry fields explicitly coerced.
    """
    numeric_hint_cols = [
        "RuntimeHours", "TemperatureC", "PressureKPa", "VibrationMM_S",
        "CurrentDrawA", "SignalNoiseLevel", "HumidityPercent", "OperationalCycles",
        "UserInteractionsPerDay", "ApproxDeviceAgeYears", "NumRepairs", "ErrorLogsCount"
    ]
    for col in df.columns:
        # explicit numeric attempt for known columns
        if col in numeric_hint_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # best-effort: if column looks numeric (all digits/float-like), to_numeric
        elif df[col].dtype == object:
            try_sample = df[col].dropna().astype(str).head(10)
            if len(try_sample) > 0 and all(any(ch.isdigit() for ch in s) for s in try_sample):
                df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def _get_expected_features_from_pipeline(pipeline) -> Optional[List[str]]:
    """
    Best-effort: find a ColumnTransformer inside the sklearn Pipeline and collect
    the input column names it expects. This helps us report missing/extra fields.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        if isinstance(pipeline, Pipeline):
            # look for first ColumnTransformer in steps
            for name, step in pipeline.steps:
                if isinstance(step, ColumnTransformer):
                    cols: List[str] = []
                    for _, trans, col_list in step.transformers:
                        if isinstance(col_list, list):
                            cols.extend(col_list)
                    return list(dict.fromkeys(cols))  # preserve order, dedupe
        # Sometimes the pipeline is nested; scan attributes shallowly
        for attr in dir(pipeline):
            obj = getattr(pipeline, attr)
            try:
                if isinstance(obj, ColumnTransformer):
                    cols = []
                    for _, _, col_list in obj.transformers:
                        if isinstance(col_list, list):
                            cols.extend(col_list)
                    return list(dict.fromkeys(cols))
            except Exception:
                continue
    except Exception:
        pass
    return None


def _label_from_int(yhat: Any, class_names: Optional[List[str]]) -> Any:
    """
    If model returns integers but we know class names, map them.
    Otherwise return as-is.
    """
    try:
        if class_names is not None:
            # scalar
            if np.isscalar(yhat):
                idx = int(yhat)
                if 0 <= idx < len(class_names):
                    return class_names[idx]
            # array of scalars
            lab = []
            for v in np.array(yhat).ravel():
                idx = int(v)
                lab.append(class_names[idx] if 0 <= idx < len(class_names) else v)
            return lab
    except Exception:
        pass
    return yhat


def _default_class_names() -> List[str]:
    # Fallback ordering used across this project
    return ["Low", "Medium", "High"]


def _risk_score_from_probs(prob_map: Dict[str, float]) -> float:
    """
    Define a risk score ∈ [0,1] as P(High) if available; else P(positive) for binary; else 1 - P(Low).
    """
    if "High" in prob_map:
        return float(prob_map["High"])
    # binary case: try to pick the second class as 'positive' (heuristic)
    keys = list(prob_map.keys())
    if len(keys) == 2:
        return float(prob_map.get(keys[1], 0.0))
    return float(1.0 - prob_map.get("Low", 0.0))


def _rule_fallback_row(row: Dict[str, Any]) -> Tuple[str, float, List[str]]:
    """
    Simple rule-based fallback matching your earlier behavior.
    Returns (label, confidence, factors)
    """
    factors = []
    temp = _to_float(row.get("TemperatureC"))
    vib = _to_float(row.get("VibrationMM_S"))
    hours = _to_float(row.get("RuntimeHours"))
    errors = _to_float(row.get("ErrorLogsCount"))
    humidity = _to_float(row.get("HumidityPercent"))

    # Basic thresholds (tune as needed)
    score = 0.0
    if temp is not None and temp >= 45:
        score += 0.6
        factors.append("Very high temperature")
    elif temp is not None and temp >= 38:
        score += 0.4
        factors.append("High temperature")

    if vib is not None and vib >= 0.1:
        score += 0.25
        factors.append("High vibration")

    if hours is not None and hours >= 3000:
        score += 0.2
        factors.append("Very high runtime")
    elif hours is not None and hours >= 1500:
        score += 0.1
        factors.append("Elevated runtime")

    if errors is not None and errors >= 5:
        score += 0.25
        factors.append("Frequent error logs")

    if humidity is not None and humidity >= 70:
        score += 0.1
        factors.append("High humidity")

    # Convert numeric score into label & confidence (heuristic scaling)
    if score >= 0.7:
        return "High", min(0.9, 0.5 + score * 0.5), factors
    if score >= 0.4:
        return "Medium", min(0.8, 0.5 + score * 0.4), factors
    return "Low", max(0.6, 1.0 - score), factors


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _build_result_row(
    row: Dict[str, Any],
    label: str,
    confidence: float,
    prob_map: Dict[str, float],
    used_model: bool,
    missing: List[str],
    extra: List[str],
) -> Dict[str, Any]:
    return {
        "device_name": row.get("DeviceName"),
        "prediction": label,
        "confidence": float(confidence) if confidence is not None else None,
        "risk_score": _risk_score_from_probs(prob_map) if prob_map else (1.0 - confidence if label == "Low" else confidence),
        "factors": [f for f in row.get("_factors", [])] if row.get("_factors") else [],
        "model_version": MODEL_VERSION or "unknown",
        "timestamp": _utcnow_iso(),
        "debug": {
            "used_model": used_model,
            "probabilities": prob_map,
            "missing_features": missing,
            "extra_features": extra,
        },
    }


# ---------- Azure ML hooks ----------

def init():
    """
    Azure ML init hook: load the model and optional metadata.
    """
    global MODEL, MODEL_VERSION, CLASS_NAMES, EXPECTED_FEATURES

    base_dir = os.getenv("AZUREML_MODEL_DIR") or os.getcwd()

    # Find and load model
    model_path = _find_model_file(base_dir)
    if model_path:
        try:
            MODEL = joblib.load(model_path)
        except Exception as e:
            MODEL = None
            print(f"[score.py] Failed to load model from {model_path}: {e}")
    else:
        print("[score.py] No model .pkl found; will use rule-based fallback.")

    # Model version discovery (sidecar files or env)
    MODEL_VERSION = (
        os.getenv("MODEL_VERSION")
        or _read_text_if_exists(os.path.join(base_dir, "MODEL_VERSION.txt"))
        or _read_text_if_exists(os.path.join(base_dir, "VERSION.txt"))
        or "rules-fallback" if MODEL is None else "1.0.0"
    )

    # Class names discovery
    # 1) ENV var as JSON or comma-separated
    raw_cls = os.getenv("CLASS_NAMES_JSON") or os.getenv("CLASS_NAMES")
    if raw_cls:
        try:
            CLASS_NAMES = json.loads(raw_cls) if raw_cls.strip().startswith("[") else [c.strip() for c in raw_cls.split(",")]
        except Exception:
            CLASS_NAMES = None

    # 2) Sidecar classes.json
    if CLASS_NAMES is None:
        sidecar = _read_text_if_exists(os.path.join(base_dir, "classes.json"))
        if sidecar:
            try:
                CLASS_NAMES = json.loads(sidecar)
            except Exception:
                CLASS_NAMES = None

    # 3) Default known ordering
    if CLASS_NAMES is None:
        CLASS_NAMES = _default_class_names()

    # Expected feature names (best effort, for debug and validation)
    if MODEL is not None:
        EXPECTED_FEATURES = _get_expected_features_from_pipeline(MODEL)


def run(raw_data):
    """
    Azure ML run hook: process a request and return JSON.
    """
    try:
        # Handle string health-checks
        if isinstance(raw_data, str) and raw_data.strip().lower() == "ping":
            return {"success": True, "model_loaded": MODEL is not None, "model_version": MODEL_VERSION, "predictions": []}

        # Parse JSON if we got raw string/bytes
        if isinstance(raw_data, (str, bytes, bytearray)):
            payload = json.loads(raw_data)
        else:
            payload = raw_data

        items = _extract_items(payload)  # may raise -> caught below

        if len(items) == 0:
            # nothing to score (health probe scenario)
            return {"success": True, "model_loaded": MODEL is not None, "model_version": MODEL_VERSION, "predictions": []}

        # Normalize to DataFrame
        df = pd.DataFrame(items)
        df = _coerce_features(df)

        # Track missing/extra relative to expected feature names (if known)
        missing = []
        extra = []
        if EXPECTED_FEATURES:
            cols_set = set(df.columns.tolist())
            exp_set = set(EXPECTED_FEATURES)
            missing = sorted(list(exp_set - cols_set))
            extra = sorted(list(cols_set - exp_set))

            # Create placeholders for missing features (None/NaN); the pipeline's imputers/scalers should handle
            for m in missing:
                df[m] = np.nan

            # Keep all columns; ColumnTransformer will select what it needs by name

        predictions_payload = []

        # Decide whether to use model or fallback per-row
        use_model = MODEL is not None

        if use_model:
            # Try model prediction
            try:
                # Predict probabilities when available
                prob_array = None
                classes_attr = getattr(MODEL, "classes_", None)

                # Pipeline may expose predict_proba at top-level
                if hasattr(MODEL, "predict_proba"):
                    prob_array = MODEL.predict_proba(df)  # shape (n, k)
                # Fallback to predict only
                y_pred = MODEL.predict(df)

                # Map labels
                # Prefer model.classes_ with CLASS_NAMES size check
                name_map = None
                if classes_attr is not None and CLASS_NAMES and len(CLASS_NAMES) == len(classes_attr):
                    # classes_ may be ints (encoded); map to provided names by index of class value
                    # Build index-by-class-value (if classes_ are [0,1,2])
                    try:
                        order = np.argsort(classes_attr)
                        ordered_classes = np.array(classes_attr)[order]
                        # assume labels [0..n-1] -> CLASS_NAMES by position of ordered_classes
                        name_map = {int(ordered_classes[i]): CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}
                    except Exception:
                        name_map = None

                for i, row in enumerate(items):
                    # Compute probabilities map
                    prob_map: Dict[str, float] = {}
                    conf = None
                    label_out = None

                    if prob_array is not None:
                        row_probs = prob_array[i]
                        # Determine class labels to pair probabilities with
                        if classes_attr is not None:
                            # If model.classes_ aligns with CLASS_NAMES
                            if name_map:
                                keys = [name_map.get(int(c), str(c)) for c in classes_attr]
                            else:
                                # classes_ might already be strings
                                keys = [str(c) for c in classes_attr]
                        else:
                            # fallback to default names of correct length
                            keys = CLASS_NAMES if len(CLASS_NAMES) == len(row_probs) else [f"class_{j}" for j in range(len(row_probs))]

                        prob_map = {str(keys[j]): float(row_probs[j]) for j in range(len(keys))}
                        # pick argmax
                        best_idx = int(np.argmax(row_probs))
                        best_key = list(prob_map.keys())[best_idx]
                        label_out = best_key
                        conf = float(row_probs[best_idx])
                    else:
                        # No proba -> use predicted label
                        yhat = y_pred[i]
                        if name_map and isinstance(yhat, (int, np.integer)):
                            label_out = name_map.get(int(yhat), str(yhat))
                        else:
                            # If prediction is int and we have default class names, map heuristically
                            if isinstance(yhat, (int, np.integer)) and int(yhat) in range(len(CLASS_NAMES)):
                                label_out = CLASS_NAMES[int(yhat)]
                            else:
                                label_out = str(yhat)
                        conf = 0.75 if label_out == "Low" else 0.7 if label_out == "Medium" else 0.8  # heuristic

                    predictions_payload.append(
                        _build_result_row(
                            row=row,
                            label=label_out,
                            confidence=conf,
                            prob_map=prob_map,
                            used_model=True,
                            missing=missing,
                            extra=extra,
                        )
                    )

                return {
                    "success": True,
                    "model_loaded": True,
                    "model_version": MODEL_VERSION,
                    "predictions": predictions_payload,
                }

            except Exception as e:
                # Model available but prediction failed -> fall back per-row
                trace = traceback.format_exc()
                fb_rows = []
                for row in items:
                    label, conf, factors = _rule_fallback_row(row)
                    row["_factors"] = factors
                    fb_rows.append(
                        _build_result_row(
                            row=row,
                            label=label,
                            confidence=conf,
                            prob_map={label: conf},
                            used_model=False,
                            missing=missing,
                            extra=extra,
                        )
                    )
                return {
                    "success": True,
                    "model_loaded": True,
                    "model_version": MODEL_VERSION,
                    "predictions": fb_rows,
                    "warning": f"Model prediction failed, used rules fallback: {str(e)}",
                    "trace": trace,
                }

        # No model loaded -> rules fallback
        fb_rows = []
        for row in items:
            label, conf, factors = _rule_fallback_row(row)
            row["_factors"] = factors
            fb_rows.append(
                _build_result_row(
                    row=row,
                    label=label,
                    confidence=conf,
                    prob_map={label: conf},
                    used_model=False,
                    missing=missing or [],
                    extra=extra or [],
                )
            )
        return {
            "success": True,
            "model_loaded": False,
            "model_version": MODEL_VERSION or "rules-fallback",
            "predictions": fb_rows,
        }

    except Exception as e:
        # Any unexpected error -> structured failure with trace
        return {
            "success": False,
            "model_loaded": MODEL is not None,
            "model_version": MODEL_VERSION,
            "error": str(e),
            "trace": traceback.format_exc(),
        }
