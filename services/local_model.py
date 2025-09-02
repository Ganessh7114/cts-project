import os
import json
import joblib
import logging
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger("services.local_model")

class LocalModel:
    """
    Loads and scores the local xgboost/sklearn pipeline (.pkl).
    Environment:
      - LOCAL_MODEL_PATH (default: xgboost_pipeline.pkl)
    Output:
      {
        ok, label, confidence, model_version, probs
      }
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv("LOCAL_MODEL_PATH", "xgboost_pipeline.pkl")
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Local model not found at {self.model_path}")
            self._model = joblib.load(self.model_path)
            logger.info("Local model loaded from %s", self.model_path)

    def _build_frame(self, row: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([row])
        # If the pipeline has feature_names_in_, align (and let ColumnTransformer handle missing/extras)
        try:
            feat_names: List[str] = list(getattr(self._model, "feature_names_in_", []))
            if feat_names:
                # align columns; fill missing with NaN
                for f in feat_names:
                    if f not in df.columns:
                        df[f] = np.nan
                df = df[feat_names]
        except Exception:
            pass
        return df

    def predict(self, telemetry_row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._ensure_loaded()
            X = self._build_frame(telemetry_row)

            label: Optional[str] = None
            conf: Optional[float] = None
            probs_map: Dict[str, float] = {}
            model_version = os.path.basename(self.model_path)

            classes = None
            if hasattr(self._model, "classes_"):
                classes = list(self._model.classes_)
            else:
                # Try to peek into final estimator in a pipeline
                try:
                    classes = list(self._model[-1].classes_)  # type: ignore[index]
                except Exception:
                    pass

            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(X)[0]
            else:
                # Fallback: try decision_function or just predict (no calibrated confidence)
                try:
                    proba = self._softmax(np.atleast_1d(self._model.decision_function(X))[0])
                except Exception:
                    pred_only = self._model.predict(X)[0]
                    label = self._map_label(pred_only)
                    return {"ok": True, "label": label, "confidence": None,
                            "model_version": model_version, "probs": {}}

            # Map class names
            if classes is None:
                classes = list(range(len(proba)))

            # Build probs map
            for c, p in zip(classes, proba):
                probs_map[self._map_label(c)] = float(p)

            # Choose best
            best_idx = int(np.argmax(proba))
            label = self._map_label(classes[best_idx])
            conf = float(proba[best_idx])

            return {"ok": True, "label": label, "confidence": conf,
                    "model_version": model_version, "probs": probs_map}

        except Exception as e:
            logger.exception("Local model prediction failed: %s", e)
            return {"ok": False, "error": str(e), "label": None, "confidence": None,
                    "model_version": None, "probs": {}}

    @staticmethod
    def _softmax(z):
        z = np.array(z, dtype=float)
        z = z - np.max(z)
        ez = np.exp(z)
        return ez / np.sum(ez)

    @staticmethod
    def _map_label(cls_val: Any) -> str:
        """
        Normalize to 'Low'|'Medium'|'High' when possible.
        """
        if isinstance(cls_val, str):
            s = cls_val.strip().lower()
            if s in ("low", "medium", "high"):
                return s.capitalize()
            # common aliases
            if s in ("0", "1", "2"):
                return {"0": "Low", "1": "Medium", "2": "High"}[s]
            return cls_val
        if isinstance(cls_val, (int, float)):
            i = int(cls_val)
            return {0: "Low", 1: "Medium", 2: "High"}.get(i, str(cls_val))
        return str(cls_val)
