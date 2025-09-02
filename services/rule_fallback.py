import os
from typing import Any, Dict, List, Optional, Tuple

CONF_DEFAULT = float(os.getenv("AML_CONF_THRESHOLD", "0.6"))

class RuleBasedAssessor:
    """
    Applies simple rules only as a fallback to Azure ML output.
    """

    def __init__(self, conf_threshold: float = CONF_DEFAULT):
        self.conf_threshold = conf_threshold

    def _rules(self, x: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """
        Very simple example rules. Tweak as needed.
        Returns (label, confidence, factors)
        """
        factors: List[str] = []
        score = 0.0

        temp = float(x.get("TemperatureC", 0))
        vib  = float(x.get("VibrationMM_S", 0))
        err  = float(x.get("ErrorLogsCount", 0))
        hrs  = float(x.get("RuntimeHours", 0))

        if temp >= 38.0:
            score += 0.4
            factors.append("High temperature")
        if vib >= 0.05:
            score += 0.3
            factors.append("High vibration")
        if err >= 3:
            score += 0.4
            factors.append("Frequent error logs")
        if hrs >= 2000:
            score += 0.2
            factors.append("Long runtime")

        if score >= 0.7:
            label = "High"
        elif score >= 0.4:
            label = "Medium"
        else:
            label = "Low"

        conf = min(0.9, 0.5 + score/2.0)  # heuristic
        return label, conf, factors

    def refine_prediction(
        self,
        telemetry_row: Dict[str, Any],
        aml_label: Optional[str],
        aml_conf: Optional[float],
    ) -> Dict[str, Any]:
        """
        If AML confidence is missing/low or AML failed, apply rules.
        """
        use_rules = (aml_label is None) or (aml_conf is None) or (aml_conf < self.conf_threshold)

        if use_rules:
            label, conf, factors = self._rules(telemetry_row)
            return {
                "source": "rules_fallback",
                "label": label,
                "confidence": conf,
                "factors": factors,
            }
        else:
            return {
                "source": "azure_ml",
                "label": aml_label,
                "confidence": aml_conf,
                "factors": [],
            }

    # Convenience if you ever need rules directly
    def direct(self, telemetry_row: Dict[str, Any]) -> Dict[str, Any]:
        label, conf, factors = self._rules(telemetry_row)
        return {"source": "rules_fallback", "label": label, "confidence": conf, "factors": factors}
