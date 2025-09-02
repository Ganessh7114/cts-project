# presentation/reasons.py
from typing import Dict, List, Any

def _bool(v: Any) -> bool:
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")

def derive_reasons(telemetry: Dict[str, Any]) -> List[str]:
    """
    Derive human-friendly factors from raw telemetry.
    Keep this aligned with your simple rules so explanations feel consistent.
    """
    f: List[str] = []

    try:
        temp = float(telemetry.get("TemperatureC", 0))
    except Exception:
        temp = 0.0
    try:
        vib = float(telemetry.get("VibrationMM_S", 0))
    except Exception:
        vib = 0.0
    try:
        err = float(telemetry.get("ErrorLogsCount", 0))
    except Exception:
        err = 0.0
    try:
        hrs = float(telemetry.get("RuntimeHours", 0))
    except Exception:
        hrs = 0.0
    try:
        hum = float(telemetry.get("HumidityPercent", 0))
    except Exception:
        hum = 0.0
    climate = telemetry.get("ClimateControl", "")

    # Align with your RuleBasedAssessor thresholds
    if temp >= 38.0:
        f.append("High temperature")
    if vib >= 0.05:
        f.append("High vibration")
    if err >= 3:
        f.append("Frequent error logs")
    if hrs >= 2000:
        f.append("Extended runtime")

    # A couple of gentle extras (safe, human-friendly)
    if hum >= 70:
        f.append("High humidity")
    if _bool(climate) is False and str(climate).lower() not in ("yes", "y", "true", "1"):
        # If climate control looks off / not present
        f.append("No climate control")

    return f
