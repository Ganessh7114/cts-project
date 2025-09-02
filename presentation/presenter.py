# presentation/presenter.py
import os
import json
from typing import Dict, Any, Optional, List
from .reasons import derive_reasons

def _round_if_float(v: Any, n: int = 2) -> Any:
    try:
        if isinstance(v, float):
            return round(v, n)
        # strings of floats
        fv = float(v)
        return round(fv, n)
    except Exception:
        return v

def _ensure_reasons(final_block: Dict[str, Any], telemetry: Dict[str, Any]) -> List[str]:
    label = str(final_block.get("label", "")).strip()
    given = final_block.get("factors") or []
    if isinstance(given, list) and len(given) > 0:
        return given

    # If Medium/High and missing reasons, derive them
    if label in ("Medium", "High"):
        derived = derive_reasons(telemetry)
        if not derived:
            # Still ensure something meaningful
            return ["Elevated risk indicators present"]
        return derived

    # Low
    return ["Optimal"]  # or "None"; choose 'Optimal' for nicer UX

def present_step(message: str, status: Optional[str] = None, log_mode: Optional[str] = None) -> None:
    """
    One-line action update without payloads. Safe for demos.
    """
    if log_mode is None:
        log_mode = (os.getenv("LOG_MODE", "presentation") or "presentation").lower()

    if log_mode == "presentation":
        if status:
            print(f"• {message} — {status}")
        else:
            print(f"• {message}")
    else:
        # In debug keep it concise too; your normal module logs have the details
        if status:
            print(f"[STEP] {message} — {status}")
        else:
            print(f"[STEP] {message}")

def present_final(record: Dict[str, Any], log_mode: Optional[str] = None) -> None:
    """
    Final, unified AI decision with no mention of source (Azure/local/rules).
    - presentation mode: clean summary
    - debug mode: pretty JSON of the whole record
    """
    if log_mode is None:
        log_mode = (os.getenv("LOG_MODE", "presentation") or "presentation").lower()

    final_block = dict(record.get("final") or {})
    telemetry = dict(record.get("telemetry") or {})

    # Fill reasons if missing
    factors = _ensure_reasons(final_block, telemetry)
    label = str(final_block.get("label", "")).strip() or "Unknown"
    conf = _round_if_float(final_block.get("confidence"))

    # NEW: fetch device name for display
    device_name = (
        final_block.get("device_name")
        or record.get("device_name")
        or telemetry.get("DeviceName")
    )

    device_type = (
        final_block.get("device_type")
        or record.get("device_type")
        or telemetry.get("DeviceType")
    )

    if log_mode == "presentation":
        print("\n=== DECISION ===")
        if device_name:
            print(f"Device      : {device_name} ({device_type})")
        print(f"Risk        : {label}")
        if conf is not None:
            print(f"Confidence  : {conf}")
        if factors:
            # remove dupes while preserving order
            seen = set()
            uniq = []
            for fac in factors:
                if fac not in seen:
                    uniq.append(fac)
                    seen.add(fac)
            if label == "Low" and uniq == ["Optimal"]:
                print("Reasons     : Optimal")
            else:
                print("Reasons     : " + ", ".join(uniq))
        print("\n")
        # Do NOT print storage locations, raw responses, or sources.
    else:
        # Debug: full JSON for developers
        print("\n=== FULL PIPELINE RECORD (DEBUG) ===")
        print(json.dumps(record, indent=2, default=str))
