# logging_config.py
"""
Centralized logging + final print utilities with two modes:
- LOG_MODE=debug         : verbose, development-friendly
- LOG_MODE=presentation  : clean, hides SDK noise and internals

Drop-in: call setup_logging() at the very start of an entrypoint.
Use present_final() to print the final decision in a clean way.
"""

from __future__ import annotations
import json
import logging
import os
import sys
import warnings
from typing import Any, Dict, Optional


# ---------------------------
# Internal helpers
# ---------------------------

class _SingleLineFormatter(logging.Formatter):
    """Single-line, timestamped formatter; cleaner in presentation mode."""
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        # Ensure no stray newlines from 3P libs
        return " ".join(str(base).splitlines())


def _remove_existing_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        logger.removeHandler(h)


def _silence_third_party(log_mode: str) -> None:
    """
    Suppress noisy third-party logs (Azure SDK wire logs, urllib3, MQTT, etc.)
    without affecting your own logs under 'services.*' or 'workflows.*'.
    """
    noisy_loggers = {
        # Azure HTTP wire logging (the URL/headers you saw)
        "azure.core.pipeline.policies.http_logging_policy": logging.ERROR,

        # Azure SDK general chatter
        "azure.core": logging.ERROR,
        "azure.storage": logging.ERROR,
        "azure.identity": logging.ERROR,

        # Networking
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,

        # MQTT client logs
        "paho": logging.WARNING,
        "paho.mqtt": logging.WARNING,
    }

    # In debug we still tone these down a bit; in presentation we silence even more
    for name, level in noisy_loggers.items():
        lg = logging.getLogger(name)
        if log_mode == "presentation":
            lg.setLevel(level)
            lg.propagate = False
        else:
            # debug mode: keep them quieter than your app logs
            lg.setLevel(max(level, logging.WARNING))

    # Optional extra quiet knobs via env (do not set by default)
    if os.getenv("QUIET_WARNINGS", "1") in ("1", "true", "yes"):
        # Suppress common benign warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Some Azure packages emit FutureWarning / ResourceWarning
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=ResourceWarning)


# ---------------------------
# Public API
# ---------------------------

def setup_logging() -> str:
    """
    Configure root logger once.
    Returns the effective LOG_MODE to let callers toggle behavior.
    """
    log_mode = os.getenv("LOG_MODE", "debug").strip().lower()
    if log_mode not in ("debug", "presentation"):
        log_mode = "debug"

    root = logging.getLogger()
    _remove_existing_handlers(root)

    # Console handler with clean format
    fmt = "%(asctime)s | %(levelname)s | %(message)s" if log_mode == "presentation" \
          else "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(_SingleLineFormatter(fmt=fmt, datefmt=datefmt))

    # debug mode is chatty; presentation is crisp
    root.setLevel(logging.DEBUG if log_mode == "debug" else logging.INFO)
    root.addHandler(ch)

    _silence_third_party(log_mode)

    logging.getLogger().info(f"Logging initialized in {log_mode.upper()} mode")
    return log_mode


def present_step(msg: str, log_mode: Optional[str] = None) -> None:
    """
    A tiny helper for 'actions-only' updates.
    In debug: includes the caller's module name via logger hierarchy.
    In presentation: one clean line, no internals.
    """
    lm = (log_mode or os.getenv("LOG_MODE", "debug")).lower()
    logger = logging.getLogger("pipeline" if lm == "presentation" else "debug.pipeline")
    logger.info(msg)


def present_final(final_or_record: Dict[str, Any],
                  log_mode: Optional[str] = None,
                  json_key_for_final: str = "final",
                  title: str = "=== FINAL DECISION ==="
                  ) -> None:
    """
    Prints the final decision minimally when in presentation mode.
    - Accepts either:
        a) the 'final' dict directly {label, confidence, factors}
        b) a full 'record' dict containing a 'final' key
    - Hides source (Azure/local/rules). Just shows outcome.

    In debug mode: prints the entire record JSON (unchanged behavior).
    """
    lm = (log_mode or os.getenv("LOG_MODE", "debug")).lower()
    data = final_or_record

    # If a full record was passed, extract final block
    if json_key_for_final in data and isinstance(data[json_key_for_final], dict):
        final = data[json_key_for_final]
    else:
        final = data  # assume caller passed the final block directly

    if lm == "presentation":
        label = final.get("label", "N/A")
        conf = final.get("confidence", None)
        factors = final.get("factors", []) or []

        print("\n" + title)
        print(f"Prediction: {label}")
        if conf is not None:
            # Round to two decimals for display
            try:
                print(f"Confidence: {round(float(conf), 2)}")
            except Exception:
                print(f"Confidence: {conf}")
        if factors:
            # Keep this short and readable; do not dump arrays or internals
            print("Notes: " + ", ".join(map(str, factors)))

    else:
        # debug: preserve your existing behavior (full JSON)
        print("\n=== FINAL RESULT (stored) ===")
        try:
            print(json.dumps(data, indent=2, default=str))
        except Exception:
            # If something is not JSON-serializable,
            # fall back to a minimal subset
            minimal = data.get(json_key_for_final, data)
            print(json.dumps(minimal, indent=2, default=str))
