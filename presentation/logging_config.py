# presentation/logging_config.py
import os
import logging
import re
from typing import List

DEFAULT_FMT_DEBUG = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_FMT_PRESENT = "%(message)s"

class MessageDropFilter(logging.Filter):
    """
    Drops log records whose message matches any of the configured regex patterns.
    Useful to hide noisy but harmless lines.
    """
    def __init__(self, patterns: List[str]):
        super().__init__()
        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns if p.strip()]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        for rx in self._compiled:
            if rx.search(msg):
                return False
        return True

def _parse_drop_patterns() -> List[str]:
    raw = os.getenv("LOG_DROP_PATTERNS", "")
    if not raw.strip():
        # default: hide transient "Publish timed out"
        return [r"Publish timed out"]
    return [p.strip() for p in raw.split(";") if p.strip()]

def setup_logging() -> str:
    """
    Configure global logging:
    - LOG_MODE=presentation (default) or debug
    - In presentation: only show [STEP], final decision, and errors
    - In debug: show everything
    """
    log_mode = (os.getenv("LOG_MODE", "presentation") or "presentation").strip().lower()
    present_mode = (log_mode == "presentation")

    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers[:]:
            root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if not present_mode else logging.INFO)
    fmt = DEFAULT_FMT_PRESENT if present_mode else DEFAULT_FMT_DEBUG
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(MessageDropFilter(_parse_drop_patterns()))
    root.addHandler(handler)

    if present_mode:
        # Only show WARNING+ by default, so INFO logs from services won't leak
        root.setLevel(logging.WARNING)
    else:
        root.setLevel(logging.DEBUG)

    # Suppress 3rd-party chatter always
    noisy_libs = [
        "azure",
        "azure.core.pipeline.policies.http_logging_policy",
        "urllib3",
        "botocore",
        "s3transfer",
        "paho",
        "hbmqtt",
        "asyncio",
    ]
    for noisy in noisy_libs:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Your services: keep them silent in presentation (except errors),
    # but verbose in debug
    for svc in [
        "services.mqtt_service",
        "services.azure_ml_client",
        "services.blob_storage",
        "services.local_model",
        "services.rule_fallback",
        "workflows.test_services",
    ]:
        logging.getLogger(svc).setLevel(logging.DEBUG if not present_mode else logging.ERROR)

    # Only announce mode in debug (not in presentation, to keep it clean)
    if not present_mode:
        logging.getLogger().info("Logging initialized in DEBUG mode")

    return log_mode
