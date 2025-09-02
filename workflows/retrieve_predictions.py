import os
import sys
import json
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Ensure services can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from services.blob_storage import BlobStorage


def extract_summary(blob_name: str, record: Dict[str, Any]) -> str:
    """
    Extract key info from a prediction record into a one-line string.
    """
    telemetry = record.get("telemetry", {})
    final = record.get("final", {})

    device_name = telemetry.get("DeviceName", "UnknownDevice")
    device_type = telemetry.get("DeviceType", "UnknownType")
    risk = final.get("label", "Unknown")
    conf = final.get("confidence", "NA")
    ts = record.get("timestamp", "NA")

    return f"{blob_name} | {device_name} ({device_type}) | Risk: {risk} | Confidence: {conf} | Time: {ts}"


def list_predictions(
    since: Optional[datetime] = None,
    device_name: Optional[str] = None,
    risk: Optional[str] = None,
    limit: int = 20,
) -> List[str]:
    """
    Fetch and filter predictions from blob storage.
    """
    blob = BlobStorage()
    blobs = blob.list_json_blobs(limit=200)

    results = []
    for b in blobs:
        record = blob.download_json(b.name)
        if not record:
            continue

        ts_str = record.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts = None

        # Apply filters
        if since and ts and ts < since:
            continue
        if device_name and str(record.get("telemetry", {}).get("DeviceName", "")).lower() != device_name.lower():
            continue
        if risk and str(record.get("final", {}).get("label", "")).lower() != risk.lower():
            continue

        results.append(extract_summary(b.name, record))
        if len(results) >= limit:
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieve predictions from blob storage")
    parser.add_argument("mode", choices=["last1h", "device", "risk"], help="Filter mode")
    parser.add_argument("value", nargs="?", help="Value for mode (device name or risk label)")
    args = parser.parse_args()

    since = None
    device = None
    risk = None

    if args.mode == "last1h":
        since = datetime.now(timezone.utc) - timedelta(hours=1)
    elif args.mode == "device":
        if not args.value:
            parser.error("Device mode requires a device name")
        device = args.value
    elif args.mode == "risk":
        if not args.value:
            parser.error("Risk mode requires a risk value (Low, Medium, High)")
        risk = args.value

    results = list_predictions(since=since, device_name=device, risk=risk, limit=10)

    if not results:
        print("No matching predictions found.")
    else:
        for line in results:
            print(line)


if __name__ == "__main__":
    main()
