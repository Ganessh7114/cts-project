# workflows/api_server.py
import os
import sys
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import logging

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Allow "services/..." and "presentation/..." imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import services
from services.blob_storage import BlobStorage
from services.mqtt_service import MQTTService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Device Predictions API", version="1.0.0")

# Enhanced CORS middleware for SSE support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize services
blob_storage = BlobStorage()
mqtt_service = MQTTService(client_id_prefix="api-server")

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def parse_time_filter(since: str) -> Optional[datetime]:
    """Parse time filter strings like 'last1h', 'last6h', 'last24h'"""
    now = datetime.now(timezone.utc)
    
    if since == "last1h":
        return now - timedelta(hours=1)
    elif since == "last6h":
        return now - timedelta(hours=6)
    elif since == "last24h":
        return now - timedelta(hours=24)
    elif since == "last7d":
        return now - timedelta(days=7)
    else:
        return None

def format_prediction_record(blob_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Format prediction record for frontend consumption"""
    telemetry = record.get("telemetry", {})
    final = record.get("final", {})
    azure_ml = record.get("azure_ml", {})
    local_model = record.get("local_model", {})
    
    return {
        "telemetry": {
            "DeviceName": telemetry.get("DeviceName", "Unknown"),
            "DeviceType": telemetry.get("DeviceType", "Unknown"),
            "TemperatureC": telemetry.get("TemperatureC", 0),
            "VibrationMM_S": telemetry.get("VibrationMM_S", 0),
            "RuntimeHours": telemetry.get("RuntimeHours", 0),
            "PressureKPa": telemetry.get("PressureKPa"),
            "CurrentDrawA": telemetry.get("CurrentDrawA"),
            "SignalNoiseLevel": telemetry.get("SignalNoiseLevel"),
            "ClimateControl": telemetry.get("ClimateControl"),
            "HumidityPercent": telemetry.get("HumidityPercent"),
            "Location": telemetry.get("Location"),
            "OperationalCycles": telemetry.get("OperationalCycles"),
            "UserInteractionsPerDay": telemetry.get("UserInteractionsPerDay"),
            "ApproxDeviceAgeYears": telemetry.get("ApproxDeviceAgeYears"),
            "NumRepairs": telemetry.get("NumRepairs"),
            "ErrorLogsCount": telemetry.get("ErrorLogsCount"),
            "SentTimestamp": telemetry.get("SentTimestamp")
        },
        "final": {
            "label": final.get("label", "Unknown"),
            "confidence": final.get("confidence", 0),
            "device_name": final.get("device_name", telemetry.get("DeviceName")),
            "device_type": final.get("device_type", telemetry.get("DeviceType")),
            "factors": final.get("factors", [])
        },
        "azure_ml": {
            "ok": azure_ml.get("ok", False),
            "label": azure_ml.get("label"),
            "confidence": azure_ml.get("confidence"),
            "error": azure_ml.get("error")
        },
        "local_model": {
            "ok": local_model.get("ok", False),
            "label": local_model.get("label"),
            "confidence": local_model.get("confidence"),
            "error": local_model.get("error")
        },
        "timestamp": record.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "pipeline": record.get("pipeline", "unknown"),
        "device_name": telemetry.get("DeviceName", "Unknown"),  # Top level for easier access
        "device_type": telemetry.get("DeviceType", "Unknown")   # Top level for easier access
    }

# ---------------------------------------------------------------------
# API Routes (matching your frontend expectations)
# ---------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Health check endpoint - matches frontend expectation"""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/predictions")
def get_predictions(
    since: Optional[str] = Query(None, description="Time filter: last1h, last6h, last24h, last7d"),
    device: Optional[str] = Query(None, description="Filter by device name"),
    risk: Optional[str] = Query(None, description="Filter by risk level: Low, Medium, High"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results")
) -> List[Dict[str, Any]]:
    """
    Get predictions with optional filters - matches frontend expectation
    """
    try:
        # Get all recent blobs
        blobs = blob_storage.list_json_blobs(limit=limit * 2)  # Get more to allow filtering
        results = []
        
        # Parse time filter
        time_filter = None
        if since:
            time_filter = parse_time_filter(since)
        
        for blob in blobs:
            record = blob_storage.download_json(blob.name)
            if not record:
                continue
                
            # Apply time filter
            if time_filter:
                try:
                    record_time = datetime.fromisoformat(record.get("timestamp", "").replace('Z', '+00:00'))
                    if record_time < time_filter:
                        continue
                except (ValueError, TypeError):
                    continue
            
            # Apply device filter
            if device:
                device_name = record.get("telemetry", {}).get("DeviceName", "")
                if device.lower() not in device_name.lower():
                    continue
            
            # Apply risk filter
            if risk:
                risk_label = record.get("final", {}).get("label", "")
                if risk.lower() != risk_label.lower():
                    continue
            
            # Format and add to results
            formatted_record = format_prediction_record(blob.name, record)
            results.append(formatted_record)
            
            if len(results) >= limit:
                break
        
        logger.info(f"Returning {len(results)} predictions")
        return results
        
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")

@app.get("/api/stream")
async def stream_predictions():
    """
    Server-Sent Events endpoint for real-time predictions - matches frontend expectation
    """
    async def event_generator():
        """Generate SSE events from MQTT or periodic polling"""
        try:
            while True:
                # Method 1: Try to get latest prediction from blob storage
                try:
                    latest_blobs = blob_storage.list_json_blobs(limit=1)
                    if latest_blobs:
                        record = blob_storage.download_json(latest_blobs[0].name)
                        if record:
                            formatted_record = format_prediction_record(latest_blobs[0].name, record)
                            yield f"data: {json.dumps(formatted_record)}\n\n"
                except Exception as e:
                    logger.error(f"Error fetching latest prediction: {e}")
                
                # Wait before next poll (adjust frequency as needed)
                await asyncio.sleep(5)  # Poll every 5 seconds
                
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

# ---------------------------------------------------------------------
# Additional Utility Endpoints
# ---------------------------------------------------------------------

@app.get("/api/devices")
def get_devices() -> List[str]:
    """Get list of all device names"""
    try:
        blobs = blob_storage.list_json_blobs(limit=100)
        devices = set()
        
        for blob in blobs:
            record = blob_storage.download_json(blob.name)
            if record:
                device_name = record.get("telemetry", {}).get("DeviceName")
                if device_name:
                    devices.add(device_name)
        
        return sorted(list(devices))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch devices: {str(e)}")

@app.get("/api/stats")
def get_dashboard_stats() -> Dict[str, Any]:
    """Get overall dashboard statistics"""
    try:
        blobs = blob_storage.list_json_blobs(limit=100)
        stats = {
            "total_devices": 0,
            "risk_distribution": {"Low": 0, "Medium": 0, "High": 0},
            "total_predictions": len(blobs),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        devices = set()
        
        for blob in blobs:
            record = blob_storage.download_json(blob.name)
            if record:
                device_name = record.get("telemetry", {}).get("DeviceName")
                if device_name:
                    devices.add(device_name)
                
                risk_level = record.get("final", {}).get("label", "Unknown")
                if risk_level in stats["risk_distribution"]:
                    stats["risk_distribution"][risk_level] += 1
        
        stats["total_devices"] = len(devices)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

# ---------------------------------------------------------------------
# Run the server
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Medical Device Predictions API server...")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
