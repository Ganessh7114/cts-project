import json
import os
import threading
import time
import configparser
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import requests
import logging
from flask import Flask, request, jsonify
from flask import send_from_directory
from flask import Blueprint, render_template, url_for
from flask_cors import CORS
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.core.exceptions import AzureError
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Azure_ML_Flask_API")

app = Flask(__name__)
CORS(app)

class AzureMLManager:
    def __init__(self):
        self.ml_client = None
        self.endpoint_url = None
        self.api_key = None
        self.deployment_name = None
        self.is_connected = False
        self.model_info = {}
        self.initialize_azure_ml()
    
    def initialize_azure_ml(self):
        """Initialize Azure ML connection"""
        try:
            # Load Azure ML configuration from environment variables
            subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
            resource_group = os.getenv('AZURE_RESOURCE_GROUP')
            workspace_name = os.getenv('AZURE_ML_WORKSPACE')
            
            # Endpoint configuration
            self.endpoint_url = os.getenv('AZURE_ML_ENDPOINT_URL')
            self.api_key = os.getenv('AZURE_ML_API_KEY')
            self.deployment_name = os.getenv('AZURE_ML_DEPLOYMENT_NAME', 'medical-device-prediction')
            
            if not all([subscription_id, resource_group, workspace_name]):
                raise ValueError("Missing required Azure ML configuration")
            
            # Initialize ML Client with service principal or default credentials
            tenant_id = os.getenv('AZURE_TENANT_ID')
            client_id = os.getenv('AZURE_CLIENT_ID')
            client_secret = os.getenv('AZURE_CLIENT_SECRET')
            
            if all([tenant_id, client_id, client_secret]):
                credential = ClientSecretCredential(tenant_id, client_id, client_secret)
                logger.info("Using service principal authentication")
            else:
                credential = DefaultAzureCredential()
                logger.info("Using default Azure credentials")
            
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            
            # Test connection
            workspaces = list(self.ml_client.workspaces.list())
            logger.info(f"Connected to Azure ML workspace: {workspace_name}")
            
            self.is_connected = True
            self.model_info = {
                'workspace': workspace_name,
                'endpoint_url': self.endpoint_url,
                'deployment': self.deployment_name,
                'connected_at': datetime.now().isoformat(),
                'status': 'connected'
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML: {e}")
            self.is_connected = False
            self.model_info = {
                'status': 'error',
                'error': str(e),
                'attempted_at': datetime.now().isoformat()
            }
    
    def prepare_prediction_payload(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data payload for Azure ML prediction"""
        # Map device data to expected model features
        payload = {
            "data": [{
                "DeviceType": device_data.get('device_type', 'Unknown'),
                "DeviceName": device_data.get('device_name', 'Unknown'),
                "RuntimeHours": float(device_data.get('runtime_hours', 0)),
                "TemperatureC": float(device_data.get('temperature', 25.0)),
                "PressureKPa": float(device_data.get('pressure', 100.0)),
                "VibrationMM_S": float(device_data.get('vibration', 0.1)),
                "CurrentDrawA": float(device_data.get('current_draw', 1.0)),
                "SignalNoiseLevel": float(device_data.get('signal_noise', 1.0)),
                "ClimateControl": device_data.get('climate_control', 'Yes'),
                "HumidityPercent": float(device_data.get('humidity', 50.0)),
                "Location": device_data.get('location', 'Unknown'),
                "OperationalCycles": int(device_data.get('operational_cycles', 1000)),
                "UserInteractionsPerDay": float(device_data.get('user_interactions', 5.0)),
                "LastServiceDate": device_data.get('last_service_date', datetime.now().strftime("%d-%m-%Y")),
                "ApproxDeviceAgeYears": float(device_data.get('device_age', 1.0)),
                "NumRepairs": int(device_data.get('repairs', 0)),
                "ErrorLogsCount": int(device_data.get('error_logs', 0))
            }]
        }
        return payload
    
    async def predict_async(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make async prediction using Azure ML endpoint"""
        if not self.is_connected or not self.endpoint_url:
            raise Exception("Azure ML not connected or endpoint not configured")
        
        payload = self.prepare_prediction_payload(device_data)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}' if self.api_key else None,
            'azureml-model-deployment': self.deployment_name
        }
        
        # Remove None authorization header if no API key
        if not self.api_key:
            del headers['Authorization']
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self.process_prediction_response(result, device_data)
                    else:
                        error_text = await response.text()
                        raise Exception(f"Azure ML API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Azure ML prediction failed: {e}")
            raise
    
    def predict_sync(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous prediction wrapper"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.predict_async(device_data))
        except Exception as e:
            logger.error(f"Sync prediction failed: {e}")
            raise
    
    def process_prediction_response(self, azure_response: Dict[str, Any], original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Azure ML response into standardized format"""
        try:
            # Extract prediction from Azure ML response
            # Adapt this based on your actual Azure ML model output format
            if 'predictions' in azure_response:
                predictions = azure_response['predictions']
                if isinstance(predictions, list) and len(predictions) > 0:
                    prediction_data = predictions[0]
                else:
                    prediction_data = predictions
            elif 'result' in azure_response:
                prediction_data = azure_response['result']
            else:
                prediction_data = azure_response
            
            # Extract risk level and confidence
            risk_level = 'Medium'  # Default
            confidence = 0.75     # Default
            risk_score = 0.5      # Default
            
            # Parse different possible response formats
            if isinstance(prediction_data, dict):
                risk_level = prediction_data.get('risk_level', prediction_data.get('prediction', 'Medium'))
                confidence = float(prediction_data.get('confidence', prediction_data.get('probability', 0.75)))
                risk_score = float(prediction_data.get('risk_score', confidence))
            elif isinstance(prediction_data, (list, np.ndarray)):
                # If it's probabilities array [low_prob, medium_prob, high_prob]
                if len(prediction_data) >= 3:
                    probs = prediction_data
                    max_idx = np.argmax(probs)
                    risk_levels = ['Low', 'Medium', 'High']
                    risk_level = risk_levels[max_idx]
                    confidence = float(max(probs))
                    risk_score = float(probs[2])  # High risk probability
                else:
                    # Binary classification
                    risk_score = float(prediction_data[0])
                    confidence = abs(risk_score - 0.5) * 2  # Convert to confidence
                    risk_level = 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low'
            
            # Analyze risk factors based on input data
            risk_factors = self.analyze_risk_factors(original_data, risk_score)
            
            return {
                'prediction': risk_level,
                'confidence': round(confidence, 3),
                'risk_score': round(risk_score, 3),
                'factors': risk_factors,
                'model_version': f"Azure ML {self.deployment_name}",
                'timestamp': datetime.now().isoformat(),
                'model_used': True,
                'azure_ml_response': prediction_data  # Include raw response for debugging
            }
            
        except Exception as e:
            logger.error(f"Error processing Azure ML response: {e}")
            # Return basic response structure
            return {
                'prediction': 'Medium',
                'confidence': 0.5,
                'risk_score': 0.5,
                'factors': ['Response processing error'],
                'model_version': f"Azure ML {self.deployment_name} (Error)",
                'timestamp': datetime.now().isoformat(),
                'model_used': True,
                'error': str(e)
            }
    
    def analyze_risk_factors(self, data: Dict[str, Any], risk_score: float) -> List[str]:
        """Analyze risk factors based on device parameters"""
        factors = []
        
        # Temperature analysis
        temp = float(data.get('temperature', 25))
        if temp > 35:
            factors.append(f"High temperature ({temp:.1f}°C)")
        elif temp < 15:
            factors.append(f"Low temperature ({temp:.1f}°C)")
        
        # Vibration analysis
        vibration = float(data.get('vibration', 0.1))
        if vibration > 0.8:
            factors.append(f"Excessive vibration ({vibration:.2f})")
        
        # Error logs analysis
        errors = int(data.get('error_logs', 0))
        if errors > 15:
            factors.append(f"High error count ({errors} logs)")
        
        # Runtime analysis
        runtime = float(data.get('runtime_hours', 1000))
        if runtime > 8000:
            factors.append(f"Extended runtime ({runtime:.0f} hours)")
        
        # Age analysis
        age = float(data.get('device_age', 1))
        if age > 5:
            factors.append(f"Aging device ({age:.1f} years)")
        
        # Repairs analysis
        repairs = int(data.get('repairs', 0))
        if repairs > 8:
            factors.append(f"Frequent repairs ({repairs} repairs)")
        
        # Pressure analysis
        pressure = float(data.get('pressure', 100))
        if pressure > 150:
            factors.append(f"High pressure ({pressure:.0f} kPa)")
        elif pressure < 80:
            factors.append(f"Low pressure ({pressure:.0f} kPa)")
        
        # Current draw analysis
        current = float(data.get('current_draw', 1))
        if current > 10:
            factors.append(f"High power consumption ({current:.1f}A)")
        
        if not factors:
            if risk_score > 0.7:
                factors.append("High risk indicated by model - check all parameters")
            elif risk_score < 0.3:
                factors.append("All parameters within normal ranges")
            else:
                factors.append("Moderate risk - monitor device closely")
        
        return factors

    async def batch_predict_async(self, devices_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch prediction using Azure ML"""
        if not self.is_connected or not self.endpoint_url:
            raise Exception("Azure ML not connected")
        
        # Prepare batch payload
        batch_payload = {
            "data": [self.prepare_prediction_payload(device)["data"][0] for device in devices_data]
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}' if self.api_key else None,
            'azureml-model-deployment': self.deployment_name
        }
        
        if not self.api_key:
            del headers['Authorization']
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint_url,
                    json=batch_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return [
                            self.process_prediction_response(pred, devices_data[i]) 
                            for i, pred in enumerate(result.get('predictions', [result]))
                        ]
                    else:
                        error_text = await response.text()
                        raise Exception(f"Azure ML batch API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Azure ML batch prediction failed: {e}")
            raise

# Initialize Azure ML Manager
azure_ml = AzureMLManager()

@app.route("/", methods=["GET"])
def home():
    return {
        "success": True,
        "message": "Welcome to the Medical Device Failure Prediction API",
        "available_endpoints": [
            "/api/status",
            "/api/model-info",
            "/api/predict"
        ]
    }


# Flask API Routes
@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API and model status"""
    return jsonify({
        'status': 'active',
        'model_loaded': azure_ml.is_connected,
        'model_version': azure_ml.model_info.get('deployment', 'Unknown'),
        'azure_ml_status': azure_ml.model_info,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    return jsonify({
        'success': True,
        'model_info': {
            'is_connected': azure_ml.is_connected,
            'workspace': azure_ml.model_info.get('workspace'),
            'deployment': azure_ml.deployment_name,
            'endpoint_url': azure_ml.endpoint_url is not None,
            'status': azure_ml.model_info.get('status'),
            'last_updated': azure_ml.model_info.get('connected_at'),
            'error': azure_ml.model_info.get('error')
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict_single():
    """Single device prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['device_name', 'temperature', 'vibration', 'error_logs', 'runtime_hours']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Make prediction using Azure ML
        try:
            prediction = azure_ml.predict_sync(data)
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'azure_ml_connected': azure_ml.is_connected,
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as ml_error:
            logger.error(f"Azure ML prediction error: {ml_error}")
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {str(ml_error)}',
                'azure_ml_connected': False,
                'timestamp': datetime.now().isoformat()
            }), 503
    
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def predict_batch():
    """Batch device prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'devices' not in data:
            return jsonify({
                'success': False,
                'error': 'No devices data provided'
            }), 400
        
        devices_data = data['devices']
        
        if not isinstance(devices_data, list) or len(devices_data) == 0:
            return jsonify({
                'success': False,
                'error': 'Devices data must be a non-empty list'
            }), 400
        
        # Make batch prediction using Azure ML
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            predictions = loop.run_until_complete(
                azure_ml.batch_predict_async(devices_data)
            )
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'count': len(predictions),
                'azure_ml_connected': azure_ml.is_connected,
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as ml_error:
            logger.error(f"Azure ML batch prediction error: {ml_error}")
            return jsonify({
                'success': False,
                'error': f'Batch prediction failed: {str(ml_error)}',
                'azure_ml_connected': False,
                'timestamp': datetime.now().isoformat()
            }), 503
    
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/reconnect', methods=['POST'])
def reconnect_azure_ml():
    """Reconnect to Azure ML"""
    try:
        azure_ml.initialize_azure_ml()
        
        return jsonify({
            'success': True,
            'connected': azure_ml.is_connected,
            'status': azure_ml.model_info,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Reconnection error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'azure_ml_connected': azure_ml.is_connected,
        'timestamp': datetime.now().isoformat(),
        'uptime': time.time()
    })

@app.route("/api-ui")
def api_ui_index():
    # serves api_ui/index.html
    return send_from_directory("api_ui", "index.html")

@app.route("/api-ui/<path:filename>")
def api_ui_assets(filename):
    # serves all other files under api_ui (html, css, js)
    return send_from_directory("api_ui", filename)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Initialize Flask app
if __name__ == '__main__':
    # Check environment variables on startup
    required_env_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP', 
        'AZURE_ML_WORKSPACE',
        'AZURE_ML_ENDPOINT_URL'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Azure ML integration may not work properly")
    
    logger.info("Starting Flask API server with Azure ML integration...")
    logger.info(f"Azure ML Connected: {azure_ml.is_connected}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)