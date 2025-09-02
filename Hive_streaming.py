import json
import random
import threading
import time
import csv
import os
from queue import Queue
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
import paho.mqtt.client as mqtt
from faker import Faker
from azure.storage.blob import BlobServiceClient
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQTT_ML_Streaming")

# ---------------------------
# MQTT Config & Queues
# ---------------------------

load_dotenv()

BROKER = "863b2bea6a5246238f2ae57eac2dc400.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "Bharathnath"
PASSWORD = "#Bharath123"
TOPIC = "iot/failure"

fake = Faker()
iot_queue = Queue()

DeviceType_list = ["Anesthesia Machine", "CT Scanner", "Defibrillator", "Dialysis Machine",
                  "ECG Monitor", "Infusion Pump", "Patient Ventilator", "Ultrasound Machine"]

DeviceName_list = ["Alaris GH", "Baxter AK 96", "Baxter Flo-Gard", "Datex Ohmeda S5", "Drager Fabius Trio",
                  "Drager V500", "Fresenius 4008", "GE Aisys", "GE Logiq E9", "GE MAC 2000", "GE Revolution",
                  "Hamilton G5", "HeartStart FRx", "Lifepak 20", "NxStage System One", "Philips EPIQ",
                  "Philips HeartStrart", "Philips Ingenuity", "Phillips PageWriter", "Puritan Bennett 980",
                  "Siemens Acuson", "Siemens S2000", "Smiths Medfusion", "Zoll R Series"]

ClimateControl_list = ["Yes", "No"]

Location_list = [
    "Hospital A - Central Region", "Hospital A - East Region", "Hospital A - North Region", "Hospital A - South Region", "Hospital A - West Region",
    "Hospital B - Central Region", "Hospital B - East Region", "Hospital B - North Region", "Hospital B - South Region", "Hospital B - West Region",
    "Hospital C - Central Region", "Hospital C - East Region", "Hospital C - North Region", "Hospital C - South Region", "Hospital C - West Region",
    "Hospital D - Central Region", "Hospital D - East Region", "Hospital D - North Region", "Hospital D - South Region", "Hospital D - West Region",
    "Hospital E - Central Region", "Hospital E - East Region", "Hospital E - North Region", "Hospital E - South Region", "Hospital E - West Region",
    "Hospital F - Central Region", "Hospital F - East Region", "Hospital F - North Region", "Hospital F - South Region", "Hospital F - West Region",
    "Hospital G - Central Region", "Hospital G - East Region", "Hospital G - North Region", "Hospital G - South Region", "Hospital G - West Region",
    "Hospital H - Central Region", "Hospital H - East Region", "Hospital H - North Region", "Hospital H - South Region", "Hospital H - West Region"
]

# Azure Blob Storage connection info
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
LOCAL_CSV_FILENAME = "predictions_output.csv"

try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    logger.info("Azure Blob Storage client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Azure Blob Storage: {e}")
    blob_service_client = None
    blob_container_client = None

# Load ML pipeline
MODEL_PATH = "xgboost_pipeline.pkl"  # Update this path as needed
try:
    if os.path.exists(MODEL_PATH):
        ml_pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Loaded ML pipeline from {MODEL_PATH}")
    else:
        logger.warning(f"ML model file not found at {MODEL_PATH}")
        logger.info("Creating a mock predictor for demonstration purposes")
        ml_pipeline = None
except Exception as e:
    logger.error(f"Failed to load ML pipeline: {e}")
    ml_pipeline = None

# ---------------------------
# ML Prediction Function
# ---------------------------
def predict_failure_risk(data):
    """Apply ML prediction to incoming data"""
    try:
        if ml_pipeline is None:
            # Mock prediction for demonstration - replace with actual logic
            # Based on some simple rules for realistic simulation
            risk_score = 0
            
            # Higher risk factors
            if data.get('RuntimeHours', 0) > 8000:
                risk_score += 0.3
            if data.get('TemperatureC', 0) > 35:
                risk_score += 0.2
            if data.get('VibrationMM_S', 0) > 0.8:
                risk_score += 0.2
            if data.get('ErrorLogsCount', 0) > 15:
                risk_score += 0.2
            if data.get('NumRepairs', 0) > 10:
                risk_score += 0.3
                
            # Determine risk level
            if risk_score >= 0.7:
                return "High"
            elif risk_score >= 0.4:
                return "Medium"
            else:
                return "Low"
        
        # Create DataFrame with single record
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = ml_pipeline.predict(df)[0]
        return str(prediction)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Prediction_Error"

# ---------------------------
# MQTT Publisher (simulate records)
# ---------------------------
def publish_simulated(max_records=10):
    """Publish simulated IoT data to MQTT"""
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        client.username_pw_set(USERNAME, PASSWORD)
        client.tls_set()
        client.connect(BROKER, PORT)
        client.loop_start()
        
        count = 0
        while count < max_records:
            record = {
                "DeviceType": random.choice(DeviceType_list),
                "DeviceName": random.choice(DeviceName_list),
                "RuntimeHours": round(random.uniform(102.32, 9999.85), 2),
                "TemperatureC": round(random.uniform(16.07, 40), 2),
                "PressureKPa": round(random.uniform(90, 120), 2),
                "VibrationMM_S": round(random.uniform(0, 1), 3),
                "CurrentDrawA": round(random.uniform(0.1, 1.5), 3),
                "SignalNoiseLevel": round(random.uniform(0, 5), 2),
                "ClimateControl": random.choice(ClimateControl_list),
                "HumidityPercent": round(random.uniform(20, 70), 2),
                "Location": random.choice(Location_list),
                "OperationalCycles": random.randint(5, 11887),
                "UserInteractionsPerDay": round(random.uniform(0, 26.4), 2),
                "LastServiceDate": fake.date_between(start_date="-2y", end_date="today").strftime("%d-%m-%Y"),
                "ApproxDeviceAgeYears": round(random.uniform(0.1, 35.89), 2),
                "NumRepairs": random.randint(0, 19),
                "ErrorLogsCount": random.randint(0, 22)
            }
            
            client.publish(TOPIC, json.dumps(record))
            count += 1
            time.sleep(0.1)  # Small delay between messages
            
        client.loop_stop()
        client.disconnect()
        logger.info(f"Finished publishing {max_records} simulated records")
        
    except Exception as e:
        logger.error(f"Error in MQTT publisher: {e}")

# ---------------------------
# MQTT Subscriber
# ---------------------------
def on_connect(client, userdata, flags, rc):
    """Callback for MQTT connection"""
    if rc == 0:
        logger.info("Connected to MQTT broker")
        client.subscribe(TOPIC)
    else:
        logger.error(f"Failed to connect to MQTT broker: {rc}")

def on_message(client, userdata, msg):
    """Callback for incoming MQTT messages"""
    try:
        data = json.loads(msg.payload.decode())
        iot_queue.put(data)
        logger.debug(f"Received message: {data.get('DeviceType', 'Unknown')}")
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

def mqtt_subscribe():
    """Subscribe to MQTT topic and listen for messages"""
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        client.username_pw_set(USERNAME, PASSWORD)
        client.tls_set()
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.connect(BROKER, PORT, 60)
        client.loop_forever()
        
    except Exception as e:
        logger.error(f"Error in MQTT subscriber: {e}")

# ---------------------------
# Save and Upload Functions
# ---------------------------
def save_to_csv(records, filename=LOCAL_CSV_FILENAME):
    """Save records to local CSV file"""
    if not records:
        return
    
    file_exists = os.path.exists(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        fieldnames = records[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(records)
    
    logger.info(f"Saved {len(records)} records to {filename}")

def upload_to_azure(filename=LOCAL_CSV_FILENAME):
    """Upload CSV file to Azure Blob Storage"""
    if blob_container_client is None:
        logger.warning("Azure Blob Storage not available, skipping upload")
        return
    
    try:
        blob_name = f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, "rb") as data:
            blob_container_client.upload_blob(blob_name, data, overwrite=True)
        
        logger.info(f"Uploaded '{blob_name}' to Azure Blob Storage")
        
    except Exception as e:
        logger.error(f"Failed to upload to Azure: {e}")

# ---------------------------
# Main Processing Function
# ---------------------------
def process_iot_data():
    """Main function to process IoT data with ML predictions"""
    batch_size = 10  # Reduced batch size for faster processing
    batch_records = []
    processed_count = 0
    
    logger.info("Starting IoT data processing...")
    
    while True:
        try:
            if not iot_queue.empty():
                # Get data from queue
                data = iot_queue.get()
                
                # Apply ML prediction
                prediction = predict_failure_risk(data)
                
                # Add prediction to data
                data['PredictedFailureRisk'] = prediction
                data['ProcessedTimestamp'] = datetime.utcnow().isoformat()
                
                # Log the result
                logger.info(f"Processed: {data['DeviceType']} - {data['DeviceName']} -> Risk: {prediction}")
                
                # Add to batch
                batch_records.append(data)
                processed_count += 1
                
                # Save batch when it reaches batch_size
                if len(batch_records) >= batch_size:
                    save_to_csv(batch_records)
                    upload_to_azure()
                    batch_records.clear()
                    logger.info(f"Processed batch. Total processed: {processed_count}")
                
                # Mark task as done
                iot_queue.task_done()
            else:
                time.sleep(0.1)  # Small delay when queue is empty
                
        except KeyboardInterrupt:
            logger.info("Stopping data processing...")
            # Save remaining records
            if batch_records:
                save_to_csv(batch_records)
                upload_to_azure()
            break
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            time.sleep(1)

# ---------------------------
# Main Execution
# ---------------------------
def main():
    """Main function to start all components"""
    logger.info("Starting IoT ML Streaming Application")
    
    # Start MQTT subscriber thread
    subscriber_thread = threading.Thread(target=mqtt_subscribe, daemon=True)
    subscriber_thread.start()
    logger.info("MQTT subscriber started")
    
    # Wait a moment for subscriber to connect
    time.sleep(2)
    
    # Start MQTT publisher thread
    publisher_thread = threading.Thread(target=publish_simulated, args=(100,), daemon=True)
    publisher_thread.start()
    logger.info("MQTT publisher started")
    
    # Wait a moment for publisher to start
    time.sleep(1)
    
    # Start main processing
    try:
        process_iot_data()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()