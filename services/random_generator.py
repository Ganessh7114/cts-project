# services/random_generator.py
import random, json, time
from datetime import datetime, timedelta
from faker import Faker


fake = Faker()

device_mapping = {
    "Alaris GH": "Infusion Pump",
    "Baxter AK 96": "Dialysis Machine",
    "Baxter Flo-Gard": "Infusion Pump",
    "Datex Ohmeda S5": "Anesthesia Machine",
    "Drager Fabius Trio": "Anesthesia Machine",
    "Drager V500": "Patient Ventilator",
    "Fresenius 4008": "Dialysis Machine",
    "GE Aisys": "Anesthesia Machine",
    "GE Logiq E9": "Ultrasound Machine",
    "GE MAC 2000": "ECG Monitor",
    "GE Revolution": "CT Scanner",
    "Hamilton G5": "Patient Ventilator",
    "HeartStart FRx": "Defibrillator",
    "Lifepak 20": "Defibrillator",
    "NxStage System One": "Dialysis Machine",
    "Philips EPIQ": "Ultrasound Machine",
    "Philips HeartStrart": "Defibrillator",
    "Philips Ingenuity": "CT Scanner",
    "Phillips PageWriter": "ECG Monitor",
    "Puritan Bennett 980": "Patient Ventilator",
    "Siemens Acuson": "Ultrasound Machine",
    "Siemens S2000": "Ultrasound Machine",
    "Smiths Medfusion": "Infusion Pump",
    "Zoll R Series": "Defibrillator"
}

ClimateControl_list = ["Yes", "No"]
Location_list = [
    f"Hospital {h} - {region} Region"
    for h in list("ABCDEFGH")
    for region in ["Central","East","North","South","West"]
]

def generate_record(device_name, device_type):
    return {
        "DeviceType": device_type,
        "DeviceName": device_name,
        "RuntimeHours": round(random.uniform(102.32, 9999.85), 2),
        "TemperatureC": round(random.uniform(16.07, 40), 2),
        "PressureKPa": round(random.uniform(90, 120), 2),
        "VibrationMM_S": round(random.uniform(0, 1), 3),
        "CurrentDrawA": round(random.uniform(0.1, 10.5), 3),
        "SignalNoiseLevel": round(random.uniform(0, 5), 2),
        "ClimateControl": random.choice(ClimateControl_list),
        "HumidityPercent": round(random.uniform(20, 70), 2),
        "Location": random.choice(Location_list),
        "OperationalCycles": random.randint(5, 11887),
        "UserInteractionsPerDay": round(random.uniform(0, 30), 2),
        "ApproxDeviceAgeYears": round(random.uniform(0.1, 35.89), 2),
        "NumRepairs": random.randint(0, 19),
        "ErrorLogsCount": random.randint(0, 22),
        "SentTimestamp": datetime.utcnow().isoformat()
    }

def generate_random_payload(device_name=None, device_type=None):
    """
    Generate one telemetry payload.
    - If device_name/device_type are provided: use them (for round-robin).
    - Otherwise: pick a random device from mapping.
    Returns (json_str, dict).
    """
    if device_name is None or device_type is None:
        device_name = random.choice(list(device_mapping.keys()))
        device_type = device_mapping[device_name]

    record = generate_record(device_name, device_type)
    record["SentTimestamp"] = datetime.utcnow().isoformat()

    return json.dumps(record, separators=(",", ":"), ensure_ascii=False), record
