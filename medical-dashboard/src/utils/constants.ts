// src/utils/constants.ts
export const RISK_COLORS = {
  Low: {
    bg: 'bg-green-50',
    border: 'border-green-300',
    text: 'text-green-800',
    indicator: 'bg-green-500',
    chart: '#10B981'
  },
  Medium: {
    bg: 'bg-yellow-50',
    border: 'border-yellow-400',
    text: 'text-yellow-800',
    indicator: 'bg-yellow-500',
    chart: '#F59E0B'
  },
  High: {
    bg: 'bg-red-50',
    border: 'border-red-400',
    text: 'text-red-800',
    indicator: 'bg-red-500',
    chart: '#EF4444'
  }
};

export const DEVICE_TYPES = [
  'CT Scanner',
  'MRI Scanner',
  'Ultrasound Machine',
  'ECG Monitor',
  'Patient Ventilator',
  'Anesthesia Machine',
  'Dialysis Machine',
  'Infusion Pump',
  'Defibrillator'
];

export const TELEMETRY_UNITS = {
  TemperatureC: '°C',
  VibrationMM_S: 'mm/s',
  RuntimeHours: 'hours',
  PressureKPa: 'kPa',
  CurrentDrawA: 'A',
  HumidityPercent: '%'
};

export const UPDATE_INTERVALS = {
  HEALTH_CHECK: 30000, // 30 seconds
  SSE_RECONNECT_BASE: 1000, // 1 second base for exponential backoff
  CHART_REFRESH: 60000 // 1 minute
};
