// src/services/types.ts
export interface TelemetryData {
  DeviceName: string;
  DeviceType: string;
  TemperatureC: number;
  VibrationMM_S: number;
  RuntimeHours: number;
  PressureKPa?: number;
  CurrentDrawA?: number;
  SignalNoiseLevel?: number;
  ClimateControl?: string;
  HumidityPercent?: number;
  Location?: string;
  OperationalCycles?: number;
  UserInteractionsPerDay?: number;
  ApproxDeviceAgeYears?: number;
  NumRepairs?: number;
  ErrorLogsCount?: number;
  SentTimestamp?: string;
}

export interface FinalPrediction {
  label: 'Low' | 'Medium' | 'High';
  confidence: number;
  device_name?: string;
  device_type?: string;
  factors?: string[];
}

export interface PredictionRecord {
  telemetry: TelemetryData;
  final: FinalPrediction;
  timestamp: string;
  azure_ml?: {
    ok: boolean;
    label?: string;
    confidence?: number;
    error?: string;
  };
  local_model?: {
    ok: boolean;
    label?: string;
    confidence?: number;
    error?: string;
  };
  device_name?: string;
  device_type?: string;
  pipeline?: string;
}

export interface APIResponse<T> {
  data: T;
  success: boolean;
  error?: string;
}

export type RiskLevel = 'Low' | 'Medium' | 'High';

export interface DeviceFilter {
  deviceName?: string;
  riskLevel?: RiskLevel;
  timeRange?: string;
}
