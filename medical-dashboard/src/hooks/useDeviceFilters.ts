// src/hooks/useDeviceFilters.ts
import { useState, useMemo } from 'react';
import { PredictionRecord, DeviceFilter, RiskLevel } from '../services/types';

export const useDeviceFilters = (data: PredictionRecord[]) => {
  const [filters, setFilters] = useState<DeviceFilter>({});

  // Get unique device names for filter dropdown
  const availableDevices = useMemo(() => {
    const devices = new Set(data.map(p => p.telemetry.DeviceName));
    return Array.from(devices).sort();
  }, [data]);

  // Get unique device types
  const availableDeviceTypes = useMemo(() => {
    const types = new Set(data.map(p => p.telemetry.DeviceType));
    return Array.from(types).sort();
  }, [data]);

  // Apply filters to data
  const filteredData = useMemo(() => {
    return data.filter(prediction => {
      if (filters.deviceName && 
          !prediction.telemetry.DeviceName.toLowerCase().includes(filters.deviceName.toLowerCase())) {
        return false;
      }
      
      if (filters.riskLevel && prediction.final.label !== filters.riskLevel) {
        return false;
      }
      
      return true;
    });
  }, [data, filters]);

  const updateFilter = (key: keyof DeviceFilter, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const clearFilters = () => {
    setFilters({});
  };

  return {
    filters,
    setFilters,
    updateFilter,
    clearFilters,
    filteredData,
    availableDevices,
    availableDeviceTypes
  };
};
