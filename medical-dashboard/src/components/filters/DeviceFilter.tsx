// src/components/filters/DeviceFilter.tsx
import React from 'react';

interface DeviceFilterProps {
  availableDevices: string[];
  selectedDevice: string;
  onDeviceChange: (device: string) => void;
}

export const DeviceFilter: React.FC<DeviceFilterProps> = ({
  availableDevices,
  selectedDevice,
  onDeviceChange
}) => {
  return (
    <div className="flex items-center space-x-2">
      <label className="text-sm font-medium text-gray-700">
        Device:
      </label>
      <select
        value={selectedDevice}
        onChange={(e) => onDeviceChange(e.target.value)}
        className="px-3 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 min-w-32"
      >
        <option value="">All Devices</option>
        {availableDevices.map(device => (
          <option key={device} value={device}>
            {device}
          </option>
        ))}
      </select>
    </div>
  );
};
