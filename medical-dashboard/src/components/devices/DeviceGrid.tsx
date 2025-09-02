// src/components/devices/DeviceGrid.tsx
import React, { useState } from 'react';
import { PredictionRecord } from '../../services/types';
import { DeviceCard } from './DeviceCard';
import { DeviceDetails } from './DeviceDetails';

interface DeviceGridProps {
  predictions: PredictionRecord[];
}

export const DeviceGrid: React.FC<DeviceGridProps> = ({ predictions }) => {
  const [selectedDevice, setSelectedDevice] = useState<PredictionRecord | null>(null);

  if (predictions.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-400 text-lg">No devices found</div>
        <p className="text-gray-500 text-sm mt-2">
          Waiting for device predictions to arrive...
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {predictions.map((prediction) => (
          <DeviceCard
            key={prediction.telemetry.DeviceName}
            prediction={prediction}
            onClick={() => setSelectedDevice(prediction)}
          />
        ))}
      </div>

      {/* Device Details Modal */}
      {selectedDevice && (
        <DeviceDetails
          prediction={selectedDevice}
          onClose={() => setSelectedDevice(null)}
        />
      )}
    </>
  );
};
