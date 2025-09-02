// src/components/alerts/AlertPanel.tsx
import React from 'react';
import { PredictionRecord } from '../../services/types';
import { AlertItem } from './AlertItem';

interface AlertPanelProps {
  alerts: PredictionRecord[];
}

export const AlertPanel: React.FC<AlertPanelProps> = ({ alerts }) => {
  if (alerts.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-green-600 text-lg font-medium">✅ All Clear</div>
        <p className="text-gray-500 text-sm mt-1">
          No high or medium risk devices detected
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3 max-h-96 overflow-y-auto">
      {alerts.map((prediction) => (
        <AlertItem
          key={prediction.telemetry.DeviceName}
          prediction={prediction}
        />
      ))}
    </div>
  );
};
