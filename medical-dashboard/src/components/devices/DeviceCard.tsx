// src/components/devices/DeviceCard.tsx
import React from 'react';
import { PredictionRecord, RiskLevel } from '../../services/types';
import { RiskIndicator } from './RiskIndicator';

interface DeviceCardProps {
  prediction: PredictionRecord;
  onClick?: () => void;
}

export const DeviceCard: React.FC<DeviceCardProps> = ({ prediction, onClick }) => {
  const { telemetry, final, timestamp } = prediction;
  
  const getRiskColorClasses = (risk: RiskLevel): string => {
    const baseClasses = "rounded-lg border-2 p-4 shadow-md transition-all duration-200 hover:shadow-lg cursor-pointer";
    
    switch (risk) {
      case 'Low':
        return `${baseClasses} bg-green-50 border-green-300 hover:bg-green-100`;
      case 'Medium':
        return `${baseClasses} bg-yellow-50 border-yellow-400 hover:bg-yellow-100`;
      case 'High':
        return `${baseClasses} bg-red-50 border-red-400 hover:bg-red-100`;
      default:
        return `${baseClasses} bg-gray-50 border-gray-300 hover:bg-gray-100`;
    }
  };

  const formatTimeAgo = (timestamp: string): string => {
    const now = new Date();
    const predTime = new Date(timestamp);
    const diffMinutes = Math.floor((now.getTime() - predTime.getTime()) / (1000 * 60));
    
    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <div className={getRiskColorClasses(final.label)} onClick={onClick}>
      {/* Header */}
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className="font-semibold text-lg text-gray-900">
            {telemetry.DeviceName}
          </h3>
          <p className="text-sm text-gray-600">{telemetry.DeviceType}</p>
        </div>
        <RiskIndicator risk={final.label} confidence={final.confidence} />
      </div>

      {/* Telemetry Grid */}
      <div className="grid grid-cols-2 gap-3 text-sm mb-3">
        <div className="flex justify-between">
          <span className="text-gray-600">Temperature:</span>
          <span className="font-medium">{telemetry.TemperatureC}°C</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-600">Vibration:</span>
          <span className="font-medium">{telemetry.VibrationMM_S}mm/s</span>
        </div>
        <div className="flex justify-between col-span-2">
          <span className="text-gray-600">Runtime:</span>
          <span className="font-medium">{telemetry.RuntimeHours.toLocaleString()}h</span>
        </div>
      </div>

      {/* Factors */}
      {final.factors && final.factors.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-gray-500 mb-1">Risk Factors:</p>
          <div className="flex flex-wrap gap-1">
            {final.factors.slice(0, 2).map((factor, index) => (
              <span key={index} className="inline-block bg-gray-200 text-gray-700 text-xs px-2 py-1 rounded">
                {factor}
              </span>
            ))}
            {final.factors.length > 2 && (
              <span className="inline-block bg-gray-200 text-gray-700 text-xs px-2 py-1 rounded">
                +{final.factors.length - 2} more
              </span>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="flex justify-between items-center pt-3 border-t border-gray-200">
        <span className="text-xs text-gray-500">
          Confidence: {Math.round(final.confidence * 100)}%
        </span>
        <span className="text-xs text-gray-500 flex items-center">
          <span className="mr-1">⏰</span>
          {formatTimeAgo(timestamp)}
        </span>
      </div>
    </div>
  );
};
