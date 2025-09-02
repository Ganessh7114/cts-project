// src/components/alerts/AlertItem.tsx
import React from 'react';
import { PredictionRecord } from '../../services/types';

interface AlertItemProps {
  prediction: PredictionRecord;
}

export const AlertItem: React.FC<AlertItemProps> = ({ prediction }) => {
  const { telemetry, final, timestamp } = prediction;
  
  const getAlertColor = (risk: string) => {
    switch (risk) {
      case 'High':
        return 'border-red-400 bg-red-50';
      case 'Medium':
        return 'border-yellow-400 bg-yellow-50';
      default:
        return 'border-gray-400 bg-gray-50';
    }
  };

  const getAlertIcon = (risk: string) => {
    switch (risk) {
      case 'High':
        return '🚨';
      case 'Medium':
        return '⚠️';
      default:
        return 'ℹ️';
    }
  };

  const formatTimeAgo = (timestamp: string): string => {
    const now = new Date();
    const alertTime = new Date(timestamp);
    const diffMinutes = Math.floor((now.getTime() - alertTime.getTime()) / (1000 * 60));
    
    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <div className={`border rounded-lg p-3 ${getAlertColor(final.label)} transition-all hover:shadow-md`}>
      <div className="flex items-start space-x-3">
        <div className="text-lg">{getAlertIcon(final.label)}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-gray-900 truncate">
              {telemetry.DeviceName}
            </h4>
            <span className={`text-xs px-2 py-1 rounded-full font-medium ${
              final.label === 'High' 
                ? 'bg-red-100 text-red-800' 
                : 'bg-yellow-100 text-yellow-800'
            }`}>
              {final.label.toUpperCase()}
            </span>
          </div>
          
          <p className="text-sm text-gray-600 mt-1">
            {telemetry.DeviceType}
          </p>
          
          <div className="text-xs text-gray-500 mt-2">
            Confidence: {Math.round(final.confidence * 100)}% • {formatTimeAgo(timestamp)}
          </div>
          
          {final.factors && final.factors.length > 0 && (
            <div className="mt-2">
              <p className="text-xs text-gray-600">
                {final.factors.slice(0, 2).join(', ')}
                {final.factors.length > 2 && ` +${final.factors.length - 2} more`}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
