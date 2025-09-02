// src/components/alerts/AlertBanner.tsx
import React, { useState } from 'react';
import { PredictionRecord } from '../../services/types';

interface AlertBannerProps {
  highRiskDevices: PredictionRecord[];
}

export const AlertBanner: React.FC<AlertBannerProps> = ({ highRiskDevices }) => {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible || highRiskDevices.length === 0) {
    return null;
  }

  const criticalCount = highRiskDevices.length;

  return (
    <div className="bg-red-600 text-white px-4 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="flex-shrink-0">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        </div>
        <div>
          <p className="font-medium">
            Critical Alert: {criticalCount} device{criticalCount > 1 ? 's' : ''} at high risk
          </p>
          <p className="text-sm text-red-100">
            {highRiskDevices.slice(0, 2).map(d => d.telemetry.DeviceName).join(', ')}
            {criticalCount > 2 && ` and ${criticalCount - 2} more`}
          </p>
        </div>
      </div>
      
      <button
        onClick={() => setIsVisible(false)}
        className="flex-shrink-0 ml-4 text-red-200 hover:text-white"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
};
