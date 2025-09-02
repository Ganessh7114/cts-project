// src/components/devices/RiskIndicator.tsx
import React from 'react';
import { RiskLevel } from '../../services/types';

interface RiskIndicatorProps {
  risk: RiskLevel;
  confidence: number;
  size?: 'sm' | 'md' | 'lg';
}

export const RiskIndicator: React.FC<RiskIndicatorProps> = ({ 
  risk, 
  confidence, 
  size = 'md' 
}) => {
  const sizeClasses = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-2'
  };

  const getRiskClasses = (risk: RiskLevel): string => {
    const baseClasses = `inline-flex items-center rounded-full font-semibold ${sizeClasses[size]}`;
    
    switch (risk) {
      case 'Low':
        return `${baseClasses} bg-green-100 text-green-800`;
      case 'Medium':
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case 'High':
        return `${baseClasses} bg-red-100 text-red-800`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  const getRiskIcon = (risk: RiskLevel): string => {
    switch (risk) {
      case 'Low': return '🟢';
      case 'Medium': return '🟡';
      case 'High': return '🔴';
      default: return '⚪';
    }
  };

  return (
    <div className="flex flex-col items-center">
      <span className={getRiskClasses(risk)}>
        <span className="mr-1">{getRiskIcon(risk)}</span>
        {risk.toUpperCase()}
      </span>
      {confidence && (
        <span className="text-xs text-gray-500 mt-1">
          {Math.round(confidence * 100)}%
        </span>
      )}
    </div>
  );
};
