// src/components/filters/RiskFilter.tsx
import React from 'react';
import { RiskLevel } from '../../services/types';

interface RiskFilterProps {
  selectedRisk: RiskLevel | '';
  onRiskChange: (risk: RiskLevel | '') => void;
}

export const RiskFilter: React.FC<RiskFilterProps> = ({
  selectedRisk,
  onRiskChange
}) => {
  return (
    <div className="flex items-center space-x-2">
      <label className="text-sm font-medium text-gray-700">
        Risk Level:
      </label>
      <select
        value={selectedRisk}
        onChange={(e) => onRiskChange(e.target.value as RiskLevel | '')}
        className="px-3 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="">All Levels</option>
        <option value="Low">Low Risk</option>
        <option value="Medium">Medium Risk</option>
        <option value="High">High Risk</option>
      </select>
    </div>
  );
};
