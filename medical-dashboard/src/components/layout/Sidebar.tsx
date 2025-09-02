// src/components/layout/Sidebar.tsx
import React from 'react';
import { getRiskStats } from '../../utils/riskHelpers';
import { PredictionRecord } from '../../services/types';

interface SidebarProps {
  predictions: PredictionRecord[];
  isCollapsed: boolean;
  onToggle: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ predictions, isCollapsed, onToggle }) => {
  const stats = getRiskStats(predictions);

  if (isCollapsed) {
    return (
      <div className="w-16 bg-white border-r border-gray-200 flex flex-col items-center py-4">
        <button
          onClick={onToggle}
          className="p-2 hover:bg-gray-100 rounded-lg"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        <h2 className="font-semibold text-gray-900">Quick Stats</h2>
        <button
          onClick={onToggle}
          className="p-1 hover:bg-gray-100 rounded"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
      </div>

      {/* Stats */}
      <div className="p-4 space-y-4">
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="text-sm text-gray-600">Total Devices</div>
          <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between p-2 bg-green-50 rounded">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm">Low Risk</span>
            </div>
            <div className="text-sm font-medium">{stats.Low}</div>
          </div>

          <div className="flex items-center justify-between p-2 bg-yellow-50 rounded">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <span className="text-sm">Medium Risk</span>
            </div>
            <div className="text-sm font-medium">{stats.Medium}</div>
          </div>

          <div className="flex items-center justify-between p-2 bg-red-50 rounded">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span className="text-sm">High Risk</span>
            </div>
            <div className="text-sm font-medium">{stats.High}</div>
          </div>
        </div>
      </div>
    </div>
  );
};
