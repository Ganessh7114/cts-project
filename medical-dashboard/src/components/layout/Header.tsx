// src/components/layout/Header.tsx
import React from 'react';
import { useHealthCheck } from '../../hooks/useHealthCheck';
import { DeviceFilter, RiskLevel } from '../../services/types';

interface HeaderProps {
  isConnected: boolean;
  onReconnect: () => void;
  totalDevices: number;
  filters: DeviceFilter;
  onFiltersChange: (filters: DeviceFilter) => void;
}

export const Header: React.FC<HeaderProps> = ({
  isConnected,
  onReconnect,
  totalDevices,
  filters,
  onFiltersChange
}) => {
  const { isHealthy, lastCheck, checkHealth } = useHealthCheck();

  const handleDeviceFilterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFiltersChange({ ...filters, deviceName: e.target.value || undefined });
  };

  const handleRiskFilterChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    onFiltersChange({ 
      ...filters, 
      riskLevel: value ? (value as RiskLevel) : undefined 
    });
  };

  const clearFilters = () => {
    onFiltersChange({});
  };

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          {/* Title and Status */}
          <div className="flex items-center space-x-6">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Medical Equipment Dashboard
              </h1>
              <p className="text-sm text-gray-600">
                Monitoring {totalDevices} devices
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Connection Status */}
              <div className={`
                flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium
                ${isConnected 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'}
              `}>
                <div className={`w-2 h-2 rounded-full ${
                  isConnected ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span>{isConnected ? 'Live Stream' : 'Disconnected'}</span>
                {!isConnected && (
                  <button
                    onClick={onReconnect}
                    className="ml-2 text-xs underline hover:no-underline"
                  >
                    Reconnect
                  </button>
                )}
              </div>

              {/* Health Status */}
              <div className={`
                flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium
                ${isHealthy 
                  ? 'bg-blue-100 text-blue-800' 
                  : 'bg-orange-100 text-orange-800'}
              `}>
                <div className={`w-2 h-2 rounded-full ${
                  isHealthy ? 'bg-blue-500' : 'bg-orange-500'
                }`} />
                <span>System {isHealthy ? 'Healthy' : 'Warning'}</span>
                <button
                  onClick={checkHealth}
                  className="ml-1 text-xs opacity-70 hover:opacity-100"
                  title="Check system health"
                >
                  🔄
                </button>
              </div>
            </div>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700">Device:</label>
              <input
                type="text"
                placeholder="Filter by name..."
                value={filters.deviceName || ''}
                onChange={handleDeviceFilterChange}
                className="px-3 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700">Risk:</label>
              <select
                value={filters.riskLevel || ''}
                onChange={handleRiskFilterChange}
                className="px-3 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Levels</option>
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
              </select>
            </div>

            {(filters.deviceName || filters.riskLevel) && (
              <button
                onClick={clearFilters}
                className="text-sm text-blue-600 hover:text-blue-800 underline"
              >
                Clear Filters
              </button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
