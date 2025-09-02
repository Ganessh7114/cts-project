// src/components/charts/ChartContainer.tsx
import React from 'react';
import { LoadingSpinner } from '../common/LoadingSpinner';

interface ChartContainerProps {
  title: string;
  children: React.ReactNode;
  loading?: boolean;
  error?: string | null;
  height?: string;
}

export const ChartContainer: React.FC<ChartContainerProps> = ({
  title,
  children,
  loading = false,
  error = null,
  height = 'h-64'
}) => {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4 text-gray-900">{title}</h3>
      
      <div className={`${height} relative`}>
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
            <LoadingSpinner text="Loading chart..." />
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-500">
              <p className="text-sm">Failed to load chart</p>
              <p className="text-xs text-gray-400 mt-1">{error}</p>
            </div>
          </div>
        )}
        
        {!loading && !error && children}
      </div>
    </div>
  );
};
