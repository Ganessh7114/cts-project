// src/components/filters/TimeRangeFilter.tsx
import React from 'react';

interface TimeRangeFilterProps {
  selectedRange: string;
  onRangeChange: (range: string) => void;
}

export const TimeRangeFilter: React.FC<TimeRangeFilterProps> = ({
  selectedRange,
  onRangeChange
}) => {
  const timeRanges = [
    { value: 'last1h', label: 'Last Hour' },
    { value: 'last6h', label: 'Last 6 Hours' },
    { value: 'last24h', label: 'Last 24 Hours' },
    { value: 'last7d', label: 'Last Week' },
  ];

  return (
    <div className="flex items-center space-x-2">
      <label className="text-sm font-medium text-gray-700">
        Time Range:
      </label>
      <select
        value={selectedRange}
        onChange={(e) => onRangeChange(e.target.value)}
        className="px-3 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        {timeRanges.map(range => (
          <option key={range.value} value={range.value}>
            {range.label}
          </option>
        ))}
      </select>
    </div>
  );
};
