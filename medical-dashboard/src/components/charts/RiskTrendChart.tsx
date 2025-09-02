// src/components/charts/RiskTrendChart.tsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { PredictionRecord } from '../../services/types';
import { format, subHours, startOfHour } from 'date-fns';

interface Props {
  data: PredictionRecord[];
}

export const RiskTrendChart: React.FC<Props> = ({ data }) => {
  const chartData = React.useMemo(() => {
    // Create hourly buckets for the last 24 hours
    const now = new Date();
    const hours: { [key: string]: { High: number; Medium: number; Low: number; total: number } } = {};
    
    // Initialize last 24 hours
    for (let i = 23; i >= 0; i--) {
      const hour = startOfHour(subHours(now, i));
      const key = format(hour, 'HH:mm');
      hours[key] = { High: 0, Medium: 0, Low: 0, total: 0 };
    }

    // Count predictions by hour and risk level
    data.forEach(prediction => {
      const predTime = new Date(prediction.timestamp);
      const hourKey = format(startOfHour(predTime), 'HH:mm');
      
      if (hours[hourKey]) {
        hours[hourKey][prediction.final.label]++;
        hours[hourKey].total++;
      }
    });

    // Convert to chart format with percentages
    return Object.entries(hours).map(([time, counts]) => ({
      time,
      High: counts.total > 0 ? Math.round((counts.High / counts.total) * 100) : 0,
      Medium: counts.total > 0 ? Math.round((counts.Medium / counts.total) * 100) : 0,
      Low: counts.total > 0 ? Math.round((counts.Low / counts.total) * 100) : 0,
      total: counts.total
    }));
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium">{label}</p>
          <p className="text-sm text-gray-600">Total devices: {data.total}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.dataKey}: {entry.value}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        No trend data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis 
          dataKey="time" 
          stroke="#666"
          fontSize={12}
          tick={{ fill: '#666' }}
        />
        <YAxis 
          stroke="#666"
          fontSize={12}
          tick={{ fill: '#666' }}
          label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="High" 
          stroke="#EF4444" 
          strokeWidth={2}
          dot={{ fill: '#EF4444', strokeWidth: 2, r: 4 }}
          activeDot={{ r: 6 }}
        />
        <Line 
          type="monotone" 
          dataKey="Medium" 
          stroke="#F59E0B" 
          strokeWidth={2}
          dot={{ fill: '#F59E0B', strokeWidth: 2, r: 4 }}
          activeDot={{ r: 6 }}
        />
        <Line 
          type="monotone" 
          dataKey="Low" 
          stroke="#10B981" 
          strokeWidth={2}
          dot={{ fill: '#10B981', strokeWidth: 2, r: 4 }}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};
