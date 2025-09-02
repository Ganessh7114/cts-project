// src/components/charts/RiskDistributionChart.tsx
import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { PredictionRecord } from '../../services/types';

interface Props {
  data: PredictionRecord[];
}

export const RiskDistributionChart: React.FC<Props> = ({ data }) => {
  const chartData = React.useMemo(() => {
    const riskCounts = { Low: 0, Medium: 0, High: 0 };
    
    data.forEach(prediction => {
      const risk = prediction.final.label;
      if (riskCounts.hasOwnProperty(risk)) {
        riskCounts[risk]++;
      }
    });

    const total = data.length;
    
    return [
      { 
        name: 'Low Risk', 
        value: riskCounts.Low, 
        percentage: total > 0 ? Math.round((riskCounts.Low / total) * 100) : 0,
        fill: '#10B981' 
      },
      { 
        name: 'Medium Risk', 
        value: riskCounts.Medium, 
        percentage: total > 0 ? Math.round((riskCounts.Medium / total) * 100) : 0,
        fill: '#F59E0B' 
      },
      { 
        name: 'High Risk', 
        value: riskCounts.High, 
        percentage: total > 0 ? Math.round((riskCounts.High / total) * 100) : 0,
        fill: '#EF4444' 
      }
    ].filter(item => item.value > 0); // Only show non-zero categories
  }, [data]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium">{data.name}</p>
          <p className="text-sm text-gray-600">
            {data.value} devices ({data.percentage}%)
          </p>
        </div>
      );
    }
    return null;
  };

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500">
        No data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={100}
          paddingAngle={2}
          dataKey="value"
        >
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.fill} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend 
          formatter={(value, entry) => `${value} (${entry.payload.value})`}
        />
      </PieChart>
    </ResponsiveContainer>
  );
};
