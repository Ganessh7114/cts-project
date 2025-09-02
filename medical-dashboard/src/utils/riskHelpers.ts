// src/utils/riskHelpers.ts
import { RiskLevel, PredictionRecord } from '../services/types';

export const getRiskPriority = (risk: RiskLevel): number => {
  switch (risk) {
    case 'High': return 3;
    case 'Medium': return 2;
    case 'Low': return 1;
    default: return 0;
  }
};

export const sortByRiskPriority = (predictions: PredictionRecord[]): PredictionRecord[] => {
  return [...predictions].sort((a, b) => {
    const aPriority = getRiskPriority(a.final.label);
    const bPriority = getRiskPriority(b.final.label);
    
    if (aPriority !== bPriority) {
      return bPriority - aPriority; // High risk first
    }
    
    // Same risk level, sort by timestamp (newest first)
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });
};

export const getRiskStats = (predictions: PredictionRecord[]) => {
  const stats = { Low: 0, Medium: 0, High: 0, total: predictions.length };
  
  predictions.forEach(prediction => {
    stats[prediction.final.label]++;
  });
  
  return {
    ...stats,
    percentages: {
      Low: stats.total > 0 ? Math.round((stats.Low / stats.total) * 100) : 0,
      Medium: stats.total > 0 ? Math.round((stats.Medium / stats.total) * 100) : 0,
      High: stats.total > 0 ? Math.round((stats.High / stats.total) * 100) : 0
    }
  };
};
