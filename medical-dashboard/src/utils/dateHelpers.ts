// src/utils/dateHelpers.ts
import { format, formatDistanceToNow, subHours, isAfter } from 'date-fns';

export const formatTimestamp = (timestamp: string): string => {
  return format(new Date(timestamp), 'MMM dd, yyyy HH:mm:ss');
};

export const formatTimeAgo = (timestamp: string): string => {
  return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
};

export const isRecentPrediction = (timestamp: string, hoursBack: number = 1): boolean => {
  const cutoff = subHours(new Date(), hoursBack);
  return isAfter(new Date(timestamp), cutoff);
};

export const groupByHour = (timestamps: string[]): { [hour: string]: number } => {
  const groups: { [hour: string]: number } = {};
  
  timestamps.forEach(timestamp => {
    const hour = format(new Date(timestamp), 'HH:00');
    groups[hour] = (groups[hour] || 0) + 1;
  });
  
  return groups;
};
