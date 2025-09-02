// src/hooks/useHealthCheck.ts
import { useEffect, useState } from 'react';
import { apiClient } from '../services/apiClient';

interface HealthStatus {
  status: string;
  isHealthy: boolean;
  lastCheck: Date | null;
  error: string | null;
}

export const useHealthCheck = (intervalMs: number = 30000) => {
  const [health, setHealth] = useState<HealthStatus>({
    status: 'unknown',
    isHealthy: false,
    lastCheck: null,
    error: null
  });

  const checkHealth = async () => {
    try {
      const result = await apiClient.healthCheck();
      setHealth({
        status: result.status,
        isHealthy: result.status === 'ok',
        lastCheck: new Date(),
        error: null
      });
    } catch (error) {
      setHealth({
        status: 'error',
        isHealthy: false,
        lastCheck: new Date(),
        error: error instanceof Error ? error.message : 'Health check failed'
      });
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, intervalMs);
    return () => clearInterval(interval);
  }, [intervalMs]);

  return { ...health, checkHealth };
};
