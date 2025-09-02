// src/hooks/usePredictions.ts
import { useEffect, useState } from 'react';
import { apiClient } from '../services/apiClient';
import { PredictionRecord } from '../services/types';

interface UsePredictionsOptions {
  since?: string;
  device?: string;
  risk?: string;
}

interface UsePredictionsResult {
  data: PredictionRecord[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export const usePredictions = (options: UsePredictionsOptions = {}): UsePredictionsResult => {
  const [data, setData] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      setError(null);
      const predictions = await apiClient.getPredictions(options);
      setData(predictions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch predictions');
      setData([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, [options.since, options.device, options.risk]);

  return {
    data,
    loading,
    error,
    refetch: fetchPredictions
  };
};
