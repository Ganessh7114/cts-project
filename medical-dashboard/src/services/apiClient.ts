import { PredictionRecord } from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

class APIClient {
  async healthCheck(): Promise<{ status: string }> {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    return response.json();
  }

  async getPredictions(params?: {
    since?: string;
    device?: string;
    risk?: string;
  }): Promise<PredictionRecord[]> {
    const url = new URL(`${API_BASE_URL}/api/predictions`);
    
    if (params?.since) url.searchParams.set('since', params.since);
    if (params?.device) url.searchParams.set('device', params.device);
    if (params?.risk) url.searchParams.set('risk', params.risk);

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error('Failed to fetch predictions');
    }

    return response.json();
  }
}

export const apiClient = new APIClient();
