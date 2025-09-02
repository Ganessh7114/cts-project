// src/hooks/useSSEStream.ts
import { useEffect, useState, useRef } from 'react';
import { PredictionRecord } from '../services/types';

interface SSEStreamState {
  data: PredictionRecord[];
  isConnected: boolean;
  error: string | null;
  reconnectCount: number;
}

export const useSSEStream = (url: string, maxReconnects: number = 5) => {
  const [state, setState] = useState<SSEStreamState>({
    data: [],
    isConnected: false,
    error: null,
    reconnectCount: 0
  });

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = () => {
    try {
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        setState(prev => ({
          ...prev,
          isConnected: true,
          error: null,
          reconnectCount: 0
        }));
      };

      eventSource.onmessage = (event) => {
        try {
          const prediction: PredictionRecord = JSON.parse(event.data);
          setState(prev => ({
            ...prev,
            data: updateDeviceData(prev.data, prediction)
          }));
        } catch (error) {
          console.error('Failed to parse SSE message:', error);
          setState(prev => ({
            ...prev,
            error: 'Failed to parse prediction data'
          }));
        }
      };

      eventSource.onerror = () => {
        setState(prev => {
          const newReconnectCount = prev.reconnectCount + 1;
          
          if (newReconnectCount <= maxReconnects) {
            // Exponential backoff: 1s, 2s, 4s, 8s, 16s
            const delay = Math.min(1000 * Math.pow(2, newReconnectCount - 1), 16000);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              connect();
            }, delay);

            return {
              ...prev,
              isConnected: false,
              error: `Connection lost. Reconnecting in ${delay/1000}s...`,
              reconnectCount: newReconnectCount
            };
          }

          return {
            ...prev,
            isConnected: false,
            error: 'Connection failed after maximum retry attempts',
            reconnectCount: newReconnectCount
          };
        });
      };

    } catch (error) {
      setState(prev => ({
        ...prev,
        isConnected: false,
        error: 'Failed to establish SSE connection'
      }));
    }
  };

  // Update device data, keeping most recent prediction per device
  const updateDeviceData = (currentData: PredictionRecord[], newPrediction: PredictionRecord): PredictionRecord[] => {
    const deviceName = newPrediction.telemetry.DeviceName;
    const filtered = currentData.filter(p => p.telemetry.DeviceName !== deviceName);
    const updated = [newPrediction, ...filtered].slice(0, 100); // Keep last 100 updates
    
    // Sort by timestamp descending
    return updated.sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  };

  useEffect(() => {
    connect();

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [url]);

  const reconnect = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    setState(prev => ({ ...prev, reconnectCount: 0 }));
    connect();
  };

  return {
    ...state,
    reconnect
  };
};
