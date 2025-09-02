// src/services/sseService.ts
import { PredictionRecord } from './types';

export type SSEEventHandler = (data: PredictionRecord) => void;
export type SSEErrorHandler = (error: Event) => void;
export type SSEOpenHandler = () => void;

export class SSEService {
  private eventSource: EventSource | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000; // Start with 1 second

  constructor(
    private url: string,
    private onMessage: SSEEventHandler,
    private onError?: SSEErrorHandler,
    private onOpen?: SSEOpenHandler
  ) {}

  connect(): void {
    try {
      this.eventSource = new EventSource(this.url);

      this.eventSource.onopen = () => {
        this.reconnectAttempts = 0;
        this.reconnectInterval = 1000;
        this.onOpen?.();
      };

      this.eventSource.onmessage = (event) => {
        try {
          const data: PredictionRecord = JSON.parse(event.data);
          this.onMessage(data);
        } catch (error) {
          console.error('Failed to parse SSE message:', error);
        }
      };

      this.eventSource.onerror = (event) => {
        this.onError?.(event);
        this.handleReconnection();
      };

    } catch (error) {
      console.error('Failed to create SSE connection:', error);
      this.handleReconnection();
    }
  }

  private handleReconnection(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      
      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        this.disconnect();
        this.connect();
      }, this.reconnectInterval);

      // Exponential backoff
      this.reconnectInterval = Math.min(this.reconnectInterval * 2, 30000);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }

  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  isConnected(): boolean {
    return this.eventSource?.readyState === EventSource.OPEN;
  }

  getReadyState(): number {
    return this.eventSource?.readyState ?? EventSource.CLOSED;
  }
}
