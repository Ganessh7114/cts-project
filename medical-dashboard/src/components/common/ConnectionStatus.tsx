// src/components/common/ConnectionStatus.tsx
import React from 'react';

interface ConnectionStatusProps {
  isConnected: boolean;
  lastUpdate?: Date | null;
  onReconnect?: () => void;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  isConnected,
  lastUpdate,
  onReconnect
}) => {
  const getStatusColor = () => {
    if (isConnected) return 'text-green-600 bg-green-100';
    return 'text-red-600 bg-red-100';
  };

  const getStatusText = () => {
    if (isConnected) {
      return lastUpdate 
        ? `Connected • Last update: ${lastUpdate.toLocaleTimeString()}`
        : 'Connected';
    }
    return 'Disconnected';
  };

  const getStatusIcon = () => {
    if (isConnected) {
      return (
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
      );
    }
    
    return (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
      </svg>
    );
  };

  return (
    <div className={`inline-flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${getStatusColor()}`}>
      {getStatusIcon()}
      <span>{getStatusText()}</span>
      {!isConnected && onReconnect && (
        <button
          onClick={onReconnect}
          className="ml-2 text-xs underline hover:no-underline focus:outline-none"
        >
          Reconnect
        </button>
      )}
    </div>
  );
};
