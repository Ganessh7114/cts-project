// src/components/devices/DeviceDetails.tsx
import React from 'react';
import { PredictionRecord } from '../../services/types';
import { RiskIndicator } from './RiskIndicator';

interface DeviceDetailsProps {
  prediction: PredictionRecord;
  onClose: () => void;
}

export const DeviceDetails: React.FC<DeviceDetailsProps> = ({ prediction, onClose }) => {
  const { telemetry, final, azure_ml, local_model, timestamp } = prediction;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex justify-between items-start p-6 border-b border-gray-200">
          <div>
            <h2 className="text-xl font-bold text-gray-900">
              {telemetry.DeviceName}
            </h2>
            <p className="text-gray-600">{telemetry.DeviceType}</p>
          </div>
          <div className="flex items-center space-x-4">
            <RiskIndicator risk={final.label} confidence={final.confidence} size="lg" />
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Telemetry Data */}
          <div>
            <h3 className="text-lg font-semibold mb-3">Current Telemetry</h3>
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(telemetry).map(([key, value]) => {
                if (key === 'DeviceName' || key === 'DeviceType') return null;
                return (
                  <div key={key} className="bg-gray-50 p-3 rounded">
                    <div className="text-sm text-gray-600 capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </div>
                    <div className="font-medium">{String(value)}</div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Risk Assessment */}
          <div>
            <h3 className="text-lg font-semibold mb-3">Risk Assessment</h3>
            <div className="bg-gray-50 p-4 rounded">
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium">Final Risk Level:</span>
                <span className="text-lg font-bold">{final.label}</span>
              </div>
              <div className="flex justify-between items-center mb-3">
                <span className="font-medium">Confidence:</span>
                <span>{Math.round(final.confidence * 100)}%</span>
              </div>
              {final.factors && final.factors.length > 0 && (
                <div>
                  <span className="font-medium block mb-2">Risk Factors:</span>
                  <div className="space-y-1">
                    {final.factors.map((factor, index) => (
                      <div key={index} className="text-sm bg-white px-2 py-1 rounded border">
                        {factor}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Model Predictions */}
          {(azure_ml || local_model) && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Model Predictions</h3>
              <div className="space-y-3">
                {azure_ml && (
                  <div className="bg-blue-50 p-3 rounded border border-blue-200">
                    <div className="font-medium text-blue-900">Azure ML Model</div>
                    <div className="text-sm text-blue-700 mt-1">
                      Status: {azure_ml.ok ? 'Success' : 'Failed'}
                      {azure_ml.label && (
                        <span className="ml-2">
                          Label: {azure_ml.label} ({Math.round((azure_ml.confidence || 0) * 100)}%)
                        </span>
                      )}
                      {azure_ml.error && (
                        <span className="text-red-600 ml-2">Error: {azure_ml.error}</span>
                      )}
                    </div>
                  </div>
                )}
                
                {local_model && (
                  <div className="bg-green-50 p-3 rounded border border-green-200">
                    <div className="font-medium text-green-900">Local Model</div>
                    <div className="text-sm text-green-700 mt-1">
                      Status: {local_model.ok ? 'Success' : 'Failed'}
                      {local_model.label && (
                        <span className="ml-2">
                          Label: {local_model.label} ({Math.round((local_model.confidence || 0) * 100)}%)
                        </span>
                      )}
                      {local_model.error && (
                        <span className="text-red-600 ml-2">Error: {local_model.error}</span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Timestamp */}
          <div className="text-sm text-gray-500 border-t pt-4">
            Last updated: {new Date(timestamp).toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
};
