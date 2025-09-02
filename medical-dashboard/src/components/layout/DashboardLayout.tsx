import React, { useEffect, useState } from 'react';
import { apiClient } from '../../services/apiClient';
import { PredictionRecord } from '../../services/types';

export const DashboardLayout: React.FC = () => {
  const [predictions, setPredictions] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await apiClient.getPredictions({ since: 'last1h' });
        setPredictions(data);
      } catch (error) {
        console.error('Failed to load data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const stats = {
    total: predictions.length,
    Low: predictions.filter(p => p.final.label === 'Low').length,
    Medium: predictions.filter(p => p.final.label === 'Medium').length,
    High: predictions.filter(p => p.final.label === 'High').length
  };

  // Inline styles to ensure styling works regardless of Tailwind issues
  const containerStyle: React.CSSProperties = {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #EBF8FF 0%, #F7FAFC 100%)',
    fontFamily: 'system-ui, -apple-system, sans-serif'
  };

  const headerStyle: React.CSSProperties = {
    background: 'white',
    boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
    borderBottom: '4px solid #2563EB',
    position: 'sticky',
    top: 0,
    zIndex: 50
  };

  const statsCardStyle = (color: string): React.CSSProperties => ({
    background: `linear-gradient(135deg, ${color} 0%, ${color}DD 100%)`,
    color: 'white',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
    transform: 'scale(1)',
    transition: 'transform 0.2s ease',
    cursor: 'pointer'
  });

  const deviceCardStyle = (riskLevel: string): React.CSSProperties => {
    const colors = {
      High: { bg: '#FEF2F2', border: '#FECACA', hover: '#FEE2E2' },
      Medium: { bg: '#FFFBEB', border: '#FDE68A', hover: '#FEF3C7' },
      Low: { bg: '#F0FDF4', border: '#BBF7D0', hover: '#DCFCE7' }
    };
    const config = colors[riskLevel as keyof typeof colors] || colors.Low;
    
    return {
      background: config.bg,
      border: `2px solid ${config.border}`,
      borderRadius: '12px',
      padding: '20px',
      boxShadow: '0 4px 15px rgba(0,0,0,0.08)',
      transition: 'all 0.2s ease',
      cursor: 'pointer',
      position: 'relative'
    };
  };

  if (loading) {
    return (
      <div style={containerStyle}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          minHeight: '100vh',
          flexDirection: 'column'
        }}>
          <div style={{
            width: '48px',
            height: '48px',
            border: '4px solid #E5E7EB',
            borderTop: '4px solid #2563EB',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }}></div>
          <p style={{ marginTop: '16px', fontSize: '18px', color: '#6B7280' }}>
            Loading Medical Dashboard...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .stats-card:hover {
          transform: scale(1.05);
        }
        .device-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
      `}</style>
      
      {/* Professional Medical Header */}
      <header style={headerStyle}>
        <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{ 
              background: '#2563EB', 
              padding: '12px', 
              borderRadius: '12px',
              fontSize: '24px'
            }}>
              🏥
            </div>
            <div>
              <h1 style={{ fontSize: '32px', fontWeight: 'bold', color: '#111827', margin: 0 }}>
                Medical Equipment Dashboard
              </h1>
              <p style={{ color: '#6B7280', margin: '4px 0 0 0' }}>
                Real-time IoT Device Monitoring & AI Predictions
              </p>
            </div>
          </div>
          
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '12px', 
            background: '#D1FAE5', 
            padding: '12px 16px', 
            borderRadius: '25px' 
          }}>
            <div style={{
              width: '12px',
              height: '12px',
              background: '#10B981',
              borderRadius: '50%',
              animation: 'pulse 2s infinite'
            }}></div>
            <span style={{ color: '#065F46', fontWeight: 'bold' }}>Connected</span>
          </div>
        </div>
      </header>

      <main style={{ maxWidth: '1280px', margin: '0 auto', padding: '32px 24px' }}>
        {/* Statistics Dashboard */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '24px', 
          marginBottom: '32px' 
        }}>
          <div className="stats-card" style={statsCardStyle('#2563EB')}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <p style={{ opacity: 0.8, fontSize: '14px', margin: '0 0 8px 0' }}>Total Devices</p>
                <p style={{ fontSize: '40px', fontWeight: 'bold', margin: 0 }}>{stats.total}</p>
              </div>
              <span style={{ fontSize: '32px' }}>⚡</span>
            </div>
          </div>

          <div className="stats-card" style={statsCardStyle('#10B981')}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <p style={{ opacity: 0.8, fontSize: '14px', margin: '0 0 8px 0' }}>Low Risk</p>
                <p style={{ fontSize: '40px', fontWeight: 'bold', margin: 0 }}>{stats.Low}</p>
              </div>
              <span style={{ fontSize: '32px' }}>✅</span>
            </div>
          </div>

          <div className="stats-card" style={statsCardStyle('#F59E0B')}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <p style={{ opacity: 0.8, fontSize: '14px', margin: '0 0 8px 0' }}>Medium Risk</p>
                <p style={{ fontSize: '40px', fontWeight: 'bold', margin: 0 }}>{stats.Medium}</p>
              </div>
              <span style={{ fontSize: '32px' }}>⚠️</span>
            </div>
          </div>

          <div className="stats-card" style={statsCardStyle('#EF4444')}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <p style={{ opacity: 0.8, fontSize: '14px', margin: '0 0 8px 0' }}>High Risk</p>
                <p style={{ fontSize: '40px', fontWeight: 'bold', margin: 0 }}>{stats.High}</p>
              </div>
              <span style={{ fontSize: '32px' }}>🚨</span>
            </div>
          </div>
        </div>

        {/* Device Grid */}
        <div style={{ 
          background: 'white', 
          borderRadius: '16px', 
          boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
          overflow: 'hidden'
        }}>
          <div style={{ 
            background: '#F9FAFB', 
            padding: '24px', 
            borderBottom: '1px solid #E5E7EB',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: '#111827', margin: 0 }}>
              Device Status ({predictions.length} devices)
            </h2>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#6B7280', fontSize: '14px' }}>
              <span style={{ fontSize: '16px' }}>🔄</span>
              <span>Last updated: {new Date().toLocaleTimeString()}</span>
            </div>
          </div>

          <div style={{ padding: '24px' }}>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', 
              gap: '20px' 
            }}>
              {predictions.map((prediction, index) => {
                const riskLevel = prediction.final.label;
                const riskIcons = { High: '🚨', Medium: '⚠️', Low: '✅' };
                
                return (
                  <div 
                    key={`${prediction.telemetry.DeviceName}-${index}`}
                    className="device-card"
                    style={deviceCardStyle(riskLevel)}
                  >
                    {/* Device Header */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                      <div style={{ flex: 1 }}>
                        <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: '#111827', margin: '0 0 4px 0' }}>
                          {prediction.telemetry.DeviceName}
                        </h3>
                        <p style={{ fontSize: '14px', color: '#6B7280', margin: 0 }}>
                          {prediction.telemetry.DeviceType}
                        </p>
                      </div>
                      
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: '12px' }}>
                        <span style={{ fontSize: '18px' }}>{riskIcons[riskLevel as keyof typeof riskIcons]}</span>
                        <span style={{
                          background: riskLevel === 'High' ? '#FEE2E2' : riskLevel === 'Medium' ? '#FEF3C7' : '#DCFCE7',
                          color: riskLevel === 'High' ? '#991B1B' : riskLevel === 'Medium' ? '#92400E' : '#166534',
                          padding: '4px 12px',
                          borderRadius: '12px',
                          fontSize: '12px',
                          fontWeight: 'bold'
                        }}>
                          {riskLevel}
                        </span>
                      </div>
                    </div>

                    {/* Telemetry Data */}
                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontSize: '14px' }}>🌡️</span>
                          <span style={{ fontSize: '14px', fontWeight: '500', color: '#374151' }}>Temperature</span>
                        </div>
                        <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#111827' }}>
                          {prediction.telemetry.TemperatureC}°C
                        </span>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontSize: '14px' }}>📳</span>
                          <span style={{ fontSize: '14px', fontWeight: '500', color: '#374151' }}>Vibration</span>
                        </div>
                        <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#111827' }}>
                          {prediction.telemetry.VibrationMM_S}mm/s
                        </span>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontSize: '14px' }}>⏱️</span>
                          <span style={{ fontSize: '14px', fontWeight: '500', color: '#374151' }}>Runtime</span>
                        </div>
                        <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#111827' }}>
                          {Math.round(prediction.telemetry.RuntimeHours)}h
                        </span>
                      </div>
                    </div>

                    {/* Confidence Bar */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span style={{ fontSize: '12px', fontWeight: '500', color: '#374151' }}>AI Confidence</span>
                        <span style={{ fontSize: '12px', fontWeight: 'bold', color: '#111827' }}>
                          {Math.round(prediction.final.confidence * 100)}%
                        </span>
                      </div>
                      
                      <div style={{ 
                        width: '100%', 
                        height: '8px', 
                        background: '#E5E7EB', 
                        borderRadius: '4px',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          height: '100%',
                          width: `${prediction.final.confidence * 100}%`,
                          background: riskLevel === 'High' ? '#EF4444' : riskLevel === 'Medium' ? '#F59E0B' : '#10B981',
                          borderRadius: '4px',
                          transition: 'width 0.5s ease'
                        }} />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
