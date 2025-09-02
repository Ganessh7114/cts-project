import React from 'react';
import { DashboardLayout } from './components/layout/DashboardLayout';
import { ErrorBoundary } from './components/common/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        <DashboardLayout />
      </div>
    </ErrorBoundary>
  );
}

export default App;
