# Medical Equipment Failure Prediction Dashboard

A real-time IoT monitoring system that uses AI/ML to predict medical equipment failures and provide actionable insights through a professional web dashboard.

![Dashboard Screenshot](docs/dashboard-screenshot.png)

## 🏥 Features

- **Real-time Device Monitoring**: Live telemetry from 50+ medical devices
- **AI-Powered Predictions**: Machine learning models predict equipment failures
- **Risk Assessment**: Color-coded risk levels (Low, Medium, High)
- **Professional Dashboard**: Hospital-grade UI for medical staff
- **Multi-Model Architecture**: Azure ML, local models, and rule-based fallback
- **Real-time Updates**: Live streaming via Server-Sent Events (SSE)

## 🏗️ Architecture

- **Backend**: Python + FastAPI + Azure services
- **Frontend**: React + TypeScript + Tailwind CSS
- **Database**: Azure Blob Storage
- **Communication**: MQTT for IoT devices
- **ML Pipeline**: Azure ML + scikit-learn

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Azure account (for cloud services)
- MQTT broker (optional, for real devices)

### Backend Setup

1. **Clone the repository**
