// Medical Device Monitoring System - Enhanced Live Streaming Application with API Integration

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';
const API_ENDPOINTS = {
    predict: `${API_BASE_URL}/predict`,
    batchPredict: `${API_BASE_URL}/batch-predict`,
    devices: `${API_BASE_URL}/devices`,
    modelInfo: `${API_BASE_URL}/model-info`,
    status: `${API_BASE_URL}/status`
};

// Device Configuration
const DEVICE_MAPPING = {
    "Alaris GH": "Infusion Pump",
    "Baxter Flo-Gard": "Infusion Pump", 
    "Smiths Medfusion": "Infusion Pump",
    "Baxter AK 96": "Dialysis Machine",
    "Fresenius 4008": "Dialysis Machine",
    "NxStage System One": "Dialysis Machine",
    "Datex Ohmeda S5": "Anesthesia Machine",
    "Drager Fabius Trio": "Anesthesia Machine", 
    "GE Aisys": "Anesthesia Machine",
    "Drager V500": "Patient Ventilator",
    "Hamilton G5": "Patient Ventilator",
    "Puritan Bennett 980": "Patient Ventilator",
    "HeartStart FRx": "Defibrillator",
    "Lifepak 20": "Defibrillator",
    "Philips HeartStrart": "Defibrillator",
    "Zoll R Series": "Defibrillator",
    "GE Logiq E9": "Ultrasound Machine",
    "Philips EPIQ": "Ultrasound Machine", 
    "Siemens Acuson": "Ultrasound Machine",
    "Siemens S2000": "Ultrasound Machine",
    "GE Revolution": "CT Scanner",
    "Philips Ingenuity": "CT Scanner",
    "GE MAC 2000": "ECG Monitor",
    "Phillips PageWriter": "ECG Monitor"
};

const DEVICE_TYPES = ["Anesthesia Machine", "CT Scanner", "Defibrillator", "Dialysis Machine", "ECG Monitor", "Infusion Pump", "Patient Ventilator", "Ultrasound Machine"];
const LOCATIONS = ["Hospital A - ICU", "Hospital A - Emergency", "Hospital B - Nephrology", "Hospital B - Cardiology", "Hospital C - Surgery"];

// Global Application State
let devices = [];
let alerts = [];
let isStreaming = false;
let streamingInterval = null;
let charts = {};
let currentTheme = 'light';
let updateInterval = 2000; // 2 seconds for fast streaming
let soundEnabled = false;
let updateCount = 0;
let lastStats = { healthy: 0, warning: 0, critical: 0 };
let chartsPaused = { risk: false, temperature: false };
let apiStatus = { connected: false, modelLoaded: false, lastCheck: null };

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
    initializeApplication();
    setupEventListeners();
    checkAPIStatus();
    generateDevices();
    setupCharts();
    startLiveStreaming(); // Auto-start live streaming
});

// Core Application Functions
function initializeApplication() {
    console.log('Initializing Enhanced Medical Device Monitoring System with API Integration...');
    
    // Detect system theme preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        currentTheme = 'dark';
        updateThemeIcon();
    }
    
    // Setup manual form device options
    setupManualForm();
    
    // Setup inventory filters
    setupInventoryFilters();
    
    // Initialize streaming controls
    updateStreamingControls();
    
    console.log('Enhanced application initialized successfully');
}

// API Integration Functions
async function checkAPIStatus() {
    try {
        const response = await fetch(API_ENDPOINTS.status);
        if (response.ok) {
            const data = await response.json();
            apiStatus = {
                connected: true,
                modelLoaded: data.model_loaded,
                lastCheck: new Date(),
                modelVersion: data.model_version
            };
            updateAPIStatusDisplay();
            console.log('API Status:', data);
        } else {
            throw new Error('API not responding');
        }
    } catch (error) {
        console.warn('API connection failed:', error);
        apiStatus = {
            connected: false,
            modelLoaded: false,
            lastCheck: new Date(),
            error: error.message
        };
        updateAPIStatusDisplay();
    }
}

function updateAPIStatusDisplay() {
    const statusElements = {
        connection: document.getElementById('apiConnectionStatus'),
        model: document.getElementById('modelStatus'),
        version: document.getElementById('modelVersion')
    };
    
    if (statusElements.connection) {
        statusElements.connection.textContent = apiStatus.connected ? 'Connected' : 'Disconnected';
        statusElements.connection.className = apiStatus.connected ? 'status--success' : 'status--error';
    }
    
    if (statusElements.model && apiStatus.connected) {
        statusElements.model.textContent = apiStatus.modelLoaded ? 'ML Model Loaded' : 'Mock Mode';
        statusElements.model.className = apiStatus.modelLoaded ? 'status--success' : 'status--warning';
    }
    
    if (statusElements.version && apiStatus.modelVersion) {
        statusElements.version.textContent = apiStatus.modelVersion;
    }
}

async function callPredictionAPI(deviceData) {
    try {
        const apiPayload = {
            device_name: deviceData.deviceName,
            temperature: deviceData.temperature,
            vibration: deviceData.vibration,
            error_logs: deviceData.errorLogs,
            runtime_hours: deviceData.runtimeHours,
            device_age: deviceData.deviceAge,
            repairs: deviceData.repairs,
            pressure: deviceData.pressure,
            current_draw: deviceData.currentDraw,
            location: deviceData.location
        };

        const response = await fetch(API_ENDPOINTS.predict, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(apiPayload)
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.success && result.prediction) {
            return {
                prediction: result.prediction.prediction,
                confidence: result.prediction.confidence,
                riskScore: result.prediction.risk_score,
                factors: result.prediction.factors || [],
                modelVersion: result.prediction.model_version,
                timestamp: result.prediction.timestamp
            };
        } else {
            throw new Error('Invalid API response');
        }
    } catch (error) {
        console.warn('API prediction failed, using fallback:', error);
        return simulateFallbackPrediction(deviceData);
    }
}

async function callBatchPredictionAPI(devicesData) {
    try {
        const apiPayload = {
            devices: devicesData.map(device => ({
                device_name: device.deviceName,
                temperature: device.temperature,
                vibration: device.vibration,
                error_logs: device.errorLogs,
                runtime_hours: device.runtimeHours,
                device_age: device.deviceAge,
                repairs: device.repairs,
                pressure: device.pressure,
                current_draw: device.currentDraw,
                location: device.location
            }))
        };

        const response = await fetch(API_ENDPOINTS.batchPredict, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(apiPayload)
        });

        if (!response.ok) {
            throw new Error(`Batch API Error: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.success && result.predictions) {
            return result.predictions.map(pred => ({
                deviceName: pred.device_name,
                prediction: pred.prediction,
                confidence: pred.confidence,
                riskScore: pred.risk_score,
                factors: pred.factors || [],
                modelVersion: pred.model_version
            }));
        } else {
            throw new Error('Invalid batch API response');
        }
    } catch (error) {
        console.warn('Batch API prediction failed, using fallback:', error);
        return devicesData.map(device => ({
            deviceName: device.deviceName,
            ...simulateFallbackPrediction(device)
        }));
    }
}

// Fallback prediction when API is unavailable
function simulateFallbackPrediction(deviceData) {
    // Simple rule-based fallback logic
    let riskScore = 0;
    
    if (deviceData.temperature > 35) riskScore += 0.3;
    if (deviceData.vibration > 0.8) riskScore += 0.2;
    if (deviceData.errorLogs > 15) riskScore += 0.2;
    if (deviceData.runtimeHours > 8000) riskScore += 0.2;
    if (deviceData.repairs > 5) riskScore += 0.1;
    
    let prediction;
    if (riskScore >= 0.6) prediction = "High";
    else if (riskScore >= 0.3) prediction = "Medium";
    else prediction = "Low";
    
    const factors = [];
    if (deviceData.temperature > 35) factors.push(`High temperature (${deviceData.temperature.toFixed(1)}°C)`);
    if (deviceData.vibration > 0.8) factors.push(`Excessive vibration (${deviceData.vibration.toFixed(2)})`);
    if (deviceData.errorLogs > 15) factors.push(`High error count (${deviceData.errorLogs})`);
    if (factors.length === 0) factors.push("All parameters within normal ranges");
    
    return {
        prediction,
        confidence: 0.75,
        riskScore,
        factors,
        modelVersion: "Fallback Model v1.0",
        timestamp: new Date().toISOString()
    };
}

function setupEventListeners() {
    // Tab navigation - Fixed event handling with proper delegation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const tabName = this.getAttribute('data-tab');
            if (tabName) {
                console.log('Switching to tab:', tabName);
                switchTab(tabName);
            }
        });
    });
    
    // Enhanced streaming controls
    document.getElementById('toggleStreamBtn').addEventListener('click', toggleStreaming);
    document.getElementById('soundToggle').addEventListener('click', toggleSound);
    
    // Speed control slider
    const speedSlider = document.getElementById('speedSlider');
    speedSlider.addEventListener('input', (e) => {
        updateInterval = parseInt(e.target.value) * 1000;
        document.getElementById('speedDisplay').textContent = e.target.value + 's';
        if (isStreaming) {
            restartStreaming();
        }
    });
    
    // Chart pause controls
    document.getElementById('pauseRiskChart').addEventListener('click', () => {
        chartsPaused.risk = !chartsPaused.risk;
        document.getElementById('pauseRiskChart').textContent = chartsPaused.risk ? '▶️' : '⏸️';
    });
    
    document.getElementById('pauseTempChart').addEventListener('click', () => {
        chartsPaused.temperature = !chartsPaused.temperature;
        document.getElementById('pauseTempChart').textContent = chartsPaused.temperature ? '▶️' : '⏸️';
    });
    
    // Other controls
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
    document.getElementById('exportBtn').addEventListener('click', exportData);
    document.getElementById('runBatchBtn').addEventListener('click', runBatchPrediction);
    document.getElementById('manualForm').addEventListener('submit', handleManualPrediction);
    document.getElementById('searchInput').addEventListener('input', filterInventory);
    document.getElementById('typeFilter').addEventListener('change', filterInventory);
    document.getElementById('locationFilter').addEventListener('change', filterInventory);
    document.getElementById('closeModal').addEventListener('click', closeModal);
    document.getElementById('deviceModal').addEventListener('click', (e) => {
        if (e.target.id === 'deviceModal') closeModal();
    });
    document.getElementById('clearAlertsBtn').addEventListener('click', clearAlerts);
}

// Tab Management - Fixed implementation
function switchTab(tabName) {
    console.log('Switching to tab:', tabName);
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeTabBtn = document.querySelector(`[data-tab="${tabName}"]`);
    if (activeTabBtn) {
        activeTabBtn.classList.add('active');
    }
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    const activeTabContent = document.getElementById(tabName);
    if (activeTabContent) {
        activeTabContent.classList.add('active');
    }
    
    // Load tab-specific content
    switch(tabName) {
        case 'dashboard':
            // Dashboard is always live
            break;
        case 'batch':
            // Batch content is static
            break;
        case 'manual':
            // Manual form is static
            break;
        case 'streaming':
            updateStreamingStats();
            break;
        case 'inventory':
            populateInventoryTable();
            break;
        case 'alerts':
            displayAlerts();
            updateAlertCounts();
            break;
    }
    
    console.log('Successfully switched to tab:', tabName);
}

// Enhanced Device Data Generation with API Integration
function generateDevices() {
    devices = [];
    const deviceNames = Object.keys(DEVICE_MAPPING);
    
    for (let i = 0; i < 24; i++) {
        const deviceName = deviceNames[i];
        const deviceType = DEVICE_MAPPING[deviceName];
        const location = LOCATIONS[Math.floor(Math.random() * LOCATIONS.length)];
        
        const device = {
            id: i + 1,
            deviceName,
            deviceType,
            location,
            alerted: false,
            lastUpdate: new Date(),
            isUpdating: false,
            ...generateRealisticParameters(deviceType)
        };
        
        devices.push(device);
    }
    
    console.log('Generated 24 medical devices with realistic parameters');
}

function generateRealisticParameters(deviceType) {
    const baseParams = {
        temperature: 20 + Math.random() * 25, // 20-45°C
        vibration: Math.random() * 1.2, // 0-1.2
        errorLogs: Math.floor(Math.random() * 30), // 0-30
        runtimeHours: 1000 + Math.random() * 8000, // 1000-9000 hours
        deviceAge: 0.5 + Math.random() * 5, // 0.5-5.5 years
        repairs: Math.floor(Math.random() * 8), // 0-8 repairs
        pressure: 80 + Math.random() * 100, // 80-180
        currentDraw: 3 + Math.random() * 8 // 3-11 amps
    };
    
    // Device-specific parameter adjustments
    switch (deviceType) {
        case "Patient Ventilator":
            baseParams.temperature = Math.min(baseParams.temperature, 35);
            baseParams.pressure = 100 + Math.random() * 50;
            break;
        case "Dialysis Machine":
            baseParams.pressure = 120 + Math.random() * 80;
            baseParams.currentDraw = 6 + Math.random() * 5;
            break;
        case "CT Scanner":
            baseParams.currentDraw = 8 + Math.random() * 7;
            baseParams.temperature = 25 + Math.random() * 15;
            break;
        case "Defibrillator":
            baseParams.currentDraw = 2 + Math.random() * 12;
            baseParams.vibration = Math.random() * 0.4;
            break;
    }
    
    return baseParams;
}

// FAST LIVE STREAMING IMPLEMENTATION WITH API INTEGRATION
function startLiveStreaming() {
    if (isStreaming) return;
    
    isStreaming = true;
    updateStreamingControls();
    updateCount = 0;
    
    // Check API status before starting
    checkAPIStatus();
    
    // Start fast streaming with API calls
    streamingInterval = setInterval(async () => {
        await updateAllDevicesWithAPI();
        updateLiveDashboard();
        updateCharts();
        checkForAlerts();
        updateStreamingMetrics();
        updateCount++;
        
        // Periodically check API status
        if (updateCount % 30 === 0) { // Every minute at 2s intervals
            await checkAPIStatus();
        }
    }, updateInterval);
    
    showToast("🔴 Live streaming started with API integration - Updates every " + (updateInterval/1000) + "s");
    console.log('Fast live streaming started with API integration at', updateInterval/1000, 'second intervals');
}

function stopLiveStreaming() {
    if (!isStreaming) return;
    
    isStreaming = false;
    if (streamingInterval) {
        clearInterval(streamingInterval);
        streamingInterval = null;
    }
    
    updateStreamingControls();
    showToast("⏸️ Live streaming stopped");
    console.log('Live streaming stopped');
}

function toggleStreaming() {
    if (isStreaming) {
        stopLiveStreaming();
    } else {
        startLiveStreaming();
    }
}

function restartStreaming() {
    if (isStreaming) {
        stopLiveStreaming();
        setTimeout(startLiveStreaming, 100);
    }
}

// ENHANCED DATA GENERATION WITH API CALLS
async function updateAllDevicesWithAPI() {
    // Update parameter values with small realistic changes
    devices.forEach(device => {
        if (!device.isUpdating) {
            // Simulate realistic parameter changes with small variations
            device.temperature += (Math.random() - 0.5) * 2; // ±1°C change
            device.vibration += (Math.random() - 0.5) * 0.1; // ±0.05 change  
            device.errorLogs += Math.floor(Math.random() * 3); // 0-2 new errors
            device.runtimeHours += 0.5; // 30 minutes runtime
            device.pressure += (Math.random() - 0.5) * 5; // ±2.5 pressure change
            device.currentDraw += (Math.random() - 0.5) * 0.5; // ±0.25A change
            
            // Keep values in realistic ranges
            device.temperature = Math.max(15, Math.min(45, device.temperature));
            device.vibration = Math.max(0, Math.min(1.2, device.vibration));
            device.errorLogs = Math.max(0, Math.min(50, device.errorLogs));
            device.pressure = Math.max(50, Math.min(250, device.pressure));
            device.currentDraw = Math.max(1, Math.min(15, device.currentDraw));
            
            device.lastUpdate = new Date();
        }
    });
    
    // Update predictions for a subset of devices each cycle to balance performance
    const devicesToUpdate = devices.slice(0, 6); // Update 6 devices per cycle
    
    try {
        // Use batch prediction for efficiency
        const predictions = await callBatchPredictionAPI(devicesToUpdate);
        
        predictions.forEach((prediction, index) => {
            const device = devicesToUpdate[index];
            device.prediction = prediction.prediction;
            device.confidence = prediction.confidence;
            device.riskScore = prediction.riskScore;
            device.factors = prediction.factors;
            device.modelVersion = prediction.modelVersion;
            device.isUpdating = false;
        });
        
        // Rotate which devices get updated next cycle
        devices = [...devices.slice(6), ...devices.slice(0, 6)];
        
    } catch (error) {
        console.warn('Batch API update failed:', error);
        // Fall back to individual predictions if batch fails
        for (let i = 0; i < Math.min(3, devicesToUpdate.length); i++) {
            try {
                const device = devicesToUpdate[i];
                device.isUpdating = true;
                const prediction = await callPredictionAPI(device);
                device.prediction = prediction.prediction;
                device.confidence = prediction.confidence;
                device.riskScore = prediction.riskScore;
                device.factors = prediction.factors;
                device.modelVersion = prediction.modelVersion;
                device.isUpdating = false;
            } catch (error) {
                console.warn(`Individual prediction failed for device ${devicesToUpdate[i].deviceName}:`, error);
                devicesToUpdate[i].isUpdating = false;
            }
        }
    }
}

// LIVE DASHBOARD UPDATES
function updateLiveDashboard() {
    // Calculate current statistics
    const stats = calculateDeviceStats();
    updateDashboardStats(stats);
    
    // Update device grid with real-time data
    updateDeviceGrid();
    
    // Update live metrics
    updateLiveMetrics();
}

function calculateDeviceStats() {
    const stats = {
        total: devices.length,
        healthy: 0,
        warning: 0,
        critical: 0
    };
    
    devices.forEach(device => {
        if (device.prediction) {
            switch(device.prediction) {
                case 'Low':
                    stats.healthy++;
                    break;
                case 'Medium':
                    stats.warning++;
                    break;
                case 'High':
                    stats.critical++;
                    break;
            }
        }
    });
    
    return stats;
}

function updateDashboardStats(stats) {
    // Update values
    document.getElementById('totalDevices').textContent = stats.total;
    document.getElementById('healthyDevices').textContent = stats.healthy;
    document.getElementById('warningDevices').textContent = stats.warning;
    document.getElementById('criticalDevices').textContent = stats.critical;
    
    // Update change indicators
    updateStatChanges(stats);
    lastStats = { ...stats };
}

function updateStatChanges(stats) {
    updateStatChange('healthyChange', stats.healthy - lastStats.healthy);
    updateStatChange('warningChange', stats.warning - lastStats.warning);
    updateStatChange('criticalChange', stats.critical - lastStats.critical);
}

function updateStatChange(elementId, change) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = change >= 0 ? `+${change}` : `${change}`;
        element.className = 'stat-change ' + (change > 0 ? 'positive' : change < 0 ? 'negative' : '');
    }
}

function updateDeviceGrid() {
    const grid = document.getElementById('deviceGrid');
    
    devices.forEach((device, index) => {
        let card = grid.children[index];
        if (!card) {
            card = createDeviceCard(device, index);
            grid.appendChild(card);
        } else {
            updateDeviceCard(card, device);
        }
    });
}

function createDeviceCard(device, index) {
    const card = document.createElement('div');
    card.className = 'device-card';
    card.dataset.deviceId = device.id;
    card.addEventListener('click', () => showDeviceDetails(device, device));
    
    updateDeviceCard(card, device);
    return card;
}

function updateDeviceCard(card, device) {
    const riskClass = device.prediction ? `risk-${device.prediction.toLowerCase()}` : 'risk-low';
    const riskLevel = device.prediction || 'Updating...';
    const isUpdating = device.isUpdating;
    
    // Add updating animation
    if (!isUpdating) {
        card.classList.add('updating');
        setTimeout(() => card.classList.remove('updating'), 200);
    }
    
    card.innerHTML = `
        <div class="device-header">
            <div>
                <div class="device-name">${device.deviceName}</div>
                <div class="device-type">${device.deviceType}</div>
            </div>
            <div class="status">
                <span class="status-dot ${riskLevel.toLowerCase().replace('updating...', 'pending')}"></span>
                <span class="status ${riskClass}">${riskLevel}</span>
                ${isUpdating ? '<span class="updating-spinner">⟳</span>' : ''}
            </div>
        </div>
        <div class="device-metrics">
            <div class="metric">
                <span class="metric-label">Temperature</span>
                <span class="metric-value">${device.temperature.toFixed(1)}°C</span>
            </div>
            <div class="metric">
                <span class="metric-label">Vibration</span>
                <span class="metric-value">${device.vibration.toFixed(2)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Runtime</span>
                <span class="metric-value">${Math.round(device.runtimeHours)}h</span>
            </div>
            <div class="metric">
                <span class="metric-label">Confidence</span>
                <span class="metric-value">${device.confidence ? Math.round(device.confidence * 100) : '--'}%</span>
            </div>
        </div>
        ${apiStatus.connected ? '' : '<div class="api-warning">⚠️ API Offline</div>'}
    `;
}

function updateLiveMetrics() {
    // Update rate per minute
    const updatesPerMinute = Math.round((60 / (updateInterval / 1000)));
    const updateRateElement = document.getElementById('updateRate');
    if (updateRateElement) {
        updateRateElement.textContent = updatesPerMinute;
    }
    
    // Data rate
    const dataRateElement = document.getElementById('dataRate');
    if (dataRateElement) {
        dataRateElement.textContent = isStreaming ? 'Real-time' : 'Stopped';
    }
    
    // Connection status
    const connectionStatusElement = document.getElementById('connectionStatus');
    if (connectionStatusElement) {
        connectionStatusElement.textContent = isStreaming ? 'Active' : 'Disconnected';
    }
}

// ENHANCED CHART UPDATES
function setupCharts() {
    const chartColors = ['#1FB8CD', '#FFC185', '#B4413C'];
    
    // Risk Distribution Chart
    charts.riskChart = new Chart(document.getElementById('riskChart'), {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: chartColors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0 // No animation for smooth real-time updates
            },
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    
    // Live Temperature Chart
    charts.temperatureChart = new Chart(document.getElementById('temperatureChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Average Temperature',
                data: [],
                borderColor: chartColors[0],
                backgroundColor: chartColors[0] + '20',
                fill: true,
                tension: 0.4,
                pointRadius: 1,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0 // No animation for smooth updates
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Temperature (°C)'
                    }
                },
                x: {
                    display: false
                }
            },
            elements: {
                line: {
                    tension: 0.4
                }
            }
        }
    });
    
    // Streaming Chart
    charts.streamingChart = new Chart(document.getElementById('streamingChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Live Temperature',
                data: [],
                borderColor: chartColors[1],
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    title: {
                        display: true,
                        text: 'Temperature (°C)'
                    }
                }
            }
        }
    });
}

function updateCharts() {
    updateRiskChart();
    updateTemperatureChart();
    updateStreamingChart();
}

function updateRiskChart() {
    if (chartsPaused.risk) return;
    
    // Update risk distribution chart
    const riskCounts = [0, 0, 0];
    devices.forEach(device => {
        if (device.prediction) {
            switch(device.prediction) {
                case 'Low': riskCounts[0]++; break;
                case 'Medium': riskCounts[1]++; break;
                case 'High': riskCounts[2]++; break;
            }
        }
    });
    
    charts.riskChart.data.datasets[0].data = riskCounts;
    charts.riskChart.update('none'); // No animation for smooth updates
}

function updateTemperatureChart() {
    if (chartsPaused.temperature) return;
    
    // Add new data point
    const now = new Date().toLocaleTimeString();
    const temps = devices.map(d => d.temperature);
    const avgTemp = temps.reduce((a, b) => a + b) / temps.length;
    
    charts.temperatureChart.data.labels.push(now);
    charts.temperatureChart.data.datasets[0].data.push(avgTemp.toFixed(1));
    
    // Keep only last 20 data points for smooth scrolling
    if (charts.temperatureChart.data.labels.length > 20) {
        charts.temperatureChart.data.labels.shift();
        charts.temperatureChart.data.datasets[0].data.shift();
    }
    
    charts.temperatureChart.update('none'); // No animation for smooth updates
}

function updateStreamingChart() {
    if (!isStreaming) return;
    
    const now = new Date().toLocaleTimeString();
    const randomDevice = devices[Math.floor(Math.random() * devices.length)];
    const temperature = randomDevice.temperature;
    
    charts.streamingChart.data.labels.push(now);
    charts.streamingChart.data.datasets[0].data.push(temperature.toFixed(1));
    
    // Keep only last 30 data points
    if (charts.streamingChart.data.labels.length > 30) {
        charts.streamingChart.data.labels.shift();
        charts.streamingChart.data.datasets[0].data.shift();
    }
    
    charts.streamingChart.update('none');
}

// POPUP ALERT SYSTEM
function checkForAlerts() {
    devices.forEach(device => {
        const shouldAlert = (
            device.prediction === 'High' || 
            device.temperature > 38 || 
            device.vibration > 0.8 ||
            device.errorLogs > 20
        );
        
        if (shouldAlert && !device.alerted) {
            showPopupAlert(device);
            device.alerted = true;
            addAlert('critical', 'CRITICAL DEVICE ALERT', 
                   `${device.deviceName} requires immediate inspection`, device.location);
        }
        
        // Reset alert after conditions improve
        if (!shouldAlert && device.alerted) {
            device.alerted = false;
        }
    });
}

function showPopupAlert(device) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert-popup critical';
    alertDiv.innerHTML = `
        <div class="alert-header">
            <h3>
                <span class="alert-icon">🚨</span>
                CRITICAL DEVICE ALERT
            </h3>
            <button class="close-alert" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
        <div class="alert-content">
            <strong>${device.deviceName}</strong> - ${device.deviceType}<br>
            Location: ${device.location}<br>
            Risk Level: <span class="risk-high">${device.prediction || 'High'}</span><br>
            Temperature: ${device.temperature.toFixed(1)}°C<br>
            Vibration: ${device.vibration.toFixed(2)}<br>
            <strong>Action Required: Immediate Inspection</strong>
            ${!apiStatus.connected ? '<br><span class="api-warning">⚠️ Alert based on local analysis - API offline</span>' : ''}
        </div>
    `;
    
    document.getElementById('popupAlertContainer').appendChild(alertDiv);
    
    // Play sound if enabled
    if (soundEnabled) {
        playAlertSound();
    }
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 10000);
}

function playAlertSound() {
    // Create audio context for alert sound
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        gainNode.gain.value = 0.1;
        
        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.2);
    } catch (error) {
        console.warn('Audio alert not available:', error);
    }
}

// STREAMING CONTROLS
function updateStreamingControls() {
    const btn = document.getElementById('toggleStreamBtn');
    const icon = document.getElementById('streamBtnIcon');
    const text = document.getElementById('streamBtnText');
    const indicator = document.getElementById('liveIndicator');
    const statusDot = document.getElementById('streamingStatusDot');
    const statusText = document.getElementById('streamingStatusText');
    
    if (isStreaming) {
        icon.textContent = '⏸️';
        text.textContent = 'Pause Live';
        btn.className = 'btn btn--secondary btn--sm';
        indicator.style.display = 'flex';
        if (statusDot) statusDot.classList.add('active');
        if (statusText) statusText.textContent = 'Streaming';
    } else {
        icon.textContent = '▶️';
        text.textContent = 'Start Live';
        btn.className = 'btn btn--primary btn--sm';
        indicator.style.display = 'none';
        if (statusDot) statusDot.classList.remove('active');
        if (statusText) statusText.textContent = 'Stopped';
    }
}

function toggleSound() {
    soundEnabled = !soundEnabled;
    const icon = document.getElementById('soundIcon');
    icon.textContent = soundEnabled ? '🔊' : '🔇';
    showToast(soundEnabled ? 'Sound alerts enabled' : 'Sound alerts disabled');
}

function updateStreamingMetrics() {
    const indicator = document.getElementById('streamingStatusIndicator');
    if (indicator) {
        indicator.style.display = 'flex';
    }
}

// Manual prediction form setup with API integration
function setupManualForm() {
    const deviceSelect = document.getElementById('manualDeviceName');
    if (!deviceSelect) return;
    
    Object.keys(DEVICE_MAPPING).forEach(deviceName => {
        const option = document.createElement('option');
        option.value = deviceName;
        option.textContent = `${deviceName} (${DEVICE_MAPPING[deviceName]})`;
        deviceSelect.appendChild(option);
    });
}

async function handleManualPrediction(event) {
    event.preventDefault();
    
    const btn = document.getElementById('manualPredictBtn');
    const btnText = document.getElementById('manualBtnText');
    const spinner = document.getElementById('manualSpinner');
    
    // Show loading state
    if (btn) btn.disabled = true;
    if (btnText) btnText.textContent = 'Analyzing...';
    if (spinner) spinner.classList.remove('hidden');
    
    const deviceData = {
        deviceName: document.getElementById('manualDeviceName').value,
        temperature: parseFloat(document.getElementById('manualTemp').value),
        vibration: parseFloat(document.getElementById('manualVibration').value),
        errorLogs: parseInt(document.getElementById('manualErrors').value),
        runtimeHours: parseInt(document.getElementById('manualRuntime').value),
        deviceAge: 2.5,
        repairs: Math.floor(Math.random() * 5),
        pressure: 100 + Math.random() * 50,
        currentDraw: 4 + Math.random() * 4,
        location: LOCATIONS[Math.floor(Math.random() * LOCATIONS.length)]
    };
    
    try {
        // Use API for prediction
        const prediction = await callPredictionAPI(deviceData);
        displayManualResults(prediction);
        showToast("Analysis completed successfully!");
    } catch (error) {
        console.error('Manual prediction error:', error);
        // Use fallback prediction
        const fallbackPrediction = simulateFallbackPrediction(deviceData);
        displayManualResults(fallbackPrediction);
        showToast("Analysis completed using fallback model", 'warning');
    } finally {
        // Reset button state
        if (btn) btn.disabled = false;
        if (btnText) btnText.textContent = 'Analyze Device';
        if (spinner) spinner.classList.add('hidden');
    }
}

function displayManualResults(prediction) {
    const resultsDiv = document.getElementById('manualResults');
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = `
        <div class="prediction-results">
            <div class="result-item">
                <span class="result-label">Risk Level:</span>
                <span class="result-value risk-${prediction.prediction.toLowerCase()}">${prediction.prediction}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Confidence:</span>
                <span class="result-value">${Math.round(prediction.confidence * 100)}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Risk Score:</span>
                <span class="result-value">${(prediction.riskScore * 100).toFixed(1)}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Model:</span>
                <span class="result-value font-mono">${prediction.modelVersion}</span>
            </div>
            <div class="result-item">
                <span class="result-label">API Status:</span>
                <span class="result-value ${apiStatus.connected ? 'status--success' : 'status--warning'}">${apiStatus.connected ? 'Connected' : 'Offline (Fallback)'}</span>
            </div>
            <div style="margin-top: 16px;">
                <strong>Risk Factors:</strong>
                <ul style="margin: 8px 0; padding-left: 20px;">
                    ${prediction.factors.map(factor => `<li>${factor}</li>`).join('')}
                </ul>
            </div>
            <div style="margin-top: 12px; font-size: 12px; color: var(--color-text-secondary);">
                Analysis completed at: ${new Date().toLocaleString()}
            </div>
        </div>
    `;
}

// Batch prediction with API integration
async function runBatchPrediction() {
    const btn = document.getElementById('runBatchBtn');
    const btnText = document.getElementById('batchBtnText');
    const spinner = document.getElementById('batchSpinner');
    
    btn.disabled = true;
    btnText.textContent = 'Analyzing...';
    spinner.classList.remove('hidden');
    
    try {
        // Use API for batch processing
        showToast("Running batch analysis via API...");
        const results = await callBatchPredictionAPI(devices);
        
        // Update devices with API results
        results.forEach((result, index) => {
            if (devices[index]) {
                devices[index].prediction = result.prediction;
                devices[index].confidence = result.confidence;
                devices[index].riskScore = result.riskScore;
                devices[index].factors = result.factors;
                devices[index].modelVersion = result.modelVersion;
            }
        });
        
        displayBatchResults(results.map((result, index) => ({...devices[index], ...result})));
        showToast("Batch analysis completed successfully!");
        
    } catch (error) {
        console.error('Batch prediction error:', error);
        showToast("Error running batch analysis - using fallback", 'error');
        
        // Use fallback for all devices
        const fallbackResults = devices.map(device => ({
            ...device,
            ...simulateFallbackPrediction(device)
        }));
        displayBatchResults(fallbackResults);
        
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Run ML Analysis';
        spinner.classList.add('hidden');
    }
}

function displayBatchResults(results) {
    const resultsDiv = document.getElementById('batchResults');
    if (!resultsDiv) return;
    
    const table = document.createElement('table');
    table.className = 'results-table';
    
    table.innerHTML = `
        <thead>
            <tr>
                <th>Device</th>
                <th>Type</th>
                <th>Location</th>
                <th>Risk Level</th>
                <th>Confidence</th>
                <th>Key Factors</th>
            </tr>
        </thead>
        <tbody>
            ${results.map(result => `
                <tr onclick="showDeviceDetails(${JSON.stringify(result).replace(/"/g, '&quot;')}, ${JSON.stringify(result).replace(/"/g, '&quot;')})" style="cursor: pointer;">
                    <td>${result.deviceName}</td>
                    <td>${result.deviceType}</td>
                    <td>${result.location}</td>
                    <td>
                        <span class="status-dot ${result.prediction.toLowerCase()}"></span>
                        <span class="risk-${result.prediction.toLowerCase()}">${result.prediction}</span>
                    </td>
                    <td>${Math.round(result.confidence * 100)}%</td>
                    <td>${result.factors.slice(0, 2).join(', ')}</td>
                </tr>
            `).join('')}
        </tbody>
    `;
    
    resultsDiv.innerHTML = `
        <div class="batch-header">
            <h4>Batch Analysis Results</h4>
            <div class="batch-status">
                <span>API Status: ${apiStatus.connected ? 'Connected' : 'Offline'}</span>
                <span>Model: ${apiStatus.modelLoaded ? 'ML Model' : 'Fallback'}</span>
                <span>Processed: ${results.length} devices</span>
            </div>
        </div>
    `;
    resultsDiv.appendChild(table);
}

// Inventory functions
function setupInventoryFilters() {
    const typeFilter = document.getElementById('typeFilter');
    const locationFilter = document.getElementById('locationFilter');
    
    if (typeFilter) {
        DEVICE_TYPES.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            typeFilter.appendChild(option);
        });
    }
    
    if (locationFilter) {
        LOCATIONS.forEach(location => {
            const option = document.createElement('option');
            option.value = location;
            option.textContent = location;
            locationFilter.appendChild(option);
        });
    }
}

function populateInventoryTable() {
    const tableDiv = document.getElementById('inventoryTable');
    if (!tableDiv) return;
    
    const table = document.createElement('table');
    table.className = 'results-table';
    
    table.innerHTML = `
        <thead>
            <tr>
                <th>ID</th>
                <th>Device Name</th>
                <th>Type</th>
                <th>Location</th>
                <th>Age (Years)</th>
                <th>Runtime (Hours)</th>
                <th>Temperature</th>
                <th>Prediction</th>
                <th>Last Update</th>
            </tr>
        </thead>
        <tbody id="inventoryBody">
            ${devices.map(device => `
                <tr>
                    <td>${device.id}</td>
                    <td>${device.deviceName}</td>
                    <td>${device.deviceType}</td>
                    <td>${device.location}</td>
                    <td>${device.deviceAge.toFixed(1)}</td>
                    <td>${Math.round(device.runtimeHours)}</td>
                    <td>${device.temperature.toFixed(1)}°C</td>
                    <td><span class="risk-${(device.prediction || 'low').toLowerCase()}">${device.prediction || 'Pending'}</span></td>
                    <td>${device.lastUpdate.toLocaleTimeString()}</td>
                </tr>
            `).join('')}
        </tbody>
    `;
    
    tableDiv.innerHTML = '';
    tableDiv.appendChild(table);
}

function filterInventory() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const typeFilter = document.getElementById('typeFilter').value;
    const locationFilter = document.getElementById('locationFilter').value;
    
    const filteredDevices = devices.filter(device => {
        const matchesSearch = device.deviceName.toLowerCase().includes(searchTerm) ||
                            device.deviceType.toLowerCase().includes(searchTerm);
        const matchesType = !typeFilter || device.deviceType === typeFilter;
        const matchesLocation = !locationFilter || device.location === locationFilter;
        
        return matchesSearch && matchesType && matchesLocation;
    });
    
    updateInventoryTable(filteredDevices);
}

function updateInventoryTable(filteredDevices) {
    const tbody = document.getElementById('inventoryBody');
    if (!tbody) return;
    
    tbody.innerHTML = filteredDevices.map(device => `
        <tr>
            <td>${device.id}</td>
            <td>${device.deviceName}</td>
            <td>${device.deviceType}</td>
            <td>${device.location}</td>
            <td>${device.deviceAge.toFixed(1)}</td>
            <td>${Math.round(device.runtimeHours)}</td>
            <td>${device.temperature.toFixed(1)}°C</td>
            <td><span class="risk-${(device.prediction || 'low').toLowerCase()}">${device.prediction || 'Pending'}</span></td>
            <td>${device.lastUpdate.toLocaleTimeString()}</td>
        </tr>
    `).join('');
}

// Device details modal with API information
function showDeviceDetails(device, prediction) {
    const modal = document.getElementById('deviceModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');
    
    if (!modal || !modalTitle || !modalBody) return;
    
    modalTitle.textContent = `${device.deviceName} - Live Details`;
    
    modalBody.innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
            <div>
                <h4>Device Information</h4>
                <div class="result-item">
                    <span class="result-label">Device Type:</span>
                    <span class="result-value">${device.deviceType}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Location:</span>
                    <span class="result-value">${device.location}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Device Age:</span>
                    <span class="result-value">${device.deviceAge.toFixed(1)} years</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Total Repairs:</span>
                    <span class="result-value">${device.repairs}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Last Update:</span>
                    <span class="result-value">${device.lastUpdate.toLocaleString()}</span>
                </div>
            </div>
            <div>
                <h4>Live Parameters</h4>
                <div class="result-item">
                    <span class="result-label">Temperature:</span>
                    <span class="result-value">${device.temperature.toFixed(1)}°C</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Vibration:</span>
                    <span class="result-value">${device.vibration.toFixed(2)}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Runtime Hours:</span>
                    <span class="result-value">${Math.round(device.runtimeHours)}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Error Logs:</span>
                    <span class="result-value">${device.errorLogs}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Pressure:</span>
                    <span class="result-value">${device.pressure.toFixed(1)} units</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Current Draw:</span>
                    <span class="result-value">${device.currentDraw.toFixed(1)}A</span>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 24px;">
            <h4>Live ML Prediction Analysis</h4>
            <div class="result-item">
                <span class="result-label">Risk Level:</span>
                <span class="result-value risk-${(device.prediction || 'Low').toLowerCase()}">${device.prediction || 'Pending'} Risk</span>
            </div>
            <div class="result-item">
                <span class="result-label">Confidence Score:</span>
                <span class="result-value">${device.confidence ? Math.round(device.confidence * 100) : '--'}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Risk Score:</span>
                <span class="result-value">${device.riskScore ? (device.riskScore * 100).toFixed(1) : '--'}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">API Status:</span>
                <span class="result-value ${apiStatus.connected ? 'status--success' : 'status--warning'}">${apiStatus.connected ? 'Connected' : 'Offline'}</span>
            </div>
            ${device.factors ? `
                <div style="margin-top: 16px;">
                    <strong>Identified Risk Factors:</strong>
                    <ul style="margin: 8px 0; padding-left: 20px;">
                        ${device.factors.map(factor => `<li>${factor}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            <div style="margin-top: 16px; font-size: 12px; color: var(--color-text-secondary);">
                Model: ${device.modelVersion || 'Pending'} | ${apiStatus.connected ? 'Live API Analysis' : 'Fallback Analysis'} | Updated: ${device.lastUpdate.toLocaleString()}
            </div>
        </div>
    `;
    
    modal.classList.remove('hidden');
}

function closeModal() {
    const modal = document.getElementById('deviceModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

// Streaming Stats with API information
function updateStreamingStats() {
    const statsDiv = document.getElementById('streamingStats');
    if (!statsDiv) return;
    
    const dataPoints = charts.streamingChart ? charts.streamingChart.data.datasets[0].data.length : 0;
    
    if (dataPoints === 0) {
        statsDiv.innerHTML = '<p class="text-muted">No streaming data available</p>';
        return;
    }
    
    const temperatures = charts.streamingChart.data.datasets[0].data.map(parseFloat);
    const avgTemp = temperatures.reduce((sum, temp) => sum + temp, 0) / temperatures.length;
    const maxTemp = Math.max(...temperatures);
    const minTemp = Math.min(...temperatures);
    
    // Count devices with predictions
    const devicesWithPredictions = devices.filter(d => d.prediction).length;
    
    statsDiv.innerHTML = `
        <div class="streaming-stat">
            <span>Data Points:</span>
            <span class="font-bold">${dataPoints}</span>
        </div>
        <div class="streaming-stat">
            <span>Update Interval:</span>
            <span class="font-bold">${updateInterval/1000}s</span>
        </div>
        <div class="streaming-stat">
            <span>Updates Count:</span>
            <span class="font-bold">${updateCount}</span>
        </div>
        <div class="streaming-stat">
            <span>API Status:</span>
            <span class="font-bold ${apiStatus.connected ? 'status--success' : 'status--error'}">${apiStatus.connected ? 'Connected' : 'Offline'}</span>
        </div>
        <div class="streaming-stat">
            <span>ML Predictions:</span>
            <span class="font-bold">${devicesWithPredictions}/${devices.length}</span>
        </div>
        <div class="streaming-stat">
            <span>Avg Temperature:</span>
            <span class="font-bold">${avgTemp.toFixed(1)}°C</span>
        </div>
        <div class="streaming-stat">
            <span>Max Temperature:</span>
            <span class="font-bold">${maxTemp.toFixed(1)}°C</span>
        </div>
        <div class="streaming-stat">
            <span>Min Temperature:</span>
            <span class="font-bold">${minTemp.toFixed(1)}°C</span>
        </div>
        <div class="streaming-stat">
            <span>Status:</span>
            <span class="font-bold status--success">${isStreaming ? 'Live Streaming' : 'Stopped'}</span>
        </div>
    `;
}

// Alert Management
function addAlert(type, title, message, location) {
    const existingAlert = alerts.find(alert => 
        alert.title === title && alert.message === message
    );
    
    if (!existingAlert) {
        alerts.unshift({
            id: Date.now(),
            type,
            title,
            message,
            location,
            timestamp: new Date(),
            source: apiStatus.connected ? 'API' : 'Fallback'
        });
        
        if (alerts.length > 50) {
            alerts = alerts.slice(0, 50);
        }
        
        updateAlertCounts();
    }
}

function updateAlertCounts() {
    const activeCount = alerts.length;
    const criticalCount = alerts.filter(alert => alert.type === 'critical').length;
    
    const activeElement = document.getElementById('activeAlertCount');
    const criticalElement = document.getElementById('criticalAlertCount');
    
    if (activeElement) activeElement.textContent = activeCount;
    if (criticalElement) criticalElement.textContent = criticalCount;
}

function displayAlerts() {
    const alertsList = document.getElementById('alertsList');
    if (!alertsList) return;
    
    if (alerts.length === 0) {
        alertsList.innerHTML = '<p class="text-muted text-center">No system alerts at this time.</p>';
        return;
    }
    
    alertsList.innerHTML = alerts.map(alert => `
        <div class="alert-item ${alert.type}">
            <div class="alert-content">
                <div class="alert-title">${alert.title}</div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-time">${alert.timestamp.toLocaleString()} - ${alert.location} [${alert.source || 'System'}]</div>
            </div>
            <button class="alert-dismiss" onclick="dismissAlert(${alert.id})">&times;</button>
        </div>
    `).join('');
}

function dismissAlert(alertId) {
    alerts = alerts.filter(alert => alert.id !== alertId);
    displayAlerts();
    updateAlertCounts();
}

function clearAlerts() {
    alerts = [];
    displayAlerts();
    updateAlertCounts();
    showToast("All alerts cleared");
}

// Utility Functions
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-color-scheme', currentTheme);
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = document.getElementById('themeIcon');
    if (icon) {
        icon.textContent = currentTheme === 'light' ? '🌙' : '☀️';
    }
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    
    if (!toast || !toastMessage) return;
    
    toastMessage.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.remove('hidden');
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

function exportData() {
    const exportData = {
        devices: devices,
        alerts: alerts,
        streamingConfig: {
            isStreaming: isStreaming,
            updateInterval: updateInterval,
            updateCount: updateCount
        },
        apiStatus: apiStatus,
        timestamp: new Date().toISOString(),
        systemInfo: {
            totalDevices: devices.length,
            devicesWithPredictions: devices.filter(d => d.prediction).length,
            activeAlerts: alerts.length,
            criticalAlerts: alerts.filter(a => a.type === 'critical').length,
            apiConnected: apiStatus.connected,
            modelLoaded: apiStatus.modelLoaded
        }
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `medical-devices-api-export-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    showToast("Live data with API status exported successfully!");
}

// Error Handling
window.addEventListener('error', function(event) {
    console.error('Application error:', event.error);
    showToast("An unexpected error occurred. Please refresh the page.", 'error');
    
    // Try to maintain API connection
    if (event.error.message && event.error.message.includes('fetch')) {
        checkAPIStatus();
    }
});

// Page visibility handling to pause/resume streaming
document.addEventListener('visibilitychange', function() {
    if (document.hidden && isStreaming) {
        console.log('Page hidden, streaming continues with reduced frequency');
        // Optionally reduce update frequency when page is hidden
    } else if (!document.hidden && isStreaming) {
        console.log('Page visible, streaming active at full speed');
        checkAPIStatus(); // Check API when page becomes visible
    }
});

// Network status monitoring
window.addEventListener('online', function() {
    console.log('Network connection restored');
    checkAPIStatus();
    showToast("Network connection restored - checking API status");
});

window.addEventListener('offline', function() {
    console.log('Network connection lost');
    apiStatus.connected = false;
    updateAPIStatusDisplay();
    showToast("Network connection lost - using fallback predictions", 'warning');
});

console.log('Enhanced Medical Device Monitoring System with Flask API Integration loaded successfully');