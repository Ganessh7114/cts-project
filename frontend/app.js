// frontend/app.js 

// -----------------------------
// API Endpoints (backend served by api_bridge.py)
// -----------------------------
const API_ENDPOINTS = {
  health: "http://localhost:8000/api/health",
  list: "http://localhost:8000/api/predictions",
  stream: "http://localhost:8000/api/stream"
};

// -----------------------------
// State
// -----------------------------
let devices = []; // unified device objects
let apiStatus = { connected: false };

// -----------------------------
// Helpers
// -----------------------------
function upsertDevice(d) {
  const idx = devices.findIndex(x => x.id === d.id);
  if (idx >= 0) {
    devices[idx] = { ...devices[idx], ...d, isUpdating: true };
  } else {
    devices.push({ ...d, isUpdating: true });
  }
}

function updateAPIStatusDisplay() {
  const el = document.getElementById("api-status");
  if (!el) return;
  if (apiStatus.connected) {
    el.textContent = "API Connected";
    el.className = "status-connected";
  } else {
    el.textContent = "API Offline";
    el.className = "status-offline";
  }
}

// -----------------------------
// Backend integration
// -----------------------------
async function checkAPIHealth() {
  try {
    const r = await fetch(API_ENDPOINTS.health);
    const data = await r.json();
    apiStatus.connected = data.status === "healthy";
  } catch (e) {
    apiStatus.connected = false;
  }
  updateAPIStatusDisplay();
}

async function loadInitialPredictions() {
  try {
    const r = await fetch(`${API_ENDPOINTS.list}?mode=last1h&limit=12`);
    const data = await r.json();

    if (data.success && Array.isArray(data.items)) {
      data.items.forEach(device => {
        // Backend already shapes records via _record_to_device
        upsertDevice(device);
      });
      updateDeviceGrid();
      updateCharts();
    }
  } catch (e) {
    console.error("Failed to load initial predictions:", e);
  }
}

function connectLiveStream() {
  const es = new EventSource(API_ENDPOINTS.stream);

  es.onopen = () => {
    apiStatus.connected = true;
    updateAPIStatusDisplay();
    showToast("Live stream connected", "success");
  };

  es.onmessage = (ev) => {
    try {
      const device = JSON.parse(ev.data);
      upsertDevice(device);
      updateDeviceGrid();
      updateCharts();
    } catch (err) {
      console.error("Stream parse failed", err);
    }
  };

  es.onerror = () => {
    es.close();
    apiStatus.connected = false;
    updateAPIStatusDisplay();
    showToast("Stream disconnected. Reconnecting…", "warning");
    setTimeout(connectLiveStream, 3000);
  };
}

// -----------------------------
// UI (Device Grid, Charts, Alerts)
// -----------------------------
function updateDeviceGrid() {
  const grid = document.getElementById("device-grid");
  if (!grid) return;

  grid.innerHTML = "";
  devices.forEach((device) => {
    const card = updateDeviceCard(device);
    grid.appendChild(card);
  });
}

function updateDeviceCard(device) {
  const card = document.createElement("div");
  card.className = "device-card";

  card.innerHTML = `
    <h3>${device.deviceName} (${device.deviceType})</h3>
    <p>Temp: ${device.temperature ?? "-"} °C</p>
    <p>Vibration: ${device.vibration ?? "-"} mm/s</p>
    <p>Runtime: ${device.runtimeHours ?? "-"} h</p>
    <p>Risk: <strong>${device.prediction}</strong> (${device.confidence})</p>
    <small>${new Date(device.timestamp).toLocaleString()}</small>
  `;

  return card;
}

// Simple placeholder until charts are wired with Chart.js or similar
function updateCharts() {
  console.log("Charts updated with", devices.length, "devices");
}

// Toasts
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// -----------------------------
// Init
// -----------------------------
async function init() {
  updateAPIStatusDisplay();
  await checkAPIHealth();
  await loadInitialPredictions();
  connectLiveStream();

  // Re-check health every 30s
  setInterval(checkAPIHealth, 30000);
}

document.addEventListener("DOMContentLoaded", init);
