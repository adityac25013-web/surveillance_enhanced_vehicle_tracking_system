const API_BASE_URL = "http://localhost:5000";
window.API_BASE_URL = API_BASE_URL;

async function pingHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

async function fetchLiveEvents() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/events/live`);
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

window.apiClient = {
  pingHealth,
  fetchLiveEvents,
};

