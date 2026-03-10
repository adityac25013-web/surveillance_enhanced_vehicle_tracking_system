const statusDot = document.getElementById("backend-status");
const healthResult = document.getElementById("health-result");
const pingButton = document.getElementById("ping-backend");
const detectionsList = document.getElementById("detections-list");
const videoStream = document.getElementById("video-stream");
const streamPlaceholder = document.getElementById("stream-placeholder");

function startVideoFeed() {
  const base = window.API_BASE_URL || "http://localhost:5000";
  videoStream.src = base + "/video_feed";
}

videoStream.onload = function () {
  streamPlaceholder.hidden = true;
  videoStream.classList.remove("load-error");
  videoStream.classList.add("loaded");
};

videoStream.onerror = function () {
  videoStream.classList.add("load-error");
};

async function handlePing() {
  const { ok, data } = await window.apiClient.pingHealth();
  statusDot.classList.remove("ok", "error");
  statusDot.classList.add(ok ? "ok" : "error");
  healthResult.textContent = JSON.stringify(data, null, 2);
  if (ok) startVideoFeed();
}

async function refreshDetections() {
  const { ok, data } = await window.apiClient.fetchLiveEvents();
  if (!ok) {
    detectionsList.textContent = "Error fetching detections.";
    return;
  }

  const events = data.events || [];
  if (events.length === 0) {
    detectionsList.textContent = "No vehicles detected. Point the camera at cars or a screen showing cars.";
    return;
  }

  const rows = events
    .map((e) => {
      const plate = (e.license_plate || "").trim();
      const plateStr = plate ? ` • Plate: ${plate}` : "";
      return `ID #${e.track_id ?? "-"} • ${e.label}${plateStr} • ${e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : ""}`;
    })
    .join("\n");

  detectionsList.textContent = rows;
}

pingButton.addEventListener("click", handlePing);

handlePing();
setInterval(refreshDetections, 1000);
setTimeout(startVideoFeed, 800);

