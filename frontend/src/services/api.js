// Configurable API base URL for production deployment.
// In development, defaults to '/api' (proxied via Vite to http://localhost:5000).
// In production, can be set via VITE_API_BASE_URL (e.g., https://my-backend.onrender.com).
const envApiUrl = import.meta.env.VITE_API_BASE_URL;
const API_BASE = envApiUrl
  ? (envApiUrl.endsWith('/api') ? envApiUrl : `${envApiUrl.replace(/\/+$/, '')}/api`)
  : '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function fetchLiveMatches(recent = false) {
  const res = await fetch(`${API_BASE}/matches?recent=${recent}`);
  if (!res.ok) throw new Error('Failed to fetch live matches');
  return res.json();
}

export async function fetchMatchDetail(matchId) {
  const res = await fetch(`${API_BASE}/matches/${matchId}`);
  if (!res.ok) throw new Error(`Failed to fetch details for match ${matchId}`);
  return res.json();
}

export async function fetchDemoMatches() {
  const res = await fetch(`${API_BASE}/demo/matches`);
  if (!res.ok) throw new Error('Failed to fetch demo matches');
  return res.json();
}

export async function fetchDemoMatchDetail(demoId) {
  const res = await fetch(`${API_BASE}/demo/matches/${demoId}`);
  if (!res.ok) throw new Error(`Failed to fetch demo match ${demoId}`);
  return res.json();
}

export async function runSimulation(stateParams) {
  const res = await fetch(`${API_BASE}/simulation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(stateParams),
  });
  if (!res.ok) throw new Error('Failed to execute next-over scenario simulation');
  return res.json();
}
