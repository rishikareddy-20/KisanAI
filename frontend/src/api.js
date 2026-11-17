// Client for the KisanAI backend.

const BASE = import.meta.env.VITE_API_BASE || "/api";

export async function diagnose({ file, language, crop }) {
  const form = new FormData();
  form.append("file", file);
  form.append("language", language);
  if (crop) form.append("crop", crop);

  const res = await fetch(`${BASE}/diagnose`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

// Resolve a backend-relative audio URL (e.g. /audio/xyz.mp3) to absolute.
export function resolveAudioUrl(url) {
  if (!url) return null;
  if (/^https?:\/\//.test(url)) return url;
  return `${BASE}${url}`;
}
