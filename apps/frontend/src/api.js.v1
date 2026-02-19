const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
const DEFAULT_API_KEY = import.meta.env.VITE_DEFAULT_API_KEY || '';

function resolveApiKey(apiKey) {
  const val = String(apiKey || '').trim();
  if (val) return val;
  return String(DEFAULT_API_KEY || '').trim();
}

function buildUrl(path) {
  if (!API_BASE) return path;
  return `${API_BASE}${path}`;
}

export async function getRuns(apiKey = '') {
  const effectiveApiKey = resolveApiKey(apiKey);
  const token = localStorage.getItem('dashboard_token') || '';
  const headers = {
    ...(effectiveApiKey ? { 'x-api-key': effectiveApiKey } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const res = await fetch(buildUrl('/dashboard/api/runs'), {
    headers,
  });
  if (!res.ok) {
    throw new Error(`Run listesi alınamadı: ${res.status}`);
  }
  return res.json();
}

export async function getOverview(runId, apiKey = '') {
  const effectiveApiKey = resolveApiKey(apiKey);
  const token = localStorage.getItem('dashboard_token') || '';
  const headers = {
    ...(effectiveApiKey ? { 'x-api-key': effectiveApiKey } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  const res = await fetch(buildUrl(`/dashboard/api/overview${query}`), {
    headers,
  });
  if (!res.ok) {
    throw new Error(`Overview alınamadı: ${res.status}`);
  }
  return res.json();
}

export async function getDbStatus(apiKey = '') {
  const effectiveApiKey = resolveApiKey(apiKey);
  const token = localStorage.getItem('dashboard_token') || '';
  const headers = {
    ...(effectiveApiKey ? { 'x-api-key': effectiveApiKey } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const res = await fetch(buildUrl('/dashboard/api/db-status'), { headers });
  if (!res.ok) {
    throw new Error(`DB durumu alınamadı: ${res.status}`);
  }
  return res.json();
}

export async function login(username, password) {
  const res = await fetch(buildUrl('/auth/login'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    let message = 'Kullanıcı adı veya şifre hatalı.';
    try {
      const payload = await res.json();
      if (typeof payload?.message === 'string' && payload.message.trim()) {
        message = payload.message;
      } else if (typeof payload?.detail === 'string' && payload.detail.trim()) {
        message = payload.detail;
      }
    } catch (_) {}
    throw new Error(message);
  }
  return res.json();
}

export async function logout() {
  const token = localStorage.getItem('dashboard_token') || '';
  await fetch(buildUrl('/auth/logout'), {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

export async function me() {
  const token = localStorage.getItem('dashboard_token') || '';
  const res = await fetch(buildUrl('/auth/me'), {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) {
    throw new Error('Oturum bulunamadı');
  }
  return res.json();
}

export async function startChatSession(payload, apiKey = '') {
  const effectiveApiKey = resolveApiKey(apiKey);
  const token = localStorage.getItem('dashboard_token') || '';
  const headers = {
    'Content-Type': 'application/json',
    ...(effectiveApiKey ? { 'x-api-key': effectiveApiKey } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const res = await fetch(buildUrl('/chat/session'), {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Chat oturumu açılamadı: ${res.status} ${text}`);
  }
  return res.json();
}

export async function sendChatMessage(payload, apiKey = '') {
  const effectiveApiKey = resolveApiKey(apiKey);
  const token = localStorage.getItem('dashboard_token') || '';
  const headers = {
    'Content-Type': 'application/json',
    ...(effectiveApiKey ? { 'x-api-key': effectiveApiKey } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const res = await fetch(buildUrl('/chat/message'), {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Chat mesajı gönderilemedi: ${res.status} ${text}`);
  }
  return res.json();
}

export async function getChatSummary(sessionId, apiKey = '') {
  const effectiveApiKey = resolveApiKey(apiKey);
  const token = localStorage.getItem('dashboard_token') || '';
  const headers = {
    ...(effectiveApiKey ? { 'x-api-key': effectiveApiKey } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const res = await fetch(buildUrl(`/chat/session/${encodeURIComponent(sessionId)}/summary`), {
    headers,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Chat özeti alınamadı: ${res.status} ${text}`);
  }
  return res.json();
}
