const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

function buildUrl(path) {
  if (!API_BASE) return path;
  return `${API_BASE}${path}`;
}

export async function getRuns(apiKey = '') {
  const token = localStorage.getItem('dashboard_token') || '';
  const headers = {
    ...(apiKey ? { 'x-api-key': apiKey } : {}),
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
  const token = localStorage.getItem('dashboard_token') || '';
  const headers = {
    ...(apiKey ? { 'x-api-key': apiKey } : {}),
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
  const token = localStorage.getItem('dashboard_token') || '';
  const headers = {
    ...(apiKey ? { 'x-api-key': apiKey } : {}),
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
    throw new Error('Kullanıcı adı veya şifre hatalı.');
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
