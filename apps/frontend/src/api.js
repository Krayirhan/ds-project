/**
 * API Client — fetchWithAuth wrapper + AbortController + timeout
 *
 * Tüm API çağrıları bu modül üzerinden yapılır.
 * - Ortak auth header'ları (x-api-key + Bearer token) otomatik eklenir
 * - Her istek için AbortController desteği (sayfa değişiminde iptal)
 * - Varsayılan 30 sn timeout (configurable)
 * - Yapılandırılmış hata nesneleri (err.status ile HTTP kodu)
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
const DEFAULT_API_KEY = import.meta.env.VITE_DEFAULT_API_KEY || '';
const DEFAULT_TIMEOUT = 30_000; // 30 seconds

function resolveApiKey(apiKey) {
  const val = String(apiKey || '').trim();
  return val || String(DEFAULT_API_KEY || '').trim();
}

function buildUrl(path) {
  return API_BASE ? `${API_BASE}${path}` : path;
}

function buildHeaders(apiKey, extra = {}) {
  const effectiveApiKey = resolveApiKey(apiKey);
  const token = localStorage.getItem('dashboard_token') || '';
  return {
    ...(effectiveApiKey ? { 'x-api-key': effectiveApiKey } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extra,
  };
}

/**
 * Core fetch wrapper.
 *
 * @param {string} path   - API path (e.g. '/dashboard/api/runs')
 * @param {object} opts
 * @param {string} opts.apiKey       - API key override
 * @param {number} opts.timeout      - Request timeout in ms (default 30 s)
 * @param {object} opts.headers      - Extra headers to merge
 * @param {AbortSignal} opts.signal  - External AbortSignal for cancellation
 * @param {string} opts.method       - HTTP method
 * @param {string} opts.body         - Request body
 * @returns {Promise<any>} Parsed JSON response
 */
export async function fetchWithAuth(path, options = {}) {
  const {
    apiKey = '',
    timeout = DEFAULT_TIMEOUT,
    headers: extraHeaders = {},
    signal: externalSignal,
    ...fetchOptions
  } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  // Link external signal for page-navigation cancellation
  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort();
    } else {
      externalSignal.addEventListener('abort', () => controller.abort(), { once: true });
    }
  }

  try {
    const res = await fetch(buildUrl(path), {
      ...fetchOptions,
      headers: buildHeaders(apiKey, extraHeaders),
      signal: controller.signal,
    });

    if (!res.ok) {
      let message = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        message = body?.detail || body?.message || message;
      } catch { /* ignore parse errors */ }
      const err = new Error(message);
      err.status = res.status;
      throw err;
    }

    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') {
      const timeoutErr = new Error('İstek zaman aşımına uğradı');
      timeoutErr.status = 408;
      timeoutErr.name = 'AbortError';
      throw timeoutErr;
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

// ── API Functions ────────────────────────────────────────────────────

export function getRuns(apiKey, { signal } = {}) {
  return fetchWithAuth('/dashboard/api/runs', { apiKey, signal });
}

export function getOverview(runId, apiKey, { signal } = {}) {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  return fetchWithAuth(`/dashboard/api/overview${query}`, { apiKey, signal });
}

export function getDbStatus(apiKey, { signal } = {}) {
  return fetchWithAuth('/dashboard/api/db-status', { apiKey, signal });
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
      const body = await res.json();
      message = body?.detail || body?.message || message;
    } catch { /* ignore */ }
    const err = new Error(message);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export async function logout() {
  const token = localStorage.getItem('dashboard_token') || '';
  await fetch(buildUrl('/auth/logout'), {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  }).catch(() => {});
}

export async function me() {
  const token = localStorage.getItem('dashboard_token') || '';
  const res = await fetch(buildUrl('/auth/me'), {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) {
    const err = new Error('Oturum bulunamadı');
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export function startChatSession(payload, apiKey, { signal } = {}) {
  return fetchWithAuth('/chat/session', {
    method: 'POST',
    apiKey,
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function sendChatMessage(payload, apiKey, { signal } = {}) {
  return fetchWithAuth('/chat/message', {
    method: 'POST',
    apiKey,
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function getChatSummary(sessionId, apiKey, { signal } = {}) {
  return fetchWithAuth(`/chat/session/${encodeURIComponent(sessionId)}/summary`, {
    apiKey,
    signal,
  });
}
