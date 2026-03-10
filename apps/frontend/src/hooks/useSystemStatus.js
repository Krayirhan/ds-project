import { useState, useCallback, useEffect, useRef } from 'react';
import { getSystemStatus } from '../api';

/**
 * @typedef {Object} SystemServiceStatus
 * @property {string} name
 * @property {string} status
 * @property {string=} reason
 * @property {string=} url
 * @property {string=} backend
 * @property {string=} model
 * @property {string=} model_name
 */

/**
 * @typedef {Object} SystemStatusPayload
 * @property {string} overall
 * @property {string} generated_at
 * @property {Record<string, SystemServiceStatus>} services
 */

/**
 * @typedef {Object} UseSystemStatusState
 * @property {SystemStatusPayload|null} status
 * @property {boolean} loading
 * @property {string} error
 * @property {() => Promise<void>} refresh
 */
/**
 * useSystemStatus â€” TÃ¼m backend servislerinin saÄŸlÄ±k durumunu Ã§eker.
 *
 * GET /dashboard/api/system â†’ { overall, generated_at, services: { database, redis, ollama, model } }
 *
 * AbortController ile sayfa deÄŸiÅŸiminde inflight istek otomatik olarak iptal edilir.
 *
 * @param {object}   [opts={}]
 * @param {string}   opts.apiKey        - API anahtarÄ± (VITE_DEFAULT_API_KEY varsayÄ±lan)
 * @param {function} opts.onAuthFailed  - 401 durumunda Ã§aÄŸrÄ±lÄ±r
 *
 * @returns {UseSystemStatusState}
 */
export function useSystemStatus({ apiKey, onAuthFailed } = {}) {
  const [status, setStatus]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const abortRef = useRef(null);

  const refresh = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError('');
    try {
      const data = await getSystemStatus(apiKey, { signal: controller.signal });
      setStatus(data);
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (err?.status === 401 || String(err?.message || '').includes('401')) {
        onAuthFailed?.(err);
        return;
      }
      setError(err.message || 'Sistem durumu alÄ±namadÄ±');
    } finally {
      setLoading(false);
    }
  }, [apiKey, onAuthFailed]);

  // Unmount'ta inflight isteÄŸi iptal et
  useEffect(() => () => abortRef.current?.abort(), []);

  return { status, loading, error, refresh };
}


