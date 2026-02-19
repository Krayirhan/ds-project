import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { getRuns, getOverview, getDbStatus } from '../api';

/**
 * useRuns — Koşu verileri, model ve DB durumu hook'u
 *
 * AbortController ile sayfa değişiminde inflight request'ler iptal edilir.
 */
export function useRuns({ onAuthFailed }) {
  const [apiKey, setApiKey]           = useState(import.meta.env.VITE_DEFAULT_API_KEY || '');
  const [runs, setRuns]               = useState([]);
  const [dbRuns, setDbRuns]           = useState([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [data, setData]               = useState(null);
  const [dbStatus, setDbStatus]       = useState(null);
  const [error, setError]             = useState('');
  const [loading, setLoading]         = useState(false);

  const abortRef      = useRef(null);
  const selectedRunRef = useRef(selectedRun);
  selectedRunRef.current = selectedRun;

  function handleApiError(err) {
    if (err.name === 'AbortError') return true;
    if (err?.status === 401 || String(err?.message || '').includes('401')) {
      onAuthFailed?.(err);
      return true;
    }
    return false;
  }

  const refreshRunsAndData = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError('');
    setLoading(true);
    try {
      const payload   = await getRuns(apiKey, { signal: controller.signal });
      const available = payload.runs || [];
      setRuns(available);
      setDbRuns(payload.db_runs || []);

      const current = selectedRunRef.current;
      const target  = current || available[0] || '';
      if (target && target !== current) setSelectedRun(target);

      const overview = await getOverview(target, apiKey, { signal: controller.signal });
      setData(overview);
    } catch (err) {
      if (!handleApiError(err)) setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiKey, onAuthFailed]);

  const refreshOverviewOnly = useCallback(async (runId) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError('');
    setLoading(true);
    try {
      const overview = await getOverview(runId, apiKey, { signal: controller.signal });
      setData(overview);
    } catch (err) {
      if (!handleApiError(err)) setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiKey, onAuthFailed]);

  const refreshDbStatus = useCallback(async () => {
    setError('');
    setLoading(true);
    try {
      const s = await getDbStatus(apiKey);
      setDbStatus(s);
    } catch (err) {
      if (!handleApiError(err)) setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiKey, onAuthFailed]);

  // Unmount'ta inflight request'leri iptal et
  useEffect(() => () => abortRef.current?.abort(), []);

  // ── Türetilmiş veriler ──────────────────────────────────────────
  const modelRows  = data?.models   || [];
  const champion   = data?.champion || {};
  const generatedAt = data?.generated_at
    ? new Date(data.generated_at).toLocaleString('tr-TR') : '-';

  const coreModels = useMemo(
    () => modelRows.filter(m => !m.model_name.endsWith('_decision')),
    [modelRows],
  );

  return {
    apiKey, setApiKey,
    runs, dbRuns, selectedRun, setSelectedRun,
    data, dbStatus, error, loading,
    refreshRunsAndData, refreshOverviewOnly, refreshDbStatus,
    // Türetilmiş
    modelRows, champion, generatedAt, coreModels,
  };
}
