import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSystemStatus } from '../hooks/useSystemStatus';

vi.mock('../api', () => ({
  getSystemStatus: vi.fn(),
}));

import { getSystemStatus } from '../api';

const mockStatus = {
  overall: 'ok',
  generated_at: '2026-02-18T14:00:00Z',
  services: {
    database: { name: 'Database', status: 'ok', reason: 'ok' },
    redis:    { name: 'Redis',    status: 'ok', reason: 'ok' },
  },
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe('useSystemStatus', () => {
  it('starts with status=null, loading=false, error=""', () => {
    getSystemStatus.mockResolvedValue(mockStatus);
    const { result } = renderHook(() => useSystemStatus({ apiKey: 'test-key' }));
    expect(result.current.status).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe('');
  });

  it('sets loading=true while fetching and false after', async () => {
    let resolve;
    getSystemStatus.mockReturnValue(new Promise(r => { resolve = r; }));
    const { result } = renderHook(() => useSystemStatus({ apiKey: 'k' }));

    await act(async () => { result.current.refresh(); });
    expect(result.current.loading).toBe(true);

    await act(async () => { resolve(mockStatus); });
    expect(result.current.loading).toBe(false);
  });

  it('sets status after successful refresh', async () => {
    getSystemStatus.mockResolvedValue(mockStatus);
    const { result } = renderHook(() => useSystemStatus({ apiKey: 'k' }));

    await act(async () => { await result.current.refresh(); });
    expect(result.current.status).toEqual(mockStatus);
    expect(result.current.error).toBe('');
  });

  it('sets error message on API failure', async () => {
    getSystemStatus.mockRejectedValue({ name: 'Error', message: 'Server down' });
    const { result } = renderHook(() => useSystemStatus({ apiKey: 'k' }));

    await act(async () => { await result.current.refresh(); });
    expect(result.current.error).toBe('Server down');
    expect(result.current.status).toBeNull();
  });

  it('calls onAuthFailed on 401 error', async () => {
    const onAuthFailed = vi.fn();
    getSystemStatus.mockRejectedValue({ status: 401, message: '401 Unauthorized' });
    const { result } = renderHook(() =>
      useSystemStatus({ apiKey: 'k', onAuthFailed }),
    );

    await act(async () => { await result.current.refresh(); });
    expect(onAuthFailed).toHaveBeenCalledTimes(1);
    expect(result.current.error).toBe('');
  });

  it('does not set error for AbortError', async () => {
    getSystemStatus.mockRejectedValue({ name: 'AbortError', message: 'Aborted' });
    const { result } = renderHook(() => useSystemStatus({ apiKey: 'k' }));

    await act(async () => { await result.current.refresh(); });
    expect(result.current.error).toBe('');
  });

  it('exposes refresh as a callable function', () => {
    getSystemStatus.mockResolvedValue(mockStatus);
    const { result } = renderHook(() => useSystemStatus());
    expect(typeof result.current.refresh).toBe('function');
  });

  it('passes apiKey to getSystemStatus', async () => {
    getSystemStatus.mockResolvedValue(mockStatus);
    const { result } = renderHook(() => useSystemStatus({ apiKey: 'my-secret-key' }));

    await act(async () => { await result.current.refresh(); });
    expect(getSystemStatus).toHaveBeenCalledWith('my-secret-key', expect.any(Object));
  });

  it('uses fallback error message when err.message is absent', async () => {
    getSystemStatus.mockRejectedValue({ name: 'Error' });
    const { result } = renderHook(() => useSystemStatus({ apiKey: 'k' }));

    await act(async () => { await result.current.refresh(); });
    expect(result.current.error).toBe('Sistem durumu alınamadı');
  });
});
