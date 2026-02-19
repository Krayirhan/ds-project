import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fetchWithAuth } from '../api';

describe('api — fetchWithAuth', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(() => 'test-token'),
      setItem: vi.fn(),
      removeItem: vi.fn(),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('sends auth headers', async () => {
    const mockResponse = { ok: true, json: () => Promise.resolve({ data: 1 }) };
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(mockResponse)));

    await fetchWithAuth('/test', { apiKey: 'my-key' });

    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, opts] = fetch.mock.calls[0];
    expect(url).toBe('/test');
    expect(opts.headers['x-api-key']).toBe('my-key');
    expect(opts.headers['Authorization']).toBe('Bearer test-token');
  });

  it('throws structured error on non-ok response', async () => {
    const mockResponse = {
      ok: false,
      status: 403,
      json: () => Promise.resolve({ detail: 'Forbidden' }),
    };
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(mockResponse)));

    await expect(fetchWithAuth('/test')).rejects.toThrow('Forbidden');
  });

  it('supports AbortController via signal option', async () => {
    const controller = new AbortController();
    controller.abort();

    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new DOMException('Aborted', 'AbortError'))));

    await expect(
      fetchWithAuth('/test', { signal: controller.signal }),
    ).rejects.toThrow();
  });
});
