import { describe, it, expect, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useRuns } from '../hooks/useRuns';

// Mock API module
vi.mock('../api', () => ({
  getRuns: vi.fn().mockResolvedValue([]),
  getOverview: vi.fn().mockResolvedValue(null),
  getDbStatus: vi.fn().mockResolvedValue(null),
}));

describe('useRuns', () => {
  const defaultProps = { onAuthFailed: vi.fn() };

  it('returns expected shape', () => {
    const { result } = renderHook(() => useRuns(defaultProps));
    expect(typeof result.current.apiKey).toBe('string');
    expect(typeof result.current.setApiKey).toBe('function');
    expect(Array.isArray(result.current.runs)).toBe(true);
    expect(Array.isArray(result.current.dbRuns)).toBe(true);
    expect(typeof result.current.selectedRun).toBe('string');
    expect(typeof result.current.setSelectedRun).toBe('function');
    expect(typeof result.current.loading).toBe('boolean');
    expect(typeof result.current.refreshRunsAndData).toBe('function');
    expect(typeof result.current.refreshOverviewOnly).toBe('function');
  });

  it('starts with empty runs', () => {
    const { result } = renderHook(() => useRuns(defaultProps));
    expect(result.current.runs).toEqual([]);
  });

  it('starts not loading', () => {
    const { result } = renderHook(() => useRuns(defaultProps));
    expect(result.current.loading).toBe(false);
  });

  it('has derived properties', () => {
    const { result } = renderHook(() => useRuns(defaultProps));
    expect(Array.isArray(result.current.modelRows)).toBe(true);
    expect(typeof result.current.champion).toBe('object');
    expect(typeof result.current.generatedAt).toBe('string');
    expect(Array.isArray(result.current.coreModels)).toBe(true);
  });
});
