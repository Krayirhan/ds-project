import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useTheme } from '../hooks/useTheme';

describe('useTheme', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(() => null),
      setItem: vi.fn(),
      removeItem: vi.fn(),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    document.documentElement.removeAttribute('data-theme');
  });

  it('defaults to classic theme', () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('classic');
    expect(result.current.isModern).toBe(false);
    expect(result.current.isDark).toBe(false);
  });

  it('cycles through themes on toggle', () => {
    const { result } = renderHook(() => useTheme());

    // classic → modern-light
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('modern-light');
    expect(result.current.isModern).toBe(true);
    expect(result.current.isDark).toBe(false);

    // modern-light → modern-dark
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('modern-dark');
    expect(result.current.isDark).toBe(true);

    // modern-dark → classic
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('classic');
  });

  it('sets data-theme attribute on document', () => {
    const { result } = renderHook(() => useTheme());

    act(() => result.current.toggleTheme());
    expect(document.documentElement.getAttribute('data-theme')).toBe('modern-light');

    act(() => result.current.toggleTheme());
    expect(document.documentElement.getAttribute('data-theme')).toBe('modern-dark');

    act(() => result.current.toggleTheme());
    expect(document.documentElement.getAttribute('data-theme')).toBeNull();
  });
});
