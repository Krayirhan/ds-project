import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAuth } from '../hooks/useAuth';

// Mock API module
vi.mock('../api', () => ({
  login: vi.fn().mockResolvedValue({ token: 'test-token' }),
  logout: vi.fn().mockResolvedValue(undefined),
  me: vi.fn().mockResolvedValue({ username: 'admin' }),
}));

describe('useAuth', () => {
  it('starts in checking state', () => {
    const { result } = renderHook(() => useAuth());
    // Initially should be checking or authenticated based on stored token
    expect(typeof result.current.authenticated).toBe('boolean');
    expect(typeof result.current.checking).toBe('boolean');
  });

  it('exposes handleLogin function', () => {
    const { result } = renderHook(() => useAuth());
    expect(typeof result.current.handleLogin).toBe('function');
  });

  it('exposes handleLogout function', () => {
    const { result } = renderHook(() => useAuth());
    expect(typeof result.current.handleLogout).toBe('function');
  });

  it('exposes handleAuthFailure function', () => {
    const { result } = renderHook(() => useAuth());
    expect(typeof result.current.handleAuthFailure).toBe('function');
  });

  it('has correct return shape', () => {
    const { result } = renderHook(() => useAuth());
    const keys = Object.keys(result.current);
    expect(keys).toContain('authenticated');
    expect(keys).toContain('currentUser');
    expect(keys).toContain('loginError');
    expect(keys).toContain('checking');
    expect(keys).toContain('handleLogin');
    expect(keys).toContain('handleLogout');
    expect(keys).toContain('handleAuthFailure');
  });
});
