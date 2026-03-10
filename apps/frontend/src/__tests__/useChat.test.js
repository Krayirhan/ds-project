import { describe, it, expect, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useChat } from '../hooks/useChat';

// Mock API module
vi.mock('../api', () => ({
  startChatSession: vi.fn().mockResolvedValue({ session_id: 'sess-1' }),
  getChatSummary: vi.fn().mockResolvedValue(null),
  streamChatMessage: vi.fn().mockResolvedValue(undefined),
  predictRiskScore: vi.fn().mockResolvedValue({ proba: [0.5] }),
  createGuest: vi.fn().mockResolvedValue({ id: 1 }),
}));

describe('useChat', () => {
  const defaultProps = {
    apiKey: 'test-key',
    onAuthFailed: vi.fn(),
    initialCustomer: {},
  };

  it('returns expected shape', () => {
    const { result } = renderHook(() => useChat(defaultProps));
    expect(typeof result.current.sessionId).toBe('string') || expect(result.current.sessionId).toBeNull();
    expect(Array.isArray(result.current.messages)).toBe(true);
    expect(typeof result.current.input).toBe('string');
    expect(typeof result.current.busy).toBe('boolean');
    expect(typeof result.current.setInput).toBe('function');
    expect(typeof result.current.openSession).toBe('function');
    expect(typeof result.current.sendMessage).toBe('function');
  });

  it('starts with empty messages', () => {
    const { result } = renderHook(() => useChat(defaultProps));
    expect(result.current.messages).toEqual([]);
  });

  it('starts not busy', () => {
    const { result } = renderHook(() => useChat(defaultProps));
    expect(result.current.busy).toBe(false);
  });

  it('exposes customer change handler', () => {
    const { result } = renderHook(() => useChat(defaultProps));
    expect(typeof result.current.handleCustomerChange).toBe('function');
  });
});
