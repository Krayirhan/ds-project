import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

const mockUseChat = vi.hoisted(() => vi.fn());
const mockGetAvailableModels = vi.hoisted(() => vi.fn());
const mockListGuests = vi.hoisted(() => vi.fn());
const mockCreateGuest = vi.hoisted(() => vi.fn());
const mockDeleteGuest = vi.hoisted(() => vi.fn());

vi.mock('../hooks/useChat', () => ({
  useChat: mockUseChat,
}));

vi.mock('../api', () => ({
  getAvailableModels: mockGetAvailableModels,
  listGuests: mockListGuests,
  createGuest: mockCreateGuest,
  deleteGuest: mockDeleteGuest,
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useOutletContext: () => ({
      runs: { apiKey: 'test-key' },
      auth: { handleAuthFailure: vi.fn() },
    }),
  };
});

import ChatPage from '../components/ChatPage';

describe('ChatPage interactions', () => {
  beforeEach(() => {
    const chatState = {
      sessionId: '',
      messages: [],
      input: '',
      quickActions: [],
      summary: null,
      busy: false,
      error: '',
      riskScore: null,
      riskLabel: 'unknown',
      predicting: false,
      selectedModel: null,
      guestId: null,
      guestSaved: false,
      customer: {},
      setInput: vi.fn(),
      setSelectedModel: vi.fn(),
      handleCustomerChange: vi.fn(),
      setGuestId: vi.fn(),
      setGuestSaved: vi.fn(),
      openSession: vi.fn(),
      sendMessage: vi.fn(),
    };

    mockUseChat.mockReset();
    mockUseChat.mockReturnValue(chatState);

    mockGetAvailableModels.mockReset();
    mockGetAvailableModels.mockResolvedValue({
      models: [{ name: 'challenger_xgboost', is_active: true }],
    });

    mockListGuests.mockReset();
    mockListGuests.mockResolvedValue({
      items: [
        {
          id: 11,
          first_name: 'Ali',
          last_name: 'Kaya',
          email: 'ali@example.com',
          phone: '+905551112233',
          nationality: 'TUR',
          hotel: 'City Hotel',
          market_segment: 'Online TA',
          vip_status: false,
          risk_label: 'medium',
          risk_score: 0.45,
          created_at: '2026-03-11T09:00:00Z',
        },
      ],
      total: 1,
    });

    mockCreateGuest.mockReset();
    mockCreateGuest.mockResolvedValue({ id: 99, first_name: 'Yeni', last_name: 'Misafir' });

    mockDeleteGuest.mockReset();
    mockDeleteGuest.mockResolvedValue(null);
  });

  it('loads guest list and maps selected guest into chat state', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );

    expect(await screen.findByText(/Ali Kaya/i)).toBeInTheDocument();

    const selectButton = screen.getByRole('button', { name: /Se/i });
    await user.click(selectButton);

    const chatState = mockUseChat.mock.results[0].value;
    expect(chatState.setGuestId).toHaveBeenCalledWith(11);
    expect(chatState.setGuestSaved).toHaveBeenCalledWith(true);
    expect(chatState.handleCustomerChange).toHaveBeenCalled();
  });

  it('allows model selection and chat session start', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <ChatPage />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(mockGetAvailableModels).toHaveBeenCalledWith('test-key');
    });

    const modelSelect = screen
      .getAllByRole('combobox')
      .find((el) => Array.from(el.options).some((opt) => opt.value === 'challenger_xgboost'));

    expect(modelSelect).toBeTruthy();
    await user.selectOptions(modelSelect, 'challenger_xgboost');

    const startButton = screen.getByRole('button', { name: /Chat Oturumu/i });
    await user.click(startButton);

    const chatState = mockUseChat.mock.results[0].value;
    expect(chatState.setSelectedModel).toHaveBeenCalledWith('challenger_xgboost');
    expect(chatState.openSession).toHaveBeenCalledTimes(1);
  });
});

