import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

const mockRefresh = vi.hoisted(() => vi.fn());
const mockUseSystemStatus = vi.hoisted(() => vi.fn());

vi.mock('../hooks/useSystemStatus', () => ({
  useSystemStatus: mockUseSystemStatus,
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useOutletContext: () => ({
      runs: {
        apiKey: 'test-key',
        runs: ['run-001'],
        dbRuns: [{ run_id: 'run-001' }],
        selectedRun: 'run-001',
        champion: { selected_model: 'challenger_xgboost' },
      },
      auth: { handleAuthFailure: vi.fn() },
    }),
  };
});

import SystemPage from '../components/SystemPage';

describe('SystemPage interactions', () => {
  beforeEach(() => {
    mockRefresh.mockReset();
    mockUseSystemStatus.mockReset();
    mockUseSystemStatus.mockReturnValue({
      status: {
        overall: 'ok',
        generated_at: '2026-03-11T10:00:00Z',
        services: {
          database: {
            name: 'Database',
            status: 'ok',
            reason: 'ok',
            backend: 'postgresql',
            url: 'postgres://localhost:5432/ds',
          },
          model: {
            name: 'Model',
            status: 'ok',
            reason: 'ok',
            model_name: 'challenger_xgboost',
          },
        },
      },
      loading: false,
      error: '',
      refresh: mockRefresh,
    });
  });

  it('requests refresh on mount and displays service cards', async () => {
    render(
      <MemoryRouter>
        <SystemPage />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(mockRefresh).toHaveBeenCalledTimes(1);
    });

    expect(screen.getByText('Database')).toBeInTheDocument();
    expect(screen.getByText('Model')).toBeInTheDocument();
  });

  it('triggers refresh again when clicking the refresh button', async () => {
    const user = userEvent.setup();

    render(
      <MemoryRouter>
        <SystemPage />
      </MemoryRouter>,
    );

    const refreshButton = screen.getByRole('button', { name: /Yenile/i });
    await user.click(refreshButton);

    expect(mockRefresh).toHaveBeenCalledTimes(2);
  });
});

