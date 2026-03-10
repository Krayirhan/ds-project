import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useOutletContext: () => ({
      runs: {
        runs: ['run-001', 'run-002'],
        dbRuns: [
          { run_id: 'run-001', selected_model: 'xgboost', updated_at: '2026-03-10' },
        ],
        selectedRun: 'run-001',
        setSelectedRun: vi.fn(),
        refreshOverviewOnly: vi.fn(),
      },
    }),
    useNavigate: () => vi.fn(),
  };
});

import RunsPage from '../components/RunsPage';

describe('RunsPage', () => {
  it('renders without crashing', () => {
    const { container } = render(
      <MemoryRouter>
        <RunsPage />
      </MemoryRouter>
    );
    expect(container).toBeTruthy();
  });

  it('renders runs information', () => {
    const { container } = render(
      <MemoryRouter>
        <RunsPage />
      </MemoryRouter>
    );
    expect(container.textContent.length).toBeGreaterThan(0);
  });
});
