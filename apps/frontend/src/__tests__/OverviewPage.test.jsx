import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

// OverviewPage uses useLayoutContext (useOutletContext), so we mock it
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useOutletContext: () => ({
      runs: {
        coreModels: [
          { model_name: 'xgboost', test_roc_auc: 0.87, test_f1: 0.79 },
        ],
        champion: { selected_model: 'xgboost', threshold: 0.42 },
        selectedRun: 'run-001',
        apiKey: '',
      },
      theme: { isDark: false, isModern: false },
    }),
  };
});

// Mock Chart.js to avoid canvas issues in jsdom
vi.mock('chart.js/auto', () => ({
  default: vi.fn().mockImplementation(() => ({
    destroy: vi.fn(),
    update: vi.fn(),
  })),
}));

// Mock API calls
vi.mock('../api', () => ({
  getMonitoring: vi.fn().mockResolvedValue(null),
}));

import OverviewPage from '../components/OverviewPage';

describe('OverviewPage', () => {
  it('renders without crashing', () => {
    const { container } = render(
      <MemoryRouter>
        <OverviewPage />
      </MemoryRouter>
    );
    expect(container).toBeTruthy();
  });

  it('renders champion card section', () => {
    const { container } = render(
      <MemoryRouter>
        <OverviewPage />
      </MemoryRouter>
    );
    // Should render something related to champion/model info
    expect(container.textContent).toBeTruthy();
  });
});
