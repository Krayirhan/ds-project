import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

const mockGetExplain = vi.hoisted(() => vi.fn());

vi.mock('../api', () => ({
  getExplain: mockGetExplain,
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useOutletContext: () => ({
      runs: {
        modelRows: [
          {
            model_name: 'challenger_xgboost',
            train_cv_roc_auc_mean: 0.91,
            train_cv_roc_auc_std: 0.01,
            cv_folds: 5,
            test_roc_auc: 0.9,
            test_f1: 0.82,
            test_precision: 0.83,
            test_recall: 0.81,
            test_threshold: 0.42,
            n_test: 250,
            positive_rate_test: 0.31,
          },
          {
            model_name: 'baseline',
            train_cv_roc_auc_mean: 0.84,
            train_cv_roc_auc_std: 0.02,
            cv_folds: 5,
            test_roc_auc: 0.83,
            test_f1: 0.75,
            test_precision: 0.74,
            test_recall: 0.76,
            test_threshold: 0.5,
            n_test: 250,
            positive_rate_test: 0.31,
          },
        ],
        champion: { selected_model: 'challenger_xgboost' },
        coreModels: [
          { model_name: 'challenger_xgboost', cv_folds: 5 },
          { model_name: 'baseline', cv_folds: 5 },
        ],
        selectedRun: 'run-001',
        apiKey: 'test-key',
      },
      theme: { isDark: false },
    }),
  };
});

import ModelsPage from '../components/ModelsPage';

describe('ModelsPage interactions', () => {
  beforeEach(() => {
    mockGetExplain.mockReset();
    mockGetExplain.mockResolvedValue({
      method: 'permutation_importance',
      scoring: 'roc_auc',
      n_repeats: 10,
      ranking: [
        { feature: 'lead_time', importance_mean: 0.12, importance_std: 0.02 },
        { feature: 'adr', importance_mean: 0.09, importance_std: 0.01 },
      ],
    });
  });

  it('loads explain data and renders feature importance rows', async () => {
    render(
      <MemoryRouter>
        <ModelsPage />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(mockGetExplain).toHaveBeenCalledWith('run-001', 'test-key', expect.any(Object));
    });

    expect(await screen.findByText('lead_time')).toBeInTheDocument();
    expect(screen.getByText('adr')).toBeInTheDocument();
  });

  it('opens model detail panel when a row is clicked', async () => {
    const user = userEvent.setup();

    render(
      <MemoryRouter>
        <ModelsPage />
      </MemoryRouter>,
    );

    const rowButton = screen.getByRole('button', { name: /xgboost/i });
    await user.click(rowButton);

    expect(screen.getByText(/Detay Bilgisi/i)).toBeInTheDocument();
    expect(screen.getByText(/challenger_xgboost/i)).toBeInTheDocument();
  });
});

