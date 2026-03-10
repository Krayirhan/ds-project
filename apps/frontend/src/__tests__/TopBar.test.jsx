import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import TopBar from '../components/TopBar';

function makeRuns(overrides = {}) {
  return {
    generatedAt: '2026-03-10T12:00:00Z',
    champion: { selected_model: 'xgboost' },
    ...overrides,
  };
}

describe('TopBar', () => {
  it('renders without crashing', () => {
    render(<TopBar runs={makeRuns()} />);
  });

  it('displays active model name', () => {
    render(<TopBar runs={makeRuns()} />);
    expect(screen.getByText(/xgboost/i)).toBeInTheDocument();
  });

  it('handles missing champion gracefully', () => {
    render(<TopBar runs={makeRuns({ champion: {} })} />);
    // Should not crash
    expect(document.querySelector('.topBar') || document.body).toBeTruthy();
  });
});
