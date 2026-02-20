import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import AppStatusBar from '../components/AppStatusBar';

function makeRuns(overrides = {}) {
  return {
    loading: false,
    coreModels: [],
    runs: [],
    ...overrides,
  };
}

describe('AppStatusBar', () => {
  it('shows "Hazır" when not loading', () => {
    render(<AppStatusBar runs={makeRuns()} />);
    expect(screen.getByText(/hazır/i)).toBeInTheDocument();
  });

  it('shows loading indicator when loading=true', () => {
    render(<AppStatusBar runs={makeRuns({ loading: true })} />);
    expect(screen.getByText('⏳ İşlem devam ediyor...')).toBeInTheDocument();
  });

  it('does not show "Hazır" while loading', () => {
    render(<AppStatusBar runs={makeRuns({ loading: true })} />);
    expect(screen.queryByText('✓ Hazır')).not.toBeInTheDocument();
  });

  it('shows correct coreModels count', () => {
    render(<AppStatusBar runs={makeRuns({ coreModels: ['a', 'b', 'c'] })} />);
    expect(screen.getByText(/3 temel/i)).toBeInTheDocument();
  });

  it('shows zero coreModels count', () => {
    render(<AppStatusBar runs={makeRuns({ coreModels: [] })} />);
    expect(screen.getByText(/0 temel/i)).toBeInTheDocument();
  });

  it('shows correct runs count', () => {
    render(<AppStatusBar runs={makeRuns({ runs: [1, 2, 3, 4, 5] })} />);
    expect(screen.getByText(/5 kayıt/i)).toBeInTheDocument();
  });

  it('shows zero runs count', () => {
    render(<AppStatusBar runs={makeRuns({ runs: [] })} />);
    expect(screen.getByText(/0 kayıt/i)).toBeInTheDocument();
  });

  it('renders a timestamp (non-empty)', () => {
    render(<AppStatusBar runs={makeRuns()} />);
    const bar = document.querySelector('.appStatusBar');
    // The last span contains the timestamp from now()
    const spans = bar.querySelectorAll('span');
    expect(spans.length).toBeGreaterThanOrEqual(4);
    // Timestamp span should not be empty
    expect(spans[spans.length - 1].textContent.trim()).not.toBe('');
  });
});
