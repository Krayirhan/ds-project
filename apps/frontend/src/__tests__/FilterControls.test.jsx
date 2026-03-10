import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import FilterControls from '../components/FilterControls';

function makeRuns(overrides = {}) {
  return {
    runs: ['run-001', 'run-002'],
    selectedRun: 'run-001',
    setSelectedRun: vi.fn(),
    apiKey: '',
    setApiKey: vi.fn(),
    loading: false,
    refreshRunsAndData: vi.fn(),
    refreshOverviewOnly: vi.fn(),
    ...overrides,
  };
}

describe('FilterControls', () => {
  it('renders without crashing', () => {
    render(<FilterControls runs={makeRuns()} />);
  });

  it('renders run selector', () => {
    render(<FilterControls runs={makeRuns()} />);
    const select = document.querySelector('select');
    expect(select).toBeTruthy();
  });

  it('renders API key input', () => {
    render(<FilterControls runs={makeRuns()} />);
    const input = document.querySelector('input');
    expect(input).toBeTruthy();
  });

  it('renders refresh button', () => {
    render(<FilterControls runs={makeRuns()} />);
    const buttons = screen.getAllByRole('button');
    expect(buttons.length).toBeGreaterThan(0);
  });

  it('calls setSelectedRun on change', () => {
    const runs = makeRuns();
    render(<FilterControls runs={runs} />);
    const select = document.querySelector('select');
    fireEvent.change(select, { target: { value: 'run-002' } });
    expect(runs.setSelectedRun).toHaveBeenCalled();
  });
});
