import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Sidebar from '../components/Sidebar';

function makeAuth() {
  return {
    handleLogout: vi.fn(),
    currentUser: 'admin',
  };
}

function makeTheme() {
  return {
    toggleTheme: vi.fn(),
    themeLabel: 'Koyu Tema',
    themeIcon: '🌙',
  };
}

function makeRuns(overrides = {}) {
  return {
    selectedRun: 'run-001',
    ...overrides,
  };
}

describe('Sidebar', () => {
  it('renders without crashing', () => {
    render(
      <MemoryRouter>
        <Sidebar auth={makeAuth()} theme={makeTheme()} runs={makeRuns()} />
      </MemoryRouter>
    );
  });

  it('displays current user', () => {
    render(
      <MemoryRouter>
        <Sidebar auth={makeAuth()} theme={makeTheme()} runs={makeRuns()} />
      </MemoryRouter>
    );
    expect(screen.getByText(/admin/i)).toBeInTheDocument();
  });

  it('renders theme toggle button', () => {
    render(
      <MemoryRouter>
        <Sidebar auth={makeAuth()} theme={makeTheme()} runs={makeRuns()} />
      </MemoryRouter>
    );
    expect(screen.getByText('🌙')).toBeInTheDocument();
  });

  it('renders navigation links', () => {
    render(
      <MemoryRouter>
        <Sidebar auth={makeAuth()} theme={makeTheme()} runs={makeRuns()} />
      </MemoryRouter>
    );
    const links = document.querySelectorAll('a');
    expect(links.length).toBeGreaterThan(0);
  });
});
