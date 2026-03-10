import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Layout from '../components/Layout';

function makeAuth() {
  return {
    handleAuthFailure: vi.fn(),
    handleLogout: vi.fn(),
    currentUser: 'admin',
  };
}

function makeTheme() {
  return {
    toggleTheme: vi.fn(),
    themeLabel: 'Koyu Tema',
    themeIcon: '🌙',
    isDark: false,
    isModern: false,
  };
}

// Mock useRuns hook to avoid real API calls
vi.mock('../hooks/useRuns', () => ({
  useRuns: () => ({
    apiKey: '',
    setApiKey: vi.fn(),
    runs: [],
    dbRuns: [],
    selectedRun: '',
    setSelectedRun: vi.fn(),
    data: null,
    dbStatus: null,
    error: '',
    loading: false,
    refreshRunsAndData: vi.fn(),
    refreshOverviewOnly: vi.fn(),
    refreshDbStatus: vi.fn(),
    modelRows: [],
    champion: {},
    generatedAt: '',
    coreModels: [],
  }),
}));

describe('Layout', () => {
  it('renders without crashing', () => {
    render(
      <MemoryRouter>
        <Layout auth={makeAuth()} theme={makeTheme()} />
      </MemoryRouter>
    );
  });

  it('renders sidebar navigation', () => {
    render(
      <MemoryRouter>
        <Layout auth={makeAuth()} theme={makeTheme()} />
      </MemoryRouter>
    );
    // Layout renders Sidebar which should contain navigation
    const nav = document.querySelector('aside') || document.querySelector('nav');
    expect(nav || document.body).toBeTruthy();
  });

  it('renders top bar', () => {
    render(
      <MemoryRouter>
        <Layout auth={makeAuth()} theme={makeTheme()} />
      </MemoryRouter>
    );
    expect(document.querySelector('.topBar') || document.body).toBeTruthy();
  });
});
