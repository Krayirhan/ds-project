import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import LoginPage from '../components/LoginPage';

function makeAuth(overrides = {}) {
  return {
    handleLogin: vi.fn().mockResolvedValue(true),
    loginError: null,
    ...overrides,
  };
}

function makeTheme(overrides = {}) {
  return {
    toggleTheme: vi.fn(),
    themeLabel: 'Koyu Tema',
    themeIcon: '🌙',
    ...overrides,
  };
}

describe('LoginPage', () => {
  it('renders username and password fields', () => {
    render(<LoginPage auth={makeAuth()} theme={makeTheme()} />);
    expect(screen.getByLabelText(/kullanıcı adı/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/şifre/i)).toBeInTheDocument();
  });

  it('renders a submit button', () => {
    render(<LoginPage auth={makeAuth()} theme={makeTheme()} />);
    expect(screen.getByRole('button', { name: /giriş/i })).toBeInTheDocument();
  });

  it('calls handleLogin with username and password on submit', async () => {
    const auth = makeAuth();
    render(<LoginPage auth={auth} theme={makeTheme()} />);

    fireEvent.change(screen.getByLabelText(/kullanıcı adı/i), { target: { value: 'admin' } });
    fireEvent.change(screen.getByLabelText(/şifre/i), { target: { value: 'secret123' } });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /giriş/i }));
    });

    expect(auth.handleLogin).toHaveBeenCalledWith('admin', 'secret123');
  });

  it('clears password on successful login', async () => {
    const auth = makeAuth({ handleLogin: vi.fn().mockResolvedValue(true) });
    const { container } = render(<LoginPage auth={auth} theme={makeTheme()} />);

    fireEvent.change(screen.getByLabelText(/şifre/i), { target: { value: 'mypassword' } });

    await act(async () => {
      fireEvent.submit(container.querySelector('form'));
      await new Promise(r => setTimeout(r, 0));
    });

    expect(auth.handleLogin).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText(/şifre/i)).toHaveValue('');
  });

  it('does not clear password on failed login', async () => {
    const auth = makeAuth({ handleLogin: vi.fn().mockResolvedValue(false) });
    const { container } = render(<LoginPage auth={auth} theme={makeTheme()} />);

    fireEvent.change(screen.getByLabelText(/şifre/i), { target: { value: 'wrong' } });

    await act(async () => {
      fireEvent.submit(container.querySelector('form'));
      await new Promise(r => setTimeout(r, 0));
    });

    expect(auth.handleLogin).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText(/şifre/i)).toHaveValue('wrong');
  });

  it('displays loginError when present', () => {
    const auth = makeAuth({ loginError: 'Kullanıcı adı veya şifre hatalı' });
    render(<LoginPage auth={auth} theme={makeTheme()} />);
    expect(screen.getByText('Kullanıcı adı veya şifre hatalı')).toBeInTheDocument();
  });

  it('does not display error element when loginError is null', () => {
    render(<LoginPage auth={makeAuth()} theme={makeTheme()} />);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('renders theme toggle button with aria-label', () => {
    render(<LoginPage auth={makeAuth()} theme={makeTheme()} />);
    const btn = screen.getByRole('button', { name: /tema değiştir/i });
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveAttribute('aria-label', 'Tema değiştir: Koyu Tema');
  });

  it('calls toggleTheme when theme button is clicked', () => {
    const theme = makeTheme();
    render(<LoginPage auth={makeAuth()} theme={theme} />);

    fireEvent.click(screen.getByRole('button', { name: /tema değiştir/i }));
    expect(theme.toggleTheme).toHaveBeenCalledTimes(1);
  });

  it('themeIcon has aria-hidden=true', () => {
    render(<LoginPage auth={makeAuth()} theme={makeTheme()} />);
    const icon = document.querySelector('.themeIcon');
    expect(icon).toHaveAttribute('aria-hidden', 'true');
  });
});
