import { useState, useCallback, useEffect } from 'react';
import { login, logout, me } from '../api';

/**
 * @typedef {Object} UseAuthState
 * @property {boolean} authenticated
 * @property {string} currentUser
 * @property {string} loginError
 * @property {boolean} checking
 * @property {(username: string, password: string) => Promise<boolean>} handleLogin
 * @property {() => Promise<void>} handleLogout
 * @property {() => void} handleAuthFailure
 */
/**
 * useAuth â€” Kimlik doÄŸrulama hook'u
 *
 * JWT token'Ä± localStorage'dan okur ve /auth/me ile doÄŸrular.
 * Login / logout / auth-failure aksiyonlarÄ±nÄ± yÃ¶netir.
 *
 * @returns {UseAuthState}
 */
export function useAuth() {
  const [authenticated, setAuthenticated] = useState(false);
  const [currentUser, setCurrentUser]     = useState('');
  const [loginError, setLoginError]       = useState('');
  const [checking, setChecking]           = useState(true);

  // Ä°lk yÃ¼klemede mevcut token'Ä± doÄŸrula
  useEffect(() => {
    const token = localStorage.getItem('dashboard_token');
    if (!token) {
      setAuthenticated(false);
      setChecking(false);
      return;
    }
    me()
      .then((p) => {
        setAuthenticated(true);
        setCurrentUser(p.username || '');
      })
      .catch(() => {
        localStorage.removeItem('dashboard_token');
        setAuthenticated(false);
      })
      .finally(() => setChecking(false));
  }, []);

  const handleLogin = useCallback(async (username, password) => {
    setLoginError('');
    try {
      const p = await login(username, password);
      localStorage.setItem('dashboard_token', p.access_token);
      setAuthenticated(true);
      setCurrentUser(p.username || username);
      return true;
    } catch (err) {
      setLoginError(err.message || 'GiriÅŸ yapÄ±lamadÄ±.');
      return false;
    }
  }, []);

  const handleLogout = useCallback(async () => {
    try { await logout(); } catch { /* ignore */ }
    localStorage.removeItem('dashboard_token');
    setAuthenticated(false);
    setCurrentUser('');
  }, []);

  const handleAuthFailure = useCallback(() => {
    localStorage.removeItem('dashboard_token');
    setAuthenticated(false);
    setLoginError('Oturum sÃ¼resi doldu. LÃ¼tfen tekrar giriÅŸ yapÄ±n.');
  }, []);

  return {
    authenticated,
    currentUser,
    loginError,
    checking,
    handleLogin,
    handleLogout,
    handleAuthFailure,
  };
}


