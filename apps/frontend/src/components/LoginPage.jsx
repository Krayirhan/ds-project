import { useState } from 'react';

/**
 * LoginPage — Giriş ekranı bileşeni
 *
 * Form state'i bu bileşene aittir (username, password).
 * Auth işlemleri props üzerinden gelen auth hook'u ile yapılır.
 */
export default function LoginPage({ auth, theme }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  async function handleSubmit(e) {
    e.preventDefault();
    const success = await auth.handleLogin(username, password);
    if (success) setPassword('');
  }

  return (
    <div className="loginPage">
      <form className="loginCard" onSubmit={handleSubmit}>
        <h1>Rezervasyon İptal Tahmin Sistemi — Giriş</h1>
        <p>Bu panel yalnızca yetkili personel içindir. Lütfen kimlik bilgilerinizi girin.</p>
        <label htmlFor="login-username">Kullanıcı Adı:</label>
        <input
          id="login-username"
          value={username}
          onChange={e => setUsername(e.target.value)}
          required
          autoFocus
          autoComplete="username"
        />
        <label htmlFor="login-password">Şifre:</label>
        <input
          id="login-password"
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          autoComplete="current-password"
        />
        {auth.loginError && <div className="error smallError">{auth.loginError}</div>}
        <button type="submit">Giriş</button>
      </form>
      <button
        className="themeToggle"
        onClick={theme.toggleTheme}
        style={{ position: 'fixed', bottom: 20, right: 20, zIndex: 100 }}
        aria-label={`Tema değiştir: ${theme.themeLabel}`}
      >
        <span className="themeIcon" aria-hidden="true">{theme.themeIcon}</span>
        {theme.themeLabel}
      </button>
    </div>
  );
}
