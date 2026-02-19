import { NavLink } from 'react-router-dom';
import { NAV_ITEMS } from '../lib/constants';
import { formatRunId } from '../lib/helpers';

/**
 * Sidebar — Sol navigasyon paneli
 */
export default function Sidebar({ auth, theme, runs }) {
  return (
    <aside className="sidebar">
      <div className="sidebarTitle">Rezervasyon Tahmin</div>
      <div className="sidebarSub">Karar Destek Paneli</div>

      <nav className="sidebarNav" role="navigation" aria-label="Ana navigasyon">
        {NAV_ITEMS.map(item => (
          <NavLink
            key={item.key}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) => `navBtn ${isActive ? 'active' : ''}`}
            title={item.desc}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebarInfo">
        <div><strong>Kullanıcı:</strong> {auth.currentUser}</div>
        <div><strong>Run:</strong> {formatRunId(runs.selectedRun)}</div>
      </div>

      <button
        className="themeToggle"
        onClick={theme.toggleTheme}
        aria-label={`Tema değiştir: ${theme.themeLabel}`}
      >
        <span className="themeIcon" aria-hidden="true">{theme.themeIcon}</span>
        {theme.themeLabel}
      </button>

      <button className="logoutBtn" onClick={auth.handleLogout}>
        ✕ Oturumu Kapat
      </button>
    </aside>
  );
}
