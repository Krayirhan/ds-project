import { useEffect } from 'react';
import { Outlet, useOutletContext } from 'react-router-dom';
import PropTypes from 'prop-types';
import { useRuns } from '../hooks/useRuns';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import FilterControls from './FilterControls';
import AppStatusBar from './AppStatusBar';

/**
 * Layout — Ana sayfa iskelet bileşeni
 *
 * Sidebar + TopBar + FilterControls + routed content (Outlet) + StatusBar
 * useRuns hook'unu burada çağırır, Outlet context olarak alt sayfalara aktarır.
 */
export default function Layout({ auth, theme }) {
  const runs = useRuns({ onAuthFailed: auth.handleAuthFailure });

  // İlk yüklemede veri çek
  useEffect(() => {
    runs.refreshRunsAndData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="appShell">
      <Sidebar auth={auth} theme={theme} runs={runs} />

      <main className="container">
        <TopBar runs={runs} />
        <FilterControls runs={runs} />

        {runs.error && (
          <div className="error card" role="alert">⚠ Hata: {runs.error}</div>
        )}

        <Outlet context={{ runs, theme, auth }} />

        <AppStatusBar runs={runs} />
      </main>
    </div>
  );
}

Layout.propTypes = {
  auth: PropTypes.shape({
    handleAuthFailure: PropTypes.func.isRequired,
    handleLogout: PropTypes.func.isRequired,
    currentUser: PropTypes.string,
  }).isRequired,
  theme: PropTypes.shape({
    toggleTheme: PropTypes.func.isRequired,
    themeLabel: PropTypes.string.isRequired,
    themeIcon: PropTypes.string.isRequired,
  }).isRequired,
};

/**
 * useLayoutContext — Alt sayfa bileşenlerinin Layout context'ine erişim hook'u
 *
 * Kullanım:
 *   const { runs, theme, auth } = useLayoutContext();
 */
export function useLayoutContext() {
  return useOutletContext();
}
