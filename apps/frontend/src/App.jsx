import { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';
import { useTheme } from './hooks/useTheme';
import ErrorBoundary from './components/ErrorBoundary';

const LoginPage = lazy(() => import('./components/LoginPage'));
const Layout = lazy(() => import('./components/Layout'));
const OverviewPage = lazy(() => import('./components/OverviewPage'));
const ModelsPage = lazy(() => import('./components/ModelsPage'));
const PipelinePage = lazy(() => import('./components/PipelinePage'));
const RunsPage = lazy(() => import('./components/RunsPage'));
const ChatPage = lazy(() => import('./components/ChatPage'));
const GuestPage = lazy(() => import('./components/GuestPage'));
const SystemPage = lazy(() => import('./components/SystemPage'));

function AppLoader({ message = 'Sayfa yukleniyor...' }) {
  return (
    <div className="loginWrap" style={{ justifyContent: 'center', alignItems: 'center', display: 'flex' }}>
      <div className="card" style={{ textAlign: 'center', padding: 40 }}>
        <div style={{ fontSize: 28, marginBottom: 10 }}>...</div>
        <div>{message}</div>
      </div>
    </div>
  );
}

function withRouteBoundary(element) {
  return <ErrorBoundary>{element}</ErrorBoundary>;
}

function routeElement(Component) {
  return withRouteBoundary(
    <Suspense fallback={<AppLoader />}>
      <Component />
    </Suspense>,
  );
}

/**
 * App - Root component.
 *
 * HashRouter is provided in main.jsx.
 */
export default function App() {
  const auth = useAuth();
  const theme = useTheme();

  if (auth.checking) {
    return <AppLoader message="Oturum kontrol ediliyor..." />;
  }

  if (!auth.authenticated) {
    return (
      <Suspense fallback={<AppLoader message="Giris ekrani yukleniyor..." />}>
        <LoginPage auth={auth} theme={theme} />
      </Suspense>
    );
  }

  return (
    <Routes>
      <Route
        element={withRouteBoundary(
          <Suspense fallback={<AppLoader message="Uygulama yukleniyor..." />}>
            <Layout auth={auth} theme={theme} />
          </Suspense>,
        )}
      >
        <Route index element={routeElement(OverviewPage)} />
        <Route path="models" element={routeElement(ModelsPage)} />
        <Route path="pipeline" element={routeElement(PipelinePage)} />
        <Route path="runs" element={routeElement(RunsPage)} />
        <Route path="chat" element={routeElement(ChatPage)} />
        <Route path="guests" element={routeElement(GuestPage)} />
        <Route path="system" element={routeElement(SystemPage)} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

