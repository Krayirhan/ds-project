import { Component } from 'react';

/**
 * ErrorBoundary — React hata sınır bileşeni
 *
 * Herhangi bir alt bileşende oluşan render hatasını yakalar,
 * tüm uygulamanın çökmesini engeller ve kullanıcıya bilgilendirici bir hata ekranı gösterir.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    // Production'da buraya hata raporlama servisi (Sentry vb.) eklenebilir
    console.error('[ErrorBoundary]', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', minHeight: '100vh', padding: 32,
          fontFamily: 'Inter, -apple-system, system-ui, sans-serif',
          background: '#f8fafc', color: '#1e293b',
        }}>
          <div style={{
            maxWidth: 480, textAlign: 'center',
            background: '#fff', border: '1px solid #e2e8f0',
            borderRadius: 12, padding: '40px 32px', boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
          }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>⚠️</div>
            <h1 style={{ fontSize: 20, fontWeight: 600, marginBottom: 8 }}>
              Beklenmeyen Bir Hata Oluştu
            </h1>
            <p style={{ fontSize: 14, color: '#64748b', lineHeight: 1.6, marginBottom: 20 }}>
              Uygulama bir hata ile karşılaştı. Sayfayı yenileyerek tekrar deneyebilirsiniz.
            </p>
            {this.state.error && (
              <pre style={{
                textAlign: 'left', fontSize: 11, background: '#fef2f2',
                border: '1px solid #fecaca', borderRadius: 6, padding: 12,
                overflow: 'auto', maxHeight: 120, marginBottom: 16, color: '#991b1b',
              }}>
                {String(this.state.error)}
              </pre>
            )}
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
              <button
                onClick={this.handleReset}
                style={{
                  padding: '8px 20px', borderRadius: 6, border: 'none',
                  background: '#1a56db', color: '#fff', cursor: 'pointer',
                  fontSize: 13, fontWeight: 500,
                }}
              >
                Tekrar Dene
              </button>
              <button
                onClick={() => window.location.reload()}
                style={{
                  padding: '8px 20px', borderRadius: 6,
                  border: '1px solid #d1d5db', background: '#fff',
                  color: '#374151', cursor: 'pointer', fontSize: 13, fontWeight: 500,
                }}
              >
                Sayfayı Yenile
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
