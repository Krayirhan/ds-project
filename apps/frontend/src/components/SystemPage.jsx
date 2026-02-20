import { useEffect } from 'react';
import { useLayoutContext } from './Layout';
import { useSystemStatus } from '../hooks/useSystemStatus';
import { displayName } from '../lib/helpers';

// ── Status display helpers ────────────────────────────────────────────────────

const STATUS_META = {
  ok:           { icon: '●', color: '#006600', label: 'Sağlıklı' },
  warning:      { icon: '◐', color: '#b06000', label: 'Uyarı' },
  unconfigured: { icon: '○', color: '#888888', label: 'Yapılandırılmamış' },
  error:        { icon: '●', color: '#cc0000', label: 'Hata' },
  degraded:     { icon: '●', color: '#cc0000', label: 'Sorunlu' },
  partial:      { icon: '◐', color: '#b06000', label: 'Kısmi' },
};

function statusMeta(s) {
  return STATUS_META[s] || STATUS_META.error;
}

// ── ServiceCard ───────────────────────────────────────────────────────────────

function ServiceCard({ svc }) {
  const m = statusMeta(svc.status);
  return (
    <div className="card" style={{ padding: '16px 20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <strong style={{ fontSize: 14 }}>{svc.name}</strong>
        <span style={{ color: m.color, fontWeight: 700, fontSize: 13 }}>
          {m.icon} {m.label}
        </span>
      </div>

      <div className="systemGrid" style={{ gap: '6px 16px' }}>
        {svc.url && (
          <div className="sysItem full">
            <span>Adres</span>
            <strong style={{ fontSize: 11, wordBreak: 'break-all' }}>{svc.url}</strong>
          </div>
        )}
        {svc.model && (
          <div className="sysItem">
            <span>Model</span>
            <strong>{svc.model}</strong>
          </div>
        )}
        {svc.backend && (
          <div className="sysItem">
            <span>Motor</span>
            <strong>{svc.backend === 'postgresql' ? 'PostgreSQL' : 'SQLite'}</strong>
          </div>
        )}
        {svc.model_name && (
          <div className="sysItem">
            <span>Aktif Model</span>
            <strong style={{ fontSize: 11 }}>{displayName(svc.model_name)}</strong>
          </div>
        )}
        <div className="sysItem full">
          <span>Açıklama</span>
          <strong style={{ color: svc.reason === 'ok' ? '#006600' : '#555', fontWeight: 400 }}>
            {svc.reason === 'ok'
              ? 'Bağlantı başarılı, servis sağlıklı çalışıyor.'
              : svc.reason || '-'}
          </strong>
        </div>
      </div>
    </div>
  );
}

// ── SystemPage ────────────────────────────────────────────────────────────────

/**
 * SystemPage — Sistem Durumu
 *
 * Tüm backend bağımlılıklarını (DB, Redis, Ollama, model) tek API çağrısıyla
 * sorgular ve kart formatında gösterir.
 */
export default function SystemPage() {
  const { runs, auth } = useLayoutContext();

  const sys = useSystemStatus({
    apiKey: runs.apiKey,
    onAuthFailed: auth?.handleAuthFailure,
  });

  useEffect(() => {
    sys.refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const services  = sys.status?.services ? Object.values(sys.status.services) : null;
  const overall   = sys.status?.overall;
  const ovMeta    = statusMeta(overall || 'unconfigured');

  return (
    <>
      <header className="pageHeader">
        <h1>🖥️ Sistem Durumu</h1>
        <p className="subtitle">
          Tüm bağımlı servislerin anlık sağlık durumu, maliyet matrisi ve genel bilgiler.
        </p>
      </header>

      {/* Overall banner */}
      <section className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="small">Genel Sistem Durumu</div>
            <strong style={{ fontSize: 18, color: ovMeta.color }}>
              {sys.loading
                ? '⏳ Sorgulanıyor…'
                : sys.error
                  ? '⚠ Durum alınamadı'
                  : overall
                    ? `${ovMeta.icon} ${
                        overall === 'ok' ? 'Tüm servisler sağlıklı'
                        : overall === 'degraded' ? 'Bir veya daha fazla servis hatalı'
                        : 'Bazı servisler kısmi çalışıyor'
                      }`
                    : '○ Henüz sorgulanmadı'}
            </strong>
            {sys.status?.generated_at && (
              <div className="explain" style={{ marginTop: 4 }}>
                Son güncelleme: {new Date(sys.status.generated_at).toLocaleString('tr-TR')}
              </div>
            )}
          </div>
          <button onClick={sys.refresh} disabled={sys.loading}>
            {sys.loading ? '⏳ Sorgulanıyor…' : '🔄 Yenile'}
          </button>
        </div>
        {sys.error && (
          <div className="error" style={{ marginTop: 8 }} role="alert">⚠ {sys.error}</div>
        )}
      </section>

      {/* Service cards — 2-column grid */}
      {services && (
        <section className="grid2">
          {services.map(svc => <ServiceCard key={svc.name} svc={svc} />)}
        </section>
      )}

      {/* Cost Matrix */}
      <section className="card">
        <div className="small">Maliyet Matrisi — Karar Parametreleri</div>
        <div className="explain">
          Bu değerler modelin "hangi müşteriye müdahale etmeli?" kararını şekillendirir.
        </div>
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th>Senaryo</th><th>Kısaltma</th><th>Değer</th><th>Açıklama</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Doğru Pozitif</td>
                <td style={{ fontFamily: 'Consolas' }}>TP</td>
                <td style={{ color: '#006600', fontWeight: 700 }}>+180 ₺</td>
                <td>İptal edecek müşteriyi doğru tahmin ettik ve kurtardık</td>
              </tr>
              <tr>
                <td>Yanlış Pozitif</td>
                <td style={{ fontFamily: 'Consolas' }}>FP</td>
                <td style={{ color: '#cc0000', fontWeight: 700 }}>−20 ₺</td>
                <td>İptal etmeyecek müşteriye gereksiz müdahale</td>
              </tr>
              <tr>
                <td>Yanlış Negatif</td>
                <td style={{ fontFamily: 'Consolas' }}>FN</td>
                <td style={{ color: '#cc0000', fontWeight: 700 }}>−200 ₺</td>
                <td>İptal edecek müşteriyi kaçırdık</td>
              </tr>
              <tr>
                <td>Doğru Negatif</td>
                <td style={{ fontFamily: 'Consolas' }}>TN</td>
                <td style={{ color: '#666' }}>0 ₺</td>
                <td>İptal etmeyecek müşteriyi doğru tahmin ettik</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* General run info */}
      <section className="card">
        <div className="small">Genel Bilgiler</div>
        <div className="systemGrid">
          <div className="sysItem">
            <span>Toplam Koşu Sayısı</span>
            <strong>{runs.runs.length}</strong>
          </div>
          <div className="sysItem">
            <span>DB Kayıtlı Koşu</span>
            <strong>{runs.dbRuns.length}</strong>
          </div>
          <div className="sysItem">
            <span>Aktif Run ID</span>
            <strong>{runs.selectedRun || '-'}</strong>
          </div>
          <div className="sysItem">
            <span>Güncel Şampiyon</span>
            <strong>{displayName(runs.champion.selected_model)}</strong>
          </div>
        </div>
      </section>
    </>
  );
}
