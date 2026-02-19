import { useEffect } from 'react';
import { useLayoutContext } from './Layout';
import { displayName } from '../lib/helpers';

/**
 * SystemPage — Sistem Durumu
 */
export default function SystemPage() {
  const { runs } = useLayoutContext();

  // Sayfa yüklendiğinde DB durumunu çek
  useEffect(() => {
    runs.refreshDbStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <header className="pageHeader">
        <h1>🖥️ Sistem Durumu</h1>
        <p className="subtitle">Veritabanı bağlantısı, altyapı bilgileri ve maliyet matrisi.</p>
      </header>

      <section className="card">
        <div className="small">Veritabanı Bağlantısı</div>
        <div className="systemGrid">
          <div className="sysItem">
            <span>Veritabanı Motoru</span>
            <strong>
              {runs.dbStatus?.database_backend === 'sqlite' ? 'SQLite (Yerel)'
                : runs.dbStatus?.database_backend === 'postgresql' ? 'PostgreSQL'
                : runs.dbStatus?.database_backend || '-'}
            </strong>
          </div>
          <div className="sysItem">
            <span>Bağlantı Durumu</span>
            <strong style={{ color: runs.dbStatus?.connected ? '#006600' : '#cc0000' }}>
              {runs.dbStatus?.connected ? '● Bağlı — Sorunsuz' : '○ Bağlantı Yok'}
            </strong>
          </div>
          <div className="sysItem full">
            <span>Bağlantı Adresi</span>
            <strong>{runs.dbStatus?.database_url || '-'}</strong>
          </div>
          <div className="sysItem full">
            <span>Durum Açıklaması</span>
            <strong>{runs.dbStatus?.reason === 'ok' ? 'Veritabanı sağlıklı çalışıyor.' : runs.dbStatus?.reason || '-'}</strong>
          </div>
        </div>
        <button onClick={runs.refreshDbStatus} disabled={runs.loading}>
          {runs.loading ? '⏳ Sorgulanıyor...' : '🔄 Bağlantıyı Test Et'}
        </button>
      </section>

      <section className="card">
        <div className="small">Maliyet Matrisi — Karar Parametreleri</div>
        <div className="explain">Bu değerler modelin "hangi müşteriye müdahale etmeli?" kararını şekillendirir.</div>
        <div className="tableWrap">
          <table>
            <thead><tr><th>Senaryo</th><th>Kısaltma</th><th>Değer</th><th>Açıklama</th></tr></thead>
            <tbody>
              <tr><td>Doğru Pozitif</td><td style={{ fontFamily: 'Consolas' }}>TP</td><td style={{ color: '#006600', fontWeight: 700 }}>+180 ₺</td><td>İptal edecek müşteriyi doğru tahmin ettik ve kurtardık</td></tr>
              <tr><td>Yanlış Pozitif</td><td style={{ fontFamily: 'Consolas' }}>FP</td><td style={{ color: '#cc0000', fontWeight: 700 }}>−20 ₺</td><td>İptal etmeyecek müşteriye gereksiz müdahale</td></tr>
              <tr><td>Yanlış Negatif</td><td style={{ fontFamily: 'Consolas' }}>FN</td><td style={{ color: '#cc0000', fontWeight: 700 }}>−200 ₺</td><td>İptal edecek müşteriyi kaçırdık</td></tr>
              <tr><td>Doğru Negatif</td><td style={{ fontFamily: 'Consolas' }}>TN</td><td style={{ color: '#666' }}>0 ₺</td><td>İptal etmeyecek müşteriyi doğru tahmin ettik</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <div className="small">Genel Bilgiler</div>
        <div className="systemGrid">
          <div className="sysItem"><span>Toplam Koşu Sayısı</span><strong>{runs.runs.length}</strong></div>
          <div className="sysItem"><span>DB Kayıtlı Koşu</span><strong>{runs.dbRuns.length}</strong></div>
          <div className="sysItem"><span>Aktif Run ID</span><strong>{runs.selectedRun || '-'}</strong></div>
          <div className="sysItem"><span>Güncel Şampiyon</span><strong>{displayName(runs.champion.selected_model)}</strong></div>
        </div>
      </section>
    </>
  );
}
