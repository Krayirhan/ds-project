import { useNavigate } from 'react-router-dom';
import { useLayoutContext } from './Layout';
import {
  f, pct, money, formatRunId, displayName, modelIcon,
} from '../lib/helpers';

/**
 * RunsPage — Koşu Geçmişi
 */
export default function RunsPage() {
  const { runs } = useLayoutContext();
  const navigate = useNavigate();

  function handleRunClick(runId) {
    runs.setSelectedRun(runId);
    runs.refreshOverviewOnly(runId);
    navigate('/');
  }

  return (
    <>
      <header className="pageHeader">
        <h1>📁 Koşu Geçmişi</h1>
        <p className="subtitle">
          Toplam {runs.runs.length} koşu kaydı bulunuyor.
          Bir koşuya tıklayarak "Genel Bakış" sayfasında detaylarını inceleyebilirsiniz.
        </p>
      </header>

      <section className="card">
        <div className="small">Koşu Kayıtları ({runs.runs.length} adet)</div>
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th style={{ width: 30 }}>#</th>
                <th>Tarih / Saat</th>
                <th>Run ID</th>
                <th>Seçilen Model</th>
                <th>Eşik</th>
                <th>Net Kazanç</th>
                <th>Kapasite</th>
                <th>Durum</th>
              </tr>
            </thead>
            <tbody>
              {runs.runs.map((r, i) => {
                const dbInfo = runs.dbRuns.find(d => d.run_id === r);
                const isCurrent = r === runs.selectedRun;
                return (
                  <tr
                    key={r}
                    style={{
                      cursor: 'pointer',
                      background: isCurrent ? 'var(--c-accent-bg, #e0f0ff)' : undefined,
                      fontWeight: isCurrent ? 600 : 400,
                    }}
                    onClick={() => handleRunClick(r)}
                    tabIndex={0}
                    onKeyDown={e => e.key === 'Enter' && handleRunClick(r)}
                    role="button"
                    aria-label={`Koşu ${formatRunId(r)} detaylarını göster`}
                  >
                    <td style={{ textAlign: 'center' }}>{i + 1}</td>
                    <td>{formatRunId(r)}</td>
                    <td style={{ fontFamily: 'Consolas', fontSize: 10 }}>{r}</td>
                    <td>
                      {dbInfo?.selected_model
                        ? <><span aria-hidden="true">{modelIcon(dbInfo.selected_model)}</span> {displayName(dbInfo.selected_model)}</>
                        : <span style={{ color: 'var(--c-text-muted, #999)' }}>—</span>}
                    </td>
                    <td style={{ fontFamily: 'Consolas' }}>{dbInfo?.threshold != null ? f(dbInfo.threshold, 3) : '—'}</td>
                    <td style={{ fontFamily: 'Consolas', textAlign: 'right' }}>{dbInfo?.expected_net_profit != null ? money(dbInfo.expected_net_profit) : '—'}</td>
                    <td>{dbInfo?.max_action_rate != null ? pct(dbInfo.max_action_rate) : '—'}</td>
                    <td>
                      {isCurrent
                        ? <span className="statusBadge ok" style={{ fontSize: 10 }}>◄ Görüntüleniyor</span>
                        : dbInfo?.selected_model
                          ? <span style={{ color: 'var(--c-success, #006600)', fontSize: 10 }}>✓ Tamamlandı</span>
                          : <span style={{ color: 'var(--c-text-muted, #999)', fontSize: 10 }}>Veri yok</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <div className="legendBox">
          <strong>💡 İpuçları:</strong>
          <ul>
            <li>Bir satıra tıkladığınızda o koşunun verileri "Genel Bakış" sayfasına yüklenir.</li>
            <li>"Net Kazanç" sütunu, modelin maliyet matrisine göre hesaplanan beklenen toplam faydadır.</li>
            <li>Koşu kimliği tarih_saat formatındadır: YYYYAAGG_SSddss</li>
          </ul>
        </div>
      </section>
    </>
  );
}
