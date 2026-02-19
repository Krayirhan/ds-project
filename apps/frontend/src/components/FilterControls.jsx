import { formatRunId } from '../lib/helpers';

/**
 * FilterControls — Koşu seçimi, API anahtarı ve yenile butonu
 */
export default function FilterControls({ runs }) {
  return (
    <section className="card controls">
      <div className="controlTitle">Filtreler</div>
      <div>
        <label htmlFor="run-select">Koşu Seçimi:</label>
        <select
          id="run-select"
          value={runs.selectedRun}
          onChange={e => {
            runs.setSelectedRun(e.target.value);
            runs.refreshOverviewOnly(e.target.value);
          }}
        >
          {runs.runs.map(r => (
            <option key={r} value={r}>{formatRunId(r)}</option>
          ))}
        </select>
      </div>
      <div>
        <label htmlFor="api-key-input">API Anahtarı (opsiyonel):</label>
        <input
          id="api-key-input"
          value={runs.apiKey}
          onChange={e => runs.setApiKey(e.target.value)}
          placeholder="x-api-key"
        />
      </div>
      <button onClick={runs.refreshRunsAndData} disabled={runs.loading}>
        {runs.loading ? '⏳ Yükleniyor...' : '🔄 Verileri Yenile'}
      </button>
    </section>
  );
}
