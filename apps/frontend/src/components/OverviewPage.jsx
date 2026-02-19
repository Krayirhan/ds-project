import { useMemo, useState, useEffect, useRef } from 'react';
import { Chart } from 'chart.js';
import { useLayoutContext } from './Layout';
import {
  f, pct, money, scoreColor, displayName,
  modelBadge, modelIcon, modelCalibration,
} from '../lib/helpers';

function ScoreBar({ score, max = 1 }) {
  if (score == null) return null;
  const pctVal = Math.min(100, (Number(score) / max) * 100);
  const color = scoreColor(score);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <div style={{ width: 60, height: 8, background: '#e0e0e0', border: '1px solid #b0b0b0' }}>
        <div style={{ width: `${pctVal}%`, height: '100%', background: color }} />
      </div>
      <span style={{ fontFamily: 'Consolas', fontSize: 11, color }}>{f(score)}</span>
    </div>
  );
}

export default function OverviewPage() {
  const { runs, theme } = useLayoutContext();
  const { coreModels, champion } = runs;

  const [sortCol, setSortCol] = useState('test_roc_auc');
  const [sortDir, setSortDir] = useState('desc');

  const aucRef = useRef(null);
  const prfRef = useRef(null);
  const chartRefs = useRef({ auc: null, prf: null });

  // Sıralama
  const sortedModels = useMemo(() => {
    const arr = [...coreModels];
    arr.sort((a, b) => {
      const va = a[sortCol] ?? -999;
      const vb = b[sortCol] ?? -999;
      return sortDir === 'desc' ? vb - va : va - vb;
    });
    return arr;
  }, [coreModels, sortCol, sortDir]);

  function toggleSort(col) {
    if (sortCol === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortCol(col); setSortDir('desc'); }
  }
  function sortIndicator(col) {
    if (sortCol !== col) return ' ⇅';
    return sortDir === 'desc' ? ' ▼' : ' ▲';
  }

  // En iyi skorlar
  const bestScores = useMemo(() => {
    if (!coreModels.length) return {};
    const fields = ['test_roc_auc', 'test_f1', 'test_precision', 'test_recall'];
    const result = {};
    fields.forEach(key => { result[key] = Math.max(...coreModels.map(m => m[key] ?? 0)); });
    return result;
  }, [coreModels]);

  const championModel = useMemo(
    () => coreModels.find(m => m.model_name === champion.selected_model) || null,
    [coreModels, champion],
  );

  // Grafik verileri
  const chartDataset = useMemo(() => {
    const labels = coreModels.map(m => displayName(m.model_name));
    return {
      labels,
      trainAuc: coreModels.map(m => m.train_cv_roc_auc_mean ?? null),
      testAuc: coreModels.map(m => m.test_roc_auc ?? null),
      testF1: coreModels.map(m => m.test_f1 ?? null),
      testPrecision: coreModels.map(m => m.test_precision ?? null),
      testRecall: coreModels.map(m => m.test_recall ?? null),
    };
  }, [coreModels]);

  // Chart.js effect — useRef ile
  useEffect(() => {
    if (!chartDataset.labels.length || !aucRef.current || !prfRef.current) return;

    chartRefs.current.auc?.destroy();
    chartRefs.current.prf?.destroy();

    const _isM  = theme.isModern;
    const _isDk = theme.isDark;
    const gridColor = _isM ? (_isDk ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)') : '#c0c0c0';
    const tickColor = _isM ? (_isDk ? '#cbd5e1' : '#4a5568') : undefined;

    chartRefs.current.auc = new Chart(aucRef.current, {
      type: 'bar',
      data: {
        labels: chartDataset.labels,
        datasets: [
          { label: 'Eğitim ROC-AUC (CV Ort.)', data: chartDataset.trainAuc, backgroundColor: _isM ? '#1a56db' : '#4472c4', borderColor: _isM ? '#1648b8' : '#2f5496', borderWidth: 1, borderRadius: _isM ? 4 : 0 },
          { label: 'Test ROC-AUC', data: chartDataset.testAuc, backgroundColor: _isM ? '#0d9488' : '#ed7d31', borderColor: _isM ? '#0f766e' : '#c65911', borderWidth: 1, borderRadius: _isM ? 4 : 0 },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: _isM ? 11 : 10 }, color: tickColor } } },
        scales: {
          y: { min: 0.5, max: 1, grid: { color: gridColor }, ticks: { font: { size: _isM ? 11 : 10 }, color: tickColor } },
          x: { grid: { color: gridColor }, ticks: { font: { size: _isM ? 10 : 9 }, maxRotation: 25, color: tickColor } },
        },
      },
    });

    chartRefs.current.prf = new Chart(prfRef.current, {
      type: 'bar',
      data: {
        labels: chartDataset.labels,
        datasets: [
          { label: 'Precision', data: chartDataset.testPrecision, backgroundColor: _isM ? '#1a56db' : '#4472c4', borderWidth: 1, borderRadius: _isM ? 4 : 0 },
          { label: 'Recall', data: chartDataset.testRecall, backgroundColor: _isM ? '#d97706' : '#ed7d31', borderWidth: 1, borderRadius: _isM ? 4 : 0 },
          { label: 'F1 Skoru', data: chartDataset.testF1, backgroundColor: _isM ? '#0d9488' : '#70ad47', borderWidth: 1, borderRadius: _isM ? 4 : 0 },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: _isM ? 11 : 10 }, color: tickColor } } },
        scales: {
          y: { min: 0, max: 1, grid: { color: gridColor }, ticks: { font: { size: _isM ? 11 : 10 }, color: tickColor } },
          x: { grid: { color: gridColor }, ticks: { font: { size: _isM ? 10 : 9 }, maxRotation: 25, color: tickColor } },
        },
      },
    });

    return () => {
      chartRefs.current.auc?.destroy();
      chartRefs.current.prf?.destroy();
    };
  }, [chartDataset, theme.theme]);

  return (
    <>
      <header className="pageHeader">
        <h1>📊 Yönetici Özeti</h1>
        <p className="subtitle">
          Bu koşuda {coreModels.length} farklı model eğitildi ve değerlendirildi.
          Sistem, <strong>"{displayName(champion.selected_model)}"</strong> modelini en kârlı olarak seçti.
        </p>
      </header>

      {/* Şampiyon Model Kartı */}
      {championModel && (
        <section className="championCard card">
          <div className="small">🏆 Seçilen Model: {displayName(champion.selected_model)}</div>
          <div className="championGrid">
            <div className="champItem">
              <span className="champLabel">Neden Bu Model?</span>
              <span className="champValue" style={{ fontSize: 11, lineHeight: 1.4 }}>
                {champion.ranking_mode === 'incremental_profit'
                  ? `Artışsal kâr hesaplamasına göre ${pct(champion.max_action_rate)} kapasite kısıtı altında en yüksek net kazancı bu model sağlıyor.`
                  : `${champion.ranking_mode} kriterine göre en başarılı model.`}
              </span>
            </div>
            <div className="champItem">
              <span className="champLabel">Beklenen Net Kazanç</span>
              <span className="champValue money">{money(champion.expected_net_profit)} ₺</span>
            </div>
            <div className="champItem">
              <span className="champLabel">Karar Eşiği</span>
              <span className="champValue">{f(champion.threshold, 3)}</span>
              <span className="champHint">Bu değerin üstündeki tahminler "müdahale et" olarak işaretlenir</span>
            </div>
            <div className="champItem">
              <span className="champLabel">Kapasite Limiti</span>
              <span className="champValue">{pct(champion.max_action_rate)}</span>
            </div>
            <div className="champItem">
              <span className="champLabel">Test AUC</span>
              <span className="champValue">{f(championModel.test_roc_auc)}</span>
            </div>
            <div className="champItem">
              <span className="champLabel">Test Seti Büyüklüğü</span>
              <span className="champValue">{championModel.n_test?.toLocaleString('tr-TR') || '-'} kayıt</span>
              <span className="champHint">İptal oranı: {pct(championModel.positive_rate_test)}</span>
            </div>
          </div>
        </section>
      )}

      {/* Durum Çubuğu */}
      <section className="statusBar card">
        <div className="statusItem">
          <span className="statusLabel">Sistem Durumu</span>
          <span className="statusBadge ok">● Çalışıyor</span>
        </div>
        <div className="statusItem">
          <span className="statusLabel">Seçim Kriteri</span>
          <span className="statusBadge neutral">{champion.ranking_mode === 'incremental_profit' ? 'Artışsal Kâr' : champion.ranking_mode || '-'}</span>
        </div>
        <div className="statusItem">
          <span className="statusLabel">Değerlendirilen Model</span>
          <span className="statusBadge neutral">{coreModels.length} adet</span>
        </div>
      </section>

      {/* Model Kıyaslama Tablosu */}
      <section className="card">
        <div className="small">Model Performans Kıyaslaması</div>
        <div className="explain">Tüm modeller aynı test seti üzerinde değerlendirildi. En yüksek skorlar yeşil renkle vurgulanır.</div>
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th style={{ width: 24 }}></th>
                <th>Model</th>
                <th>Tür</th>
                <th>Kalibrasyon</th>
                <th onClick={() => toggleSort('test_roc_auc')} style={{ cursor: 'pointer' }} role="button" tabIndex={0} aria-label="Test AUC sırala">Test AUC{sortIndicator('test_roc_auc')}</th>
                <th onClick={() => toggleSort('test_f1')} style={{ cursor: 'pointer' }} role="button" tabIndex={0}>F1{sortIndicator('test_f1')}</th>
                <th onClick={() => toggleSort('test_precision')} style={{ cursor: 'pointer' }} role="button" tabIndex={0}>Precision{sortIndicator('test_precision')}</th>
                <th onClick={() => toggleSort('test_recall')} style={{ cursor: 'pointer' }} role="button" tabIndex={0}>Recall{sortIndicator('test_recall')}</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.map(m => {
                const isChamp = m.model_name === champion.selected_model;
                return (
                  <tr key={m.model_name} style={isChamp ? { background: '#fffff0', fontWeight: 600 } : {}}>
                    <td style={{ textAlign: 'center' }} aria-label={isChamp ? 'Şampiyon' : ''}>{isChamp ? '★' : <span aria-hidden="true">{modelIcon(m.model_name)}</span>}</td>
                    <td><strong>{displayName(m.model_name)}</strong></td>
                    <td><span className={`typeBadge ${modelBadge(m.model_name) === 'Gelişmiş' ? 'advanced' : 'base'}`}>{modelBadge(m.model_name)}</span></td>
                    <td>{modelCalibration(m.model_name)}</td>
                    <td style={{ color: m.test_roc_auc === bestScores.test_roc_auc ? '#006600' : undefined, fontWeight: m.test_roc_auc === bestScores.test_roc_auc ? 700 : 400 }}><ScoreBar score={m.test_roc_auc} /></td>
                    <td style={{ color: m.test_f1 === bestScores.test_f1 ? '#006600' : undefined, fontWeight: m.test_f1 === bestScores.test_f1 ? 700 : 400 }}><ScoreBar score={m.test_f1} /></td>
                    <td><ScoreBar score={m.test_precision} /></td>
                    <td><ScoreBar score={m.test_recall} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Grafikler */}
      <section className="grid2">
        <div className="card">
          <div className="small">Eğitim vs. Test Başarısı (ROC-AUC)</div>
          <div className="explain">Eğitim ve test skorlarının yakın olması aşırı öğrenme olmadığını gösterir.</div>
          <canvas ref={aucRef} height="160" />
        </div>
        <div className="card">
          <div className="small">Test Metrikleri Karşılaştırması</div>
          <div className="explain">Precision: doğruluk, Recall: kapsayıcılık, F1: ikisinin dengeli özeti.</div>
          <canvas ref={prfRef} height="160" />
        </div>
      </section>
    </>
  );
}
