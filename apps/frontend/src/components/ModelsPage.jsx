import { useState, useMemo } from 'react';
import { useLayoutContext } from './Layout';
import {
  f, pct, scoreColor, displayName,
  modelBadge, modelIcon, modelCalibration, modelType,
} from '../lib/helpers';

function ScoreBar({ score }) {
  if (score == null) return null;
  const pctVal = Math.min(100, Number(score) * 100);
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

export default function ModelsPage() {
  const { runs } = useLayoutContext();
  const { modelRows, champion, coreModels } = runs;
  const [selectedModelIdx, setSelectedModelIdx] = useState(null);

  const selectedModel = selectedModelIdx !== null ? modelRows[selectedModelIdx] : null;
  const isSelectedChamp = selectedModel?.model_name === champion.selected_model;

  return (
    <>
      <header className="pageHeader">
        <h1>📋 Model Karşılaştırma — Detaylı Analiz</h1>
        <p className="subtitle">
          Her modelin eğitim kararlılığı, test performansı ve kalibrasyon bilgisi.
          Satıra tıklayarak detay görebilirsiniz.
          <strong> "{displayName(champion.selected_model)}"</strong> şampiyon olarak seçildi.
        </p>
      </header>

      {/* Ana Tablo */}
      <section className="card">
        <div className="small">Tüm Modeller — {modelRows.length} varyant ({coreModels.length} temel + {modelRows.length - coreModels.length} karar eşiği versiyonu)</div>
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th style={{ width: 20 }}>#</th>
                <th>Model</th>
                <th>Kategori</th>
                <th>Kalibrasyon</th>
                <th>Eğitim AUC (CV ± Std)</th>
                <th>CV Katlanma</th>
                <th>Test AUC</th>
                <th>F1</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Eşik</th>
                <th>Test N</th>
                <th>İptal Oranı</th>
              </tr>
            </thead>
            <tbody>
              {modelRows.map((m, i) => {
                const isChamp = m.model_name === champion.selected_model;
                const isSelected = selectedModelIdx === i;
                return (
                  <tr
                    key={m.model_name}
                    className={isSelected ? 'selected' : ''}
                    style={{
                      cursor: 'pointer',
                      background: isChamp && !isSelected ? '#fffff0' : undefined,
                      fontWeight: isChamp ? 600 : 400,
                    }}
                    onClick={() => setSelectedModelIdx(i)}
                    tabIndex={0}
                    onKeyDown={e => e.key === 'Enter' && setSelectedModelIdx(i)}
                    role="button"
                    aria-label={`${displayName(m.model_name)} detaylarını göster`}
                  >
                    <td style={{ textAlign: 'center' }}>{isChamp ? '★' : i + 1}</td>
                    <td><span aria-hidden="true">{modelIcon(m.model_name)}</span> <strong>{displayName(m.model_name)}</strong></td>
                    <td><span className={`typeBadge ${modelBadge(m.model_name) === 'Gelişmiş' ? 'advanced' : 'base'}`}>{modelBadge(m.model_name)}</span></td>
                    <td>{modelCalibration(m.model_name)}</td>
                    <td>{f(m.train_cv_roc_auc_mean)} ± {f(m.train_cv_roc_auc_std)}</td>
                    <td style={{ textAlign: 'center' }}>{m.cv_folds ?? '-'}</td>
                    <td><ScoreBar score={m.test_roc_auc} /></td>
                    <td><ScoreBar score={m.test_f1} /></td>
                    <td><ScoreBar score={m.test_precision} /></td>
                    <td><ScoreBar score={m.test_recall} /></td>
                    <td style={{ fontFamily: 'Consolas' }}>{f(m.test_threshold, 3)}</td>
                    <td style={{ textAlign: 'right' }}>{m.n_test?.toLocaleString('tr-TR') || '-'}</td>
                    <td>{pct(m.positive_rate_test)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Seçili Model Detay Paneli */}
      {selectedModel && (
        <section className="card detailPanel">
          <div className="small">
            <span aria-hidden="true">{modelIcon(selectedModel.model_name)}</span> {displayName(selectedModel.model_name)} — Detay Bilgisi
            {isSelectedChamp && <span style={{ marginLeft: 8, color: '#996600' }}>★ Şampiyon Model</span>}
          </div>
          <div className="detailGrid">
            <div className="detailItem"><span>Teknik Ad</span><strong style={{ fontSize: 10, wordBreak: 'break-all' }}>{selectedModel.model_name}</strong></div>
            <div className="detailItem"><span>Model Tipi</span><strong>{modelType(selectedModel.model_name)}</strong></div>
            <div className="detailItem"><span>Kalibrasyon</span><strong>{modelCalibration(selectedModel.model_name)}</strong></div>
            <div className="detailItem"><span>Eğitim AUC (Ort)</span><strong>{f(selectedModel.train_cv_roc_auc_mean)}</strong></div>
            <div className="detailItem"><span>Eğitim AUC (Std)</span><strong>{f(selectedModel.train_cv_roc_auc_std)}</strong></div>
            <div className="detailItem"><span>CV Katlanma</span><strong>{selectedModel.cv_folds ?? '-'}</strong></div>
            <div className="detailItem highlight"><span>Test ROC-AUC</span><strong style={{ color: scoreColor(selectedModel.test_roc_auc) }}>{f(selectedModel.test_roc_auc)}</strong></div>
            <div className="detailItem highlight"><span>F1 Skoru</span><strong style={{ color: scoreColor(selectedModel.test_f1) }}>{f(selectedModel.test_f1)}</strong></div>
            <div className="detailItem"><span>Precision</span><strong>{f(selectedModel.test_precision)}</strong></div>
            <div className="detailItem"><span>Recall</span><strong>{f(selectedModel.test_recall)}</strong></div>
            <div className="detailItem"><span>Karar Eşiği</span><strong>{f(selectedModel.test_threshold, 3)}</strong></div>
            <div className="detailItem"><span>Test Seti</span><strong>{selectedModel.n_test?.toLocaleString('tr-TR') || '-'} kayıt</strong></div>
            <div className="detailItem full">
              <span>Yorum</span>
              <strong style={{ fontSize: 11, fontWeight: 400 }}>
                {selectedModel.test_roc_auc > 0.93 ? 'Yüksek ayırt edicilik. Model, iptal edecek ve etmeyecek müşterileri çok iyi ayırt edebiliyor.'
                  : selectedModel.test_roc_auc > 0.85 ? 'İyi düzeyde ayırt edicilik. Pratikte kullanılabilir performans.'
                  : 'Düşük-orta ayırt edicilik. Daha güçlü modeller tercih edilmeli.'}
                {' '}
                {Math.abs((selectedModel.train_cv_roc_auc_mean || 0) - (selectedModel.test_roc_auc || 0)) < 0.02
                  ? 'Eğitim-test farkı çok düşük, aşırı öğrenme riski yok.'
                  : 'Eğitim ve test arasında fark var, dikkat edilmeli.'}
              </strong>
            </div>
          </div>
        </section>
      )}

      {/* Terim Açıklamaları */}
      <section className="card">
        <div className="legendBox">
          <strong>📖 Metrik Açıklamaları:</strong>
          <ul>
            <li><strong>ROC-AUC</strong>: Modelin iptal / iptal-değil ayrımındaki genel başarısı. 1.0 mükemmel, 0.5 rastgele tahmin.</li>
            <li><strong>Precision</strong>: "İptal edecek" dediğimiz müşterilerin gerçekten ne kadarı iptal etti?</li>
            <li><strong>Recall</strong>: Gerçekten iptal eden müşterilerin ne kadarını yakaladık?</li>
            <li><strong>F1</strong>: Precision ve Recall'un harmonik ortalaması.</li>
            <li><strong>CV (Çapraz Doğrulama)</strong>: Eğitim verisini {coreModels[0]?.cv_folds || 5} parçaya bölerek her parçada ayrı test yapma.</li>
            <li><strong>Kalibrasyon</strong>: Modelin olasılık çıktısını gerçek oranlarla uyumlu hale getiren işlem.</li>
            <li><strong>Eşik</strong>: Bu değerin üstündeki tahminler "iptal riski var, müdahale et" olarak işaretlenir.</li>
          </ul>
        </div>
      </section>
    </>
  );
}
