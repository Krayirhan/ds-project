import { useLayoutContext } from './Layout';
import { displayName } from '../lib/helpers';

/**
 * PipelinePage — Veri İşleme Hattı (Pipeline) Dokümantasyon Sayfası
 *
 * Büyük ölçüde statik içerik — ham veriden tahmine kadar tüm adımları açıklar.
 */
export default function PipelinePage() {
  const { runs } = useLayoutContext();
  const { coreModels } = runs;

  return (
    <>
      <header className="pageHeader">
        <h1>🔧 Veri İşleme Hattı (Pipeline)</h1>
        <p className="subtitle">
          Ham veriden tahmine kadar tüm adımlar. Her model aşağıdaki önişleme, özellik dönüşümü ve eğitim sürecinden geçer.
        </p>
      </header>

      {/* Pipeline Akış Şeması */}
      <section className="card">
        <div className="small">📐 Uçtan Uca Pipeline Akışı</div>
        <div className="explain">Her kutu bir DVC aşamasını temsil eder. Veriler soldan sağa doğru akar.</div>
        <div className="pipelineFlow">
          <div className="pipeStep raw"><div className="pipeStepIcon">📄</div><div className="pipeStepTitle">Ham Veri</div><div className="pipeStepDesc">hotel_bookings.csv<br />Orijinal 32+ sütun</div></div>
          <div className="pipeArrow">→</div>
          <div className="pipeStep validate"><div className="pipeStepIcon">✅</div><div className="pipeStepTitle">Doğrulama</div><div className="pipeStepDesc">5 katman / 30+ kural<br />Pandera + temel şema</div></div>
          <div className="pipeArrow">→</div>
          <div className="pipeStep preprocess"><div className="pipeStepIcon">🔧</div><div className="pipeStepTitle">Önişleme</div><div className="pipeStepDesc">Sızıntı temizliği<br />Eksik veri doldurma</div></div>
          <div className="pipeArrow">→</div>
          <div className="pipeStep split"><div className="pipeStepIcon">✂️</div><div className="pipeStepTitle">Veri Bölme</div><div className="pipeStepDesc">%64 eğitim / %16 kalibrasyon<br />%20 test</div></div>
          <div className="pipeArrow">→</div>
          <div className="pipeStep train"><div className="pipeStepIcon">🧠</div><div className="pipeStepTitle">Eğitim</div><div className="pipeStepDesc">Feature transform<br />Model uydurma + CV</div></div>
          <div className="pipeArrow">→</div>
          <div className="pipeStep calibrate"><div className="pipeStepIcon">⚖️</div><div className="pipeStepTitle">Kalibrasyon</div><div className="pipeStepDesc">Sigmoid / İzotonik<br />Olasılık düzeltme</div></div>
          <div className="pipeArrow">→</div>
          <div className="pipeStep evaluate"><div className="pipeStepIcon">📊</div><div className="pipeStepTitle">Değerlendirme</div><div className="pipeStepDesc">Eşik taraması<br />Kâr optimizasyonu</div></div>
          <div className="pipeArrow">→</div>
          <div className="pipeStep champion"><div className="pipeStepIcon">🏆</div><div className="pipeStepTitle">Şampiyon Seçim</div><div className="pipeStepDesc">Tercih sırası<br />Karar politikası</div></div>
        </div>
      </section>

      {/* Adım 1: Doğrulama */}
      <section className="card">
        <div className="small">1️⃣ Veri Doğrulama — 5 Katmanlı Savunma</div>
        <div className="tableWrap">
          <table>
            <thead><tr><th style={{ width: 28 }}>#</th><th>Katman</th><th>Ne Zaman?</th><th>Kaynak</th><th>Kontroller</th><th>Durum</th></tr></thead>
            <tbody>
              <tr><td style={{ textAlign: 'center', fontWeight: 'bold', color: '#b8860b' }}>1</td><td><strong>Temel Şema</strong></td><td>Önişleme başında</td><td><code>validate.py</code></td><td>Boş veri · Hedef sütun varlığı · Yinelenen sütun · Etiket kümesi</td><td style={{ color: 'green' }}>✅ Aktif</td></tr>
              <tr><td style={{ textAlign: 'center', fontWeight: 'bold', color: '#0055aa' }}>2</td><td><strong>Pandera Ham Veri</strong></td><td>Önişleme başında</td><td><code>data_validation.py</code></td><td>17 sütun tip kontrolü · Sayısal aralık · Kategori kümesi</td><td style={{ color: 'green' }}>✅ Aktif</td></tr>
              <tr><td style={{ textAlign: 'center', fontWeight: 'bold', color: '#880088' }}>3</td><td><strong>İşlenmiş Veri</strong></td><td>Eğitim öncesi</td><td><code>data_validation.py</code></td><td>Hedef 0/1 tamsayı · NaN/Inf yok · İmpütasyon sonrası kontrol</td><td style={{ color: 'green' }}>✅ Aktif</td></tr>
              <tr><td style={{ textAlign: 'center', fontWeight: 'bold', color: '#cc3300' }}>4</td><td><strong>Inference Payload</strong></td><td>Her API isteğinde</td><td><code>predict.py</code></td><td>Eksik/fazla sütun · Tip zorlaması · Pandera şema</td><td style={{ color: 'green' }}>✅ Aktif</td></tr>
              <tr><td style={{ textAlign: 'center', fontWeight: 'bold', color: '#006644' }}>5</td><td><strong>Dağılım İzleme</strong></td><td>Monitor CLI</td><td><code>data_validation.py</code></td><td>Referans ortalama/std · Aralık dışı değer · Unseen category</td><td style={{ color: 'green' }}>✅ Aktif</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Adım 2: Önişleme */}
      <section className="card">
        <div className="small">2️⃣ Önişleme Adımları (Preprocessing)</div>
        <div className="pipelineStepList">
          <div className="stepItem"><div className="stepNum">A</div><div className="stepBody"><strong>Sütun İsmi Temizleme</strong><span>Başta/sonda boşluklar kaldırılır.</span></div></div>
          <div className="stepItem"><div className="stepNum">B</div><div className="stepBody"><strong>Veri Sızıntısı Temizliği</strong><span>reservation_status ve reservation_status_date çıkarılır.</span></div></div>
          <div className="stepItem"><div className="stepNum">C</div><div className="stepBody"><strong>Hedef Etiket Dönüşümü</strong><span>"yes" → 1, "no" → 0</span></div></div>
          <div className="stepItem"><div className="stepNum">D</div><div className="stepBody"><strong>Tamamen Boş Sütunların Kaldırılması</strong><span>%100 NaN içeren sütunlar çıkarılır.</span></div></div>
          <div className="stepItem"><div className="stepNum">E</div><div className="stepBody"><strong>Temel Eksik Veri Doldurma</strong><span>Sayısal → medyan, Kategorik → mod</span></div></div>
        </div>
        <div className="stepOutput"><strong>Çıktı:</strong> <code>data/processed/dataset.parquet</code></div>
      </section>

      {/* Adım 3: Veri Bölme */}
      <section className="card">
        <div className="small">3️⃣ Veri Bölme Stratejisi (Train / Calibration / Test)</div>
        <div className="splitDiagram">
          <div className="splitBlock full">
            <div className="splitLabel">Tüm Veri (%100)</div>
            <div className="splitChildren">
              <div className="splitBlock train-full">
                <div className="splitLabel">Eğitim Havuzu (%80)</div>
                <div className="splitChildren">
                  <div className="splitBlock train"><div className="splitLabel">Eğitim<br />(%64)</div><div className="splitDesc">Model uydurma<br />CV doğrulama</div></div>
                  <div className="splitBlock cal"><div className="splitLabel">Kalibrasyon<br />(%16)</div><div className="splitDesc">Olasılık<br />düzeltme</div></div>
                </div>
              </div>
              <div className="splitBlock test"><div className="splitLabel">Test<br />(%20)</div><div className="splitDesc">Nihai<br />değerlendirme</div></div>
            </div>
          </div>
        </div>
      </section>

      {/* Adım 4: Feature Engineering */}
      <section className="card">
        <div className="small">4️⃣ Özellik Çıkarımı ve Dönüşüm (Feature Engineering)</div>
        <div className="explain">features.py — Sklearn ColumnTransformer ile pipeline içinde uygulanır.</div>
        <div className="grid2" style={{ margin: 0, gap: 2 }}>
          <div className="card" style={{ margin: 0 }}>
            <div className="small">Sayısal Özellikler — 19 sütun</div>
            <div className="featurePipeline">
              <div className="fpStep">SimpleImputer(strategy='median')</div>
              <div className="fpArrow">↓</div>
              <div className="fpStep">StandardScaler (z-score)</div>
            </div>
          </div>
          <div className="card" style={{ margin: 0 }}>
            <div className="small">Kategorik Özellikler — 10 sütun</div>
            <div className="featurePipeline">
              <div className="fpStep">SimpleImputer(strategy='most_frequent')</div>
              <div className="fpArrow">↓</div>
              <div className="fpStep">OneHotEncoder(handle_unknown='ignore')</div>
            </div>
          </div>
        </div>
      </section>

      {/* Adım 5: Model Eğitimi */}
      <section className="card">
        <div className="small">5️⃣ Model Eğitimi (Training)</div>
        <div className="grid2" style={{ margin: 0, gap: 2 }}>
          <div className="card" style={{ margin: 0, borderColor: '#88aacc' }}>
            <div className="small">🔵 Temel Model — Lojistik Regresyon</div>
            <div className="tableWrap">
              <table><thead><tr><th>Parametre</th><th>Değer</th></tr></thead>
                <tbody>
                  <tr><td>Algoritma</td><td>LogisticRegression</td></tr>
                  <tr><td>max_iter</td><td>3000</td></tr>
                  <tr><td>solver</td><td>lbfgs</td></tr>
                  <tr><td>random_state</td><td>42</td></tr>
                </tbody>
              </table>
            </div>
          </div>
          <div className="card" style={{ margin: 0, borderColor: '#cc9944' }}>
            <div className="small">🟠 Gelişmiş Model — XGBoost</div>
            <div className="tableWrap">
              <table><thead><tr><th>Parametre</th><th>Değer</th></tr></thead>
                <tbody>
                  <tr><td>n_estimators</td><td>500</td></tr>
                  <tr><td>learning_rate</td><td>0.05</td></tr>
                  <tr><td>max_depth</td><td>6</td></tr>
                  <tr><td>subsample</td><td>0.9</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* Adım 6: Kalibrasyon */}
      <section className="card">
        <div className="small">6️⃣ Olasılık Kalibrasyonu</div>
        <div className="grid2" style={{ margin: 0, gap: 2 }}>
          <div className="card" style={{ margin: 0 }}>
            <div className="small">Sigmoid (Platt Scaling)</div>
            <div className="pipelineStepList">
              <div className="stepItem compact"><div className="stepBody"><strong>Yöntem:</strong> <span>Lojistik regresyon uydurma</span></div></div>
              <div className="stepItem compact"><div className="stepBody"><strong>Avantaj:</strong> <span>Küçük setlerde kararlı</span></div></div>
            </div>
          </div>
          <div className="card" style={{ margin: 0 }}>
            <div className="small">İzotonik Regresyon</div>
            <div className="pipelineStepList">
              <div className="stepItem compact"><div className="stepBody"><strong>Yöntem:</strong> <span>Parametrik olmayan monoton regresyon</span></div></div>
              <div className="stepItem compact"><div className="stepBody"><strong>Avantaj:</strong> <span>Büyük setlerde esnek</span></div></div>
            </div>
          </div>
        </div>
      </section>

      {/* Adım 7: Değerlendirme */}
      <section className="card">
        <div className="small">7️⃣ Değerlendirme ve Eşik Optimizasyonu</div>
        <div className="pipelineStepList">
          <div className="stepItem"><div className="stepNum">I</div><div className="stepBody"><strong>Temel Metrik Hesaplama</strong><span>ROC-AUC, F1, Precision, Recall, Confusion Matrix</span></div></div>
          <div className="stepItem"><div className="stepNum">II</div><div className="stepBody"><strong>Eşik Taraması</strong><span>0.001–0.999 arasında 999 eşik taranır</span></div></div>
          <div className="stepItem"><div className="stepNum">III</div><div className="stepBody"><strong>Kapasite Kısıtlı Optimizasyon</strong><span>%5, %10, %15, %20, %30 aksiyon oranları</span></div></div>
        </div>
      </section>

      {/* Adım 8: Şampiyon Seçimi */}
      <section className="card">
        <div className="small">8️⃣ Şampiyon Model Seçimi</div>
        <div className="tableWrap">
          <table>
            <thead><tr><th>Sıra</th><th>Model Adayı</th><th>Açıklama</th></tr></thead>
            <tbody>
              <tr><td style={{ textAlign: 'center' }}>1</td><td>{displayName('challenger_xgboost_calibrated_sigmoid')}</td><td>En kararlı kalibrasyon + en güçlü model</td></tr>
              <tr><td style={{ textAlign: 'center' }}>2</td><td>LightGBM + Sigmoid</td><td>Yedek GBM</td></tr>
              <tr><td style={{ textAlign: 'center' }}>3</td><td>CatBoost + Sigmoid</td><td>Üçüncü alternatif</td></tr>
              <tr><td style={{ textAlign: 'center' }}>4</td><td>HistGradientBoosting + Sigmoid</td><td>Sklearn yerleşik</td></tr>
              <tr><td style={{ textAlign: 'center' }}>5</td><td>{displayName('baseline_calibrated_sigmoid')}</td><td>Temel model kalibre versiyonu</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Dosya Haritası */}
      <section className="card">
        <div className="small">📂 Pipeline Kaynak Dosya Haritası</div>
        <div className="tableWrap">
          <table>
            <thead><tr><th>Aşama</th><th>Dosya</th><th>Giriş</th><th>Çıkış</th></tr></thead>
            <tbody>
              <tr><td>Doğrulama</td><td>src/data_validation.py</td><td>hotel_bookings.csv</td><td>Doğrulanmış DataFrame</td></tr>
              <tr><td>Önişleme</td><td>src/preprocess.py</td><td>hotel_bookings.csv</td><td>data/processed/dataset.parquet</td></tr>
              <tr><td>Veri Bölme</td><td>src/split.py</td><td>dataset.parquet</td><td>train/cal/test.parquet</td></tr>
              <tr><td>Feature Eng.</td><td>src/features.py</td><td>train.parquet</td><td>ColumnTransformer</td></tr>
              <tr><td>Eğitim</td><td>src/train.py</td><td>train + cal</td><td>models/*.joblib</td></tr>
              <tr><td>Kalibrasyon</td><td>src/calibration.py</td><td>cal + ham model</td><td>*_calibrated_*.joblib</td></tr>
              <tr><td>Değerlendirme</td><td>src/evaluate.py</td><td>test + modeller</td><td>reports/metrics/*.json</td></tr>
              <tr><td>Politika</td><td>src/policy.py</td><td>Metrikler</td><td>decision_policy.json</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
