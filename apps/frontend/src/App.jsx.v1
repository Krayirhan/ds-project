import { useEffect, useMemo, useState, useCallback } from 'react';
import {
  CategoryScale,
  Chart,
  BarController,
  BarElement,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Legend,
  Tooltip,
  Title,
} from 'chart.js';
import {
  getDbStatus,
  getOverview,
  getRuns,
  login,
  logout,
  me,
  startChatSession,
  sendChatMessage,
  getChatSummary,
} from './api';
import './modern.css';

Chart.register(
  CategoryScale, LinearScale, BarController, BarElement,
  LineController, LineElement, PointElement, Legend, Tooltip, Title,
);
Chart.defaults.font.family = 'Tahoma, "Segoe UI", sans-serif';
Chart.defaults.font.size = 10;

function applyChartTheme(themeVal) {
  const isModern = themeVal.startsWith('modern');
  const isDark = themeVal === 'modern-dark';
  if (isModern) {
    Chart.defaults.font.family = 'Inter, -apple-system, system-ui, sans-serif';
    Chart.defaults.font.size = 11;
    Chart.defaults.color = isDark ? '#cbd5e1' : '#4a5568';
    Chart.defaults.borderColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)';
  } else {
    Chart.defaults.font.family = 'Tahoma, "Segoe UI", sans-serif';
    Chart.defaults.font.size = 10;
    Chart.defaults.color = '#666';
    Chart.defaults.borderColor = 'rgba(0,0,0,0.1)';
  }
}

/* ================================================================
   MODEL İSİMLENDİRME — Teknik isimleri Türkçe anlaşılır hale çevir
   ================================================================ */
const MODEL_DISPLAY = {
  'baseline':                                       { short: 'Lojistik Regresyon',              badge: 'Temel',   type: 'Temel Model',      calibration: '—',         icon: '🔵' },
  'baseline_decision':                              { short: 'Lojistik Regresyon (Karar)',      badge: 'Temel',   type: 'Temel Model',      calibration: 'Karar Eşiği', icon: '🔵' },
  'baseline_calibrated_sigmoid':                    { short: 'Lojistik + Sigmoid Kalibrasyon',  badge: 'Temel',   type: 'Kalibre Model',    calibration: 'Sigmoid',   icon: '🟢' },
  'baseline_calibrated_sigmoid_decision':           { short: 'Lojistik + Sigmoid (Karar)',      badge: 'Temel',   type: 'Kalibre Model',    calibration: 'Sigmoid',   icon: '🟢' },
  'baseline_calibrated_isotonic':                   { short: 'Lojistik + İzotonik Kalibrasyon', badge: 'Temel',   type: 'Kalibre Model',    calibration: 'İzotonik',  icon: '🟢' },
  'baseline_calibrated_isotonic_decision':          { short: 'Lojistik + İzotonik (Karar)',     badge: 'Temel',   type: 'Kalibre Model',    calibration: 'İzotonik',  icon: '🟢' },
  'challenger_xgboost':                             { short: 'XGBoost',                         badge: 'Gelişmiş', type: 'Gelişmiş Model',  calibration: '—',         icon: '🟠' },
  'challenger_xgboost_decision':                    { short: 'XGBoost (Karar)',                 badge: 'Gelişmiş', type: 'Gelişmiş Model',  calibration: 'Karar Eşiği', icon: '🟠' },
  'challenger_xgboost_calibrated_sigmoid':          { short: 'XGBoost + Sigmoid Kalibrasyon',   badge: 'Gelişmiş', type: 'Kalibre Gelişmiş', calibration: 'Sigmoid', icon: '🟤' },
  'challenger_xgboost_calibrated_sigmoid_decision': { short: 'XGBoost + Sigmoid (Karar)',       badge: 'Gelişmiş', type: 'Kalibre Gelişmiş', calibration: 'Sigmoid', icon: '🟤' },
  'challenger_xgboost_calibrated_isotonic':         { short: 'XGBoost + İzotonik Kalibrasyon',  badge: 'Gelişmiş', type: 'Kalibre Gelişmiş', calibration: 'İzotonik', icon: '🟤' },
  'challenger_xgboost_calibrated_isotonic_decision':{ short: 'XGBoost + İzotonik (Karar)',      badge: 'Gelişmiş', type: 'Kalibre Gelişmiş', calibration: 'İzotonik', icon: '🟤' },
};

function displayName(raw) {
  return MODEL_DISPLAY[raw]?.short || raw;
}
function modelBadge(raw) {
  return MODEL_DISPLAY[raw]?.badge || '';
}
function modelIcon(raw) {
  return MODEL_DISPLAY[raw]?.icon || '⚪';
}
function modelCalibration(raw) {
  return MODEL_DISPLAY[raw]?.calibration || '—';
}
function modelType(raw) {
  return MODEL_DISPLAY[raw]?.type || 'Bilinmiyor';
}

/* ================================================================
   YARDIMCI FONKSİYONLAR
   ================================================================ */
function f(value, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(digits);
}
function pct(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return `%${(Number(value) * 100).toFixed(digits)}`;
}
function money(value) {
  if (value == null) return '-';
  return Number(value).toLocaleString('tr-TR', { maximumFractionDigits: 0 });
}
function formatRunId(runId) {
  if (!runId || runId.length < 15) return runId || '-';
  const d = runId.slice(0, 8);
  const t = runId.slice(9);
  return `${d.slice(6,8)}.${d.slice(4,6)}.${d.slice(0,4)}  ${t.slice(0,2)}:${t.slice(2,4)}`;
}
function scoreColor(score) {
  if (score == null || Number.isNaN(Number(score))) return '#666';
  const v = Number(score);
  if (v >= 0.90) return '#006600';
  if (v >= 0.80) return '#337700';
  if (v >= 0.70) return '#996600';
  return '#cc0000';
}
function scoreBar(score, max = 1) {
  if (score == null) return null;
  const pctVal = Math.min(100, (Number(score) / max) * 100);
  const color = scoreColor(score);
  return (
    <div style={{display:'flex',alignItems:'center',gap:4}}>
      <div style={{width:60,height:8,background:'#e0e0e0',border:'1px solid #b0b0b0'}}>
        <div style={{width:`${pctVal}%`,height:'100%',background:color}} />
      </div>
      <span style={{fontFamily:'Consolas',fontSize:11,color}}>{f(score)}</span>
    </div>
  );
}
function now() {
  return new Date().toLocaleString('tr-TR');
}

/* ================================================================
   ANA UYGULAMA
   ================================================================ */
export default function App() {
  const [authenticated, setAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  const [activePage, setActivePage] = useState('overview');
  const [apiKey, setApiKey] = useState(import.meta.env.VITE_DEFAULT_API_KEY || '');
  const [runs, setRuns] = useState([]);
  const [dbRuns, setDbRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [data, setData] = useState(null);
  const [dbStatus, setDbStatus] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedModelIdx, setSelectedModelIdx] = useState(null);
  const [sortCol, setSortCol] = useState('test_roc_auc');
  const [sortDir, setSortDir] = useState('desc');

  /* ---- Tema Yönetimi ---- */
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('ds_theme') || 'classic';
    if (saved === 'modern') return 'modern-light';   // eski değer → aydınlık modern
    return saved;
  });

  useEffect(() => {
    if (theme === 'modern-light' || theme === 'modern-dark') {
      document.documentElement.setAttribute('data-theme', theme);
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    applyChartTheme(theme);
    localStorage.setItem('ds_theme', theme);
  }, [theme]);

  function toggleTheme() {
    setTheme(prev => {
      if (prev === 'classic') return 'modern-light';
      if (prev === 'modern-light') return 'modern-dark';
      return 'classic';
    });
  }

  const isModern = theme.startsWith('modern');
  const isDark = theme === 'modern-dark';

  const [chatSessionId, setChatSessionId] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatQuickActions, setChatQuickActions] = useState([]);
  const [chatSummary, setChatSummary] = useState(null);
  const [chatBusy, setChatBusy] = useState(false);
  const [chatError, setChatError] = useState('');
  const [chatRiskScore, setChatRiskScore] = useState(0.5);
  const [chatCustomer, setChatCustomer] = useState({
    hotel: 'City Hotel',
    lead_time: 30,
    deposit_type: 'No Deposit',
    previous_cancellations: 0,
    market_segment: 'Online TA',
    adults: 2,
    children: 0,
    stays_in_week_nights: 2,
    stays_in_weekend_nights: 1,
  });

  function authFailed(err) {
    const msg = String(err?.message || err || '');
    if (msg.includes('401')) {
      localStorage.removeItem('dashboard_token');
      setAuthenticated(false);
      setLoginError('Oturum süresi doldu. Lütfen tekrar giriş yapın.');
      return true;
    }
    return false;
  }

  const refreshRunsAndData = useCallback(async () => {
    setError('');
    setLoading(true);
    try {
      const runPayload = await getRuns(apiKey);
      const availableRuns = runPayload.runs || [];
      setRuns(availableRuns);
      setDbRuns(runPayload.db_runs || []);
      const runForOverview = selectedRun || availableRuns[0] || '';
      if (runForOverview && runForOverview !== selectedRun) setSelectedRun(runForOverview);
      const overview = await getOverview(runForOverview, apiKey);
      setData(overview);
    } catch (err) {
      if (!authFailed(err)) setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiKey, selectedRun]);

  async function refreshOverviewOnly(runId) {
    setError('');
    setLoading(true);
    try {
      const overview = await getOverview(runId, apiKey);
      setData(overview);
    } catch (err) {
      if (!authFailed(err)) setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function refreshDbStatus() {
    setError('');
    setLoading(true);
    try {
      const s = await getDbStatus(apiKey);
      setDbStatus(s);
    } catch (err) {
      if (!authFailed(err)) setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function chatRiskLabelFromScore(score) {
    const val = Number(score);
    if (val >= 0.65) return 'high';
    if (val >= 0.35) return 'medium';
    return 'low';
  }

  function handleChatCustomerChange(key, value) {
    setChatCustomer(prev => ({ ...prev, [key]: value }));
  }

  async function openChatSession() {
    setChatError('');
    setChatBusy(true);
    try {
      const payload = {
        customer_data: {
          ...chatCustomer,
          lead_time: Number(chatCustomer.lead_time || 0),
          previous_cancellations: Number(chatCustomer.previous_cancellations || 0),
          adults: Number(chatCustomer.adults || 1),
          children: Number(chatCustomer.children || 0),
          stays_in_week_nights: Number(chatCustomer.stays_in_week_nights || 0),
          stays_in_weekend_nights: Number(chatCustomer.stays_in_weekend_nights || 0),
        },
        risk_score: Number(chatRiskScore),
        risk_label: chatRiskLabelFromScore(chatRiskScore),
      };
      const created = await startChatSession(payload, apiKey);
      setChatSessionId(created.session_id);
      setChatQuickActions(created.quick_actions || []);
      setChatMessages([
        {
          role: 'assistant',
          content: created.bot_message || 'Oturum açıldı.',
        },
      ]);
      const summary = await getChatSummary(created.session_id, apiKey);
      setChatSummary(summary);
    } catch (err) {
      if (!authFailed(err)) setChatError(err.message || 'Chat oturumu açılamadı.');
    } finally {
      setChatBusy(false);
    }
  }

  async function sendUserChatMessage(text) {
    const messageText = String(text || '').trim();
    if (!messageText || !chatSessionId) return;

    setChatError('');
    setChatBusy(true);
    setChatMessages(prev => [...prev, { role: 'user', content: messageText }]);
    setChatInput('');

    try {
      const response = await sendChatMessage(
        {
          session_id: chatSessionId,
          message: messageText,
        },
        apiKey,
      );
      setChatMessages(prev => [
        ...prev,
        { role: 'assistant', content: response.bot_message || 'Yanıt alınamadı.' },
      ]);
      setChatQuickActions(response.quick_actions || []);
      const summary = await getChatSummary(chatSessionId, apiKey);
      setChatSummary(summary);
    } catch (err) {
      if (!authFailed(err)) setChatError(err.message || 'Mesaj gönderilemedi.');
    } finally {
      setChatBusy(false);
    }
  }

  async function handleLogin(e) {
    e.preventDefault();
    setLoginError('');
    try {
      const p = await login(username, password);
      localStorage.setItem('dashboard_token', p.access_token);
      setAuthenticated(true);
      setCurrentUser(p.username || username);
      setPassword('');
      await refreshRunsAndData();
    } catch (err) {
      setLoginError(err.message || 'Giriş yapılamadı.');
    }
  }

  async function handleLogout() {
    try { await logout(); } catch (_) {}
    localStorage.removeItem('dashboard_token');
    setAuthenticated(false);
    setCurrentUser('');
    setData(null);
    setRuns([]);
    setDbRuns([]);
    setDbStatus(null);
  }

  useEffect(() => {
    const token = localStorage.getItem('dashboard_token');
    if (!token) { setAuthenticated(false); return; }
    me().then((p) => {
      setAuthenticated(true);
      setCurrentUser(p.username || '');
      refreshRunsAndData();
    }).catch(() => {
      localStorage.removeItem('dashboard_token');
      setAuthenticated(false);
    });
  }, []);

  useEffect(() => {
    if (!authenticated) return;
    if (activePage === 'system') refreshDbStatus();
  }, [activePage, authenticated]);

  /* ---- Türetilmiş Veriler ---- */
  const modelRows = data?.models || [];
  const champion = data?.champion || {};
  const generatedAt = data?.generated_at ? new Date(data.generated_at).toLocaleString('tr-TR') : '-';

  // Karar modelleri hariç filtrele (genel bakışta)
  const coreModels = useMemo(() => modelRows.filter(m => !m.model_name.endsWith('_decision')), [modelRows]);

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

  // En iyi skor bulucu
  const bestScores = useMemo(() => {
    if (!coreModels.length) return {};
    const fields = ['test_roc_auc', 'test_f1', 'test_precision', 'test_recall'];
    const result = {};
    fields.forEach(f => {
      result[f] = Math.max(...coreModels.map(m => m[f] ?? 0));
    });
    return result;
  }, [coreModels]);

  // Şampiyon modelin bilgisi
  const championModel = useMemo(() => {
    return coreModels.find(m => m.model_name === champion.selected_model) || null;
  }, [coreModels, champion]);

  /* ---- Grafikler ---- */
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

  useEffect(() => {
    if (!chartDataset.labels.length || activePage !== 'overview') return;
    const aucCtx = document.getElementById('aucChart');
    const prfCtx = document.getElementById('prfChart');
    if (!aucCtx || !prfCtx) return;

    const _isM = theme.startsWith('modern');
    const _isDk = theme === 'modern-dark';
    const gridColor = _isM ? (_isDk ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)') : '#c0c0c0';
    const tickColor = _isM ? (_isDk ? '#cbd5e1' : '#4a5568') : undefined;
    const auc = new Chart(aucCtx, {
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

    const prf = new Chart(prfCtx, {
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
    return () => { auc.destroy(); prf.destroy(); };
  }, [chartDataset, activePage, theme]);

  /* ================================================================
     LOGIN EKRANI
     ================================================================ */
  if (!authenticated) {
    return (
      <div className="loginPage">
        <form className="loginCard" onSubmit={handleLogin}>
          <h1>Rezervasyon İptal Tahmin Sistemi — Giriş</h1>
          <p>Bu panel yalnızca yetkili personel içindir. Lütfen kimlik bilgilerinizi girin.</p>
          <p style={{ marginTop: 4, fontSize: 12, color: isModern ? (isDark ? '#cbd5e1' : '#4a5568') : '#666' }}>
            Docker ortamı için giriş: <b>admin / admin123</b>
          </p>
          <label>Kullanıcı Adı:</label>
          <input value={username} onChange={e => setUsername(e.target.value)} required autoFocus />
          <label>Şifre:</label>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} required />
          {loginError && <div className="error smallError">{loginError}</div>}
          <button type="submit">Giriş</button>
        </form>
        <button
          className="themeToggle"
          onClick={toggleTheme}
          style={{ position: 'fixed', bottom: 20, right: 20, zIndex: 100 }}
        >
          <span className="themeIcon">{theme === 'classic' ? '☀️' : isDark ? '🖥️' : '🌙'}</span>
          {theme === 'classic' ? 'Modern Aydınlık' : isDark ? 'Klasik Görünüm' : 'Modern Karanlık'}
        </button>
      </div>
    );
  }

  /* ================================================================
     ANA ARAYÜZ
     ================================================================ */
  const navItems = [
    { key: 'overview', label: 'Genel Bakış',       desc: 'Aktif model ve özet göstergeler' },
    { key: 'models',   label: 'Model Karşılaştırma', desc: 'Tüm modellerin detaylı analizi' },
    { key: 'pipeline', label: 'Veri İşleme Hattı',  desc: 'Önişleme, özellik çıkarımı ve model eğitim adımları' },
    { key: 'runs',     label: 'Koşu Geçmişi',      desc: 'Geçmiş çalıştırma kayıtları' },
    { key: 'chat',     label: 'Chat Asistanı',      desc: 'Müşteri bazlı iptal azaltma danışmanı' },
    { key: 'system',   label: 'Sistem Durumu',      desc: 'Veritabanı ve altyapı bilgisi' },
  ];

  return (
    <div className="appShell">
      {/* ===== SOL PANEL ===== */}
      <aside className="sidebar">
        <div className="sidebarTitle">Rezervasyon Tahmin</div>
        <div className="sidebarSub">Karar Destek Paneli</div>
        <nav className="sidebarNav">
          {navItems.map(item => (
            <button
              key={item.key}
              className={`navBtn ${activePage === item.key ? 'active' : ''}`}
              onClick={() => setActivePage(item.key)}
              title={item.desc}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebarInfo">
          <div><strong>Kullanıcı:</strong> {currentUser}</div>
          <div><strong>Run:</strong> {formatRunId(selectedRun)}</div>
        </div>
        <button className="themeToggle" onClick={toggleTheme}>
          <span className="themeIcon">{theme === 'classic' ? '☀️' : isDark ? '🖥️' : '🌙'}</span>
          {theme === 'classic' ? 'Modern Aydınlık' : isDark ? 'Klasik Görünüm' : 'Modern Karanlık'}
        </button>
        <button className="logoutBtn" onClick={handleLogout}>✕ Oturumu Kapat</button>
      </aside>

      {/* ===== ANA ALAN ===== */}
      <main className="container">
        {/* Araç Çubuğu */}
        <div className="topBar">
          <div className="brandBlock">
            <div className="brandTitle">DS Project — Rezervasyon İptal Tahmin Sistemi</div>
          </div>
          <div className="metaBlock">
            <span className="metaItem"><strong>Son Güncelleme:</strong> {generatedAt}</span>
            <span className="metaItem">|</span>
            <span className="metaItem"><strong>Aktif Model:</strong> {displayName(champion.selected_model)}</span>
          </div>
        </div>

        {/* Filtre */}
        <section className="card controls">
          <div className="controlTitle">Filtreler</div>
          <div>
            <label>Koşu Seçimi:</label>
            <select value={selectedRun} onChange={e => { setSelectedRun(e.target.value); refreshOverviewOnly(e.target.value); }}>
              {runs.map(r => <option key={r} value={r}>{formatRunId(r)}</option>)}
            </select>
          </div>
          <div>
            <label>API Anahtarı (opsiyonel):</label>
            <input value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="x-api-key" />
          </div>
          <button onClick={refreshRunsAndData} disabled={loading}>
            {loading ? '⏳ Yükleniyor...' : '🔄 Verileri Yenile'}
          </button>
        </section>

        {error && <div className="error card">⚠ Hata: {error}</div>}

        {/* ===============================================================
            SAYFA 1: GENEL BAKIŞ — Yönetici Özeti
            =============================================================== */}
        {activePage === 'overview' && (
          <>
            <header className="pageHeader">
              <h1>📊 Yönetici Özeti</h1>
              <p className="subtitle">
                Bu koşuda {coreModels.length} farklı model eğitildi ve değerlendirildi.
                Sistem, <strong>"{displayName(champion.selected_model)}"</strong> modelini en kârlı olarak seçti.
                Aşağıda seçim kararının gerekçesi ve temel göstergeler yer alıyor.
              </p>
            </header>

            {/* Şampiyon Model Kartı */}
            {championModel && (
              <section className="championCard card">
                <div className="small">🏆 Seçilen Model: {displayName(champion.selected_model)}</div>
                <div className="championGrid">
                  <div className="champItem">
                    <span className="champLabel">Neden Bu Model?</span>
                    <span className="champValue" style={{fontSize:11,lineHeight:1.4}}>
                      {champion.ranking_mode === 'incremental_profit'
                        ? `Artışsal kâr (incremental profit) hesaplamasına göre ${pct(champion.max_action_rate)} kapasite kısıtı altında en yüksek net kazancı bu model sağlıyor.`
                        : `${champion.ranking_mode} kriterine göre en başarılı model.`
                      }
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
                    <span className="champHint">Müşterilerin en fazla bu kadarına müdahale edilebilir</span>
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
              <div className="explain">Tüm modeller aynı test seti üzerinde değerlendirildi. En yüksek skorlar yeşil renkle vurgulanır. Şampiyon model ★ ile işaretlidir.</div>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th style={{width:24}}></th>
                      <th>Model</th>
                      <th>Tür</th>
                      <th>Kalibrasyon</th>
                      <th onClick={() => toggleSort('test_roc_auc')} style={{cursor:'pointer'}}>Test AUC{sortIndicator('test_roc_auc')}</th>
                      <th onClick={() => toggleSort('test_f1')} style={{cursor:'pointer'}}>F1{sortIndicator('test_f1')}</th>
                      <th onClick={() => toggleSort('test_precision')} style={{cursor:'pointer'}}>Precision{sortIndicator('test_precision')}</th>
                      <th onClick={() => toggleSort('test_recall')} style={{cursor:'pointer'}}>Recall{sortIndicator('test_recall')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedModels.map((m, i) => {
                      const isChamp = m.model_name === champion.selected_model;
                      return (
                        <tr key={m.model_name} style={isChamp ? {background:'#fffff0',fontWeight:600} : {}}>
                          <td style={{textAlign:'center'}}>{isChamp ? '★' : modelIcon(m.model_name)}</td>
                          <td><strong>{displayName(m.model_name)}</strong></td>
                          <td><span className={`typeBadge ${modelBadge(m.model_name) === 'Gelişmiş' ? 'advanced' : 'base'}`}>{modelBadge(m.model_name)}</span></td>
                          <td>{modelCalibration(m.model_name)}</td>
                          <td style={{color: m.test_roc_auc === bestScores.test_roc_auc ? '#006600' : undefined, fontWeight: m.test_roc_auc === bestScores.test_roc_auc ? 700 : 400}}>{scoreBar(m.test_roc_auc)}</td>
                          <td style={{color: m.test_f1 === bestScores.test_f1 ? '#006600' : undefined, fontWeight: m.test_f1 === bestScores.test_f1 ? 700 : 400}}>{scoreBar(m.test_f1)}</td>
                          <td>{scoreBar(m.test_precision)}</td>
                          <td>{scoreBar(m.test_recall)}</td>
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
                <div className="explain">Eğitim ve test skorlarının yakın olması modelin aşırı öğrenme (overfitting) yapmadığını gösterir.</div>
                <canvas id="aucChart" height="160" />
              </div>
              <div className="card">
                <div className="small">Test Metrikleri Karşılaştırması</div>
                <div className="explain">Precision: doğruluk, Recall: kapsayıcılık, F1: ikisinin dengeli özeti.</div>
                <canvas id="prfChart" height="160" />
              </div>
            </section>
          </>
        )}

        {/* ===============================================================
            SAYFA 2: MODEL KARŞILAŞTIRMA — Detaylı Analiz
            =============================================================== */}
        {activePage === 'models' && (
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
                      <th style={{width:20}}>#</th>
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
                        >
                          <td style={{textAlign:'center'}}>{isChamp ? '★' : i + 1}</td>
                          <td>{modelIcon(m.model_name)} <strong>{displayName(m.model_name)}</strong></td>
                          <td><span className={`typeBadge ${modelBadge(m.model_name) === 'Gelişmiş' ? 'advanced' : 'base'}`}>{modelBadge(m.model_name)}</span></td>
                          <td>{modelCalibration(m.model_name)}</td>
                          <td>{f(m.train_cv_roc_auc_mean)} ± {f(m.train_cv_roc_auc_std)}</td>
                          <td style={{textAlign:'center'}}>{m.cv_folds ?? '-'}</td>
                          <td>{scoreBar(m.test_roc_auc)}</td>
                          <td>{scoreBar(m.test_f1)}</td>
                          <td>{scoreBar(m.test_precision)}</td>
                          <td>{scoreBar(m.test_recall)}</td>
                          <td style={{fontFamily:'Consolas'}}>{f(m.test_threshold, 3)}</td>
                          <td style={{textAlign:'right'}}>{m.n_test?.toLocaleString('tr-TR') || '-'}</td>
                          <td>{pct(m.positive_rate_test)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Seçili Model Detay Paneli */}
            {selectedModelIdx !== null && modelRows[selectedModelIdx] && (() => {
              const m = modelRows[selectedModelIdx];
              const isChamp = m.model_name === champion.selected_model;
              return (
                <section className="card detailPanel">
                  <div className="small">
                    {modelIcon(m.model_name)} {displayName(m.model_name)} — Detay Bilgisi
                    {isChamp && <span style={{marginLeft: 8, color: '#996600'}}>★ Şampiyon Model</span>}
                  </div>
                  <div className="detailGrid">
                    <div className="detailItem">
                      <span>Teknik Ad</span>
                      <strong style={{fontSize:10, wordBreak:'break-all'}}>{m.model_name}</strong>
                    </div>
                    <div className="detailItem">
                      <span>Model Tipi</span>
                      <strong>{modelType(m.model_name)}</strong>
                    </div>
                    <div className="detailItem">
                      <span>Kalibrasyon</span>
                      <strong>{modelCalibration(m.model_name)}</strong>
                    </div>
                    <div className="detailItem">
                      <span>Eğitim AUC (Ort)</span>
                      <strong>{f(m.train_cv_roc_auc_mean)}</strong>
                    </div>
                    <div className="detailItem">
                      <span>Eğitim AUC (Std)</span>
                      <strong>{f(m.train_cv_roc_auc_std)}</strong>
                    </div>
                    <div className="detailItem">
                      <span>CV Katlanma</span>
                      <strong>{m.cv_folds ?? '-'}</strong>
                    </div>
                    <div className="detailItem highlight">
                      <span>Test ROC-AUC</span>
                      <strong style={{color: scoreColor(m.test_roc_auc)}}>{f(m.test_roc_auc)}</strong>
                    </div>
                    <div className="detailItem highlight">
                      <span>F1 Skoru</span>
                      <strong style={{color: scoreColor(m.test_f1)}}>{f(m.test_f1)}</strong>
                    </div>
                    <div className="detailItem">
                      <span>Precision</span>
                      <strong>{f(m.test_precision)}</strong>
                    </div>
                    <div className="detailItem">
                      <span>Recall</span>
                      <strong>{f(m.test_recall)}</strong>
                    </div>
                    <div className="detailItem">
                      <span>Karar Eşiği</span>
                      <strong>{f(m.test_threshold, 3)}</strong>
                    </div>
                    <div className="detailItem">
                      <span>Test Seti</span>
                      <strong>{m.n_test?.toLocaleString('tr-TR') || '-'} kayıt</strong>
                    </div>
                    <div className="detailItem full">
                      <span>Yorum</span>
                      <strong style={{fontSize:11, fontWeight:400}}>
                        {m.test_roc_auc > 0.93
                          ? 'Yüksek ayırt edicilik. Model, iptal edecek ve etmeyecek müşterileri çok iyi ayırt edebiliyor.'
                          : m.test_roc_auc > 0.85
                            ? 'İyi düzeyde ayırt edicilik. Pratikte kullanılabilir performans.'
                            : 'Düşük-orta ayırt edicilik. Daha güçlü modeller tercih edilmeli.'}
                        {' '}
                        {Math.abs((m.train_cv_roc_auc_mean || 0) - (m.test_roc_auc || 0)) < 0.02
                          ? 'Eğitim-test farkı çok düşük, aşırı öğrenme (overfitting) riski yok.'
                          : 'Eğitim ve test arasında fark var, dikkat edilmeli.'}
                      </strong>
                    </div>
                  </div>
                </section>
              );
            })()}

            {/* Terim Açıklamaları */}
            <section className="card">
              <div className="legendBox">
                <strong>📖 Metrik Açıklamaları:</strong>
                <ul>
                  <li><strong>ROC-AUC</strong>: Modelin iptal / iptal-değil ayrımındaki genel başarısı. 1.0 mükemmel, 0.5 rastgele tahmin.</li>
                  <li><strong>Precision</strong>: "İptal edecek" dediğimiz müşterilerin gerçekten ne kadarı iptal etti? Yüksekse → az yanlış alarm.</li>
                  <li><strong>Recall</strong>: Gerçekten iptal eden müşterilerin ne kadarını yakaladık? Yüksekse → az kaçırma.</li>
                  <li><strong>F1</strong>: Precision ve Recall'un harmonik ortalaması. İkisini dengeli değerlendirmek için kullanılır.</li>
                  <li><strong>CV (Çapraz Doğrulama)</strong>: Eğitim verisini {coreModels[0]?.cv_folds || 5} parçaya bölerek her parçada ayrı test yapma. Sonucun güvenilir olduğunu doğrular.</li>
                  <li><strong>Kalibrasyon</strong>: Modelin "% olasılık" çıktısının gerçek iptal oranıyla ne kadar uyumlu olduğunu iyileştiren işlem.</li>
                  <li><strong>Eşik</strong>: Bu değerin üstündeki tahminler "iptal riski var, müdahale et" olarak işaretlenir.</li>
                </ul>
              </div>
            </section>
          </>
        )}

        {/* ===============================================================
            SAYFA 3: VERİ İŞLEME HATTI — Pipeline Görünümü
            =============================================================== */}
        {activePage === 'pipeline' && (
          <>
            <header className="pageHeader">
              <h1>🔧 Veri İşleme Hattı (Pipeline)</h1>
              <p className="subtitle">
                Ham veriden tahmine kadar tüm adımlar. Her model aşağıdaki önişleme, özellik dönüşümü
                ve eğitim sürecinden geçerek nihai karar modelini oluşturur.
              </p>
            </header>

            {/* === Pipeline Akış Şeması === */}
            <section className="card">
              <div className="small">📐 Uçtan Uca Pipeline Akışı</div>
              <div className="explain">Her kutu bir DVC aşamasını temsil eder. Veriler soldan sağa doğru akar.</div>
              <div className="pipelineFlow">
                <div className="pipeStep raw">
                  <div className="pipeStepIcon">📄</div>
                  <div className="pipeStepTitle">Ham Veri</div>
                  <div className="pipeStepDesc">hotel_bookings.csv<br />Orijinal 32+ sütun</div>
                </div>
                <div className="pipeArrow">→</div>
                <div className="pipeStep validate">
                  <div className="pipeStepIcon">✅</div>
                  <div className="pipeStepTitle">Doğrulama</div>
                  <div className="pipeStepDesc">5 katman / 30+ kural<br />Pandera + temel şema</div>
                </div>
                <div className="pipeArrow">→</div>
                <div className="pipeStep preprocess">
                  <div className="pipeStepIcon">🔧</div>
                  <div className="pipeStepTitle">Önişleme</div>
                  <div className="pipeStepDesc">Sızıntı temizliği<br />Eksik veri doldurma</div>
                </div>
                <div className="pipeArrow">→</div>
                <div className="pipeStep split">
                  <div className="pipeStepIcon">✂️</div>
                  <div className="pipeStepTitle">Veri Bölme</div>
                  <div className="pipeStepDesc">%64 eğitim / %16 kalibrasyon<br />%20 test</div>
                </div>
                <div className="pipeArrow">→</div>
                <div className="pipeStep train">
                  <div className="pipeStepIcon">🧠</div>
                  <div className="pipeStepTitle">Eğitim</div>
                  <div className="pipeStepDesc">Feature transform<br />Model uydurma + CV</div>
                </div>
                <div className="pipeArrow">→</div>
                <div className="pipeStep calibrate">
                  <div className="pipeStepIcon">⚖️</div>
                  <div className="pipeStepTitle">Kalibrasyon</div>
                  <div className="pipeStepDesc">Sigmoid / İzotonik<br />Olasılık düzeltme</div>
                </div>
                <div className="pipeArrow">→</div>
                <div className="pipeStep evaluate">
                  <div className="pipeStepIcon">📊</div>
                  <div className="pipeStepTitle">Değerlendirme</div>
                  <div className="pipeStepDesc">Eşik taraması<br />Kâr optimizasyonu</div>
                </div>
                <div className="pipeArrow">→</div>
                <div className="pipeStep champion">
                  <div className="pipeStepIcon">🏆</div>
                  <div className="pipeStepTitle">Şampiyon Seçim</div>
                  <div className="pipeStepDesc">Tercih sırası<br />Karar politikası</div>
                </div>
              </div>
            </section>

            {/* === Adım 1: Doğrulama === */}
            <section className="card">
              <div className="small">1️⃣ Veri Doğrulama — 5 Katmanlı Savunma (Data Validation)</div>
              <div className="explain">
                Ham veriden inference'a kadar 5 ayrı noktada doğrulama devreye girer.
                Her katman farklı bir aşamada veri kalitesini güvence altına alır.
              </div>

              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th style={{width:28}}>#</th>
                      <th>Katman</th>
                      <th>Ne Zaman?</th>
                      <th>Kaynak</th>
                      <th>Kontroller</th>
                      <th>Durum</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style={{textAlign:'center',fontWeight:'bold',color:'#b8860b'}}>1</td>
                      <td><strong>Temel Şema</strong></td>
                      <td>Önişleme başında</td>
                      <td><code>validate.py</code></td>
                      <td>Boş veri · Hedef sütun varlığı · Yinelenen sütun · Etiket kümesi · Null oranı raporu</td>
                      <td style={{color:'green',whiteSpace:'nowrap'}}>✅ Aktif</td>
                    </tr>
                    <tr>
                      <td style={{textAlign:'center',fontWeight:'bold',color:'#0055aa'}}>2</td>
                      <td><strong>Pandera Ham Veri</strong></td>
                      <td>Önişleme başında</td>
                      <td><code>data_validation.py</code></td>
                      <td>17 sütun için tip kontrolü · Sayısal aralık (lead_time ≥ 0, adr ≥ -10 …) · Kategori kümesi (hotel, meal …) · is_canceled ∈ {'{yes,no}'}​</td>
                      <td style={{color:'green',whiteSpace:'nowrap'}}>✅ Aktif</td>
                    </tr>
                    <tr>
                      <td style={{textAlign:'center',fontWeight:'bold',color:'#880088'}}>3</td>
                      <td><strong>İşlenmiş Veri</strong></td>
                      <td>Önişleme + eğitim öncesi</td>
                      <td><code>data_validation.py</code></td>
                      <td>Hedef 0/1 tamsayı · Sayısal sütunlarda NaN/Inf yok · İmpütasyon sonrası NaN → ValueError</td>
                      <td style={{color:'green',whiteSpace:'nowrap'}}>✅ Aktif</td>
                    </tr>
                    <tr>
                      <td style={{textAlign:'center',fontWeight:'bold',color:'#cc3300'}}>4</td>
                      <td><strong>Inference Payload</strong></td>
                      <td>Her API isteğinde</td>
                      <td><code>predict.py</code></td>
                      <td>Eksik / fazla sütun tespiti · Sayısal tip zorlaması · Kategorik → string · Pandera şema (non-blocking, uyarı loglar) · Drift kontrolü</td>
                      <td style={{color:'green',whiteSpace:'nowrap'}}>✅ Aktif</td>
                    </tr>
                    <tr>
                      <td style={{textAlign:'center',fontWeight:'bold',color:'#006644'}}>5</td>
                      <td><strong>Dağılım İzleme</strong></td>
                      <td>Monitor CLI / canlı izleme</td>
                      <td><code>data_validation.py</code></td>
                      <td>Referans ortalama/std/min/max (reference_stats.json) · |Δmean|/std &gt; eşik → uyarı · Aralık dışı değer · Referans kategori seti · Unseen category tespiti</td>
                      <td style={{color:'green',whiteSpace:'nowrap'}}>✅ Aktif</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div style={{marginTop:14,padding:'9px 14px',background:'#f0fff0',border:'1px solid #8fbc8f',fontSize:12,lineHeight:1.6}}>
                <strong>Toplam:</strong> 5 katman · <strong>30+ kural</strong> · Ham veri → Önişleme → Eğitim → Inference → İzleme
                <span style={{marginLeft:16,color:'green',fontWeight:'bold'}}>✅ 5/5 katman aktif</span>
              </div>
            </section>

            {/* === Adım 2: Önişleme === */}
            <section className="card">
              <div className="small">2️⃣ Önişleme Adımları (Preprocessing)</div>
              <div className="explain">preprocess.py — Ham veriden temiz parquet dosyasına dönüşüm süreci.</div>
              <div className="pipelineStepList">
                <div className="stepItem">
                  <div className="stepNum">A</div>
                  <div className="stepBody">
                    <strong>Sütun İsmi Temizleme</strong>
                    <span>Tüm sütun isimlerindeki başta/sonda boşluklar kaldırılır (strip).</span>
                  </div>
                </div>
                <div className="stepItem">
                  <div className="stepNum">B</div>
                  <div className="stepBody">
                    <strong>Veri Sızıntısı Temizliği (Leakage Removal)</strong>
                    <span>Hedef değişkeni doğrudan açığa çıkaran sütunlar çıkarılır:<br />
                      <code>reservation_status</code> — iptal durumunu doğrudan gösterir<br />
                      <code>reservation_status_date</code> — iptal tarihini içerir<br />
                      Bu sütunlar modele verilseydi AUC=1.0 olur ama gerçek dünyada kullanılamaz (sahte başarı).
                    </span>
                  </div>
                </div>
                <div className="stepItem">
                  <div className="stepNum">C</div>
                  <div className="stepBody">
                    <strong>Hedef Etiket Dönüşümü</strong>
                    <span>"yes" → 1, "no" → 0 şeklinde sayısal formata çevrilir. Küçük harfe dönüştürülüp boşluklar kaldırılır.</span>
                  </div>
                </div>
                <div className="stepItem">
                  <div className="stepNum">D</div>
                  <div className="stepBody">
                    <strong>Tamamen Boş Sütunların Kaldırılması</strong>
                    <span>%100 NaN içeren sütunlar veri setinden çıkarılır — herhangi bir bilgi taşımadıkları için.</span>
                  </div>
                </div>
                <div className="stepItem">
                  <div className="stepNum">E</div>
                  <div className="stepBody">
                    <strong>Temel Eksik Veri Doldurma (Imputation)</strong>
                    <span>
                      Sayısal sütunlar → <strong>medyan</strong> ile doldurulur<br />
                      Kategorik sütunlar → <strong>mod (en sık değer)</strong> ile doldurulur; mod yoksa "UNKNOWN"<br />
                      <em>Not: Sklearn Pipeline içinde de tekrar imputation yapılır (güvenlik katmanı).</em>
                    </span>
                  </div>
                </div>
              </div>
              <div className="stepOutput">
                <strong>Çıktı:</strong> <code>data/processed/dataset.parquet</code> — Temizlenmiş, doldurulmuş veri seti
              </div>
            </section>

            {/* === Adım 3: Veri Bölme === */}
            <section className="card">
              <div className="small">3️⃣ Veri Bölme Stratejisi (Train / Calibration / Test Split)</div>
              <div className="explain">split.py — Katmanlaştırılmış (stratified) bölme ile sınıf oranları korunur.</div>
              <div className="splitDiagram">
                <div className="splitBlock full">
                  <div className="splitLabel">Tüm Veri (%100)</div>
                  <div className="splitChildren">
                    <div className="splitBlock train-full">
                      <div className="splitLabel">Eğitim Havuzu (%80)</div>
                      <div className="splitChildren">
                        <div className="splitBlock train">
                          <div className="splitLabel">Eğitim<br />(%64)</div>
                          <div className="splitDesc">Model uydurma<br />CV doğrulama</div>
                        </div>
                        <div className="splitBlock cal">
                          <div className="splitLabel">Kalibrasyon<br />(%16)</div>
                          <div className="splitDesc">Olasılık<br />düzeltme</div>
                        </div>
                      </div>
                    </div>
                    <div className="splitBlock test">
                      <div className="splitLabel">Test<br />(%20)</div>
                      <div className="splitDesc">Nihai<br />değerlendirme</div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="legendBox" style={{marginTop:6}}>
                <strong>Neden 3 parça?</strong>
                <ul>
                  <li><strong>Eğitim:</strong> Modelin öğrendiği veri. 5-katlı çapraz doğrulama bu set üzerinde yapılır.</li>
                  <li><strong>Kalibrasyon:</strong> Modelin olasılık çıktısını düzeltmek için ayrı tutulan veri. Eğitime karışmaz.</li>
                  <li><strong>Test:</strong> Hiçbir aşamada kullanılmamış saf veri. Gerçek performansı ölçer.</li>
                </ul>
              </div>
            </section>

            {/* === Adım 4: Feature Engineering === */}
            <section className="card">
              <div className="small">4️⃣ Özellik Çıkarımı ve Dönüşüm (Feature Engineering)</div>
              <div className="explain">features.py — Sklearn ColumnTransformer ile model pipeline'ı içinde uygulanır (eğitim-sunucu tutarsızlığını önler).</div>
              <div className="grid2" style={{margin:0,gap:2}}>
                {/* Sayısal Özellikler */}
                <div className="card" style={{margin:0}}>
                  <div className="small">Sayısal Özellikler — 19 sütun</div>
                  <div className="featurePipeline">
                    <div className="fpStep">SimpleImputer(strategy='median')</div>
                    <div className="fpArrow">↓</div>
                    <div className="fpStep">StandardScaler (z-score normalizasyon)</div>
                  </div>
                  <div className="explain" style={{marginTop:4}}>Formül: z = (x − μ) / σ → Ortalama 0, standart sapma 1</div>
                  <div className="featureList">
                    <div className="featureTag num">lead_time <span>Rezervasyon öncesi gün</span></div>
                    <div className="featureTag num">arrival_date_year <span>Varış yılı</span></div>
                    <div className="featureTag num">arrival_date_week_number <span>Hafta numarası</span></div>
                    <div className="featureTag num">arrival_date_day_of_month <span>Ayın günü</span></div>
                    <div className="featureTag num">stays_in_weekend_nights <span>Hafta sonu gece</span></div>
                    <div className="featureTag num">stays_in_week_nights <span>Hafta içi gece</span></div>
                    <div className="featureTag num">adults <span>Yetişkin sayısı</span></div>
                    <div className="featureTag num">children <span>Çocuk sayısı</span></div>
                    <div className="featureTag num">babies <span>Bebek sayısı</span></div>
                    <div className="featureTag num">is_repeated_guest <span>Tekrar misafir (0/1)</span></div>
                    <div className="featureTag num">previous_cancellations <span>Önceki iptaller</span></div>
                    <div className="featureTag num">previous_bookings_not_canceled <span>Önceki tamamlananlar</span></div>
                    <div className="featureTag num">booking_changes <span>Rezervasyon değişiklikleri</span></div>
                    <div className="featureTag num">agent <span>Acente ID</span></div>
                    <div className="featureTag num">company <span>Şirket ID</span></div>
                    <div className="featureTag num">days_in_waiting_list <span>Bekleme listesi günü</span></div>
                    <div className="featureTag num">adr <span>Ortalama günlük ücret</span></div>
                    <div className="featureTag num">required_car_parking_spaces <span>Otopark talebi</span></div>
                    <div className="featureTag num">total_of_special_requests <span>Özel istek sayısı</span></div>
                  </div>
                </div>
                {/* Kategorik Özellikler */}
                <div className="card" style={{margin:0}}>
                  <div className="small">Kategorik Özellikler — 10 sütun</div>
                  <div className="featurePipeline">
                    <div className="fpStep">SimpleImputer(strategy='most_frequent')</div>
                    <div className="fpArrow">↓</div>
                    <div className="fpStep">OneHotEncoder(handle_unknown='ignore')</div>
                  </div>
                  <div className="explain" style={{marginTop:4}}>Her kategori ayrı 0/1 sütuna dönüşür. Bilinmeyen kategoriler yok sayılır.</div>
                  <div className="featureList">
                    <div className="featureTag cat">hotel <span>Otel tipi</span></div>
                    <div className="featureTag cat">arrival_date_month <span>Varış ayı</span></div>
                    <div className="featureTag cat">meal <span>Yemek paketi</span></div>
                    <div className="featureTag cat">country <span>Ülke kodu</span></div>
                    <div className="featureTag cat">market_segment <span>Pazar segmenti</span></div>
                    <div className="featureTag cat">distribution_channel <span>Dağıtım kanalı</span></div>
                    <div className="featureTag cat">reserved_room_type <span>Rezerve oda tipi</span></div>
                    <div className="featureTag cat">assigned_room_type <span>Atanan oda tipi</span></div>
                    <div className="featureTag cat">deposit_type <span>Depozito tipi</span></div>
                    <div className="featureTag cat">customer_type <span>Müşteri tipi</span></div>
                  </div>
                </div>
              </div>
              <div className="legendBox" style={{marginTop:6}}>
                <strong>⚠ Önemli Tasarım Kararı:</strong> Tüm feature transform'lar sklearn Pipeline <em>içinde</em> tanımlanır.
                Bu sayede eğitim ve tahmin aşamasında aynı dönüşümler otomatik uygulanır — eğitim/sunucu tutarsızlığı (train-serving skew) önlenir.
                Modelle birlikte .joblib dosyasına kaydedilir.
              </div>
            </section>

            {/* === Adım 5: Model Eğitimi === */}
            <section className="card">
              <div className="small">5️⃣ Model Eğitimi (Training)</div>
              <div className="explain">train.py — Her koşuda iki model ailesi eğitilir: Temel (baseline) ve Gelişmiş (challenger).</div>
              <div className="grid2" style={{margin:0,gap:2}}>
                <div className="card" style={{margin:0,borderColor:'#88aacc'}}>
                  <div className="small">🔵 Temel Model — Lojistik Regresyon</div>
                  <div className="tableWrap">
                    <table>
                      <thead><tr><th>Parametre</th><th>Değer</th></tr></thead>
                      <tbody>
                        <tr><td>Algoritma</td><td>LogisticRegression (sklearn)</td></tr>
                        <tr><td>max_iter</td><td>3000</td></tr>
                        <tr><td>solver</td><td>lbfgs</td></tr>
                        <tr><td>random_state</td><td>42</td></tr>
                      </tbody>
                    </table>
                  </div>
                  <div className="explain" style={{marginTop:4}}>Hızlı, yorumlanabilir, kararlı bir referans modeli. Diğer modellerin bunu geçmesi beklenir.</div>
                </div>
                <div className="card" style={{margin:0,borderColor:'#cc9944'}}>
                  <div className="small">🟠 Gelişmiş Model — XGBoost</div>
                  <div className="tableWrap">
                    <table>
                      <thead><tr><th>Parametre</th><th>Değer</th></tr></thead>
                      <tbody>
                        <tr><td>Algoritma</td><td>XGBClassifier (gradient boosting)</td></tr>
                        <tr><td>n_estimators</td><td>500</td></tr>
                        <tr><td>learning_rate</td><td>0.05</td></tr>
                        <tr><td>max_depth</td><td>6</td></tr>
                        <tr><td>subsample</td><td>0.9</td></tr>
                        <tr><td>colsample_bytree</td><td>0.9</td></tr>
                        <tr><td>objective</td><td>binary:logistic</td></tr>
                        <tr><td>eval_metric</td><td>logloss</td></tr>
                      </tbody>
                    </table>
                  </div>
                  <div className="explain" style={{marginTop:4}}>Güçlü ensemble yöntem. LightGBM, CatBoost ve HistGradientBoosting yedek seçeneklerdir.</div>
                </div>
              </div>
              <div className="legendBox" style={{marginTop:6}}>
                <strong>Çapraz Doğrulama (Cross-Validation):</strong>
                <ul>
                  <li><strong>Yöntem:</strong> StratifiedKFold — sınıf oranları her katlama da korunur</li>
                  <li><strong>Katlama sayısı:</strong> 5</li>
                  <li><strong>Skor metriği:</strong> ROC-AUC</li>
                  <li>CV sonrası model tüm eğitim setine yeniden uydurulur (refit)</li>
                </ul>
              </div>
            </section>

            {/* === Adım 6: Kalibrasyon === */}
            <section className="card">
              <div className="small">6️⃣ Olasılık Kalibrasyonu (Probability Calibration)</div>
              <div className="explain">calibration.py — Modelin olasılık çıktısını gerçek iptal oranlarıyla uyumlu hale getirir.</div>
              <div className="grid2" style={{margin:0,gap:2}}>
                <div className="card" style={{margin:0}}>
                  <div className="small">Sigmoid (Platt Scaling)</div>
                  <div className="pipelineStepList">
                    <div className="stepItem compact">
                      <div className="stepBody">
                        <strong>Yöntem:</strong> <span>Modelin ham olasılıklarına lojistik regresyon uydurma</span>
                      </div>
                    </div>
                    <div className="stepItem compact">
                      <div className="stepBody">
                        <strong>Avantaj:</strong> <span>Küçük kalibrasyon setlerinde daha kararlı</span>
                      </div>
                    </div>
                    <div className="stepItem compact">
                      <div className="stepBody">
                        <strong>Tercih:</strong> <span>Kurumsal tercih listesinde 1. sırada (önerilen)</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="card" style={{margin:0}}>
                  <div className="small">İzotonik Regresyon</div>
                  <div className="pipelineStepList">
                    <div className="stepItem compact">
                      <div className="stepBody">
                        <strong>Yöntem:</strong> <span>Parametrik olmayan monoton regresyon</span>
                      </div>
                    </div>
                    <div className="stepItem compact">
                      <div className="stepBody">
                        <strong>Avantaj:</strong> <span>Büyük veri setlerinde daha esnek</span>
                      </div>
                    </div>
                    <div className="stepItem compact">
                      <div className="stepBody">
                        <strong>Risk:</strong> <span>Küçük kalibrasyon setlerinde aşırı öğrenme riski</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="legendBox" style={{marginTop:6}}>
                <strong>Çarpan etkisi:</strong> Her temel model × 2 kalibrasyon = toplam 6 model varyantı oluşur.
                Karar eşiği uygulandığında (decision) bu sayı 11'e çıkar.
                <code style={{display:'block',marginTop:4,fontSize:10}}>
                  baseline → baseline_calibrated_sigmoid, baseline_calibrated_isotonic<br />
                  challenger_xgboost → challenger_xgboost_calibrated_sigmoid, challenger_xgboost_calibrated_isotonic
                </code>
              </div>
            </section>

            {/* === Adım 7: Değerlendirme ve Eşik Seçimi === */}
            <section className="card">
              <div className="small">7️⃣ Değerlendirme ve Eşik Optimizasyonu (Evaluation)</div>
              <div className="explain">evaluate.py — Test seti üzerinde performans ölçümü ve iş odaklı eşik belirleme.</div>
              <div className="pipelineStepList">
                <div className="stepItem">
                  <div className="stepNum">I</div>
                  <div className="stepBody">
                    <strong>Temel Metrik Hesaplama</strong>
                    <span>ROC-AUC, F1, Precision, Recall ve Confusion Matrix hesaplanır.</span>
                  </div>
                </div>
                <div className="stepItem">
                  <div className="stepNum">II</div>
                  <div className="stepBody">
                    <strong>Eşik Taraması (Threshold Sweep)</strong>
                    <span>
                      0.001–0.999 aralığında 999 eşik değeri taranır.<br />
                      Her eşikte TP, FP, FN, TN hesaplanır ve maliyet matrisine göre net kâr bulunur.
                    </span>
                  </div>
                </div>
                <div className="stepItem">
                  <div className="stepNum">III</div>
                  <div className="stepBody">
                    <strong>Kapasite Kısıtlı Optimizasyon</strong>
                    <span>
                      Aksiyon oranı kısıtları uygulanır: %5, %10, %15, %20, %30<br />
                      Sadece kısıt altında uygulanabilir eşikler değerlendirilir.
                      Uygun eşik bulunamazsa quantile geri-dönüşü kullanılır.
                    </span>
                  </div>
                </div>
                <div className="stepItem">
                  <div className="stepNum">IV</div>
                  <div className="stepBody">
                    <strong>Kural Tabanlı Eşik (Yedek)</strong>
                    <span>
                      F1 maksimizasyonu + "Recall ≥ %80 şartıyla en yüksek Precision" kuralı.
                    </span>
                  </div>
                </div>
              </div>
            </section>

            {/* === Adım 8: Şampiyon Seçimi === */}
            <section className="card">
              <div className="small">8️⃣ Şampiyon Model Seçimi (Champion Selection)</div>
              <div className="explain">policy.py — Tercih listesine göre en kârlı model seçilir ve karar politikası oluşturulur.</div>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th>Sıra</th>
                      <th>Model Adayı</th>
                      <th>Açıklama</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td style={{textAlign:'center'}}>1</td><td>{displayName('challenger_xgboost_calibrated_sigmoid')}</td><td>En kararlı kalibrasyon + en güçlü model</td></tr>
                    <tr><td style={{textAlign:'center'}}>2</td><td>LightGBM + Sigmoid Kalibrasyon</td><td>XGBoost yoksa yedek GBM</td></tr>
                    <tr><td style={{textAlign:'center'}}>3</td><td>CatBoost + Sigmoid Kalibrasyon</td><td>Üçüncü GBM alternatifi</td></tr>
                    <tr><td style={{textAlign:'center'}}>4</td><td>HistGradientBoosting + Sigmoid Kalibrasyon</td><td>Sklearn yerleşik GBM (ek kurulum gerektirmez)</td></tr>
                    <tr><td style={{textAlign:'center'}}>5</td><td>{displayName('baseline_calibrated_sigmoid')}</td><td>Temel model kalibre versiyonu</td></tr>
                    <tr><td style={{textAlign:'center'}}>6+</td><td>Ham modeller (kalibre edilmemiş)</td><td>Son çare — kalibrasyon yoksa ham olasılıklar kullanılır</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="legendBox" style={{marginTop:6}}>
                <strong>Seçim Kriteri:</strong> <code>incremental_profit</code> — Maliyet matrisine göre en yüksek net kârı sağlayan model,
                tercih sırası içinden seçilir. Karar politikası <code>decision_policy.json</code> olarak kaydedilir.
              </div>
            </section>

            {/* === Çıkarılan/Engellenen Sütunlar === */}
            <section className="card">
              <div className="small">🚫 Pipeline'dan Çıkarılan Sütunlar</div>
              <div className="explain">Bu sütunlar model eğitiminde kesinlikle kullanılmaz.</div>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th>Sütun</th>
                      <th>Çıkarılma Sebebi</th>
                      <th>Aşama</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>reservation_status</td><td>Veri sızıntısı — iptal durumunu doğrudan açıklar</td><td>Önişleme (B adımı)</td></tr>
                    <tr><td>reservation_status_date</td><td>Veri sızıntısı — iptal tarihini içerir</td><td>Önişleme (B adımı)</td></tr>
                    <tr><td>is_canceled</td><td>Hedef değişken — özellik olarak kullanılmaz</td><td>Feature Engineering</td></tr>
                    <tr><td>%100 NaN sütunlar</td><td>Herhangi bir bilgi taşımadıkları için</td><td>Önişleme (D adımı)</td></tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* === Dosya Haritası === */}
            <section className="card">
              <div className="small">📂 Pipeline Kaynak Dosya Haritası</div>
              <div className="explain">Her aşamanın hangi Python dosyasında tanımlı olduğu.</div>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr><th>Aşama</th><th>Dosya</th><th>Giriş</th><th>Çıkış</th></tr>
                  </thead>
                  <tbody>
                    <tr><td>Doğrulama</td><td>src/data_validation.py</td><td>hotel_bookings.csv</td><td>Doğrulanmış DataFrame</td></tr>
                    <tr><td>Önişleme</td><td>src/preprocess.py</td><td>hotel_bookings.csv</td><td>data/processed/dataset.parquet</td></tr>
                    <tr><td>Veri Bölme</td><td>src/split.py</td><td>dataset.parquet</td><td>train.parquet, cal.parquet, test.parquet</td></tr>
                    <tr><td>Feature Eng.</td><td>src/features.py</td><td>train.parquet</td><td>ColumnTransformer (Pipeline içinde)</td></tr>
                    <tr><td>Eğitim</td><td>src/train.py</td><td>train.parquet, cal.parquet</td><td>models/*.joblib</td></tr>
                    <tr><td>Kalibrasyon</td><td>src/calibration.py</td><td>cal.parquet + ham model</td><td>*_calibrated_*.joblib</td></tr>
                    <tr><td>Değerlendirme</td><td>src/evaluate.py</td><td>test.parquet + modeller</td><td>reports/metrics/*.json</td></tr>
                    <tr><td>Politika</td><td>src/policy.py</td><td>Metrikler + tercih listesi</td><td>decision_policy.json</td></tr>
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}

        {/* ===============================================================
            SAYFA 4: KOŞU GEÇMİŞİ — Run Kayıtları
            =============================================================== */}
        {activePage === 'runs' && (
          <>
            <header className="pageHeader">
              <h1>📁 Koşu Geçmişi</h1>
              <p className="subtitle">
                Her "koşu" bir model eğitim + değerlendirme + seçim döngüsünü temsil eder.
                Toplam {runs.length} koşu kaydı bulunuyor. Bir koşuya tıklayarak detaylarını "Genel Bakış" sayfasında inceleyebilirsiniz.
              </p>
            </header>

            <section className="card">
              <div className="small">Koşu Kayıtları ({runs.length} adet)</div>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th style={{width:30}}>#</th>
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
                    {runs.map((r, i) => {
                      const dbInfo = dbRuns.find(d => d.run_id === r);
                      const isCurrent = r === selectedRun;
                      return (
                        <tr
                          key={r}
                          style={{ cursor: 'pointer', background: isCurrent ? '#e0f0ff' : undefined, fontWeight: isCurrent ? 600 : 400 }}
                          onClick={() => { setSelectedRun(r); refreshOverviewOnly(r); setActivePage('overview'); }}
                          title="Tıklayarak bu koşunun detaylarını görüntüleyin"
                        >
                          <td style={{textAlign:'center'}}>{i + 1}</td>
                          <td>{formatRunId(r)}</td>
                          <td style={{fontFamily:'Consolas',fontSize:10}}>{r}</td>
                          <td>{dbInfo?.selected_model ? `${modelIcon(dbInfo.selected_model)} ${displayName(dbInfo.selected_model)}` : <span style={{color:'#999'}}>—</span>}</td>
                          <td style={{fontFamily:'Consolas'}}>{dbInfo?.threshold != null ? f(dbInfo.threshold, 3) : '—'}</td>
                          <td style={{fontFamily:'Consolas',textAlign:'right'}}>{dbInfo?.expected_net_profit != null ? money(dbInfo.expected_net_profit) : '—'}</td>
                          <td>{dbInfo?.max_action_rate != null ? pct(dbInfo.max_action_rate) : '—'}</td>
                          <td>
                            {isCurrent
                              ? <span className="statusBadge ok" style={{fontSize:10}}>◄ Görüntüleniyor</span>
                              : dbInfo?.selected_model
                                ? <span style={{color:'#006600',fontSize:10}}>✓ Tamamlandı</span>
                                : <span style={{color:'#999',fontSize:10}}>Veri yok</span>
                            }
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
                  <li>"Seçilen Model" sütunu, o koşuda şampiyon seçilen modeli gösterir.</li>
                  <li>"Net Kazanç" sütunu, modelin maliyet matrisine göre hesaplanan beklenen toplam faydadır.</li>
                  <li>Koşu kimliği (Run ID) tarih_saat formatındadır: YYYYAAGG_SSddss</li>
                </ul>
              </div>
            </section>
          </>
        )}

        {/* ===============================================================
            SAYFA 5: CHAT ASİSTANI
            =============================================================== */}
        {activePage === 'chat' && (
          <>
            <header className="pageHeader">
              <h1>💬 Chat Asistanı — İptal Azaltma</h1>
              <p className="subtitle">
                Önce müşteri formunu doldurun, ardından chat oturumunu başlatın.
                Asistan müşteri profiline göre somut aksiyon önerileri sunar.
              </p>
            </header>

            <section className="card chatGrid">
              <div>
                <div className="small">Müşteri Formu</div>
                <div className="chatFormGrid">
                  <div>
                    <label>Otel</label>
                    <select value={chatCustomer.hotel} onChange={e => handleChatCustomerChange('hotel', e.target.value)}>
                      <option value="City Hotel">City Hotel</option>
                      <option value="Resort Hotel">Resort Hotel</option>
                    </select>
                  </div>
                  <div>
                    <label>Lead Time (gün)</label>
                    <input type="number" min="0" value={chatCustomer.lead_time} onChange={e => handleChatCustomerChange('lead_time', e.target.value)} />
                  </div>
                  <div>
                    <label>Depozito</label>
                    <select value={chatCustomer.deposit_type} onChange={e => handleChatCustomerChange('deposit_type', e.target.value)}>
                      <option value="No Deposit">No Deposit</option>
                      <option value="Non Refund">Non Refund</option>
                      <option value="Refundable">Refundable</option>
                    </select>
                  </div>
                  <div>
                    <label>Market Segment</label>
                    <select value={chatCustomer.market_segment} onChange={e => handleChatCustomerChange('market_segment', e.target.value)}>
                      <option value="Online TA">Online TA</option>
                      <option value="Direct">Direct</option>
                      <option value="Corporate">Corporate</option>
                      <option value="Groups">Groups</option>
                    </select>
                  </div>
                  <div>
                    <label>Yetişkin</label>
                    <input type="number" min="1" value={chatCustomer.adults} onChange={e => handleChatCustomerChange('adults', e.target.value)} />
                  </div>
                  <div>
                    <label>Çocuk</label>
                    <input type="number" min="0" value={chatCustomer.children} onChange={e => handleChatCustomerChange('children', e.target.value)} />
                  </div>
                  <div>
                    <label>Hafta içi gece</label>
                    <input type="number" min="0" value={chatCustomer.stays_in_week_nights} onChange={e => handleChatCustomerChange('stays_in_week_nights', e.target.value)} />
                  </div>
                  <div>
                    <label>Hafta sonu gece</label>
                    <input type="number" min="0" value={chatCustomer.stays_in_weekend_nights} onChange={e => handleChatCustomerChange('stays_in_weekend_nights', e.target.value)} />
                  </div>
                  <div>
                    <label>Geçmiş İptal</label>
                    <input type="number" min="0" value={chatCustomer.previous_cancellations} onChange={e => handleChatCustomerChange('previous_cancellations', e.target.value)} />
                  </div>
                  <div>
                    <label>Risk skoru (0-1)</label>
                    <input type="number" min="0" max="1" step="0.01" value={chatRiskScore} onChange={e => setChatRiskScore(e.target.value)} />
                  </div>
                </div>

                <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                  <button onClick={openChatSession} disabled={chatBusy}>
                    {chatBusy ? '⏳ Açılıyor...' : '🚀 Chat Oturumu Başlat'}
                  </button>
                  {chatSummary && (
                    <span className="metaItem">
                      <strong>Mesaj:</strong> {chatSummary.message_count}
                    </span>
                  )}
                </div>
              </div>

              <div>
                <div className="small">Sohbet</div>
                <div className="chatPanel">
                  {chatMessages.length === 0 && (
                    <div className="chatEmpty">Oturum başlatıldığında asistan mesajı burada görünecek.</div>
                  )}
                  {chatMessages.map((m, idx) => (
                    <div key={`${m.role}-${idx}`} className={`chatBubble ${m.role === 'user' ? 'user' : 'assistant'}`}>
                      <div className="chatRole">{m.role === 'user' ? 'Temsilci' : 'Asistan'}</div>
                      <div>{m.content}</div>
                    </div>
                  ))}
                </div>

                {chatQuickActions.length > 0 && (
                  <div className="chatQuickActions">
                    {chatQuickActions.map((a, idx) => (
                      <button key={`${a.label}-${idx}`} onClick={() => sendUserChatMessage(a.message)} disabled={chatBusy || !chatSessionId}>
                        {a.label}
                      </button>
                    ))}
                  </div>
                )}

                <form
                  className="chatComposer"
                  onSubmit={e => {
                    e.preventDefault();
                    sendUserChatMessage(chatInput);
                  }}
                >
                  <input
                    value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    placeholder="Örn: Bu müşteri için ilk adım ne olmalı?"
                    disabled={!chatSessionId}
                  />
                  <button type="submit" disabled={chatBusy || !chatSessionId || !chatInput.trim()}>
                    Gönder
                  </button>
                </form>

                {chatError && <div className="error" style={{ marginTop: 8 }}>{chatError}</div>}
              </div>
            </section>
          </>
        )}

        {/* ===============================================================
            SAYFA 4: SİSTEM DURUMU
            =============================================================== */}
        {activePage === 'system' && (
          <>
            <header className="pageHeader">
              <h1>🖥️ Sistem Durumu</h1>
              <p className="subtitle">
                Veritabanı bağlantısı, altyapı bilgileri ve maliyet matrisi parametreleri.
              </p>
            </header>

            <section className="card">
              <div className="small">Veritabanı Bağlantısı</div>
              <div className="systemGrid">
                <div className="sysItem">
                  <span>Veritabanı Motoru</span>
                  <strong>{dbStatus?.database_backend === 'sqlite' ? 'SQLite (Yerel)' : dbStatus?.database_backend === 'postgresql' ? 'PostgreSQL' : dbStatus?.database_backend || '-'}</strong>
                </div>
                <div className="sysItem">
                  <span>Bağlantı Durumu</span>
                  <strong style={{color: dbStatus?.connected ? '#006600' : '#cc0000'}}>
                    {dbStatus?.connected ? '● Bağlı — Sorunsuz' : '○ Bağlantı Yok'}
                  </strong>
                </div>
                <div className="sysItem full">
                  <span>Bağlantı Adresi (URL)</span>
                  <strong>{dbStatus?.database_url || '-'}</strong>
                </div>
                <div className="sysItem full">
                  <span>Durum Açıklaması</span>
                  <strong>{dbStatus?.reason === 'ok' ? 'Veritabanı sağlıklı çalışıyor.' : dbStatus?.reason || '-'}</strong>
                </div>
              </div>
              <button onClick={refreshDbStatus} disabled={loading}>
                {loading ? '⏳ Sorgulanıyor...' : '🔄 Bağlantıyı Test Et'}
              </button>
            </section>

            <section className="card">
              <div className="small">Maliyet Matrisi — Karar Parametreleri</div>
              <div className="explain">Bu değerler modelin "hangi müşteriye müdahale etmeli?" kararını şekillendirir.</div>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th>Senaryo</th>
                      <th>Kısaltma</th>
                      <th>Değer</th>
                      <th>Açıklama</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Doğru Pozitif</td>
                      <td style={{fontFamily:'Consolas'}}>TP</td>
                      <td style={{color:'#006600',fontWeight:700}}>+180 ₺</td>
                      <td>İptal edecek müşteriyi doğru tahmin ettik ve müdahale ile kurtardık</td>
                    </tr>
                    <tr>
                      <td>Yanlış Pozitif</td>
                      <td style={{fontFamily:'Consolas'}}>FP</td>
                      <td style={{color:'#cc0000',fontWeight:700}}>−20 ₺</td>
                      <td>İptal etmeyecek müşteriye gereksiz yere müdahale ettik (kampanya maliyeti)</td>
                    </tr>
                    <tr>
                      <td>Yanlış Negatif</td>
                      <td style={{fontFamily:'Consolas'}}>FN</td>
                      <td style={{color:'#cc0000',fontWeight:700}}>−200 ₺</td>
                      <td>İptal edecek müşteriyi kaçırdık, rezervasyon kaybedildi</td>
                    </tr>
                    <tr>
                      <td>Doğru Negatif</td>
                      <td style={{fontFamily:'Consolas'}}>TN</td>
                      <td style={{color:'#666'}}>0 ₺</td>
                      <td>İptal etmeyecek müşteriyi doğru tahmin ettik, ek işlem yok</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            <section className="card">
              <div className="small">Genel Bilgiler</div>
              <div className="systemGrid">
                <div className="sysItem">
                  <span>Toplam Koşu Sayısı</span>
                  <strong>{runs.length}</strong>
                </div>
                <div className="sysItem">
                  <span>DB Kayıtlı Koşu</span>
                  <strong>{dbRuns.length}</strong>
                </div>
                <div className="sysItem">
                  <span>Aktif Run ID</span>
                  <strong>{selectedRun || '-'}</strong>
                </div>
                <div className="sysItem">
                  <span>Güncel Şampiyon</span>
                  <strong>{displayName(champion.selected_model)}</strong>
                </div>
              </div>
            </section>
          </>
        )}

        {/* Alt Durum Çubuğu */}
        <div className="appStatusBar">
          <span>{loading ? '⏳ İşlem devam ediyor...' : '✓ Hazır'}</span>
          <span>Model: {coreModels.length} temel</span>
          <span>Koşu: {runs.length} kayıt</span>
          <span>{now()}</span>
        </div>
      </main>
    </div>
  );
}
