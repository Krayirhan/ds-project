import { useState, useEffect, useCallback } from 'react';
import { useLayoutContext } from './Layout';
import { useChat } from '../hooks/useChat';
import { createGuest, listGuests, deleteGuest, getAvailableModels } from '../api';

// ─── Sabitler ────────────────────────────────────────────────────────────────
const HOTELS        = ['City Hotel', 'Resort Hotel'];
const DEPOSIT_TYPES = ['No Deposit', 'Non Refund', 'Refundable'];
const SEGMENTS      = ['Online TA', 'Direct', 'Corporate', 'Groups', 'Offline TA/TO'];
const GENDERS       = [
  { value: '', label: 'Belirtilmedi' },
  { value: 'M', label: 'Erkek' },
  { value: 'F', label: 'Kadın' },
  { value: 'other', label: 'Diğer' },
];
// Kullanıcı dostu model isimleri
const MODEL_LABELS = {
  'baseline':                          'Logistic Regression (Temel)',
  'baseline_calibrated_sigmoid':       'Logistic Regression + Sigmoid Kalibrasyon',
  'baseline_calibrated_isotonic':      'Logistic Regression + Isotonic Kalibrasyon',
  'challenger_xgboost':                'XGBoost',
  'challenger_xgboost_calibrated_sigmoid':  'XGBoost + Sigmoid Kalibrasyon',
  'challenger_xgboost_calibrated_isotonic': 'XGBoost + Isotonic Kalibrasyon',
};

const INITIAL_FORM = {
  first_name: '', last_name: '', email: '', phone: '',
  nationality: '', identity_no: '', birth_date: '', gender: '',
  vip_status: false, notes: '',
  hotel: 'City Hotel', lead_time: 30, deposit_type: 'No Deposit',
  market_segment: 'Online TA', adults: 2, children: 0, babies: 0,
  stays_in_week_nights: 2, stays_in_weekend_nights: 1,
  is_repeated_guest: 0, previous_cancellations: 0, adr: '',
};

// ─── Yardımcı bileşenler ─────────────────────────────────────────────────────
function RiskCard({ predicting, riskScore, riskLabel }) {
  const cardMod  = predicting ? '' : riskLabel === 'high' ? 'riskHigh' : riskLabel === 'medium' ? 'riskMed' : riskScore !== null ? 'riskLow' : '';
  const labelMod = cardMod;
  const icon = predicting ? '⏳' : riskLabel === 'high' ? '🔴' : riskLabel === 'medium' ? '🟡' : riskScore !== null ? '🟢' : '❓';
  const label = predicting
    ? 'Hesaplanıyor…'
    : riskScore !== null
      ? `%${Math.round(riskScore * 100)} — ${riskLabel === 'high' ? 'YÜKSEK RİSK' : riskLabel === 'medium' ? 'ORTA RİSK' : 'DÜŞÜK RİSK'}`
    : 'Rezervasyon bilgilerini girin';
  return (
    <div className={`riskCard ${cardMod}`}>
      <span style={{ fontSize: 18 }}>{icon}</span>
      <div>
        <div className="riskCardHint">Tahmini iptal riski</div>
        <div className={`riskCardLabel ${labelMod}`}>{label}</div>
      </div>
    </div>
  );
}

function RiskBadge({ label, score }) {
  if (!label) return <span className="textMuted" style={{ fontSize: 12 }}>—</span>;
  const mod  = label === 'high' ? 'riskHigh' : label === 'medium' ? 'riskMed' : 'riskLow';
  const text = label === 'high' ? 'YÜKSEK' : label === 'medium' ? 'ORTA' : 'DÜŞÜK';
  return (
    <span className={`riskBadge ${mod}`}>
      {text}{score != null ? ` %${Math.round(score * 100)}` : ''}
    </span>
  );
}

/** Satır içi **bold** → <strong> dönüşümü */
function InlineText({ text }) {
  const parts = text.split(/(\*\*[\s\S]*?\*\*)/g);
  return (
    <>
      {parts.map((part, i) =>
        part.startsWith('**') && part.endsWith('**') && part.length > 4
          ? <strong key={i} style={{ fontWeight: 700 }}>{part.slice(2, -2)}</strong>
          : <span key={i}>{part}</span>
      )}
    </>
  );
}

/**
 * Markdown metni → JSX
 * Desteklenen formatlar:
 *   • "1. madde\n2. madde" — satır başı numaralı liste
 *   • "1. madde 2. madde 3. madde" — tek satırda bitişik liste (LLM çıktısı)
 *   • **bold** — satır içi kalın
 *   • **İlk Adım:** gibi satır başı başlık
 */
function MarkdownText({ content }) {
  if (!content) return null;

  // Önce \r\n → \n normalize et
  const normalized = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

  // Satırları al
  const rawLines = normalized.split('\n');

  // Her satırı, içinde "2. " veya "3. " gibi bitişik maddeler varsa böl
  // Örn: "1. **A:** foo 2. **B:** bar" → ["1. **A:** foo ", "2. **B:** bar"]
  const lines = [];
  for (const raw of rawLines) {
    // Eğer satırda birden fazla numaralı madde başlığı varsa böl
    const split = raw.split(/(?=\s\d+\.\s+\*{0,2}[A-ZÇĞİÖŞÜa-zçğışöşü])/);
    for (const seg of split) {
      const s = seg.trim();
      if (s) lines.push(s);
    }
  }

  const elements = [];
  const listItems = [];
  let k = 0;

  function flushList() {
    if (listItems.length === 0) return;
    elements.push(
      <ol key={k++} style={{ paddingLeft: 20, margin: '4px 0', lineHeight: 1.7 }}>
        {listItems.map((item, i) => (
          <li key={i} style={{ marginBottom: 4 }}><InlineText text={item} /></li>
        ))}
      </ol>
    );
    listItems.length = 0;
  }

  for (const line of lines) {
    const listMatch = line.match(/^\d+\.\s+(.+)/);
    if (listMatch) {
      listItems.push(listMatch[1]);
    } else {
      flushList();
      if (line.trim() === '') {
        if (elements.length > 0) elements.push(<div key={k++} style={{ height: 6 }} />);
      } else {
        elements.push(
          <p key={k++} style={{ margin: '3px 0', lineHeight: 1.6 }}>
            <InlineText text={line} />
          </p>
        );
      }
    }
  }
  flushList();

  return <div style={{ fontSize: 'inherit' }}>{elements}</div>;
}

const thS = 'guestTh';
const tdS = 'guestTd';

// ─── Ana bileşen ─────────────────────────────────────────────────────────────
/**
 * ChatPage — Misafir Yönetimi + Chat Asistanı
 *
 * Üst  : Misafir listesi (arama + tablo + sayfalama)
 * Alt  : chatGrid — Sol = TEK birleşik form (kişisel + rezervasyon + risk)
 *                   Sağ = Sohbet paneli
 *
 * Tek form hem yeni kayıt hem de seçili misafirin bilgilerini gösterir.
 * formChange → chat.handleCustomerChange → risk otomatik güncellenir.
 */
export default function ChatPage() {
  const { runs, auth } = useLayoutContext();
  const apiKey = runs.apiKey;

  // ── Kullanılabilir modeller ─────────────────────────────────────────
  const [availableModels, setAvailableModels] = useState([]);

  useEffect(() => {
    if (!apiKey) return;
    getAvailableModels(apiKey).then(data => {
      setAvailableModels(data?.models ?? []);
    }).catch(() => {});
  }, [apiKey]);

  // ── Seçili misafir ──────────────────────────────────────────────────────────
  const [activeGuest, setActiveGuest] = useState(null);

  // ── Tek form state (kişisel + rezervasyon) ──────────────────────────────────
  const [form, setForm]           = useState(INITIAL_FORM);
  const [saving, setSaving]       = useState(false);
  const [saveError, setSaveError] = useState('');
  const [saveOk, setSaveOk]       = useState('');

  // ── Chat hook — form'dan beslenir ───────────────────────────────────────────
  const chat = useChat({
    apiKey,
    onAuthFailed: auth.handleAuthFailure,
    initialCustomer: INITIAL_FORM,
  });

  // Form alanı değişince hem form state'ini hem de chat.customer'ı güncelle
  function formChange(key, val) {
    setForm(prev => ({ ...prev, [key]: val }));
    chat.handleCustomerChange(key, val);
    setSaveOk(''); setSaveError('');
  }

  // Listeden misafir seç → formu ve chat'i doldur
  function selectGuest(g) {
    setActiveGuest(g);
    // null değerler controlled input'u uncontrolled'a döndürür → '' ile normalize et
    const sanitized = Object.fromEntries(
      Object.entries(g).map(([k, v]) => [k, v === null || v === undefined ? (INITIAL_FORM[k] ?? '') : v])
    );
    const data = { ...INITIAL_FORM, ...sanitized };
    setForm(data);
    Object.entries(data).forEach(([k, v]) => chat.handleCustomerChange(k, v ?? ''));
    chat.setGuestId(g.id ?? null);
    chat.setGuestSaved(!!g.id);
    setSaveOk(''); setSaveError('');
  }

  // Yeni misafir / formu temizle
  function clearForm() {
    setActiveGuest(null);
    setForm(INITIAL_FORM);
    Object.entries(INITIAL_FORM).forEach(([k, v]) => chat.handleCustomerChange(k, v));
    chat.setGuestId(null);
    chat.setGuestSaved(false);
    setSaveOk(''); setSaveError('');
  }

  // Sil → onay sor → API'ye gönder → listeden kaldır
  const [deleting, setDeleting] = useState(null);       // silinen guest id
  const [confirmDelete, setConfirmDelete] = useState(null); // onay bekleyen guest id
  async function handleDelete(g) {
    setDeleting(g.id);
    try {
      await deleteGuest(g.id, apiKey);
      if (activeGuest?.id === g.id) clearForm();
      await loadGuests(search, offset);
    } catch (err) {
      setSaveError('Silme hatası: ' + (err.message || 'Bilinmeyen hata'));
    } finally { setDeleting(null); setConfirmDelete(null); }
  }

  // Kaydet → DB'ye yaz → seç
  async function handleSave(e) {
    e.preventDefault();
    if (!form.first_name.trim() || !form.last_name.trim()) {
      setSaveError('Ad ve soyad zorunludur.'); return;
    }
    setSaving(true); setSaveError(''); setSaveOk('');
    try {
      const payload = {
        ...form,
        lead_time:               Number(form.lead_time || 0),
        adults:                  Number(form.adults || 1),
        children:                Number(form.children || 0),
        babies:                  Number(form.babies || 0),
        stays_in_week_nights:    Number(form.stays_in_week_nights || 0),
        stays_in_weekend_nights: Number(form.stays_in_weekend_nights || 0),
        is_repeated_guest:       Number(form.is_repeated_guest || 0),
        previous_cancellations:  Number(form.previous_cancellations || 0),
        adr:         form.adr !== '' ? Number(form.adr) : null,
        birth_date:  form.birth_date  || null,
        gender:      form.gender      || null,
        nationality: form.nationality || null,
        identity_no: form.identity_no || null,
        email:       form.email       || null,
        phone:       form.phone       || null,
        notes:       form.notes       || null,
      };
      const saved = await createGuest(payload, apiKey);
      selectGuest(saved);
      chat.setGuestId(saved.id);
      chat.setGuestSaved(true);
      setSaveOk(`✅ ${saved.first_name} ${saved.last_name} kaydedildi.`);
      await loadGuests('', 0);
    } catch (err) {
      setSaveError(err.message || 'Kayıt sırasında hata oluştu.');
    } finally { setSaving(false); }
  }

  // ── Misafir listesi ─────────────────────────────────────────────────────────
  const PAGE_SIZE = 10;
  const [guests, setGuests]       = useState([]);
  const [total, setTotal]         = useState(0);
  const [search, setSearch]       = useState('');
  const [offset, setOffset]       = useState(0);
  const [loading, setLoading]     = useState(false);
  const [listError, setListError] = useState('');

  const loadGuests = useCallback(async (q, off) => {
    setLoading(true); setListError('');
    try {
      const res = await listGuests({ search: q || undefined, limit: PAGE_SIZE, offset: off }, apiKey);
      setGuests(res.items || []);
      setTotal(res.total || 0);
    } catch (e) {
      setListError(e.message || 'Misafir listesi alınamadı.');
    } finally { setLoading(false); }
  }, [apiKey]);

  // İlk yükleme
  useEffect(() => { loadGuests('', 0); }, [apiKey, loadGuests]);

  function handleSearch(e) { const q = e.target.value; setSearch(q); setOffset(0); loadGuests(q, 0); }
  function prevPage() { const o = Math.max(0, offset - PAGE_SIZE); setOffset(o); loadGuests(search, o); }
  function nextPage() { if (offset + PAGE_SIZE < total) { const o = offset + PAGE_SIZE; setOffset(o); loadGuests(search, o); } }

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <>
      <header className="pageHeader">
        <h1>🏨 Misafir & Chat Asistanı</h1>
        <p className="subtitle">
          Listeden misafir seçin veya formu doldurup kaydedin — ardından chat oturumu başlatın.
        </p>
      </header>

      {/* ══ Misafir Listesi ═════════════════════════════════════════════════════ */}
      <section className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
          <div className="small" style={{ flex: 1 }}>Kayıtlı Misafirler ({total})</div>
          <input
            value={search} onChange={handleSearch}
            placeholder="🔍 İsim veya e-posta ara…"
            style={{ width: 220 }}
          />
          <button onClick={clearForm}>＋ Yeni Misafir</button>
        </div>

        {listError && <div className="error" style={{ marginBottom: 8 }}>{listError}</div>}
        {loading && <div style={{ textAlign: 'center', padding: 16 }} className="textMuted">⏳ Yükleniyor…</div>}
        {!loading && guests.length === 0 && (
          <div style={{ textAlign: 'center', padding: 20 }} className="textMuted">
            {search ? 'Arama sonucu bulunamadı.' : 'Henüz kayıtlı misafir yok.'}
          </div>
        )}
        {!loading && guests.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th className={thS}>Ad Soyad</th>
                  <th className={thS}>E-posta / Tel</th>
                  <th className={thS}>Otel</th>
                  <th className={thS}>Segment</th>
                  <th className={thS}>Risk</th>
                  <th className={thS}>VIP</th>
                  <th className={thS}>Kayıt</th>
                  <th className={thS}>Seç</th>
                  <th className={thS}>Sil</th>
                </tr>
              </thead>
              <tbody>
                {guests.map(g => {
                  const isActive = activeGuest?.id === g.id;
                  return (
                    <tr key={g.id} className={isActive ? 'guestRowActive' : undefined}>
                      <td className={tdS}>
                        <strong>{g.first_name} {g.last_name}</strong>
                        {g.nationality && <span className="guestSecondary" style={{ marginLeft: 4 }}>({g.nationality})</span>}
                      </td>
                      <td className={tdS}>
                        <div>{g.email || <span className="textMuted">—</span>}</div>
                        <div className="guestSecondary">{g.phone || ''}</div>
                      </td>
                      <td className={tdS}>{g.hotel}</td>
                      <td className={tdS}>{g.market_segment}</td>
                      <td className={tdS}><RiskBadge label={g.risk_label} score={g.risk_score} /></td>
                      <td className={tdS} style={{ textAlign: 'center' }}>{g.vip_status ? '⭐' : '—'}</td>
                      <td className={`${tdS} guestSecondary`}>
                        {g.created_at ? new Date(g.created_at).toLocaleDateString('tr-TR') : '—'}
                      </td>
                      <td className={tdS}>
                        <button
                          className={isActive ? 'btnPrimary' : undefined}
                          style={{ padding: '3px 12px', fontSize: 12 }}
                          onClick={() => selectGuest(g)}
                        >
                          {isActive ? '✓ Seçildi' : '💬 Seç'}
                        </button>
                      </td>
                      <td className={tdS}>
                        {confirmDelete === g.id ? (
                          <span style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                            <span className="textDanger" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>Emin misin?</span>
                            <button
                              className="btnDanger"
                              onClick={() => handleDelete(g)}
                              disabled={deleting === g.id}
                              style={{ padding: '2px 8px', fontSize: 11 }}
                            >
                              {deleting === g.id ? '⏳' : '✔ Evet'}
                            </button>
                            <button
                              className="btnNeutral"
                              onClick={() => setConfirmDelete(null)}
                              style={{ padding: '2px 8px', fontSize: 11 }}
                            >
                              ✖ İptal
                            </button>
                          </span>
                        ) : (
                          <button
                            className="btnDanger"
                            onClick={() => setConfirmDelete(g.id)}
                            disabled={deleting === g.id}
                            style={{ padding: '3px 10px', fontSize: 12 }}
                          >
                            🗑
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {total > PAGE_SIZE && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 10, fontSize: 13 }}>
            <button onClick={prevPage} disabled={offset === 0} style={{ padding: '4px 12px' }}>← Önceki</button>
            <span className="textMuted">{offset + 1}–{Math.min(offset + PAGE_SIZE, total)} / {total}</span>
            <button onClick={nextPage} disabled={offset + PAGE_SIZE >= total} style={{ padding: '4px 12px' }}>Sonraki →</button>
          </div>
        )}
      </section>

      {/* ══ Form + Chat ══════════════════════════════════════════════════════════ */}
      <section className="card chatGrid">

        {/* Sol: TEK BİRLEŞİK FORM ─────────────────────────────────────────── */}
        <div>
          <div className="small" style={{ marginBottom: 10 }}>
            {activeGuest
              ? <>Misafir: <strong>{activeGuest.first_name} {activeGuest.last_name}</strong>
                  <span className="textMuted" style={{ fontWeight: 400 }}> #{activeGuest.id}</span></>
              : 'Yeni Misafir'}
          </div>

          <form onSubmit={handleSave}>
            {/* Kişisel Bilgiler */}
            <div className="textMuted" style={{ fontSize: 11, marginBottom: 6 }}>👤 Kişisel Bilgiler</div>
            <div className="chatFormGrid">
              <div>
                <label>Ad *</label>
                <input value={form.first_name} onChange={e => formChange('first_name', e.target.value)} placeholder="Ahmet" required />
              </div>
              <div>
                <label>Soyad *</label>
                <input value={form.last_name} onChange={e => formChange('last_name', e.target.value)} placeholder="Yılmaz" required />
              </div>
              <div>
                <label>E-posta</label>
                <input type="email" value={form.email} onChange={e => formChange('email', e.target.value)} placeholder="ornek@mail.com" />
              </div>
              <div>
                <label>Telefon</label>
                <input type="tel" value={form.phone} onChange={e => formChange('phone', e.target.value)} placeholder="+90 555 000 00 00" />
              </div>
              <div>
                <label>Uyruk (ISO-3)</label>
                <input value={form.nationality} onChange={e => formChange('nationality', e.target.value.toUpperCase())} placeholder="TUR" maxLength={3} />
              </div>
              <div>
                <label>TC / Pasaport No</label>
                <input value={form.identity_no} onChange={e => formChange('identity_no', e.target.value)} placeholder="12345678901" />
              </div>
              <div>
                <label>Doğum Tarihi</label>
                <input type="date" value={form.birth_date} onChange={e => formChange('birth_date', e.target.value)} />
              </div>
              <div>
                <label>Cinsiyet</label>
                <select value={form.gender} onChange={e => formChange('gender', e.target.value)}>
                  {GENDERS.map(g => <option key={g.value} value={g.value}>{g.label}</option>)}
                </select>
              </div>
            </div>
            <div style={{ margin: '6px 0' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 13 }}>
                <input type="checkbox" checked={form.vip_status} onChange={e => formChange('vip_status', e.target.checked)} />
                ⭐ VIP Misafir
              </label>
            </div>
            <div style={{ marginBottom: 10 }}>
              <label>Notlar</label>
              <textarea value={form.notes} onChange={e => formChange('notes', e.target.value)}
                rows={2} placeholder="Özel istek…"
                style={{ width: '100%', resize: 'vertical', boxSizing: 'border-box' }} />
            </div>

            {/* Rezervasyon Bilgileri */}
            <div className="textMuted" style={{ fontSize: 11, marginBottom: 6 }}>📋 Rezervasyon Bilgileri</div>
            <div className="chatFormGrid">
              <div>
                <label>Otel</label>
                <select value={form.hotel} onChange={e => formChange('hotel', e.target.value)}>
                  {HOTELS.map(h => <option key={h}>{h}</option>)}
                </select>
              </div>
              <div>
                <label>Lead Time (gün)</label>
                <input type="number" min="0" value={form.lead_time} onChange={e => formChange('lead_time', e.target.value)} />
              </div>
              <div>
                <label>Depozito</label>
                <select value={form.deposit_type} onChange={e => formChange('deposit_type', e.target.value)}>
                  {DEPOSIT_TYPES.map(d => <option key={d}>{d}</option>)}
                </select>
              </div>
              <div>
                <label>Market Segment</label>
                <select value={form.market_segment} onChange={e => formChange('market_segment', e.target.value)}>
                  {SEGMENTS.map(s => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label>Yetişkin</label>
                <input type="number" min="1" value={form.adults} onChange={e => formChange('adults', e.target.value)} />
              </div>
              <div>
                <label>Çocuk</label>
                <input type="number" min="0" value={form.children} onChange={e => formChange('children', e.target.value)} />
              </div>
              <div>
                <label>Bebek</label>
                <input type="number" min="0" value={form.babies} onChange={e => formChange('babies', e.target.value)} />
              </div>
              <div>
                <label>Hafta içi gece</label>
                <input type="number" min="0" value={form.stays_in_week_nights} onChange={e => formChange('stays_in_week_nights', e.target.value)} />
              </div>
              <div>
                <label>Hafta sonu gece</label>
                <input type="number" min="0" value={form.stays_in_weekend_nights} onChange={e => formChange('stays_in_weekend_nights', e.target.value)} />
              </div>
              <div>
                <label>Sadık Müşteri</label>
                <select value={form.is_repeated_guest} onChange={e => formChange('is_repeated_guest', e.target.value)}>
                  <option value={0}>Hayır</option>
                  <option value={1}>Evet</option>
                </select>
              </div>
              <div>
                <label>Geçmiş İptal</label>
                <input type="number" min="0" value={form.previous_cancellations} onChange={e => formChange('previous_cancellations', e.target.value)} />
              </div>
              <div>
                <label>ADR (Ort. Ücret)</label>
                <input type="number" min="0" step="0.01" value={form.adr} onChange={e => formChange('adr', e.target.value)} placeholder="isteğe bağlı" />
              </div>
            </div>

            {/* Risk & Model Seçimi */}
            <div style={{ marginTop: 10, marginBottom: 10 }}>
              {availableModels.length > 0 && (
                <div className="modelSelectorWrap">
                  <label className="modelSelectorLabel">🤖 Tahmin Modeli</label>
                  <select
                    className="modelSelectorSelect"
                    value={chat.selectedModel || ''}
                    onChange={e => chat.setSelectedModel(e.target.value || null)}
                  >
                    <option value=''>🏆 Varsayılan (Aktif Şampiyon)</option>
                    {availableModels.map(m => (
                      <option key={m.name} value={m.name}>
                        {m.is_active ? '★ ' : ''}{MODEL_LABELS[m.name] || m.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              <RiskCard predicting={chat.predicting} riskScore={chat.riskScore} riskLabel={chat.riskLabel} />
            </div>

            {/* Eylem butonları */}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <button type="submit" disabled={saving || chat.predicting}>
                {saving ? '⏳ Kaydediliyor…' : '💾 Kaydet'}
              </button>
              <button type="button" className="btnGhost" onClick={clearForm}>
                Temizle
              </button>
              <button type="button" className="btnSuccess"
                onClick={chat.openSession}
                disabled={chat.busy || chat.predicting}>
                {chat.busy ? '⏳ Açılıyor…' : '🚀 Chat Oturumu Başlat'}
              </button>
            </div>

            {saveOk    && <div className="formSuccess">{saveOk}</div>}
            {saveError && <div className="error" style={{ marginTop: 8 }}>{saveError}</div>}
            {chat.guestSaved && (
              <div className="formSuccess">
                ✅ Misafir #{chat.guestId} oturuma bağlandı
              </div>
            )}
            {chat.summary && (
              <div className="formNote">
                Mesaj sayısı: {chat.summary.message_count}
              </div>
            )}
          </form>
        </div>

        {/* Sağ: Sohbet paneli ─────────────────────────────────────────────── */}
        <div>
          <div className="small">Sohbet</div>
          <div className="chatPanel" aria-live="polite" aria-label="Chat mesajları">
            {chat.messages.length === 0 && (
              <div className="chatEmpty">
                {activeGuest
                  ? `${activeGuest.first_name} ${activeGuest.last_name} için oturum başlatın.`
                  : 'Formu doldurun ve chat oturumu başlatın.'}
              </div>
            )}
            {chat.messages.map((m, idx) => (
              <div key={m.id ?? `${m.role}-${idx}`}
                className={`chatBubble ${m.role === 'user' ? 'user' : 'assistant'}`}>
                <div className="chatRole">{m.role === 'user' ? 'Temsilci' : 'Asistan'}</div>
                {m.role === 'assistant'
                  ? m.streaming
                    ? <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, wordBreak: 'break-word' }}>
                        {m.content || '\u00a0'}
                        <span style={{ display: 'inline-block', opacity: 0.7, marginLeft: 1 }}>▮</span>
                      </div>
                    : <MarkdownText content={m.content} />
                  : <div>{m.content}</div>}
              </div>
            ))}
          </div>

          {chat.quickActions.length > 0 && (
            <div className="chatQuickActions">
              {chat.quickActions.map((a, idx) => (
                <button key={`${a.label}-${idx}`}
                  onClick={() => chat.sendMessage(a.message)}
                  disabled={chat.busy || !chat.sessionId}>
                  {a.label}
                </button>
              ))}
            </div>
          )}

          <form className="chatComposer"
            onSubmit={e => { e.preventDefault(); chat.sendMessage(chat.input); }}>
            <input
              value={chat.input}
              onChange={e => chat.setInput(e.target.value)}
              placeholder="Örn: Bu müşteri için ilk adım ne olmalı?"
              disabled={!chat.sessionId}
              aria-label="Chat mesajı yaz"
            />
            <button type="submit" disabled={chat.busy || !chat.sessionId || !chat.input.trim()}>
              Gönder
            </button>
          </form>

          {chat.error && (
            <div className="error" role="alert" style={{ marginTop: 8 }}>{chat.error}</div>
          )}
        </div>
      </section>
    </>
  );
}
