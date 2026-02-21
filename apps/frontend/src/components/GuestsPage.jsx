import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLayoutContext } from './Layout';
import { createGuest, listGuests, predictRiskScore } from '../api';

/**
 * GuestsPage — Misafir Yönetimi
 *
 * Sol panel: Yeni misafir kayıt formu (kişisel bilgi + rezervasyon bilgileri)
 * Sağ panel: Kayıtlı misafir listesi (arama + sayfalama)
 *
 * Booking alanları değişince iptal riski otomatik hesaplanır (debounced 600ms).
 * Kayıt butonuna basılınca tüm bilgiler DB'ye gönderilir.
 */

const HOTELS         = ['City Hotel', 'Resort Hotel'];
const DEPOSIT_TYPES  = ['No Deposit', 'Non Refund', 'Refundable'];
const SEGMENTS       = ['Online TA', 'Direct', 'Corporate', 'Groups', 'Offline TA/TO'];
const GENDERS        = [{ value: '', label: 'Belirtilmedi' }, { value: 'M', label: 'Erkek' }, { value: 'F', label: 'Kadın' }, { value: 'other', label: 'Diğer' }];

const INITIAL_FORM = {
  // Personal
  first_name:  '',
  last_name:   '',
  email:       '',
  phone:       '',
  nationality: '',
  identity_no: '',
  birth_date:  '',
  gender:      '',
  vip_status:  false,
  notes:       '',
  // Booking / model
  hotel:                   'City Hotel',
  lead_time:               30,
  deposit_type:            'No Deposit',
  market_segment:          'Online TA',
  adults:                  2,
  children:                0,
  babies:                  0,
  stays_in_week_nights:    2,
  stays_in_weekend_nights: 1,
  is_repeated_guest:       0,
  previous_cancellations:  0,
  adr:                     '',
};

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

export default function GuestsPage() {
  const { runs } = useLayoutContext();
  const apiKey   = runs.apiKey;
  const navigate = useNavigate();

  // ── Form state ────────────────────────────────────────────────────────────
  const [form, setForm]           = useState(INITIAL_FORM);
  const [saving, setSaving]       = useState(false);
  const [saveError, setSaveError] = useState('');
  const [saveOk, setSaveOk]       = useState('');

  // ── Auto risk prediction ──────────────────────────────────────────────────
  const [riskScore, setRiskScore]     = useState(null);
  const [riskLabel, setRiskLabel]     = useState('unknown');
  const [predicting, setPredicting]   = useState(false);
  const predictAbort  = useRef(null);
  const debounceTimer = useRef(null);

  const bookingSnapshot = JSON.stringify({
    hotel: form.hotel, lead_time: form.lead_time, deposit_type: form.deposit_type,
    market_segment: form.market_segment, adults: form.adults, children: form.children,
    stays_in_week_nights: form.stays_in_week_nights,
    stays_in_weekend_nights: form.stays_in_weekend_nights,
    is_repeated_guest: form.is_repeated_guest,
    previous_cancellations: form.previous_cancellations,
  });

  useEffect(() => {
    clearTimeout(debounceTimer.current);
    predictAbort.current?.abort();
    debounceTimer.current = setTimeout(async () => {
      const ctrl = new AbortController();
      predictAbort.current = ctrl;
      setPredicting(true);
      try {
        const result = await predictRiskScore({
          hotel:                   form.hotel,
          lead_time:               Number(form.lead_time || 0),
          deposit_type:            form.deposit_type,
          market_segment:          form.market_segment,
          adults:                  Number(form.adults || 1),
          children:                Number(form.children || 0),
          stays_in_week_nights:    Number(form.stays_in_week_nights || 0),
          stays_in_weekend_nights: Number(form.stays_in_weekend_nights || 0),
          previous_cancellations:  Number(form.previous_cancellations || 0),
          is_repeated_guest:       Number(form.is_repeated_guest || 0),
        }, apiKey, { signal: ctrl.signal });
        setRiskScore(result.risk_score);
        setRiskLabel(result.risk_label);
      } catch (e) {
        if (e.name !== 'AbortError') { setRiskScore(null); setRiskLabel('unknown'); }
      } finally {
        setPredicting(false);
      }
    }, 600);
    return () => clearTimeout(debounceTimer.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bookingSnapshot, apiKey]);

  // ── Guest list state ──────────────────────────────────────────────────────
  const [guests, setGuests]       = useState([]);
  const [total, setTotal]         = useState(0);
  const [search, setSearch]       = useState('');
  const [offset, setOffset]       = useState(0);
  const PAGE_SIZE = 20;
  const [loading, setLoading]     = useState(false);
  const [listError, setListError] = useState('');

  const loadGuests = useCallback(async (q = search, off = offset) => {
    setLoading(true);
    setListError('');
    try {
      const res = await listGuests({ search: q || undefined, limit: PAGE_SIZE, offset: off }, apiKey);
      setGuests(res.items || []);
      setTotal(res.total || 0);
    } catch (e) {
      setListError(e.message || 'Misafir listesi alınamadı.');
    } finally {
      setLoading(false);
    }
  }, [apiKey, search, offset]);

  useEffect(() => { loadGuests(); }, [apiKey]); // initial load

  // ── Form helpers ──────────────────────────────────────────────────────────
  function change(key, value) {
    setForm(prev => ({ ...prev, [key]: value }));
    setSaveOk('');
    setSaveError('');
  }

  async function handleSave(e) {
    e.preventDefault();
    if (!form.first_name.trim() || !form.last_name.trim()) {
      setSaveError('Ad ve soyad zorunludur.');
      return;
    }
    setSaving(true);
    setSaveError('');
    setSaveOk('');
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
        adr:                     form.adr !== '' ? Number(form.adr) : null,
        birth_date:              form.birth_date || null,
        gender:                  form.gender || null,
        nationality:             form.nationality || null,
        identity_no:             form.identity_no || null,
        email:                   form.email || null,
        phone:                   form.phone || null,
        notes:                   form.notes || null,
      };
      await createGuest(payload, apiKey);
      setSaveOk('✅ Misafir başarıyla kaydedildi.');
      setForm(INITIAL_FORM);
      setRiskScore(null);
      setRiskLabel('unknown');
      // Refresh list
      setOffset(0);
      setSearch('');
      await loadGuests('', 0);
    } catch (err) {
      setSaveError(err.message || 'Kayıt sırasında hata oluştu.');
    } finally {
      setSaving(false);
    }
  }

  function handleSearch(e) {
    const q = e.target.value;
    setSearch(q);
    setOffset(0);
    loadGuests(q, 0);
  }

  function prevPage() {
    const newOff = Math.max(0, offset - PAGE_SIZE);
    setOffset(newOff);
    loadGuests(search, newOff);
  }

  function nextPage() {
    const newOff = offset + PAGE_SIZE;
    if (newOff < total) { setOffset(newOff); loadGuests(search, newOff); }
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <>
      <header className="pageHeader">
        <h1>🏨 Misafir Yönetimi</h1>
        <p className="subtitle">
          Yeni misafir kaydı oluşturun. Rezervasyon bilgilerinden iptal riski otomatik hesaplanır.
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>

        {/* ── Sol: Kayıt Formu ─────────────────────────────────────── */}
        <form className="card" onSubmit={handleSave}>
          <div className="small" style={{ marginBottom: 12 }}>Yeni Misafir Kaydı</div>

          {/* Kişisel Bilgiler */}
          <div className="textMuted" style={{ fontSize: 11, marginBottom: 6 }}>👤 Kişisel Bilgiler</div>
          <div className="chatFormGrid">
            <div>
              <label>Ad *</label>
              <input value={form.first_name} onChange={e => change('first_name', e.target.value)} placeholder="Ahmet" required />
            </div>
            <div>
              <label>Soyad *</label>
              <input value={form.last_name} onChange={e => change('last_name', e.target.value)} placeholder="Yılmaz" required />
            </div>
            <div>
              <label>E-posta</label>
              <input type="email" value={form.email} onChange={e => change('email', e.target.value)} placeholder="ornek@mail.com" />
            </div>
            <div>
              <label>Telefon</label>
              <input type="tel" value={form.phone} onChange={e => change('phone', e.target.value)} placeholder="+90 555 000 00 00" />
            </div>
            <div>
              <label>Uyruk (ISO-3)</label>
              <input value={form.nationality} onChange={e => change('nationality', e.target.value)} placeholder="TUR" maxLength={3} style={{ textTransform: 'uppercase' }} />
            </div>
            <div>
              <label>TC / Pasaport No</label>
              <input value={form.identity_no} onChange={e => change('identity_no', e.target.value)} placeholder="12345678901" />
            </div>
            <div>
              <label>Doğum Tarihi</label>
              <input type="date" value={form.birth_date} onChange={e => change('birth_date', e.target.value)} />
            </div>
            <div>
              <label>Cinsiyet</label>
              <select value={form.gender} onChange={e => change('gender', e.target.value)}>
                {GENDERS.map(g => <option key={g.value} value={g.value}>{g.label}</option>)}
              </select>
            </div>
          </div>

          {/* VIP + Notlar — tam genişlik */}
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 8, marginBottom: 4 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', userSelect: 'none' }}>
              <input type="checkbox" checked={form.vip_status} onChange={e => change('vip_status', e.target.checked)} />
              ⭐ VIP Misafir
            </label>
          </div>
          <div style={{ marginBottom: 10 }}>
            <label>Notlar</label>
            <textarea value={form.notes} onChange={e => change('notes', e.target.value)} rows={2} placeholder="Özel istek veya notlar…" style={{ width: '100%', resize: 'vertical', boxSizing: 'border-box' }} />
          </div>

          {/* Rezervasyon Bilgileri */}
          <div className="textMuted" style={{ fontSize: 11, marginBottom: 6, marginTop: 4 }}>📋 Rezervasyon Bilgileri</div>
          <div className="chatFormGrid">
            <div>
              <label>Otel</label>
              <select value={form.hotel} onChange={e => change('hotel', e.target.value)}>
                {HOTELS.map(h => <option key={h}>{h}</option>)}
              </select>
            </div>
            <div>
              <label>Lead Time (gün)</label>
              <input type="number" min="0" value={form.lead_time} onChange={e => change('lead_time', e.target.value)} />
            </div>
            <div>
              <label>Depozito</label>
              <select value={form.deposit_type} onChange={e => change('deposit_type', e.target.value)}>
                {DEPOSIT_TYPES.map(d => <option key={d}>{d}</option>)}
              </select>
            </div>
            <div>
              <label>Market Segment</label>
              <select value={form.market_segment} onChange={e => change('market_segment', e.target.value)}>
                {SEGMENTS.map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <label>Yetişkin</label>
              <input type="number" min="1" value={form.adults} onChange={e => change('adults', e.target.value)} />
            </div>
            <div>
              <label>Çocuk</label>
              <input type="number" min="0" value={form.children} onChange={e => change('children', e.target.value)} />
            </div>
            <div>
              <label>Bebek</label>
              <input type="number" min="0" value={form.babies} onChange={e => change('babies', e.target.value)} />
            </div>
            <div>
              <label>Hafta içi gece</label>
              <input type="number" min="0" value={form.stays_in_week_nights} onChange={e => change('stays_in_week_nights', e.target.value)} />
            </div>
            <div>
              <label>Hafta sonu gece</label>
              <input type="number" min="0" value={form.stays_in_weekend_nights} onChange={e => change('stays_in_weekend_nights', e.target.value)} />
            </div>
            <div>
              <label>Sadık Müşteri</label>
              <select value={form.is_repeated_guest} onChange={e => change('is_repeated_guest', e.target.value)}>
                <option value={0}>Hayır (İlk ziyaret)</option>
                <option value={1}>Evet (Tekrar gelen)</option>
              </select>
            </div>
            <div>
              <label>Geçmiş İptal</label>
              <input type="number" min="0" value={form.previous_cancellations} onChange={e => change('previous_cancellations', e.target.value)} />
            </div>
            <div>
              <label>Ort. Gecelik Ücret (ADR)</label>
              <input type="number" min="0" step="0.01" value={form.adr} onChange={e => change('adr', e.target.value)} placeholder="isteğe bağlı" />
            </div>
          </div>

          {/* Risk Kartı */}
          <div style={{ marginTop: 12 }}>
            <RiskCard predicting={predicting} riskScore={riskScore} riskLabel={riskLabel} />
          </div>

          {/* Aksiyon */}
          <div style={{ marginTop: 12, display: 'flex', gap: 10, alignItems: 'center' }}>
            <button type="submit" disabled={saving || predicting}>
              {saving ? '⏳ Kaydediliyor…' : '💾 Misafiri Kaydet'}
            </button>
            <button type="button" className="btnGhost"
              onClick={() => { setForm(INITIAL_FORM); setRiskScore(null); setRiskLabel('unknown'); setSaveError(''); setSaveOk(''); }}>
              Temizle
            </button>
          </div>
          {saveOk    && <div className="formSuccess" style={{ marginTop: 8 }}>{saveOk}</div>}
          {saveError && <div className="error" style={{ marginTop: 8 }}>{saveError}</div>}
        </form>

        {/* ── Sağ: Misafir Listesi ──────────────────────────────────── */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--c-border, #eee)' }}>
            <div className="small" style={{ marginBottom: 8 }}>Kayıtlı Misafirler ({total})</div>
            <input
              value={search}
              onChange={handleSearch}
              placeholder="🔍 İsim veya e-posta ile ara…"
              style={{ width: '100%', boxSizing: 'border-box' }}
            />
          </div>

          {listError && <div className="error" style={{ margin: 12 }}>{listError}</div>}
          {loading   && <div className="textMuted" style={{ padding: 20, textAlign: 'center' }}>⏳ Yüleniyor…</div>}

          {!loading && guests.length === 0 && (
            <div className="textMuted" style={{ padding: 24, textAlign: 'center' }}>
              {search ? 'Arama sonucu bulunamadı.' : 'Henüz misafir kaydı yok.'}
            </div>
          )}

          {!loading && guests.length > 0 && (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr>
                    <th className="guestTh">Ad Soyad</th>
                    <th className="guestTh">E-posta / Tel</th>
                    <th className="guestTh">Otel</th>
                    <th className="guestTh">Segment</th>
                    <th className="guestTh">Risk</th>
                    <th className="guestTh">VIP</th>
                    <th className="guestTh">Kayıt</th>
                    <th className="guestTh">İşlem</th>
                  </tr>
                </thead>
                <tbody>
                  {guests.map(g => (
                    <tr key={g.id}>
                      <td className="guestTd">
                        <strong>{g.first_name} {g.last_name}</strong>
                        {g.nationality && <span className="guestSecondary" style={{ marginLeft: 4 }}>({g.nationality})</span>}
                      </td>
                      <td className="guestTd">
                        <div>{g.email || <span className="textMuted">—</span>}</div>
                        <div className="guestSecondary">{g.phone || ''}</div>
                      </td>
                      <td className="guestTd">{g.hotel}</td>
                      <td className="guestTd">{g.market_segment}</td>
                      <td className="guestTd">
                        <RiskBadge label={g.risk_label} score={g.risk_score} />
                      </td>
                      <td className="guestTd" style={{ textAlign: 'center' }}>{g.vip_status ? '⭐' : '—'}</td>
                      <td className="guestTd guestSecondary">
                        {g.created_at ? new Date(g.created_at).toLocaleDateString('tr-TR') : '—'}
                      </td>
                      <td className="guestTd">
                        <button
                          style={{ padding: '3px 10px', fontSize: 12 }}
                          onClick={() => navigate('/chat', { state: { guest: g } })}
                          title="Bu misafir ile chat oturumu başlat"
                        >
                          💬 Chat
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {total > PAGE_SIZE && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 16px', borderTop: '1px solid var(--c-border, #eee)', fontSize: 13 }}>
              <button onClick={prevPage} disabled={offset === 0} style={{ padding: '4px 12px' }}>← Önceki</button>
              <span className="textMuted">{offset + 1}–{Math.min(offset + PAGE_SIZE, total)} / {total}</span>
              <button onClick={nextPage} disabled={offset + PAGE_SIZE >= total} style={{ padding: '4px 12px' }}>Sonraki →</button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

