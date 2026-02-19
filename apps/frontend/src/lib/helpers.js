/**
 * Yardımcı fonksiyonlar — saf (pure) utility'ler + Chart.js tema ayarı
 */
import { Chart } from 'chart.js';
import { MODEL_DISPLAY } from './constants';

// ── Formatlama ──────────────────────────────────────────────────────

export function f(value, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(digits);
}

export function pct(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return `%${(Number(value) * 100).toFixed(digits)}`;
}

export function money(value) {
  if (value == null) return '-';
  return Number(value).toLocaleString('tr-TR', { maximumFractionDigits: 0 });
}

export function formatRunId(runId) {
  if (!runId || runId.length < 15) return runId || '-';
  const d = runId.slice(0, 8);
  const t = runId.slice(9);
  return `${d.slice(6,8)}.${d.slice(4,6)}.${d.slice(0,4)}  ${t.slice(0,2)}:${t.slice(2,4)}`;
}

export function now() {
  return new Date().toLocaleString('tr-TR');
}

// ── Skor renklendirme ───────────────────────────────────────────────

export function scoreColor(score) {
  if (score == null || Number.isNaN(Number(score))) return '#666';
  const v = Number(score);
  if (v >= 0.90) return '#006600';
  if (v >= 0.80) return '#337700';
  if (v >= 0.70) return '#996600';
  return '#cc0000';
}

// ── Model bilgi erişim fonksiyonları ────────────────────────────────

export function displayName(raw)      { return MODEL_DISPLAY[raw]?.short       || raw; }
export function modelBadge(raw)       { return MODEL_DISPLAY[raw]?.badge       || ''; }
export function modelIcon(raw)        { return MODEL_DISPLAY[raw]?.icon        || '⚪'; }
export function modelCalibration(raw) { return MODEL_DISPLAY[raw]?.calibration || '—'; }
export function modelType(raw)        { return MODEL_DISPLAY[raw]?.type        || 'Bilinmiyor'; }

// ── Chart.js tema ayarı ─────────────────────────────────────────────

export function applyChartTheme(themeVal) {
  const isModern = themeVal.startsWith('modern');
  const isDark   = themeVal === 'modern-dark';
  if (isModern) {
    Chart.defaults.font.family = 'Inter, -apple-system, system-ui, sans-serif';
    Chart.defaults.font.size   = 11;
    Chart.defaults.color       = isDark ? '#cbd5e1' : '#4a5568';
    Chart.defaults.borderColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)';
  } else {
    Chart.defaults.font.family = 'Tahoma, "Segoe UI", sans-serif';
    Chart.defaults.font.size   = 10;
    Chart.defaults.color       = '#666';
    Chart.defaults.borderColor = 'rgba(0,0,0,0.1)';
  }
}
