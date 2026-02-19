import { now } from '../lib/helpers';

/**
 * AppStatusBar — Alt durum çubuğu
 */
export default function AppStatusBar({ runs }) {
  return (
    <div className="appStatusBar">
      <span>{runs.loading ? '⏳ İşlem devam ediyor...' : '✓ Hazır'}</span>
      <span>Model: {runs.coreModels.length} temel</span>
      <span>Koşu: {runs.runs.length} kayıt</span>
      <span>{now()}</span>
    </div>
  );
}
