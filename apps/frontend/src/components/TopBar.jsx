import PropTypes from 'prop-types';
import { displayName } from '../lib/helpers';

/**
 * TopBar — Üst bilgi çubuğu
 */
export default function TopBar({ runs }) {
  return (
    <div className="topBar">
      <div className="brandBlock">
        <div className="brandTitle">DS Project — Rezervasyon İptal Tahmin Sistemi</div>
      </div>
      <div className="metaBlock">
        <span className="metaItem">
          <strong>Son Güncelleme:</strong> {runs.generatedAt}
        </span>
        <span className="metaItem">|</span>
        <span className="metaItem">
          <strong>Aktif Model:</strong> {displayName(runs.champion.selected_model)}
        </span>
      </div>
    </div>
  );
}

TopBar.propTypes = {
  runs: PropTypes.shape({
    generatedAt: PropTypes.string,
    champion: PropTypes.shape({
      selected_model: PropTypes.string,
    }),
  }).isRequired,
};
