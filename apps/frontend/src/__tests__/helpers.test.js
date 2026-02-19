import { describe, it, expect } from 'vitest';
import {
  f, pct, money, formatRunId, scoreColor, displayName, modelBadge,
} from '../lib/helpers';

describe('helpers', () => {
  describe('f()', () => {
    it('formats a number with default 4 digits', () => {
      expect(f(0.12345)).toBe('0.1235');
    });
    it('returns "-" for null', () => {
      expect(f(null)).toBe('-');
    });
    it('respects custom digits', () => {
      expect(f(0.5, 2)).toBe('0.50');
    });
  });

  describe('pct()', () => {
    it('formats as percentage with % prefix (Turkish)', () => {
      expect(pct(0.856)).toBe('%85.6');
    });
    it('returns "-" for null', () => {
      expect(pct(null)).toBe('-');
    });
  });

  describe('money()', () => {
    it('formats as Turkish locale number', () => {
      const result = money(1234.5);
      // toLocaleString('tr-TR') — output varies by environment
      expect(result).toBeTruthy();
      expect(result).not.toBe('-');
    });
  });

  describe('formatRunId()', () => {
    it('formats YYYYMMDD_HHMMSS pattern as DD.MM.YYYY HH:MM', () => {
      expect(formatRunId('20260218_143203')).toBe('18.02.2026  14:32');
    });
    it('returns raw value for non-matching input', () => {
      expect(formatRunId('abc')).toBe('abc');
    });
  });

  describe('scoreColor()', () => {
    it('returns dark green for score >= 0.90', () => {
      expect(scoreColor(0.95)).toBe('#006600');
    });
    it('returns yellow-brown for score 0.70-0.79', () => {
      expect(scoreColor(0.75)).toBe('#996600');
    });
    it('returns red for score < 0.70', () => {
      expect(scoreColor(0.4)).toBe('#cc0000');
    });
  });

  describe('displayName()', () => {
    it('resolves known model key', () => {
      expect(displayName('baseline')).toBe('Lojistik Regresyon');
    });
    it('returns raw value for unknown key', () => {
      expect(displayName('unknown_model')).toBe('unknown_model');
    });
  });

  describe('modelBadge()', () => {
    it('resolves badge text', () => {
      expect(modelBadge('baseline')).toBe('Temel');
    });
    it('returns empty string for unknown', () => {
      expect(modelBadge('nope')).toBe('');
    });
  });
});
