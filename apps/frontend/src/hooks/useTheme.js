import { useState, useCallback, useEffect } from 'react';
import { applyChartTheme } from '../lib/helpers';

/**
 * useTheme — Tema yönetimi hook'u
 *
 * 3 tema destekler: classic → modern-light → modern-dark
 * CSS variable'ları data-theme attribute'u ile kontrol edilir.
 */
export function useTheme() {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('ds_theme') || 'classic';
    if (saved === 'modern') return 'modern-light'; // eski değer geçişi
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

  const isModern = theme.startsWith('modern');
  const isDark   = theme === 'modern-dark';

  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      if (prev === 'classic')      return 'modern-light';
      if (prev === 'modern-light') return 'modern-dark';
      return 'classic';
    });
  }, []);

  const themeLabel = theme === 'classic' ? 'Modern Aydınlık'
    : isDark ? 'Klasik Görünüm' : 'Modern Karanlık';
  const themeIcon  = theme === 'classic' ? '☀️' : isDark ? '🖥️' : '🌙';

  return { theme, isModern, isDark, toggleTheme, themeLabel, themeIcon };
}
