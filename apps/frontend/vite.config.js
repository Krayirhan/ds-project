import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const BACKEND_TARGET = process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/__tests__/setup.js'],
    css: true,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/dashboard/api': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      '/auth': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      '/predict_proba': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      '/decide': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      '/reload': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      '/chat': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      '/health': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      '/ready': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      '/metrics': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      '/guests': {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
    },
  },
});
