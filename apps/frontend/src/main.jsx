import React from 'react';
import { createRoot } from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import {
  Chart,
  CategoryScale,
  LinearScale,
  BarController,
  BarElement,
  LineController,
  LineElement,
  PointElement,
  Legend,
  Tooltip,
  Title,
} from 'chart.js';
import ErrorBoundary from './components/ErrorBoundary';
import App from './App';
import './styles.css';
import './modern.css';

/* Chart.js bileşenlerini global olarak kaydet */
Chart.register(
  CategoryScale,
  LinearScale,
  BarController,
  BarElement,
  LineController,
  LineElement,
  PointElement,
  Legend,
  Tooltip,
  Title,
);

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <HashRouter>
        <App />
      </HashRouter>
    </ErrorBoundary>
  </React.StrictMode>,
);
