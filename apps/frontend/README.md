# Frontend (React + Vite)

Bu dizin, dashboard arayüzünün backend'den ayrılmış fullstack frontend katmanıdır.

## Geliştirme

```bash
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

## Ortam Değişkenleri

- `VITE_API_BASE_URL` (opsiyonel)
  - Örnek: `http://localhost:8000`
  - Boş bırakılırsa aynı origin üzerinden `/dashboard/api/*` çağrılır.
