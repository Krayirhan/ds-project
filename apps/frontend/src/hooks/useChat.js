import { useState, useCallback, useEffect, useRef } from 'react';
import { startChatSession, sendChatMessage, getChatSummary } from '../api';

/**
 * useChat — Chat asistanı hook'u
 *
 * Müşteri formu, oturum yönetimi ve mesajlaşma state'i.
 * AbortController ile oturum açma cancel edilebilir.
 */
export function useChat({ apiKey, onAuthFailed }) {
  const [sessionId, setSessionId]         = useState('');
  const [messages, setMessages]           = useState([]);
  const [input, setInput]                 = useState('');
  const [quickActions, setQuickActions]   = useState([]);
  const [summary, setSummary]             = useState(null);
  const [busy, setBusy]                   = useState(false);
  const [error, setError]                 = useState('');
  const [riskScore, setRiskScore]         = useState(0.5);
  const [customer, setCustomer]           = useState({
    hotel: 'City Hotel',
    lead_time: 30,
    deposit_type: 'No Deposit',
    previous_cancellations: 0,
    market_segment: 'Online TA',
    adults: 2,
    children: 0,
    stays_in_week_nights: 2,
    stays_in_weekend_nights: 1,
  });

  const abortRef    = useRef(null);
  const sessionRef  = useRef(sessionId);
  sessionRef.current = sessionId;

  function handleCustomerChange(key, value) {
    setCustomer(prev => ({ ...prev, [key]: value }));
  }

  function riskLabel(score) {
    const val = Number(score);
    if (val >= 0.65) return 'high';
    if (val >= 0.35) return 'medium';
    return 'low';
  }

  const openSession = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError('');
    setBusy(true);
    try {
      const payload = {
        customer_data: {
          ...customer,
          lead_time:                Number(customer.lead_time || 0),
          previous_cancellations:   Number(customer.previous_cancellations || 0),
          adults:                   Number(customer.adults || 1),
          children:                 Number(customer.children || 0),
          stays_in_week_nights:     Number(customer.stays_in_week_nights || 0),
          stays_in_weekend_nights:  Number(customer.stays_in_weekend_nights || 0),
        },
        risk_score: Number(riskScore),
        risk_label: riskLabel(riskScore),
      };
      const created = await startChatSession(payload, apiKey, { signal: controller.signal });
      setSessionId(created.session_id);
      setQuickActions(created.quick_actions || []);
      setMessages([{ role: 'assistant', content: created.bot_message || 'Oturum açıldı.' }]);

      const s = await getChatSummary(created.session_id, apiKey, { signal: controller.signal });
      setSummary(s);
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (err?.status === 401) { onAuthFailed?.(err); return; }
      setError(err.message || 'Chat oturumu açılamadı.');
    } finally {
      setBusy(false);
    }
  }, [customer, riskScore, apiKey, onAuthFailed]);

  const sendMessage = useCallback(async (text) => {
    const messageText = String(text || '').trim();
    if (!messageText || !sessionRef.current) return;

    setError('');
    setBusy(true);
    setMessages(prev => [...prev, { role: 'user', content: messageText }]);
    setInput('');

    try {
      const response = await sendChatMessage(
        { session_id: sessionRef.current, message: messageText },
        apiKey,
      );
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: response.bot_message || 'Yanıt alınamadı.' },
      ]);
      setQuickActions(response.quick_actions || []);

      const s = await getChatSummary(sessionRef.current, apiKey);
      setSummary(s);
    } catch (err) {
      if (err?.status === 401) { onAuthFailed?.(err); return; }
      setError(err.message || 'Mesaj gönderilemedi.');
    } finally {
      setBusy(false);
    }
  }, [apiKey, onAuthFailed]);

  useEffect(() => () => abortRef.current?.abort(), []);

  return {
    sessionId, messages, input, setInput,
    quickActions, summary, busy, error,
    riskScore, setRiskScore,
    customer, handleCustomerChange,
    openSession, sendMessage,
  };
}
