import { useLayoutContext } from './Layout';
import { useChat } from '../hooks/useChat';

/**
 * ChatPage — Chat Asistanı
 *
 * useChat hook'u ile kendi state'ini yönetir.
 * apiKey ve onAuthFailed Layout context'inden alınır.
 */
export default function ChatPage() {
  const { runs, auth } = useLayoutContext();
  const chat = useChat({ apiKey: runs.apiKey, onAuthFailed: auth.handleAuthFailure });

  return (
    <>
      <header className="pageHeader">
        <h1>💬 Chat Asistanı — İptal Azaltma</h1>
        <p className="subtitle">
          Önce müşteri formunu doldurun, ardından chat oturumunu başlatın.
          Asistan müşteri profiline göre somut aksiyon önerileri sunar.
        </p>
      </header>

      <section className="card chatGrid">
        {/* Sol: Müşteri Formu */}
        <div>
          <div className="small">Müşteri Formu</div>
          <div className="chatFormGrid">
            <div>
              <label htmlFor="chat-hotel">Otel</label>
              <select id="chat-hotel" value={chat.customer.hotel} onChange={e => chat.handleCustomerChange('hotel', e.target.value)}>
                <option value="City Hotel">City Hotel</option>
                <option value="Resort Hotel">Resort Hotel</option>
              </select>
            </div>
            <div>
              <label htmlFor="chat-lead-time">Lead Time (gün)</label>
              <input id="chat-lead-time" type="number" min="0" value={chat.customer.lead_time} onChange={e => chat.handleCustomerChange('lead_time', e.target.value)} />
            </div>
            <div>
              <label htmlFor="chat-deposit">Depozito</label>
              <select id="chat-deposit" value={chat.customer.deposit_type} onChange={e => chat.handleCustomerChange('deposit_type', e.target.value)}>
                <option value="No Deposit">No Deposit</option>
                <option value="Non Refund">Non Refund</option>
                <option value="Refundable">Refundable</option>
              </select>
            </div>
            <div>
              <label htmlFor="chat-segment">Market Segment</label>
              <select id="chat-segment" value={chat.customer.market_segment} onChange={e => chat.handleCustomerChange('market_segment', e.target.value)}>
                <option value="Online TA">Online TA</option>
                <option value="Direct">Direct</option>
                <option value="Corporate">Corporate</option>
                <option value="Groups">Groups</option>
              </select>
            </div>
            <div>
              <label htmlFor="chat-adults">Yetişkin</label>
              <input id="chat-adults" type="number" min="1" value={chat.customer.adults} onChange={e => chat.handleCustomerChange('adults', e.target.value)} />
            </div>
            <div>
              <label htmlFor="chat-children">Çocuk</label>
              <input id="chat-children" type="number" min="0" value={chat.customer.children} onChange={e => chat.handleCustomerChange('children', e.target.value)} />
            </div>
            <div>
              <label htmlFor="chat-week">Hafta içi gece</label>
              <input id="chat-week" type="number" min="0" value={chat.customer.stays_in_week_nights} onChange={e => chat.handleCustomerChange('stays_in_week_nights', e.target.value)} />
            </div>
            <div>
              <label htmlFor="chat-weekend">Hafta sonu gece</label>
              <input id="chat-weekend" type="number" min="0" value={chat.customer.stays_in_weekend_nights} onChange={e => chat.handleCustomerChange('stays_in_weekend_nights', e.target.value)} />
            </div>
            <div>
              <label htmlFor="chat-prev-cancel">Geçmiş İptal</label>
              <input id="chat-prev-cancel" type="number" min="0" value={chat.customer.previous_cancellations} onChange={e => chat.handleCustomerChange('previous_cancellations', e.target.value)} />
            </div>
            <div>
              <label htmlFor="chat-risk">Risk skoru (0-1)</label>
              <input id="chat-risk" type="number" min="0" max="1" step="0.01" value={chat.riskScore} onChange={e => chat.setRiskScore(e.target.value)} />
            </div>
          </div>

          <div style={{ marginTop: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
            <button onClick={chat.openSession} disabled={chat.busy}>
              {chat.busy ? '⏳ Açılıyor...' : '🚀 Chat Oturumu Başlat'}
            </button>
            {chat.summary && (
              <span className="metaItem"><strong>Mesaj:</strong> {chat.summary.message_count}</span>
            )}
          </div>
        </div>

        {/* Sağ: Sohbet Paneli */}
        <div>
          <div className="small">Sohbet</div>
          <div className="chatPanel" aria-live="polite" aria-label="Chat mesajları">
            {chat.messages.length === 0 && (
              <div className="chatEmpty">Oturum başlatıldığında asistan mesajı burada görünecek.</div>
            )}
            {chat.messages.map((m, idx) => (
              <div key={`${m.role}-${idx}`} className={`chatBubble ${m.role === 'user' ? 'user' : 'assistant'}`}>
                <div className="chatRole">{m.role === 'user' ? 'Temsilci' : 'Asistan'}</div>
                <div>{m.content}</div>
              </div>
            ))}
          </div>

          {chat.quickActions.length > 0 && (
            <div className="chatQuickActions">
              {chat.quickActions.map((a, idx) => (
                <button key={`${a.label}-${idx}`} onClick={() => chat.sendMessage(a.message)} disabled={chat.busy || !chat.sessionId}>
                  {a.label}
                </button>
              ))}
            </div>
          )}

          <form
            className="chatComposer"
            onSubmit={e => { e.preventDefault(); chat.sendMessage(chat.input); }}
          >
            <input
              value={chat.input}
              onChange={e => chat.setInput(e.target.value)}
              placeholder="Örn: Bu müşteri için ilk adım ne olmalı?"
              disabled={!chat.sessionId}
              aria-label="Chat mesajı yaz"
            />
            <button type="submit" disabled={chat.busy || !chat.sessionId || !chat.input.trim()}>
              Gönder
            </button>
          </form>

          {chat.error && <div className="error" role="alert" style={{ marginTop: 8 }}>{chat.error}</div>}
        </div>
      </section>
    </>
  );
}
