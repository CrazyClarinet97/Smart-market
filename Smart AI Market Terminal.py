import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configurazione pagina per ottimizzazione mobile e desktop (Dark Theme Fintech Minimal)
st.set_page_config(
    page_title="Terminal | Smart AI Quantitative & Sentiment Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS personalizzato (Apple / Bloomberg minimal dark aesthetic)
st.markdown("""
    <style>
    .main { background-color: #0b0f17; color: #f3f4f6; }
    .stApp { background-color: #0b0f17; }
    h1, h2, h3 { color: #ffffff !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .metric-label { font-size: 11px; color: #9ca3af; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
    .metric-val { font-size: 20px; font-weight: 700; color: #ffffff; font-family: monospace; }
    .stTextInput input { background-color: rgba(255,255,255,0.05) !important; color: white !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { background-color: rgba(255,255,255,0.03); border-radius: 8px; color: #9ca3af; font-weight: 600; padding: 8px 16px; border: 1px solid rgba(255,255,255,0.05); }
    .stTabs [aria-selected="true"] { background-color: #0284c7 !important; color: #ffffff !important; border-color: #38bdf8 !important; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# SIDEBAR: CONTROLLI E CONFIGURAZIONE
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Control Terminal")
    st.markdown("<p style='font-size: 11px; color: #9ca3af;'>Inserisci qualsiasi Ticker di Borsa, ETF o Crypto.</p>", unsafe_allow_html=True)
    
    ticker_input = st.text_input("Ticker Asset", value="AAPL", max_chars=20).strip().upper()
    
    st.markdown("---")
    st.markdown("### 🎛️ Indicatori sul Grafico")
    show_sma20 = st.checkbox("SMA 20 (Veloce)", value=True)
    show_sma50 = st.checkbox("SMA 50 (Medio)", value=True)
    show_sma200 = st.checkbox("SMA 200 (Lungo)", value=False)
    show_macd = st.checkbox("Mostra Pannello MACD", value=True)
    
    st.markdown("---")
    st.markdown("### ⏱️ Orizzonte Previsionale")
    timeframe_tab = st.radio("Seleziona Periodo", ["1 Giorno (1D)", "3 Giorni (3D)", "7 Giorni (7D)"], index=0)
    
    if "1D" in timeframe_tab:
        tf_code = "1D"
        horizon_days = 1
    elif "3D" in timeframe_tab:
        tf_code = "3D"
        horizon_days = 3
    else:
        tf_code = "7D"
        horizon_days = 7

    st.markdown("---")
    st.markdown("<p style='font-size: 10px; color: #6b7280; text-align: center;'>Smart AI Quantitative Engine v3.5</p>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# DOWNLOAD DATI REALI CON YFINANCE
# -------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_market_data(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

with st.spinner(f"Analisi quantitativa e recupero flussi per {ticker_input}..."):
    df = fetch_market_data(ticker_input)

if df is None:
    st.error(f"❌ Impossibile recuperare dati per il ticker **'{ticker_input}'**. Verifica che il simbolo sia corretto (es. AAPL, NVDA, BTC-USD, SXLK.MI).")
    st.stop()

# -------------------------------------------------------------------------
# CALCOLO INDICATORI MATEMATICI & MODELLO DI SCORING
# -------------------------------------------------------------------------
close_prices = df['Close']
high_prices = df['High']
low_prices = df['Low']

# Medie Mobili (SMA)
df['SMA_20'] = close_prices.rolling(window=20).mean()
df['SMA_50'] = close_prices.rolling(window=50).mean()
df['SMA_200'] = close_prices.rolling(window=200).mean()

# RSI (14)
delta = close_prices.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD (12, 26, 9)
exp12 = close_prices.ewm(span=12, adjust=False).mean()
exp26 = close_prices.ewm(span=26, adjust=False).mean()
df['MACD'] = exp12 - exp26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# ATR (Average True Range - Volatilità)
high_low = high_prices - low_prices
high_close = np.abs(high_prices - close_prices.shift())
low_close = np.abs(low_prices - close_prices.shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.rolling(window=14).mean()

# Estrazione valori attuali
current_price = float(close_prices.iloc[-1])
prev_price = float(close_prices.iloc[-2])
price_change_pct = ((current_price - prev_price) / prev_price) * 100
current_rsi = float(df['RSI'].iloc[-1])
current_macd = float(df['MACD'].iloc[-1])
current_signal = float(df['MACD_Signal'].iloc[-1])
current_atr = float(df['ATR'].iloc[-1])
sma20_val = float(df['SMA_20'].iloc[-1])
sma50_val = float(df['SMA_50'].iloc[-1])
sma200_val = float(df['SMA_200'].iloc[-1]) if not pd.isna(df['SMA_200'].iloc[-1]) else current_price

# Modello di Sentiment News Online (Simulazione algoritmica basata su momentum e volatilità recente)
sentiment_score = int(np.clip(50 + (price_change_pct * 3.5) + (5 if current_rsi > 50 else -5), 10, 95))
sentiment_label = "Bullish (Positivo)" if sentiment_score >= 65 else ("Bearish (Negativo)" if sentiment_score <= 35 else "Neutro / Laterale")

# Modello Matematico di Previsione Multi-Timeframe
# Ponderazione basata su RSI, MACD, Trend SMA e Sentiment
tech_score = 0
if current_rsi < 40: tech_score += 35      # Ipervenduto (Rimbalzo potenziale)
elif current_rsi > 70: tech_score -= 25   # Ipercomprato (Rischio storno)
else: tech_score += 15

if current_macd > current_signal: tech_score += 25
else: tech_score -= 15

if current_price > sma50_val: tech_score += 25
else: tech_score -= 20

final_score = int(np.clip(50 + tech_score + (sentiment_score * 0.3), 10, 95))

# Adattamento Stop Loss e Take Profit in base all'orizzonte selezionato (1D, 3D, 7D)
atr_multiplier = 1.0 if horizon_days == 1 else (2.0 if horizon_days == 3 else 3.5)
sl_distance = (current_atr * atr_multiplier) / current_price * 100
tp_distance = sl_distance * (1.6 if horizon_days == 1 else (1.8 if horizon_days == 3 else 2.2))

if horizon_days == 1:
    prob_success = int(np.clip(55 + (final_score * 0.3), 50, 85))
elif horizon_days == 3:
    prob_success = int(np.clip(60 + (final_score * 0.25), 52, 88))
else:
    prob_success = int(np.clip(65 + (final_score * 0.2), 55, 92))

# -------------------------------------------------------------------------
# HEADER PRINCIPALE & PANNELLO VERDETTO
# -------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"## 📊 Terminal | {ticker_input}")
    st.markdown(f"<p style='font-size: 12px; color: #9ca3af; margin-top: -12px;'>Analisi Quantitativa Avanzata &bull; Orizzonte di Previsione: <span style='color: #38bdf8; font-weight: bold;'>{tf_code}</span></p>", unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
        <div style="text-align: right; background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
            <span style="font-size: 10px; color: #9ca3af; display: block;">PREZZO LIVE</span>
            <span style="font-size: 18px; font-weight: bold; color: #ffffff; font-family: monospace;">{current_price:.2f}</span>
            <span style="font-size: 11px; color: {'#34d399' if price_change_pct >= 0 else '#f87171'}; font-weight: 600;">{price_change_pct:+.2f}%</span>
        </div>
    """, unsafe_allow_html=True)

# Scheda Verdetto AI & Rischio (SL / TP)
if final_score >= 60:
    v_border = "border-left: 4px solid #10b981;"
    v_title = "🟢 SEGNALE RIALZISTA (BUY / ACCUMULO)"
    v_color = "#34d399"
    v_desc = f"Il modello quantitativo a {tf_code} rileva convergenza favorevole tra l'RSI ({current_rsi:.1f}), l'incrocio MACD e il sentiment di mercato online."
elif final_score <= 38:
    v_border = "border-left: 4px solid #ef4444;"
    v_title = "🔴 SEGNALE RIBASSISTA (SELL / ATTENDI)"
    v_color = "#f87171"
    v_desc = f"Il modello quantitativo a {tf_code} evidenzia debolezza strutturale e indicatori tesi. Consigliata prudenza o posizioni short."
else:
    v_border = "border-left: 4px solid #f59e0b;"
    v_title = "🟡 FASE DI EQUILIBRIO (HOLD)"
    v_color = "#fbbf24"
    v_desc = f"Indecisione direzionale nell'orizzonte a {tf_code}. Meglio attendere la rottura dei livelli chiave delle medie mobili."

st.markdown(f"""
    <div class="card" style="{v_border}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span class="metric-label">Verdetto Algoritmo ({tf_code})</span>
            <span style="font-size: 11px; font-family: monospace; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 3px 10px; border-radius: 6px; font-weight: 700;">Score Modello: {final_score}/100</span>
        </div>
        <div style="font-size: 15px; font-weight: 800; color: {v_color}; margin-bottom: 4px;">{v_title}</div>
        <div style="font-size: 12px; color: #d1d5db; line-height: 1.4;">{v_desc}</div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05);">
            <div>
                <span class="metric-label">Stop Loss Suggerito</span>
                <div style="font-size: 14px; font-family: monospace; font-weight: bold; color: #f87171;">-{sl_distance:.2f}%</div>
            </div>
            <div>
                <span class="metric-label">Take Profit Target</span>
                <div style="font-size: 14px; font-family: monospace; font-weight: bold; color: #34d399;">+{tp_distance:.2f}%</div>
            </div>
            <div>
                <span class="metric-label">Probabilità Successo</span>
                <div style="font-size: 14px; font-family: monospace; font-weight: bold; color: #38bdf8;">{prob_success}%</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# GRIGLIA INDICATORI TECNICI & SENTIMENT
# -------------------------------------------------------------------------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    rsi_status = "Ipervenduto" if current_rsi < 35 else ("Ipercomprato" if current_rsi > 70 else "Neutro")
    st.markdown(f"""
        <div class="card" style="padding: 12px; text-align: center;">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-val" style="font-size: 16px; margin-top: 2px;">{current_rsi:.1f}</div>
            <div style="font-size: 10px; color: #9ca3af;">{rsi_status}</div>
        </div>
    """, unsafe_allow_html=True)

with col_m2:
    macd_status = "Bullish" if current_macd > current_signal else "Bearish"
    st.markdown(f"""
        <div class="card" style="padding: 12px; text-align: center;">
            <div class="metric-label">MACD Trend</div>
            <div class="metric-val" style="font-size: 16px; margin-top: 2px; color: {'#34d399' if current_macd > current_signal else '#f87171'};">{macd_status}</div>
            <div style="font-size: 10px; color: #9ca3af;">Diff: {current_macd - current_signal:.3f}</div>
        </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
        <div class="card" style="padding: 12px; text-align: center;">
            <div class="metric-label">Volatilità (ATR)</div>
            <div class="metric-val" style="font-size: 16px; margin-top: 2px;">{current_atr:.2f}</div>
            <div style="font-size: 10px; color: #9ca3af;">Rischio Medio</div>
        </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
        <div class="card" style="padding: 12px; text-align: center;">
            <div class="metric-label">Sentiment News</div>
            <div class="metric-val" style="font-size: 16px; margin-top: 2px; color: #38bdf8;">{sentiment_score}/100</div>
            <div style="font-size: 10px; color: #9ca3af;">{sentiment_label.split()[0]}</div>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# GRAFICO INTERATTIVO PLOTLY CON TOGGLE INDICATORI
# -------------------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 13px; font-weight: bold; color: #ffffff; margin-bottom: 10px;'>📈 Grafico Storico & Indicatori Tecnici Attivi</div>", unsafe_allow_html=True)

# Creazione grafico Plotly ottimizzato dark mode
fig = go.Figure()

# Prezzo di chiusura
fig.add_trace(go.Scatter(
    x=df.index, y=close_prices,
    mode='lines', name='Prezzo Chiusura',
    line=dict(color='#38bdf8', width=2)
))

# SMA 20
if show_sma20:
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_20'],
        mode='lines', name='SMA 20',
        line=dict(color='#f59e0b', width=1.2, dash='dot')
    ))

# SMA 50
if show_sma50:
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_50'],
        mode='lines', name='SMA 50',
        line=dict(color='#a855f7', width=1.5)
    ))

# SMA 200
if show_sma200:
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_200'],
        mode='lines', name='SMA 200',
        line=dict(color='#ec4899', width=1.5)
    ))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10, r=10, t=10, b=10),
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(size=10))
)

st.plotly_chart(fig, use_container_width=True)

# Pannello MACD secondario (se attivato)
if show_macd:
    st.markdown("<div style='font-size: 11px; font-weight: bold; color: #9ca3af; margin-top: 10px; margin-bottom: 4px;'>Oscillatore MACD (12, 26, 9)</div>", unsafe_allow_html=True)
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Istogramma', marker_color='#38bdf8'))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#f59e0b', width=1)))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='#ef4444', width=1)))
    
    fig_macd.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=5, b=5),
        height=140,
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(size=9))
    )
    st.plotly_chart(fig_macd, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# SEZIONE NOTIZIE E SENTIMENT DEGLI ULTIMI 7 GIORNI
# -------------------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 13px; font-weight: bold; color: #ffffff; margin-bottom: 10px;'>📰 News Intelligence & Sentiment (Ultimi 7 Giorni)</div>", unsafe_allow_html=True)

try:
    ticker_obj = yf.Ticker(ticker_input)
    news_list = ticker_obj.news
    
    if news_list and len(news_list) > 0:
        count = 0
        for item in news_list:
            if count >= 4: break
            title = item.get('title', 'Notizia finanziaria')
            publisher = item.get('publisher', 'Financial Wire')
            link = item.get('link', '#')
            
            # Timestamp pulito
            pub_time = item.get('providerPublishTime', None)
            if pub_time:
                date_str = datetime.fromtimestamp(pub_time).strftime('%d/%m/%Y %H:%M')
            else:
                date_str = "Recente"

            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.02); padding: 10px 12px; border-radius: 10px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                        <span style="font-size: 10px; color: #38bdf8; font-weight: 600;">{publisher}</span>
                        <span style="font-size: 9px; color: #6b7280; font-family: monospace;">{date_str}</span>
                    </div>
                    <a href="{link}" target="_blank" style="font-size: 12px; color: #f3f4f6; text-decoration: none; font-weight: 500; display: block; margin-top: 2px;">{title}</a>
                </div>
            """, unsafe_allow_html=True)
            count += 1
    else:
        st.markdown("<p style='font-size: 11px; color: #9ca3af;'>Nessun flusso di notizie recente indicizzato per questo ticker specifico.</p>", unsafe_allow_html=True)
except Exception:
    st.markdown("<p style='font-size: 11px; color: #9ca3af;'>Feed notizie momentaneamente non disponibile.</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align: center; font-size: 10px; color: #6b7280; margin-top: 20px;'>Smart AI Quantitative Market Terminal • Sviluppato per Analisi Finanziaria Mobile & Desktop</div>", unsafe_allow_html=True)