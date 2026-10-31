import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from textblob import TextBlob

# Configurazione Pagina & Layout Minimal Serio
st.set_page_config(
    page_title="Market Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# STILE CSS PERSONALIZZATO (Minimal, Font ridotti, Dark Mode pulita)
st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #d1d5db;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    h1 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #f3f4f6 !important;
        margin-bottom: 0px !important;
    }
    h2, h3 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #f3f4f6 !important;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 17px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 10px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    input, div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        color: #f3f4f6 !important;
        border-color: #374151 !important;
        font-size: 13px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MOTORE DI CALCOLO TECNICO AVANZATO
# ---------------------------------------------------------
def calcola_indicatori(df):
    # Medie Mobili
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI (14 e 7)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    gain7 = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss7 = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs7 = gain7 / (loss7 + 1e-9)
    df['RSI7'] = 100 - (100 / (1 + rs7))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bande di Bollinger
    df['BB_Middle'] = df['SMA20']
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)

    # Stocastico
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low14) / (high14 - low14 + 1e-9))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # ATR (Average True Range)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # Volume Ratio (Confronto volume odierno con media 20 giorni)
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA20'] + 1e-9)

    return df

# ---------------------------------------------------------
# SENTIMENT WEB
# ---------------------------------------------------------
def estrai_news_e_sentiment(ticker):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers).json()
        news_list = response.get('news', [])[:5]
        
        sentiments = []
        articoli = []
        for item in news_list:
            title = item.get('title', '')
            link = item.get('link', '#')
            publisher = item.get('publisher', 'Fonte')
            analysis = TextBlob(title)
            polarity = analysis.sentiment.polarity
            sentiments.append(polarity)
            articoli.append({'title': title, 'link': link, 'publisher': publisher, 'score': polarity})
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        return avg_sentiment, articoli
    except:
        return 0.0, []

# ---------------------------------------------------------
# MODELLO PREDITTIVO MULTI-FATTORE
# ---------------------------------------------------------
def calcola_modello(df, sentiment_score, horizon):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    dettagli = []
    score = 50

    # RSI
    rsi = last['RSI']
    if rsi < 32:
        score += 15
        dettagli.append({"Fattore": "RSI (14)", "Impatto": "+15 pt", "Stato": f"Ipervenduto profondo ({rsi:.1f})"})
    elif rsi > 68:
        score -= 15
        dettagli.append({"Fattore": "RSI (14)", "Impatto": "-15 pt", "Stato": f"Ipercomprato ({rsi:.1f})"})
    else:
        score += 5
        dettagli.append({"Fattore": "RSI (14)", "Impatto": "+5 pt", "Stato": f"Neutro/Equilibrato ({rsi:.1f})"})

    # Trend SMA50 & EMA20
    if last['Close'] > last['SMA50'] and last['Close'] > last['EMA20']:
        score += 15
        dettagli.append({"Fattore": "Trend Prezzo", "Impatto": "+15 pt", "Stato": "Sopra SMA50 e EMA20 (Rialzista)"})
    else:
        score -= 15
        dettagli.append({"Fattore": "Trend Prezzo", "Impatto": "-15 pt", "Stato": "Sotto SMA50 (Debolezza strutturale)"})

    # Volumi
    vol_ratio = last['Vol_Ratio']
    if vol_ratio > 1.2 and last['Close'] > prev['Close']:
        score += 10
        dettagli.append({"Fattore": "Volumi", "Impatto": "+10 pt", "Stato": f"Volumi in aumento con prezzo in salita ({vol_ratio:.1f}x)"})
    elif vol_ratio > 1.2 and last['Close'] < prev['Close']:
        score -= 10
        dettagli.append({"Fattore": "Volumi", "Impatto": "-10 pt", "Stato": f"Volumi di scarico elevati ({vol_ratio:.1f}x)"})

    # MACD
    if last['MACD'] > last['MACD_Signal']:
        score += 10
        dettagli.append({"Fattore": "MACD", "Impatto": "+10 pt", "Stato": "Sopra la linea di segnale"})
    else:
        score -= 10
        dettagli.append({"Fattore": "MACD", "Impatto": "-10 pt", "Stato": "Sotto la linea di segnale"})

    # Sentiment
    if sentiment_score > 0.05:
        score += 10
        dettagli.append({"Fattore": "News Sentiment", "Impatto": "+10 pt", "Stato": f"Positivo (+{sentiment_score:.2f})"})
    elif sentiment_score < -0.05:
        score -= 10
        dettagli.append({"Fattore": "News Sentiment", "Impatto": "-10 pt", "Stato": f"Negativo ({sentiment_score:.2f})"})

    score = max(0, min(100, score))

    # SL / TP Dinamici su ATR
    atr = last['ATR']
    close = last['Close']
    mult = 1.0 if horizon == "1D" else (1.8 if horizon == "3D" else 2.6)

    sl_val = close - (atr * mult * 1.2)
    tp_val = close + (atr * mult * 2.2)
    sl_pct = ((sl_val - close) / close) * 100
    tp_pct = ((tp_val - close) / close) * 100
    risk_reward = abs(tp_pct / sl_pct) if sl_pct != 0 else 0

    if score >= 63:
        verdetto, colore = "BUY (ACQUISTO)", "#238636"
    elif score <= 38:
        verdetto, colore = "SELL (VENDITA)", "#da3633"
    else:
        verdetto, colore = "HOLD (ATTENDI)", "#d29922"

    return verdetto, score, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, dettagli

# ---------------------------------------------------------
# INTERFACCIA UTENTE
# ---------------------------------------------------------
st.markdown("<h1>Terminal & Predictive Engine</h1>", unsafe_allow_html=True)
st.write("")

c1, c2 = st.columns([2, 1])
with c1:
    ticker = st.text_input("Ticker:", value="AAPL").upper().strip()
with c2:
    horizon = st.selectbox("Orizzonte:", ["1D", "3D", "7D"], index=0)

if ticker:
    with st.spinner("Analisi in corso..."):
        try:
            df = yf.download(ticker, period="1y", interval="1d")
            if df.empty:
                st.error("Ticker non trovato.")
                st.stop()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = calcola_indicatori(df)
            avg_sentiment, articoli = estrai_news_e_sentiment(ticker)
            verdetto, score, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, dettagli = calcola_modello(df, avg_sentiment, horizon)
            last_price = df['Close'].iloc[-1]
            change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        except Exception as e:
            st.error(f"Errore: {e}")
            st.stop()

    # SCHEDA VERDETTO
    st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {colore};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="metric-label">Verdetto ({horizon})</span>
                    <h3 style="margin: 0; color: {colore} !important;">{verdetto}</h3>
                </div>
                <div style="text-align: right;">
                    <span class="metric-label">Score</span>
                    <div style="font-size: 20px; font-weight: bold; color: #f3f4f6;">{score}/100</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # KPI RIGHE
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="metric-card"><span class="metric-label">Prezzo</span><div class="metric-value">${last_price:.2f}</div><span style="color:{'#238636' if change_pct>=0 else '#da3633'}; font-size:11px;">{change_pct:+.2f}%</span></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card"><span class="metric-label">Stop Loss</span><div class="metric-value" style="color:#da3633;">${sl_val:.2f}</div><span style="color:#da3633; font-size:11px;">{sl_pct:.2f}%</span></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="metric-card"><span class="metric-label">Take Profit</span><div class="metric-value" style="color:#238636;">${tp_val:.2f}</div><span style="color:#238636; font-size:11px;">{tp_pct:+.2f}%</span></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="metric-card"><span class="metric-label">R/R Ratio</span><div class="metric-value" style="color:#38bdf8;">1 : {risk_reward:.1f}</div><span style="color:#9ca3af; font-size:11px;">Rischio/Beneficio</span></div>""", unsafe_allow_html=True)

    # GRAFICO INTERATTIVO CON ZOOM LIBERO (MOBILE FRIENDLY)
    st.write("")
    st.subheader("Grafico & Indicatori")
    
    ind_options = st.multiselect(
        "Seleziona Indicatori:",
        ["SMA20", "SMA50", "EMA20", "Bande di Bollinger", "RSI", "MACD", "Stocastico"],
        default=["SMA20", "Bande di Bollinger", "RSI"]
    )

    num_plots = 1 + ("RSI" in ind_options) + ("MACD" in ind_options) + ("Stocastico" in ind_options)
    row_heights = [0.52] + [0.16] * (num_plots - 1)
    
    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Prezzo'
    ), row=1, col=1)

    if "SMA20" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#eab308', width=1), name='SMA20'), row=1, col=1)
    if "SMA50" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='#a855f7', width=1), name='SMA50'), row=1, col=1)
    if "EMA20" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#38bdf8', width=1), name='EMA20'), row=1, col=1)
    if "Bande di Bollinger" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(255,255,255,0.2)', width=1), name='BB Up'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(255,255,255,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.03)', name='BB Low'), row=1, col=1)

    curr_row = 2
    if "RSI" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#f97316', width=1.2), name='RSI'), row=curr_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=curr_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=curr_row, col=1)
        curr_row += 1

    if "MACD" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#38bdf8', width=1.2), name='MACD'), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#f97316', width=1.2), name='Signal'), row=curr_row, col=1)
        curr_row += 1

    if "Stocastico" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='#38bdf8', width=1.2), name='%K'), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='#f97316', width=1.2), name='%D'), row=curr_row, col=1)
        curr_row += 1

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        margin=dict(l=5, r=5, t=5, b=5),
        height=480,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        dragmode="zoom"
    )
    
    # Abilita lo zoom fluido e configurazione mobile
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

    # NOTIZIE RECENTI (ULTIMI 7 GIORNI)
    st.write("")
    st.subheader("Notizie & Sentiment Web")
    if articoli:
        for art in articoli:
            sc_col = "#22c55e" if art['score'] > 0 else ("#ef4444" if art['score'] < 0 else "#9ca3af")
            st.markdown(f"""
                <div class="metric-card">
                    <a href="{art['link']}" target="_blank" style="color:#38bdf8; font-weight:500; font-size:13px; text-decoration:none;">{art['title']}</a>
                    <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:10px; color:#9ca3af;">
                        <span>{art['publisher']}</span>
                        <span style="color:{sc_col}; font-weight:bold;">Sentiment: {art['score']:+.2f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nessuna notizia recente.")

    # GUIDA SPOSTATA RIGOROSAMENTE IN FONDO
    st.write("")
    with st.expander("📖 Guida ai Calcoli e Logica del Modello"):
        st.write("Il punteggio **Score (0-100)** analizza la confluenza di molteplici indicatori:")
        st.table(pd.DataFrame(dettagli))
        st.write("""
        * **Rischio/Rendimento (R/R):** Indica il rapporto tra il potenziale guadagno e la perdita stimata in base alla volatilità (ATR). Un valore superiore a 1.5 è considerato ottimale per il trading.
        * **Orizzonte (1D, 3D, 7D):** Ridimensiona l'ampiezza dell'ATR per adattare i target operativi al time-frame prescelto.
        """)
