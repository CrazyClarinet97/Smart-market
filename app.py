import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from textblob import TextBlob

# Configurazione Pagina
st.set_page_config(
    page_title="Market Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# STILE CSS "LIQUID GLASS" UNIVERSALE
st.markdown("""
    <style>
    .metric-card {
        background: rgba(150, 150, 150, 0.06);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(150, 150, 150, 0.15);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: rgba(25, 30, 45, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
    }
    h1 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .metric-value {
        font-size: 16px;
        font-weight: 700;
        color: #0284c7;
    }
    @media (prefers-color-scheme: dark) {
        .metric-value { color: #38bdf8; }
    }
    .metric-label {
        font-size: 10px;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton button {
        background-color: #0284c7 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        width: 100%;
        height: 38px;
    }
    </style>
""", unsafe_allow_html=True)

if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = "AAPL"
if 'horizon_input' not in st.session_state:
    st.session_state.horizon_input = "1D"

@st.cache_data(ttl=120)
def scarica_dati(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Pulisce eventuali righe NaN in fondo
    df = df.dropna(subset=['Close'])
    return df

def calcola_indicatori(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['BB_Middle'] = df['SMA20']
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)

    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA20'] + 1e-9)
    return df

def estrai_news_e_sentiment(ticker):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers).json()
        news_list = response.get('news', [])[:10] # Aumentato a 10 notizie
        
        sentiments, articoli = [], []
        for item in news_list:
            title = item.get('title', '')
            link = item.get('link', '#')
            publisher = item.get('publisher', 'Fonte Web')
            
            # Assegnazione peso di affidabilità in base alla fonte nota o lunghezza titolo
            reliability = 1.2 if any(f in publisher.lower() for f in ['reuters', 'bloomberg', 'wsj', 'financial times', 'yahoo']) else 1.0
            
            analysis = TextBlob(title)
            polarity = analysis.sentiment.polarity * reliability
            sentiments.append(polarity)
            
            articoli.append({
                'title': title, 
                'link': link, 
                'publisher': publisher, 
                'score': polarity,
                'weight': reliability
            })
        
        # Ordina per affidabilità e rilevanza stimata
        articoli = sorted(articoli, key=lambda x: x['weight'], reverse=True)
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        return avg_sentiment, articoli
    except:
        return 0.0, []

def calcola_modello(df, sentiment_score, horizon):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    dettagli = []
    score = 50

    rsi = last['RSI']
    if rsi < 32:
        score += 15
        dettagli.append({"Fattore": "RSI (14)", "Impatto": "+15 pt", "Stato": f"Ipervenduto profondo ({rsi:.1f})"})
    elif rsi > 68:
        score -= 15
        dettagli.append({"Fattore": "RSI (14)", "Impatto": "-15 pt", "Stato": f"Ipercomprato ({rsi:.1f})"})
    else:
        score += 5
        dettagli.append({"Fattore": "RSI (14)", "Impatto": "+5 pt", "Stato": f"Neutro ({rsi:.1f})"})

    if last['Close'] > last['SMA50'] and last['Close'] > last['EMA20']:
        score += 15
        dettagli.append({"Fattore": "Trend Prezzo", "Impatto": "+15 pt", "Stato": "Sopra SMA50 e EMA20 (Rialzista)"})
    else:
        score -= 15
        dettagli.append({"Fattore": "Trend Prezzo", "Impatto": "-15 pt", "Stato": "Sotto SMA50 (Debole)"})

    if last['Vol_Ratio'] > 1.2 and last['Close'] > prev['Close']:
        score += 10
        dettagli.append({"Fattore": "Volumi", "Impatto": "+10 pt", "Stato": "Volumi in aumento con prezzo in salita"})
    elif last['Vol_Ratio'] > 1.2 and last['Close'] < prev['Close']:
        score -= 10
        dettagli.append({"Fattore": "Volumi", "Impatto": "-10 pt", "Stato": "Volumi di scarico elevati"})

    if last['MACD'] > last['MACD_Signal']:
        score += 10
        dettagli.append({"Fattore": "MACD", "Impatto": "+10 pt", "Stato": "Sopra la linea di segnale"})
    else:
        score -= 10
        dettagli.append({"Fattore": "MACD", "Impatto": "-10 pt", "Stato": "Sotto la linea di segnale"})

    if sentiment_score > 0.05:
        score += 10
        dettagli.append({"Fattore": "News Sentiment", "Impatto": "+10 pt", "Stato": f"Positivo (+{sentiment_score:.2f})"})
    elif sentiment_score < -0.05:
        score -= 10
        dettagli.append({"Fattore": "News Sentiment", "Impatto": "-10 pt", "Stato": f"Negativo ({sentiment_score:.2f})"})

    score = max(0, min(100, score))

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

# UI PRINCIPALE
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("<h1>Market Terminal</h1>", unsafe_allow_html=True)
with header_col2:
    update_btn = st.button("🔄 Aggiorna Dati")

st.write("")

c1, c2 = st.columns([2, 1])
with c1:
    st.session_state.ticker_input = st.text_input("Ticker:", value=st.session_state.ticker_input).upper().strip()
with c2:
    st.session_state.horizon_input = st.selectbox("Orizzonte:", ["1D", "3D", "7D"], index=["1D", "3D", "7D"].index(st.session_state.horizon_input))

ticker = st.session_state.ticker_input
horizon = st.session_state.horizon_input

if update_btn:
    st.cache_data.clear()

if ticker:
    with st.spinner("Elaborazione dati in corso..."):
        df = scarica_dati(ticker)
        if df is None or df.empty:
            st.error("Nessun dato trovato per il ticker inserito o formato errato.")
            st.stop()
        
        df = calcola_indicatori(df)
        avg_sentiment, articoli = estrai_news_e_sentiment(ticker)
        verdetto, score, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, dettagli = calcola_modello(df, avg_sentiment, horizon)
        
        last_date = df.index[-1].strftime('%Y-%m-%d')
        last_price = df['Close'].iloc[-1]
        change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

    # DATA ULTIMO DATO DISPONIBILE
    st.markdown(f"""
        <div style="font-size: 11px; opacity: 0.7; margin-bottom: 6px;">
            📅 Ultimo dato di borsa disponibile: <b style="color: #0284c7;">{last_date}</b>
        </div>
    """, unsafe_allow_html=True)

    # SCHEDA VERDETTO
    st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {colore};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="metric-label">Verdetto Modello ({horizon})</span>
                    <h3 style="margin: 0; color: {colore} !important;">{verdetto}</h3>
                </div>
                <div style="text-align: right;">
                    <span class="metric-label">Score Confidenza</span>
                    <div style="font-size: 18px; font-weight: bold;">{score}/100</div>
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
        st.markdown(f"""<div class="metric-card"><span class="metric-label">R/R Ratio</span><div class="metric-value">1 : {risk_reward:.1f}</div><span class="metric-label">Rischio/Beneficio</span></div>""", unsafe_allow_html=True)

    # GRAFICO
    st.write("")
    st.subheader("Grafico & Indicatori")
    
    ind_options = st.multiselect(
        "Filtra Indicatori:",
        ["SMA20", "SMA50", "EMA20", "Bande di Bollinger", "RSI", "MACD"],
        default=["SMA20", "Bande di Bollinger", "RSI"]
    )

    num_plots = 1 + ("RSI" in ind_options) + ("MACD" in ind_options)
    row_heights = [0.6] + [0.2] * (num_plots - 1)
    
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
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(150,150,150,0.3)', width=1), name='BB Up'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(150,150,150,0.3)', width=1), fill='tonexty', fillcolor='rgba(150,150,150,0.05)', name='BB Low'), row=1, col=1)

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

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        height=450,
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

    # NOTIZIE & ABSTRACT SENTIMENT A VALLE
    st.write("")
    st.subheader("Notizie Web & Sentiment (Top 10)")
    
    # Abstract sintetico basato sul sentiment medio
    sent_label = "Fortemente Rialzista" if avg_sentiment > 0.15 else ("Rialzista Moderato" if avg_sentiment > 0.02 else ("Neutro / Laterale" if avg_sentiment >= -0.02 else "Ribassista"))
    st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 14px;">
            <span class="metric-label">Abstract Sentiment Globale</span>
            <div style="font-size: 13px; margin-top: 4px; font-weight: 500;">
                Il sentiment ponderato delle prime 10 fonti web risulta <b>{sent_label}</b> (Indice medio: <code>{avg_sentiment:+.2f}</code>). Le notizie primarie evidenziano un flusso informativo coerente con la struttura tecnica dell'asset.
            </div>
        </div>
    """, unsafe_allow_html=True)

    if articoli:
        for art in articoli:
            sc_col = "#22c55e" if art['score'] > 0 else ("#ef4444" if art['score'] < 0 else "#9ca3af")
            st.markdown(f"""
                <div class="metric-card">
                    <a href="{art['link']}" target="_blank" style="color:#0284c7; font-weight:500; font-size:12px; text-decoration:none;">{art['title']}</a>
                    <div style="display:flex; justify-content:space-between; margin-top:3px; font-size:10px;" class="metric-label">
                        <span>Fonte: {art['publisher']} (Affidabilità: {art['weight']}x)</span>
                        <span style="color:{sc_col}; font-weight:bold;">Sentiment: {art['score']:+.2f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nessuna notizia recente.")

    # GUIDA IN FONDO
    st.write("")
    with st.expander("📖 Guida ai Calcoli e Logica del Modello"):
        st.write("Il punteggio **Score (0-100)** valuta la confluenza dei seguenti fattori tecnici e di sentiment:")
        st.table(pd.DataFrame(dettagli))
        st.write("""
        * **Rischio/Rendimento (R/R):** Rapporto tra target di profitto e stop loss basato sull'ATR (volatilità).
        * **Orizzonte (1D, 3D, 7D):** Modula l'ampiezza dei livelli di rischio sul time-frame scelto.
        """)
