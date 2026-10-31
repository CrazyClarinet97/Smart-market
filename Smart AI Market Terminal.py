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
    page_title="AI Market Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# FORZATURA TEMA DARK PERFETTO (Nessun problema con temi chiari)
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    h1, h2, h3, h4 {
        color: #f0f6fc !important;
    }
    .stSelectbox label, .stTextInput label, .stMultiSelect label {
        color: #8b949e !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border-color: #30363d !important;
    }
    input {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border-color: #30363d !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CALCOLI TECNICI & INDICATORI
# ---------------------------------------------------------
def calcola_indicatori(df):
    # Medie Mobili
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

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

    # Stocastico (%K e %D)
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low14) / (high14 - low14 + 1e-9))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # ATR (Average True Range) per volatilità/SL/TP
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    return df

# ---------------------------------------------------------
# NOTIZIE & SENTIMENT
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
            publisher = item.get('publisher', 'Fonte sconosciuta')
            analysis = TextBlob(title)
            polarity = analysis.sentiment.polarity
            sentiments.append(polarity)
            articoli.append({
                'title': title,
                'link': link,
                'publisher': publisher,
                'score': polarity
            })
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        return avg_sentiment, articoli
    except:
        return 0.0, []

# ---------------------------------------------------------
# MODELLO ALGORITMICO DI SCORE (0-100)
# ---------------------------------------------------------
def calcola_modello_predittivo(df, sentiment_score, horizon):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    dettagli = []
    score_tecnico = 50 # Base neutra

    # 1. RSI (Max +15 / -15)
    rsi = last['RSI']
    if rsi < 30:
        score_tecnico += 15
        dettagli.append({"Fattore": "RSI (<30)", "Impatto": "+15 pt", "Stato": f"Ipervenduto ({rsi:.1f}) - Forte segnale d'acquisto"})
    elif rsi > 70:
        score_tecnico -= 15
        dettagli.append({"Fattore": "RSI (>70)", "Impatto": "-15 pt", "Stato": f"Ipercomprato ({rsi:.1f}) - Rischio correzione"})
    elif 45 <= rsi <= 60:
        score_tecnico += 5
        dettagli.append({"Fattore": "RSI Neutro/Bullish", "Impatto": "+5 pt", "Stato": f"Zonale equilibrato ({rsi:.1f})"})

    # 2. Medie Mobili (Max +15 / -15)
    if last['Close'] > last['SMA50'] and last['Close'] > last['EMA20']:
        score_tecnico += 15
        dettagli.append({"Fattore": "Trend Medie", "Impatto": "+15 pt", "Stato": "Prezzo sopra SMA50 e EMA20 (Trend Rialzista)"})
    elif last['Close'] < last['SMA50']:
        score_tecnico -= 15
        dettagli.append({"Fattore": "Trend Medie", "Impatto": "-15 pt", "Stato": "Prezzo sotto SMA50 (Trend Ribassista)"})

    # 3. MACD (Max +10 / -10)
    if last['MACD'] > last['MACD_Signal'] and last['MACD_Hist'] > prev['MACD_Hist']:
        score_tecnico += 10
        dettagli.append({"Fattore": "MACD", "Impatto": "+10 pt", "Stato": "Crossover e Momentum Positivo"})
    elif last['MACD'] < last['MACD_Signal']:
        score_tecnico -= 10
        dettagli.append({"Fattore": "MACD", "Impatto": "-10 pt", "Stato": "Cross Ribassista sopra la Signal Line"})

    # 4. Stocastico (Max +10 / -10)
    if last['Stoch_K'] < 20:
        score_tecnico += 10
        dettagli.append({"Fattore": "Stocastico", "Impatto": "+10 pt", "Stato": f"Ipervenduto ({last['Stoch_K']:.1f})"})
    elif last['Stoch_K'] > 80:
        score_tecnico -= 10
        dettagli.append({"Fattore": "Stocastico", "Impatto": "-10 pt", "Stato": f"Ipercomprato ({last['Stoch_K']:.1f})"})

    # 5. Sentiment News (Max +10 / -10)
    if sentiment_score > 0.1:
        score_tecnico += 10
        dettagli.append({"Fattore": "Sentiment Notizie", "Impatto": "+10 pt", "Stato": f"Positivo (Polarità: +{sentiment_score:.2f})"})
    elif sentiment_score < -0.1:
        score_tecnico -= 10
        dettagli.append({"Fattore": "Sentiment Notizie", "Impatto": "-10 pt", "Stato": f"Negativo (Polarità: {sentiment_score:.2f})"})

    score = max(0, min(100, score_tecnico))

    # Calcolo SL e TP dinamici basati sull'ATR
    atr = last['ATR']
    close = last['Close']
    atr_pct = (atr / close)

    multiplier = 1.0 if horizon == "1D" else (1.8 if horizon == "3D" else 2.5)
    
    sl_val = close - (atr * multiplier * 1.2)
    tp_val = close + (atr * multiplier * 2.2)

    sl_pct = ((sl_val - close) / close) * 100
    tp_pct = ((tp_val - close) / close) * 100

    if score >= 65:
        verdetto = "BUY (ACQUISTO)"
        colore = "#238636"
    elif score <= 40:
        verdetto = "SELL (VENDITA)"
        colore = "#da3633"
    else:
        verdetto = "HOLD (ATTENDI)"
        colore = "#d29922"

    return verdetto, score, colore, sl_val, sl_pct, tp_val, tp_pct, dettagli

# ---------------------------------------------------------
# INTERFACCIA UTENTE (STREAMLIT)
# ---------------------------------------------------------

st.title("📈 Market Terminal & Predictive Engine")

# Controlli in alto
c1, c2 = st.columns([2, 1])
with c1:
    ticker = st.text_input("Inserisci Ticker (es. AAPL, NVDA, SXLK.MI, BTC-USD):", value="AAPL").upper().strip()
with c2:
    horizon = st.selectbox("Orizzonte Temporale:", ["1D", "3D", "7D"], index=0)

if ticker:
    with st.spinner("Estrazione dati di borsa ed elaborazione indicatori..."):
        try:
            df = yf.download(ticker, period="1y", interval="1d")
            if df.empty:
                st.error("Nessun dato trovato per il ticker specificato.")
                st.stop()
            
            # Gestione MultiIndex se restituito da yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = calcola_indicatori(df)
            avg_sentiment, articoli = estrai_news_e_sentiment(ticker)
            verdetto, score, colore, sl_val, sl_pct, tp_val, tp_pct, dettagli = calcola_modello_predittivo(df, avg_sentiment, horizon)
            last_price = df['Close'].iloc[-1]
            change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

        except Exception as e:
            st.error(f"Errore durante l'elaborazione dei dati: {e}")
            st.stop()

    # SCHEDA VERDETTO & RISCHIO
    st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid {colore};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="metric-label">Verdetto Modello Algoritmico ({horizon})</span>
                    <h2 style="margin: 0; color: {colore} !important;">{verdetto}</h2>
                </div>
                <div style="text-align: right;">
                    <span class="metric-label">Score Confidenza</span>
                    <div style="font-size: 24px; font-weight: bold; color: #f0f6fc;">{score}/100</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Riquadri KPI
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="metric-card"><span class="metric-label">Prezzo Attuale</span><div class="metric-value">${last_price:.2f}</div><span style="color:{'#238636' if change_pct>=0 else '#da3633'}; font-size:12px;">{change_pct:+.2f}% oggi</span></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card"><span class="metric-label">Stop Loss Suggerito</span><div class="metric-value" style="color:#da3633;">${sl_val:.2f}</div><span style="color:#da3633; font-size:12px;">{sl_pct:.2f}%</span></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="metric-card"><span class="metric-label">Take Profit Suggerito</span><div class="metric-value" style="color:#238636;">${tp_val:.2f}</div><span style="color:#238636; font-size:12px;">{tp_pct:+.2f}%</span></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="metric-card"><span class="metric-label">Sentiment News</span><div class="metric-value" style="color:#58a6ff;">{avg_sentiment:+.2f}</div><span style="color:#8b949e; font-size:12px;">Analisi Web Feed</span></div>""", unsafe_allow_html=True)

    # GRAFICO INTERATTIVO MULTI-INDICATORE
    st.subheader("📊 Grafico Avanzato & Indicatori")
    
    ind_options = st.multiselect(
        "Attiva/Disattiva Indicatori sul Grafico:",
        ["SMA20", "SMA50", "EMA20", "Bande di Bollinger", "RSI", "MACD", "Stocastico"],
        default=["SMA20", "Bande di Bollinger", "RSI"]
    )

    # Subplots Plotly
    num_plots = 1 + ("RSI" in ind_options) + ("MACD" in ind_options) + ("Stocastico" in ind_options)
    row_heights = [0.5] + [0.16] * (num_plots - 1)
    
    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=row_heights)

    # Prezzo Candele
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Prezzo'
    ), row=1, col=1)

    # Medie
    if "SMA20" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#e3b341', width=1), name='SMA 20'), row=1, col=1)
    if "SMA50" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='#a371f7', width=1), name='SMA 50'), row=1, col=1)
    if "EMA20" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#58a6ff', width=1), name='EMA 20'), row=1, col=1)

    # Bollinger
    if "Bande di Bollinger" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(255,255,255,0.3)', width=1), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(255,255,255,0.3)', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name='BB Lower'), row=1, col=1)

    curr_row = 2
    if "RSI" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#f0883e', width=1.5), name='RSI (14)'), row=curr_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=curr_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=curr_row, col=1)
        curr_row += 1

    if "MACD" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#58a6ff', width=1.5), name='MACD'), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#f0883e', width=1.5), name='Signal'), row=curr_row, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Hist'), row=curr_row, col=1)
        curr_row += 1

    if "Stocastico" in ind_options:
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='#58a6ff', width=1.5), name='%K'), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='#f0883e', width=1.5), name='%D'), row=curr_row, col=1)
        curr_row += 1

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        margin=dict(l=10, r=10, t=10, b=10),
        height=550,
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # SPIEGAZIONE LOGICA & CALCOLI
    with st.expander("🧠 Spiegazione Logica & Calcoli del Modello AI"):
        st.write("Il punteggio di confidenza **Score (0-100)** è ottenuto partendo da una base neutra (50) e sommando/sottraendo i seguenti pesi matematici:")
        st.table(pd.DataFrame(dettagli))
        st.write("""
        * **Stop Loss & Take Profit:** Sono calcolati moltiplicando la volatilità reale dell'asset (**ATR - Average True Range**) per il coefficiente dell'orizzonte temporale selezionato.
        * **Orizzonte (1D, 3D, 7D):** Aumentando i giorni, il modello amplia i margini di SL/TP per evitare falsi segnali dovuti al rumore di mercato.
        """)

    # NOTIZIE RECENTI
    st.subheader("📰 Notizie Principali & Sentiment")
    if articoli:
        for art in articoli:
            sent_color = "#238636" if art['score'] > 0 else ("#da3633" if art['score'] < 0 else "#8b949e")
            st.markdown(f"""
                <div class="metric-card">
                    <a href="{art['link']}" target="_blank" style="color:#58a6ff; font-weight:bold; font-size:14px; text-decoration:none;">{art['title']}</a>
                    <div style="display:flex; justify-between; margin-top:6px; font-size:11px; color:#8b949e;">
                        <span>Fonte: {art['publisher']}</span>
                        <span style="color:{sent_color}; font-weight:bold;">Sentiment Score: {art['score']:+.2f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nessuna notizia rilevata di recente per questo ticker.")
