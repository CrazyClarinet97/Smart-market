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

# STILE CSS "LIQUID GLASS" UNIVERSALE E PROPORZIONATO
st.markdown("""
    <style>
    .metric-card {
        background: rgba(150, 150, 150, 0.06);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(150, 150, 150, 0.15);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        height: 100%;
    }
    .date-card {
        background: rgba(2, 132, 199, 0.08);
        border: 1px solid rgba(2, 132, 199, 0.25);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 14px;
        font-size: 13px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: rgba(25, 30, 45, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        .date-card {
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.2);
            color: #f3f4f6;
        }
    }
    h1 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .metric-label {
        font-size: 11px;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 17px;
        font-weight: 700;
        color: #0284c7;
    }
    @media (prefers-color-scheme: dark) {
        .metric-val { color: #38bdf8; }
    }
    .metric-sub {
        font-size: 11px;
        font-weight: 600;
        margin-top: 2px;
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
    df = df.dropna(subset=['Close'])
    
    # Download Benchmark per Idea 1 (Forza Relativa)
    bench_ticker = "FTSEMIB.MI" if ticker.endswith(".MI") else "^GSPC"
    df_bench = yf.download(bench_ticker, period="1y", interval="1d", progress=False)
    if not df_bench.empty:
        if isinstance(df_bench.columns, pd.MultiIndex):
            df_bench.columns = df_bench.columns.get_level_values(0)
        # Allineamento date
        df['Bench_Close'] = df_bench['Close'].reindex(df.index, method='ffill')
    else:
        df['Bench_Close'] = df['Close'] # Fallback neutro
        
    return df

def calcola_indicatori(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI 14 e RSI 7
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

    # ATR (Average True Range)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # Idea 3: Percentile ATR (Regime di Volatilità a 252 periodi)
    df['ATR_Pctile'] = df['ATR'].rolling(window=252, min_periods=20).rank(pct=True)

    # Idea 1: Relative Strength Line (Asset / Benchmark) e SMA di RS
    df['RS'] = df['Close'] / (df['Bench_Close'] + 1e-9)
    df['RS_SMA20'] = df['RS'].rolling(window=20).mean()
    df['RS_SMA50'] = df['RS'].rolling(window=50).mean()

    # Volume Ratio
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA20'] + 1e-9)
    return df

def estrai_news_e_sentiment(ticker):
    try:
        tk = yf.Ticker(ticker)
        news_list = tk.news
        if not news_list:
            # Fallback su query base senza suffisso .MI se vuoto
            base_ticker = ticker.split('.')[0]
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={base_ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers).json()
            news_list = response.get('news', [])
            
        news_list = news_list[:10]
        sentiments, articoli = [], []
        for item in news_list:
            title = item.get('title', item.get('content', {}).get('title', ''))
            link = item.get('link', item.get('content', {}).get('clickThroughUrl', {}).get('url', '#'))
            publisher = item.get('publisher', item.get('content', {}).get('provider', {}).get('displayName', 'Fonte Web'))
            if not title:
                continue
                
            reliability = 1.3 if any(f in publisher.lower() for f in ['reuters', 'bloomberg', 'wsj', 'financial times', 'yahoo', 'ansa', 'il sole 24 ore']) else 1.0
            
            analysis = TextBlob(title)
            polarity = analysis.sentiment.polarity * reliability
            sentiments.append(polarity)
            
            articoli.append({
                'title': title, 'link': link, 'publisher': publisher, 'score': polarity, 'weight': reliability
            })
        
        articoli = sorted(articoli, key=lambda x: x['weight'], reverse=True)
        return (np.mean(sentiments) if sentiments else 0.0), articoli
    except:
        return 0.0, []

def calcola_modello_quant(df, sentiment_score, horizon):
    """
    MODELLO QUANTITATIVO AVANZATO CON IDEA 1 (FORZA RELATIVA) & IDEA 3 (REGIME VOLATILITÀ)
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    dettagli = []
    score = 50.0  # Base neutra

    # 1. RSI Confluence (14 & 7)
    rsi14 = last['RSI']
    rsi7 = last['RSI7']
    if rsi14 < 32 and rsi7 < 30:
        score += 15
        dettagli.append({"Fattore": "RSI Confluence (14 & 7)", "Peso": "+15", "Stato": f"Ipervenduto profondo (RSI14: {rsi14:.1f})"})
    elif rsi14 > 68 and rsi7 > 70:
        score -= 15
        dettagli.append({"Fattore": "RSI Confluence (14 & 7)", "Peso": "-15", "Stato": f"Ipercomprato critico (RSI14: {rsi14:.1f})"})
    else:
        score += 3
        dettagli.append({"Fattore": "RSI (14)", "Peso": "+3", "Stato": f"Fascia di equilibrio ({rsi14:.1f})"})

    # 2. Trend Strutturale (SMA50 & EMA20)
    if last['Close'] > last['SMA50'] and last['EMA20'] > last['SMA50']:
        score += 15
        dettagli.append({"Fattore": "Trend Strutturale (SMA50/EMA20)", "Peso": "+15", "Stato": "Uptrend confermato sopra le medie"})
    elif last['Close'] < last['SMA50']:
        score -= 15
        dettagli.append({"Fattore": "Trend Strutturale (SMA50)", "Peso": "-15", "Stato": "Downtrend sotto la SMA50"})

    # 3. IDEA 1: Forza Relativa vs Benchmark (Relative Strength Line)
    rs_val = last['RS']
    rs_sma20 = last['RS_SMA20']
    rs_sma50 = last['RS_SMA50']
    if rs_val > rs_sma20 and rs_sma20 > rs_sma50:
        score += 12
        dettagli.append({"Fattore": "Forza Relativa (Idea 1 vs Benchmark)", "Peso": "+12", "Stato": "Asset sovraperforma il mercato di riferimento"})
    elif rs_val < rs_sma20:
        score -= 12
        dettagli.append({"Fattore": "Forza Relativa (Idea 1 vs Benchmark)", "Peso": "-12", "Stato": "Asset sottoperforma l'indice di mercato"})

    # 4. IDEA 3: Filtro di Regime di Volatilità (ATR Percentile)
    atr_pctile = last['ATR_Pctile']
    if not np.isnan(atr_pctile):
        if atr_pctile > 0.85:
            score -= 12
            dettagli.append({"Fattore": "Regime Volatilità ATR (Idea 3)", "Peso": "-12", "Stato": f"Alta instabilità / panico (Percentile {atr_pctile*100:.0f}%)"})
        elif atr_pctile < 0.20:
            score += 6
            dettagli.append({"Fattore": "Regime Volatilità ATR (Idea 3)", "Peso": "+6", "Stato": f"Bassa volatilità / stabilità (Percentile {atr_pctile*100:.0f}%)"})

    # 5. Order Flow (Volume Ratio)
    vol_ratio = last['Vol_Ratio']
    if vol_ratio > 1.3 and last['Close'] > prev['Close']:
        score += 10
        dettagli.append({"Fattore": "Order Flow (Volumi)", "Peso": "+10", "Stato": f"Pressione d'acquisto (Vol {vol_ratio:.1f}x)"})
    elif vol_ratio > 1.3 and last['Close'] < prev['Close']:
        score -= 10
        dettagli.append({"Fattore": "Order Flow (Volumi)", "Peso": "-10", "Stato": f"Pressione di scarico (Vol {vol_ratio:.1f}x)"})

    # 6. MACD Momentum
    if last['MACD'] > last['MACD_Signal'] and last['MACD_Hist'] > prev['MACD_Hist']:
        score += 8
        dettagli.append({"Fattore": "MACD Momentum", "Peso": "+8", "Stato": "Crossover e istogramma espansivo"})
    elif last['MACD'] < last['MACD_Signal']:
        score -= 8
        dettagli.append({"Fattore": "MACD Momentum", "Peso": "-8", "Stato": "Sotto la linea di segnale"})

    # 7. Sentiment Ponderato
    if sentiment_score > 0.04:
        score += 8
        dettagli.append({"Fattore": "Sentiment Web Ponderato", "Peso": "+8", "Stato": f"Flusso notizie positivo (+{sentiment_score:.2f})"})
    elif sentiment_score < -0.04:
        score -= 8
        dettagli.append({"Fattore": "Sentiment Web Ponderato", "Peso": "-8", "Stato": f"Flusso notizie negativo ({sentiment_score:.2f})"})

    score = max(0.0, min(100.0, score))

    # ATR e Orizzonte con compensazione dinamica per Idea 3 (Regime di volatilità)
    atr = last['ATR']
    close = last['Close']
    
    if horizon == "1D":
        atr_mult, sl_buf, tp_buf = 1.0, 1.1, 1.8
    elif horizon == "3D":
        atr_mult, sl_buf, tp_buf = 1.8, 1.2, 2.2
    else:
        atr_mult, sl_buf, tp_buf = 2.6, 1.3, 2.8

    # Se siamo in regime di alta volatilità (Idea 3), allarghiamo lo stop loss per evitare falsi stop
    if not np.isnan(atr_pctile) and atr_pctile > 0.85:
        sl_buf *= 1.3

    sl_val = close - (atr * atr_mult * sl_buf)
    tp_val = close + (atr * atr_mult * tp_buf)
    sl_pct = ((sl_val - close) / close) * 100
    tp_pct = ((tp_val - close) / close) * 100
    risk_reward = abs(tp_pct / sl_pct) if sl_pct != 0 else 0

    if score >= 62.0:
        verdetto, colore = "BUY", "#238636"
    elif score <= 38.0:
        verdetto, colore = "SELL", "#da3633"
    else:
        verdetto, colore = "HOLD", "#d29922"

    return verdetto, score, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, dettagli

# UI PRINCIPALE
st.markdown("<h1>Market Terminal</h1>", unsafe_allow_html=True)
st.write("")

c1, c2 = st.columns([2, 1])
with c1:
    st.session_state.ticker_input = st.text_input("Ticker:", value=st.session_state.ticker_input).upper().strip()
with c2:
    st.session_state.horizon_input = st.selectbox("Orizzonte:", ["1D", "3D", "7D"], index=["1D", "3D", "7D"].index(st.session_state.horizon_input))

ticker = st.session_state.ticker_input
horizon = st.session_state.horizon_input

if ticker:
    with st.spinner("Elaborazione dati quantitativi in corso..."):
        df = scarica_dati(ticker)
        if df is None or df.empty:
            st.error("Nessun dato trovato per il ticker inserito o formato errato.")
            st.stop()
        
        df = calcola_indicatori(df)
        avg_sentiment, articoli = estrai_news_e_sentiment(ticker)
        verdetto, score, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, dettagli = calcola_modello_quant(df, avg_sentiment, horizon)
        
        last_date = df.index[-1].strftime('%Y-%m-%d')
        last_price = df['Close'].iloc[-1]
        change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

    # BOX DATA
    st.markdown(f"""
        <div class="date-card">
            <span>📅</span>
            <span>Ultima data: <b>{last_date}</b></span>
        </div>
    """, unsafe_allow_html=True)

    # SCHEDA VERDETTO
    st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {colore};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="metric-label">Signal ({horizon})</span>
                    <div style="font-size: 20px; font-weight: 800; color: {colore}; margin-top: 2px;">{verdetto}</div>
                </div>
                <div style="text-align: right;">
                    <span class="metric-label">Quant Score</span>
                    <div style="font-size: 18px; font-weight: bold;">{score:.0f} / 100</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # KPI RIGHE
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ch_color = '#238636' if change_pct >= 0 else '#da3633'
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prezzo Attuale</div>
                <div class="metric-val">${last_price:.2f}</div>
                <div class="metric-sub" style="color: {ch_color};">{change_pct:+.2f}% oggi</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Stop Loss</div>
                <div class="metric-val" style="color: #da3633;">${sl_val:.2f}</div>
                <div class="metric-sub" style="color: #da3633;">{sl_pct:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Take Profit</div>
                <div class="metric-val" style="color: #238636;">${tp_val:.2f}</div>
                <div class="metric-sub" style="color: #238636;">{tp_pct:+.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">R/R Ratio</div>
                <div class="metric-val">1 : {risk_reward:.1f}</div>
                <div class="metric-sub" style="opacity: 0.7;">Rischio / Premio</div>
            </div>
        """, unsafe_allow_html=True)

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

    # NOTIZIE & ABSTRACT SENTIMENT
    st.write("")
    st.subheader("Notizie Web & Sentiment (Top 10)")
    
    sent_label = "Fortemente Rialzista" if avg_sentiment > 0.15 else ("Rialzista Moderato" if avg_sentiment > 0.02 else ("Neutro / Laterale" if avg_sentiment >= -0.02 else "Ribassista"))
    st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 14px;">
            <div class="metric-label">Abstract Sentiment Globale</div>
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

    # GUIDA MATEMATICA AGGIORNATA
    st.write("")
    with st.expander("📖 Documentazione Matematica e Modello Quantitativo (Aggiornato Idea 1 & 3)"):
        st.markdown("""
        ### 1. Architettura del Modello Quantitativo con Relative Strength e Regime di Volatilità
        Il terminale calcola lo **Score Globale** $S \in [0, 100]$ partendo da $50.0$. Ora include:
        """)
        st.table(pd.DataFrame(dettagli))
        st.markdown("""
        ### 2. Dettaglio Matematico delle Nuove Implementazioni
        * **Idea 1 - Forza Relativa (Relative Strength):**
          Calcola il rapporto tra il prezzo dell'asset e il benchmark di riferimento ($RS = \\frac{P_{\\text{asset}}}{P_{\\text{bench}}}$, dove il benchmark è il *FTSEMIB.MI* per i ticker italiani o l'*S&P 500* per gli esteri). Se la media mobile a 20 periodi di RS è superiore alla 50 periodi, l'asset dimostra una forza superiore al mercato, aggiungendo punti al quant score.
        
        * **Idea 3 - Filtro di Regime di Volatilità (ATR Percentile):**
          Calcola il rango percentuale dell'ATR odierno rispetto agli ultimi 252 giorni di negoziazione:
          $$\\text{Rank}_t = \\frac{1}{252} \\sum_{i=0}^{251} \\mathbb{I}(\\text{ATR}_{t-i} \\le \\text{ATR}_t)$$
          Se $\\text{Rank}_t > 0.85$ (regime di alta volatilità / panico), il modello penalizza lo score e dilata automaticamente il buffer di stop loss per evitare falsi stop guidati dal rumore di mercato.

        ### 3. Logica delle Fasce Operative
        * **BUY ($S \\ge 62.0$):** Confluenza positiva di trend, volumi, forza relativa e sentiment.
        * **HOLD ($38.0 < S < 62.0$):** Indecisione o divergenza tra indicatori.
        * **SELL ($S \\le 38.0$):** Pressione ribassista o volatilità critica in distribuzione.
        """)
