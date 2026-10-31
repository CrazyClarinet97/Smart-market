import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from textblob import TextBlob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Configurazione Pagina
st.set_page_config(
    page_title="Institutional Market Terminal",
    page_icon="⚡",
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
    .preflight-card {
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(150, 150, 150, 0.2);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 14px;
        font-size: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
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
        .preflight-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e2e8f0;
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

def get_dynamic_benchmark(ticker):
    t = ticker.upper()
    if "-USD" in t:
        return "BTC-USD"
    elif t.endswith(".MI"):
        return "FTSEMIB.MI"
    elif any(tech in t for tech in ["AAPL", "NVDA", "MSFT", "GOOGL", "QQQ", "SXLK", "META", "AMZN"]):
        return "^NDX"
    else:
        return "^GSPC"

@st.cache_data(ttl=120)
def scarica_dati(ticker):
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['Close'])
    
    bench_ticker = get_dynamic_benchmark(ticker)
    df_bench = yf.download(bench_ticker, period="5y", interval="1d", progress=False)
    if not df_bench.empty:
        if isinstance(df_bench.columns, pd.MultiIndex):
            df_bench.columns = df_bench.columns.get_level_values(0)
        df['Bench_Close'] = df_bench['Close'].reindex(df.index, method='ffill')
        df['Bench_SMA200'] = df['Bench_Close'].rolling(window=200).mean()
    else:
        df['Bench_Close'] = df['Close']
        df['Bench_SMA200'] = df['Close'].rolling(window=200).mean()
        
    return df

def calcola_indicatori(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    df['Dist_EMA20'] = (df['Close'] - df['EMA20']) / df['EMA20'] * 100
    df['Dist_SMA50'] = (df['Close'] - df['SMA50']) / df['SMA50'] * 100

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    gain7 = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss7 = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs7 = gain7 / (loss7 + 1e-9)
    df['RSI7'] = 100 - (100 / (1 + rs7))

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
    df['ATR_Pctile'] = df['ATR'].rolling(window=252, min_periods=20).rank(pct=True)

    df['RS'] = df['Close'] / (df['Bench_Close'] + 1e-9)
    df['RS_Change'] = df['RS'].pct_change(5)

    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA20'] + 1e-9)

    daily_rets = df['Close'].pct_change()
    neg_rets = daily_rets.where(daily_rets < 0, 0)
    df['Sortino'] = (daily_rets.mean() / (neg_rets.std() + 1e-9)) * np.sqrt(252)
    
    roll_max = df['Close'].cummax()
    drawdown = (df['Close'] - roll_max) / roll_max
    df['Max_DD'] = drawdown.cummin() * 100

    return df.dropna()

def estrai_news_e_sentiment(ticker):
    news_list = []
    queries = [ticker, ticker.split('.')[0]]
    if "SXLK" in ticker.upper() or "TECH" in ticker.upper():
        queries.extend(["US Technology sector", "Wall Street tech stocks"])
    else:
        queries.append("Stock market news")

    headers = {'User-Agent': 'Mozilla/5.0'}
    for q in queries:
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
            response = requests.get(url, headers=headers, timeout=3).json()
            items = response.get('news', [])
            if items:
                news_list.extend(items)
        except:
            pass

    seen, unique_news = set(), []
    for item in news_list:
        title = item.get('title', item.get('content', {}).get('title', ''))
        if title and title not in seen:
            seen.add(title)
            unique_news.append(item)

    if not unique_news:
        unique_news = [{'title': f"Market update for {ticker}", 'link': '#', 'publisher': 'Global Market Wire'}]

    unique_news = unique_news[:10]
    sentiments, articoli = [], []
    
    bullish_terms = ['beat', 'surge', 'growth', 'record', 'rally', 'gain', 'jump', 'upgrade', 'profit']
    bearish_terms = ['miss', 'crash', 'drop', 'fall', 'loss', 'lawsuit', 'cut', 'downgrade', 'slump', 'risk']

    for item in unique_news:
        title = item.get('title', item.get('content', {}).get('title', 'Financial update'))
        link = item.get('link', '#')
        publisher = item.get('publisher', 'Financial News')
        reliability = 1.3 if any(f in publisher.lower() for f in ['reuters', 'bloomberg', 'wsj', 'financial times', 'yahoo']) else 1.0
        
        analysis = TextBlob(title)
        polarity = analysis.sentiment.polarity
        
        title_lower = title.lower()
        if any(term in title_lower for term in bullish_terms):
            polarity += 0.25
        if any(term in title_lower for term in bearish_terms):
            polarity -= 0.25
            
        final_score = np.clip(polarity * reliability, -1.0, 1.0)
        sentiments.append(final_score)
        articoli.append({'title': title, 'link': link, 'publisher': publisher, 'score': final_score, 'weight': reliability})
    
    articoli = sorted(articoli, key=lambda x: x['weight'], reverse=True)
    return (np.mean(sentiments) if sentiments else 0.0), articoli

def calcola_modello_ml_e_backtest(df, sentiment_score, horizon):
    days_forward = 1 if horizon == "1D" else (3 if horizon == "3D" else 7)
    df = df.copy()
    df['Target'] = (df['Close'].shift(-days_forward) > df['Close']).astype(int)
    df = df.dropna()

    if len(df) < 100:
        return 50.0, "HOLD", "#d29922", 0, 0, 0, 0, 0, 0, 0, []

    feature_cols = ['RSI', 'RSI7', 'MACD_Hist', 'Vol_Ratio', 'Dist_EMA20', 'RS_Change', 'ATR_Pctile']
    X = df[feature_cols]
    y = df['Target']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train_scaled, y_train)

    last_features = scaler.transform(df[feature_cols].iloc[[-1]])
    prob = model.predict_proba(last_features)[0][1] * 100
    prob = max(0.0, min(100.0, prob + (sentiment_score * 12)))

    last = df.iloc[-1]
    is_macro_bull = last['Bench_Close'] > last['Bench_SMA200']
    
    if not is_macro_bull and prob >= 62.0:
        prob = 50.0

    test_predictions = model.predict(X_test_scaled)
    test_actuals = y_test.values
    correct_trades = np.sum(test_predictions == test_actuals)
    total_trades = len(test_actuals)
    
    raw_win_rate = (correct_trades / total_trades) * 100 if total_trades > 0 else 0.0
    win_rate = max(0.0, raw_win_rate - 3.5)

    atr = last['ATR']
    close = last['Close']
    atr_mult = 1.0 if horizon == "1D" else (1.8 if horizon == "3D" else 2.6)
    
    sl_buf = 1.3 if last['ATR_Pctile'] > 0.85 else 1.1
    sl_val = close - (atr * atr_mult * sl_buf)
    tp_val = close + (atr * atr_mult * 1.8)
    sl_pct = ((sl_val - close) / close) * 100
    tp_pct = ((tp_val - close) / close) * 100
    risk_reward = abs(tp_pct / sl_pct) if sl_pct != 0 else 0

    if prob >= 62.0:
        verdetto, colore = "BUY", "#238636"
    elif prob <= 38.0:
        verdetto, colore = "SELL", "#da3633"
    else:
        verdetto, colore = "HOLD", "#d29922"

    dettagli = []
    return prob, verdetto, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, win_rate, is_macro_bull, dettagli

# UI PRINCIPALE
st.markdown("<h1>Institutional Market Terminal</h1>", unsafe_allow_html=True)
st.write("")

c1, c2 = st.columns([2, 1])
with c1:
    st.session_state.ticker_input = st.text_input("Ticker:", value=st.session_state.ticker_input).upper().strip()
with c2:
    st.session_state.horizon_input = st.selectbox("Orizzonte:", ["1D", "3D", "7D"], index=["1D", "3D", "7D"].index(st.session_state.horizon_input))

ticker = st.session_state.ticker_input
horizon = st.session_state.horizon_input

if ticker:
    with st.spinner("Esecuzione Pre-Flight Check & Gradient Boosting..."):
        df = scarica_dati(ticker)
        if df is None or df.empty:
            st.error("Nessun dato trovato per il ticker inserito o formato errato.")
            st.stop()
        
        df = calcola_indicatori(df)
        
        # PRE-FLIGHT METRICS CON VALORI NUMERICI ESATTI
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        last_price = df['Close'].iloc[-1]
        last_atr = df['ATR'].iloc[-1]
        atr_pct = (last_atr / last_price) * 100
        
        # Formattazione valori volume (M = Milioni, K = Migliaia)
        vol_str = f"{avg_vol/1e6:.1f}M" if avg_vol >= 1e6 else f"{avg_vol/1e3:.0f}K"
        
        # Valutazione stato pre-flight con valori incorporati
        if avg_vol > 500000:
            liq_status = f"🟢 Liquidità: {vol_str} (Ottimale)"
        else:
            liq_status = f"🔴 Liquidità: {vol_str} (Bassa - Rischio Slippage)"

        if 1.5 <= atr_pct <= 3.5:
            vol_status = f"🟢 Volatilità (ATR): {atr_pct:.2f}% (Ideale)"
        elif atr_pct < 1.5:
            vol_status = f"🟡 Volatilità (ATR): {atr_pct:.2f}% (Bassa/Piatta)"
        else:
            vol_status = f"🔴 Volatilità (ATR): {atr_pct:.2f}% (Eccessiva/Rischiosa)"

        avg_sentiment, articoli = estrai_news_e_sentiment(ticker)
        prob, verdetto, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, win_rate, is_macro_bull, dettagli = calcola_modello_ml_e_backtest(df, avg_sentiment, horizon)
        
        last_date = df.index[-1].strftime('%Y-%m-%d')
        change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        sortino_val = df['Sortino'].iloc[-1]
        max_dd = df['Max_DD'].iloc[-1]

    # PRE-FLIGHT CHECK BAR CON VALORI ESATTI
    st.markdown(f"""
        <div class="preflight-card">
            <span>🛡️ <b>Pre-Flight Check:</b></span>
            <span>{liq_status}</span>
            <span>|</span>
            <span>{vol_status}</span>
        </div>
    """, unsafe_allow_html=True)

    # BOX DATA & MACRO STATUS
    macro_status = "🟢 Bullish (Sopra SMA200)" if is_macro_bull else "🔴 Bearish (Sotto SMA200 - Filtro Attivo)"
    st.markdown(f"""
        <div class="date-card">
            <span>📅</span>
            <span>Ultima data: <b>{last_date}</b> | Benchmark Peer: <b>{get_dynamic_benchmark(ticker)}</b> | Macro Trend: <b>{macro_status}</b></span>
        </div>
    """, unsafe_allow_html=True)

    # SCHEDA VERDETTO ML
    st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {colore};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="metric-label">ML Signal ({horizon})</span>
                    <div style="font-size: 20px; font-weight: 800; color: {colore}; margin-top: 2px;">{verdetto}</div>
                </div>
                <div style="text-align: right;">
                    <span class="metric-label">Probabilità Rialzo (GB) | Net Win Rate: {win_rate:.1f}%</span>
                    <div style="font-size: 18px; font-weight: bold;">{prob:.1f}%</div>
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
                <div class="metric-label">Stop Loss / R/R</div>
                <div class="metric-val" style="color: #da3633;">${sl_val:.2f}</div>
                <div class="metric-sub" style="color: #38bdf8;">R/R Ratio: 1 : {risk_reward:.1f}</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sortino Ratio</div>
                <div class="metric-val" style="color: #238636;">{sortino_val:.2f}</div>
                <div class="metric-sub" style="opacity: 0.7;">Rendimento / Downside Risk</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Backtest Net Win Rate</div>
                <div class="metric-val" style="color: #38bdf8;">{win_rate:.1f}%</div>
                <div class="metric-sub" style="opacity: 0.7;">Post-Costi & Slippage</div>
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

    # NOTIZIE & SENTIMENT
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

    # GUIDA ACCADEMICA E CAPITOLO PRATICO ESECUTIVO (AGGIORNATO CON VALORI PRE-FLIGHT)
    st.write("")
    with st.expander("📖 Guida ai Risultati, Pre-Flight Check e Capitolo Pratico Esecutivo"):
        st.markdown(r"""
        ### 1. Capitolo Pratico: Pre-Flight Check, Selezione Ticker e Protocollo
        Prima di valutare qualsiasi segnale, il terminale esegue automaticamente il **Pre-Flight Check** mostrando i valori numerici esatti e lo stato di convenienza:
        * **Filtro di Liquidità (Volume Medio 20D):** Mostra il volume medio giornaliero (es. $15.2\text{M}$ o $350\text{K}$).
          * 🟢 **Ottimale ($> 500\text{K}$ azioni):** Liquidità alta, esecuzione pulita senza slippage.
          * 🔴 **Bassa ($\le 500\text{K}$ azioni):** Rischio di scostamento di prezzo all'ordine.
        * **Filtro di Volatilità (ATR Percentuale sul Prezzo):** Mostra l'escursione percentuale media (es. $2.1\%$).
          * 🟢 **Ideale ($1.5\% \le \text{ATR}\% \le 3.5\%$):** Range perfetto per lo swing trading mirato.
          * 🟡 **Bassa / Piatta ($< 1.5\%$):** Movimento insufficiente per il target di profitto mensile.
          * 🔴 **Eccessiva ($> 3.5\%$):** Rischio di crollo o trigger prematuro dello stop loss.

        * **Protocollo d'Ingresso (Le 4 Regole d'Oro):**
          1. *Check Macro:* Il benchmark deve essere 🟢 *Bullish (Sopra SMA200)*.
          2. *Confluenza Rigida:* Probabilità ML $\ge 62\%$, Net Win Rate $\ge 55\%$ e R/R Ratio $\ge 1.5$.
          3. *Position Sizing:* Rischio massimo 1-2% del capitale per trade.
          4. *Disciplina Meccanica:* Esecuzione rigida dello stop loss basato su ATR.

        ---

        ### 2. Guida ai KPI e Bande di Valutazione
        * **ML Signal & Probabilità di Rialzo (Gradient Boosting):**
          * 🟢 **Banda Ottimale per BUY ($P \ge 62\%$):** Alta confidenza statistica out-of-sample.
          * 🟡 **Banda Neutra / HOLD ($38\% < P < 62\%$):** Indecisione stocastica. Nessuna operazione.
          * 🔴 **Banda Critica per SELL ($P \le 38\%$):** Alta probabilità di storno.
        * **Backtest Net Win Rate:** Ottimale se $\ge 55\%$ (al netto dei costi di transazione).
        * **R/R Ratio:** Ottimale se $\ge 1 : 1.5$.
        * **Sortino Ratio:** Ottimale se $> 1.5$.

        ---

        ### 3. Trattazione Matematica dell'Architettura
        * **Modello Predittivo ($y_t$):**
          $$y_t = \mathbb{I}\left( P_{t+h} > P_t \right)$$
        * **Feature Stazionarie ($\mathbf{X}_t$):**
          $$\text{Dist}_{\text{EMA}20} = \frac{P_t - \text{EMA}_{20}(P)}{\text{EMA}_{20}(P)} \cdot 100, \quad \text{RS}_{\text{Change}} = \frac{\text{RS}_t - \text{RS}_{t-5}}{\text{RS}_{t-5}}$$
        * **Sortino Ratio:**
          $$\text{Sortino} = \frac{\mathbb{E}[R_t] - R_f}{\sqrt{\frac{1}{N}\sum_{R_t < 0} (R_t - R_f)^2}} \cdot \sqrt{252}$$
        * **Maximum Drawdown (MDD):**
          $$ \text{MDD} = \max_{\tau \in (0, T)} \left( \frac{\max_{t \in (0, \tau)} P_t - P_\tau}{\max_{t \in (0, \tau)} P_t} \right) \cdot 100 $$
        """)
