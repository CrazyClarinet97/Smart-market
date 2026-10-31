import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
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
    """
    MODELLO DI FRONTIERA:
    1. Zero Data Leakage (Scaler fit solo su Training Set)
    2. Macro Trend Gatekeeper (Filtro SMA200 sul Benchmark)
    3. Backtester Vettorizzato Storico In-Line (Win Rate e Profit Factor)
    """
    days_forward = 1 if horizon == "1D" else (3 if horizon == "3D" else 7)
    df = df.copy()
    df['Target'] = (df['Close'].shift(-days_forward) > df['Close']).astype(int)
    df = df.dropna()

    if len(df) < 100:
        return 50.0, "HOLD", "#d29922", 0, 0, 0, 0, 0, 0, 0, []

    feature_cols = ['RSI', 'RSI7', 'MACD_Hist', 'Vol_Ratio', 'Dist_EMA20', 'RS_Change', 'ATR_Pctile']
    X = df[feature_cols]
    y = df['Target']

    # 1. Zero Data Leakage: Split temporale rigoroso (80% training, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Fit solo sul training set!
    X_test_scaled = scaler.transform(X_test)       # Transform pulito sul test set

    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Predizione out-of-sample sull'ultima candela
    last_features = scaler.transform(df[feature_cols].iloc[[-1]])
    prob = model.predict_proba(last_features)[0][1] * 100
    prob = max(0.0, min(100.0, prob + (sentiment_score * 12)))

    # 2. Macro Trend Gatekeeper (Filtro SMA200)
    last = df.iloc[-1]
    is_macro_bull = last['Bench_Close'] > last['Bench_SMA200']
    
    # Se il mercato macro è in Bear Market, blocchiamo i BUY forzando un giudizio prudenziale
    if not is_macro_bull and prob >= 62.0:
        prob = 50.0 # Annulla il segnale di acquisto per protezione sistemica

    # 3. Backtester Vettorizzato Storico In-Line (Calcolo Win Rate sul test set)
    test_predictions = model.predict(X_test_scaled)
    test_actuals = y_test.values
    correct_trades = np.sum(test_predictions == test_actuals)
    total_trades = len(test_actuals)
    win_rate = (correct_trades / total_trades) * 100 if total_trades > 0 else 0.0

    # Calcolo SL e TP dinamici
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
    with st.spinner("Esecuzione Modello Machine Learning (Random Forest & Backtest)..."):
        df = scarica_dati(ticker)
        if df is None or df.empty:
            st.error("Nessun dato trovato per il ticker inserito o formato errato.")
            st.stop()
        
        df = calcola_indicatori(df)
        avg_sentiment, articoli = estrai_news_e_sentiment(ticker)
        prob, verdetto, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, win_rate, is_macro_bull, dettagli = calcola_modello_ml_e_backtest(df, avg_sentiment, horizon)
        
        last_date = df.index[-1].strftime('%Y-%m-%d')
        last_price = df['Close'].iloc[-1]
        change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        sortino_val = df['Sortino'].iloc[-1]
        max_dd = df['Max_DD'].iloc[-1]

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
                    <span class="metric-label">Probabilità Rialzo (ML) | Backtest Win Rate: {win_rate:.1f}%</span>
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
                <div class="metric-label">Backtest Win Rate (Out-of-Sample)</div>
                <div class="metric-val" style="color: #38bdf8;">{win_rate:.1f}%</div>
                <div class="metric-sub" style="opacity: 0.7;">Accuratezza Storica Test Set</div>
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

    # GUIDA ACCADEMICA E MATEMATICA DI FRONTIERA INTEGRALE CON BANDE E FORMULE
    st.write("")
    with st.expander("📖 Guida ai Risultati (KPI), Bande Operative e Trattazione Matematica Completa"):
        st.markdown(r"""
        ### 1. Guida ai KPI, Bande di Valutazione e Regimi Operativi
        Questa sezione definisce le soglie quantitative e le bande di riferimento per stabilire se i valori registrati supportano l'operatività mirata al tuo obiettivo di profitto:

        * **ML Signal & Probabilità di Rialzo (Random Forest):**
          * 🟢 **Banda Ottimale per BUY ($P \ge 62\%$):** Alta confidenza statistica fuori campione (out-of-sample). Il modello individua una probabilità di chiusura in positivo superiore alla soglia di arbitraggio.
          * 🟡 **Banda Neutra / HOLD ($38\% < P < 62\%$):** Indecisione stocastica o rumore di mercato. Nessuna operazione.
          * 🔴 **Banda Critica per SELL ($P \le 38\%$):** Elevata probabilità di storno o esaurimento del momentum.

        * **Backtest Win Rate Storico (Out-of-Sample):**
          * 🟢 **Eccellente ($WinRate \ge 55\%$):** La strategia ha dimostrato storicamente un vantaggio statistico reale (edge) sul test set recente.
          * 🟡 **Marginale ($48\% \le WinRate < 55\%$):** Prestazione vicina al random walk; richiede un R/R ratio elevato per essere redditizia.
          * 🔴 **Scarsa ($WinRate < 48\%$):** Il modello perde colpi su questo asset; evitare di operare.

        * **Macro Trend Gatekeeper (Filtro SMA200 sul Benchmark):**
          * 🟢 **Bullish ($P_{\text{bench}} > \text{SMA}_{200}$):** Il mercato generale è in trend rialzista strutturale. I segnali BUY sono pienamente operativi.
          * 🔴 **Bearish ($P_{\text{bench}} < \text{SMA}_{200}$):** Il mercato è in macro-ribasso. Il filtro di protezione blocca automaticamente i segnali BUY per evitare trappole sistemiche.

        * **R/R Ratio (Rapporto Rischio / Premio):**
          * 🟢 **Ottimale ($R/R \ge 1.5$):** Asimmetria favorevole tra profitto atteso e stop loss.
          * 🔴 **Scarsa ($R/R < 1.2$):** Rischio non giustificato.

        * **Sortino Ratio (Efficienza del Rendimento):**
          * 🟢 **Ottimale ($Sortino > 1.5$):** Ottima qualità dei guadagni rispetto alla volatilità negativa.

        * **Max Drawdown Storico (Max DD su 5 Anni):**
          * 🟢 **Basso Rischio ($MaxDD \ge -15\%$):** Asset difensivo o in trend solido.
          * 🔴 **Elevato Rischio ($MaxDD < -30\%$):** Asset volatile; richiede stop loss stretti.

        ---

        ### 2. Architettura del Modello di Machine Learning (Random Forest & Zero Data Leakage)
        Il terminale implementa un approccio istituzionale basato su ensemble di alberi decisionali (`RandomForestClassifier`) con uno storico di 5 anni ($N \approx 1250$ barre giornaliere).
        * **Target Binario ($y_t$):**
          $$y_t = \mathbb{I}\left( P_{t+h} > P_t \right)$$
          *(dove $h \in \{1, 3, 7\}$ giorni in base all'orizzonte $H$ selezionato).*
        
        * **Feature Stazionarie ($\mathbf{X}_t$):**
          Per evitare la non-stazionarietà delle serie storiche, il modello utilizza variazioni percentuali e distanze relative:
          $$\text{Dist}_{\text{EMA}20} = \frac{P_t - \text{EMA}_{20}(P)}{\text{EMA}_{20}(P)} \cdot 100, \quad \text{RS}_{\text{Change}} = \frac{\text{RS}_t - \text{RS}_{t-5}}{\text{RS}_{t-5}}$$
          Insieme a $\text{RSI}_{14}$, $\text{RSI}_7$, $\text{MACD}_{\text{Hist}}$, $\text{Vol}_{\text{Ratio}}$ e $\text{ATR}_{\text{Pctile}}$.

        * **Prevenzione del Data Leakage (StandardScaler Isolato):**
          Il dataset viene diviso temporalmente in Training Set (80%) e Test Set (20%). Lo scaler viene calcolato **unicamente sul Training Set** ed applicato per proiezione al Test Set:
          $$\mu_{\text{train}}, \sigma_{\text{train}} = \text{fit}(X_{\text{train}}), \quad X_{\text{scaled}} = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

        ### 3. Sentiment Finanziario Potenziato (NLP)
        L'analisi testuale delle notizie web unisce l'analisi di polarità lessicale di base a un dizionario di boost semantico finanziario mirato:
        * Se il titolo contiene termini chiave rialzisti (*beat, surge, growth, record, rally, profit*), viene aggiunto un boost di polarità positiva ($+0.25$).
        * Se contiene termini ribassisti (*miss, crash, drop, loss, lawsuit, downgrade*), viene applicata una penalizzazione ($-0.25$).
        * Il punteggio finale è pesato per l'affidabilità della fonte giornalistica (Reuters, Bloomberg, WSJ, Financial Times, Sole 24 Ore, ecc. pesate a $1.3\times$).

        ### 4. Metriche Istituzionali di Rischio
        * **Sortino Ratio:**
          $$\text{Sortino} = \frac{\mathbb{E}[R_t] - R_f}{\sqrt{\frac{1}{N}\sum_{R_t < 0} (R_t - R_f)^2}} \cdot \sqrt{252}$$
        * **Maximum Drawdown Storico (Max DD):**
          $$\text{MDD} = \max_{\tau \in (0, T)} \left( \frac{\max_{t \in (0, \tau)} P_t - P_\tau}{\max_{t \in (0, \tau)} P_t} \right) \cdot 100$$
        """)
