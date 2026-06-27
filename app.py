import streamlit as st
import pandas as pd
import numpy as np
import nevergrad as ng
from scipy.stats import ttest_1samp, wilcoxon
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# ==========================================================================
# 0. CONFIGURATION & DESIGN SYSTEM (Premium CSS & Styling)
# ==========================================================================
st.set_page_config(
    page_title="Tối Ưu Hóa Danh Mục Đầu Tư HOSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Set main font */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradient headers */
    .main-title {
        background: linear-gradient(135deg, #185FA5 0%, #0F6E56 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 5px;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 25px;
        text-align: center;
    }
    
    /* Custom container styling (cards) */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
        transition: all 0.3s ease;
        margin-bottom: 15px;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(24, 95, 165, 0.3);
    }
    
    .metric-title {
        color: #888780;
        font-size: 0.9rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Highlight styles */
    .best-badge {
        background: linear-gradient(135deg, #185FA5 0%, #0F6E56 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 5px;
    }
    
    /* Tab styling custom */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px 8px 0px 0px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 10px 20px;
        font-weight: 600;
        color: #a0a0a0;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.07);
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(24, 95, 165, 0.15) !important;
        border-bottom: 2px solid #185FA5 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Titles
st.markdown('<div class="main-title">TỐI ƯU HÓA DANH MỤC ĐẦU TƯ HOSE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ứng dụng Web tối ưu hóa danh mục bằng chỉ báo kỹ thuật kết hợp thuật toán bầy đàn (PSO)</div>', unsafe_allow_html=True)


# ==========================================================================
# 1. CORE LOGIC (Imported & Optimized from Jupyter Notebook)
# ==========================================================================

# Loading and caching data function
@st.cache_data
def load_data(file_source):
    df = pd.read_csv(file_source, low_memory=False)
    # Drop unnamed columns (usually trailing commas in CSV)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Try different date formats
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        df["date"] = pd.to_datetime(df["date"], format=fmt, errors="coerce")
        if not df["date"].isnull().all():
            break
            
    if df["date"].isnull().all():
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date"])
    
    # Detect if adjusted prices are available
    use_adj = all(c in df.columns for c in ["adj_open", "adj_close"])
    df["Open"]  = df["adj_open"]  if use_adj else df["open"]
    df["Close"] = df["adj_close"] if use_adj else df["close"]
    
    df = df.rename(columns={"date": "Date", "ticker": "Ticker"})
    df = df[["Date", "Ticker", "Open", "Close"]].dropna(subset=["Open", "Close"])
    df = df[df["Close"] > 0]
    
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True), use_adj

# TECHNICAL INDICATORS
def rsi_wilder(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def macd_lines(close, fast, slow, sign):
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd, macd.ewm(span=sign, adjust=False).mean()

def bollinger(close, window, dev):
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    return ma + dev * sd, ma - dev * sd

# STRATEGY PARAMETERS & CONTROLS
def default_params(strat):
    if strat == "rsi":           
        return dict(rsi_window=14, lower=30.0, upper=70.0)
    if strat == "rsi_macd":      
        return dict(rsi_window=14, upper=70.0, macd_fast=12, macd_slow=26, macd_sign=9)
    if strat == "bollinger_rsi": 
        return dict(bb_window=20, bb_dev=2.0, rsi_window=14, lower=45.0, upper=70.0)
    raise ValueError(strat)

def parametrization(strat):
    I = lambda lo, hi: ng.p.Scalar(lower=lo, upper=hi).set_integer_casting()
    F = lambda lo, hi: ng.p.Scalar(lower=lo, upper=hi)
    if strat == "rsi":
        return ng.p.Dict(rsi_window=I(5, 30), lower=F(15, 40), upper=F(60, 85))
    if strat == "rsi_macd":
        return ng.p.Dict(rsi_window=I(5, 30), upper=F(55, 85), macd_fast=I(5, 20), macd_slow=I(21, 40), macd_sign=I(5, 15))
    if strat == "bollinger_rsi":
        return ng.p.Dict(bb_window=I(10, 30), bb_dev=F(1.5, 3.0), rsi_window=I(5, 25), lower=F(30, 55), upper=F(60, 85))
    raise ValueError(strat)

def clean_params(strat, p):
    p = dict(p)
    for k in ["rsi_window", "bb_window", "macd_fast", "macd_slow", "macd_sign"]:
        if k in p: 
            p[k] = int(round(float(p[k])))
    for k in ["bb_dev", "lower", "upper"]:
        if k in p: 
            p[k] = float(p[k])
    if "macd_fast" in p and "macd_slow" in p and p["macd_fast"] >= p["macd_slow"]:
        p["macd_slow"] = p["macd_fast"] + 5
    if "lower" in p and "upper" in p and p["lower"] >= p["upper"]:
        p["lower"], p["upper"] = min(p["lower"], p["upper"]), max(p["lower"], p["upper"]) + 5
    return p

# SIGNAL GENERATION
def generate_signals(df, strat, params, trend_window=100):
    p = clean_params(strat, params)
    c = df["Close"]
    trend = c > c.rolling(trend_window, min_periods=trend_window).mean()
    
    if strat == "rsi":
        r = rsi_wilder(c, p["rsi_window"])
        buy, sell = r < p["lower"], r > p["upper"]
    elif strat == "rsi_macd":
        r = rsi_wilder(c, p["rsi_window"])
        m, s = macd_lines(c, p["macd_fast"], p["macd_slow"], p["macd_sign"])
        buy = (m > s) & (r < p["upper"]) & trend
        sell = (m < s) | (r > p["upper"])
    elif strat == "bollinger_rsi":
        ub, lb = bollinger(c, p["bb_window"], p["bb_dev"])
        r = rsi_wilder(c, p["rsi_window"])
        buy = (c < lb) & (r < p["lower"]) & trend
        sell = (c > ub) | (r > p["upper"])
    else:
        raise ValueError(strat)
    return buy.fillna(False).values, sell.fillna(False).values

def signal_arrays(df, strat, params, trend_window=100):
    buy, sell = generate_signals(df, strat, params, trend_window)
    o = df["Open"].values.astype(float)
    c = df["Close"].values.astype(float)
    dt = df["Date"].values.astype("datetime64[ns]")
    
    buy_act = np.empty(len(buy), bool)
    buy_act[0] = False
    buy_act[1:] = buy[:-1]
    
    sell_act = np.empty(len(sell), bool)
    sell_act[0] = False
    sell_act[1:] = sell[:-1]
    
    cp = np.empty(len(c))
    cp[0] = np.nan
    cp[1:] = c[:-1]
    
    return dt, o, c, buy_act, sell_act, cp

# BACKTEST SIMULATION ENGINE (1 STOCK)
def sim_core(arrays, start, end, capital, fee=0.0015, sl=0.15, record=False):
    dt, o, c, buy_act, sell_act, cp = arrays
    idx = np.nonzero((dt >= np.datetime64(start)) & (dt < np.datetime64(end)))[0]
    if len(idx) < 2:
        return None, []
        
    cash, shares, in_pos, bp = float(capital), 0.0, False, 0.0
    vals = np.empty(len(idx))
    trades = []
    
    for j in range(len(idx)):
        i = idx[j]
        oi = o[i]
        ci = c[i]
        cpi = cp[i]
        
        if (not in_pos) and buy_act[i] and oi > 0 and cash > 0:
            shares = cash * (1 - fee) / oi
            bp = oi
            cash = 0.0
            in_pos = True
            if record: 
                trades.append({"Type": "MUA", "Date": dt[i], "Price": float(oi), "Reason": "TIN HIEU"})
        elif in_pos:
            stop = (not np.isnan(cpi)) and (cpi < bp * (1 - sl))
            if sell_act[i] or stop:
                proceeds = shares * oi * (1 - fee)
                if record: 
                    trades.append({
                        "Type": "BAN", 
                        "Date": dt[i], 
                        "Price": float(oi),
                        "PnL": float(proceeds - shares * bp),
                        "Reason": "STOP-LOSS" if (stop and not sell_act[i]) else "TIN HIEU"
                    })
                cash = proceeds
                shares = 0.0
                in_pos = False
        vals[j] = cash + shares * ci
        
    if in_pos and shares > 0:
        # Liquidate at end of period
        vals[-1] = shares * c[idx[-1]] * (1 - fee)
        if record:
            trades.append({
                "Type": "BAN",
                "Date": dt[idx[-1]],
                "Price": float(c[idx[-1]]),
                "PnL": float(shares * c[idx[-1]] * (1 - fee) - shares * bp),
                "Reason": "CUOI KY"
            })
            
    return pd.Series(vals, index=pd.to_datetime(dt[idx])), trades

def simulate_stock(df, start, end, strat, params, capital, cfg, record=True):
    return sim_core(
        signal_arrays(df, strat, params, cfg['trend_window']), 
        start, end, capital, cfg['fee_rate'], cfg['stop_loss_pct'], record=record
    )

def equity_stats(eq):
    if eq is None or len(eq) < 5: 
        return None
    dr = eq.pct_change().dropna()
    if dr.std() < 1e-12: 
        return None
    return (eq.iloc[-1] / eq.iloc[0] - 1) * 100, np.sqrt(252) * dr.mean() / dr.std()

# METRICS COMPUTATION
def metrics(equity, label=""):
    v = pd.Series(equity).dropna()
    if len(v) < 2:
        return dict(label=label, v_start=0.0, v_end=0.0, total=0.0, cagr=0.0, vol=0.0, sharpe=0.0, sortino=0.0, mdd=0.0, calmar=0.0, win=0.0)
    dr = v.pct_change().dropna()
    tot = (v.iloc[-1] / v.iloc[0] - 1) * 100
    ny = (v.index[-1] - v.index[0]).days / 365.25
    cagr = ((v.iloc[-1] / v.iloc[0]) ** (1 / ny) - 1) * 100 if ny > 0 else 0.0
    
    vol = dr.std() * np.sqrt(252) * 100 if dr.std() > 1e-12 else 0.0
    shp = np.sqrt(252) * dr.mean() / dr.std() if dr.std() > 1e-12 else 0.0
    
    neg = dr[dr < 0]
    sor = np.sqrt(252) * dr.mean() / neg.std() if len(neg) > 1 and neg.std() > 1e-12 else 0.0
    
    mdd = ((v - v.cummax()) / v.cummax()).min() * 100
    cal = cagr / abs(mdd) if mdd != 0 else 0.0
    win = (dr > 0).mean() * 100 if len(dr) else 0.0
    
    return dict(
        label=label, 
        v_start=float(v.iloc[0]), 
        v_end=float(v.iloc[-1]),
        total=tot, 
        cagr=cagr, 
        vol=vol, 
        sharpe=shp, 
        sortino=sor, 
        mdd=mdd, 
        calmar=cal, 
        win=win
    )

# CALENDAR & WINDOWS
def build_calendar(ticker_dfs):
    return np.array(sorted(set().union(*[set(d["Date"]) for d in ticker_dfs.values()])))

def rebalance_dates(all_dates, invest_years, rebalance_months):
    invest = [d for d in all_dates if pd.Timestamp(d).year in invest_years]
    reb = []
    for y in invest_years:
        for q in rebalance_months:
            after = [d for d in invest if pd.Timestamp(d) >= pd.Timestamp(y, q, 1)]
            if after: 
                reb.append(pd.Timestamp(after[0]))
    return sorted(set(reb))

def lookback_window(all_dates, d, n=252):
    arr = pd.to_datetime(all_dates)
    idx = int(np.searchsorted(arr, pd.Timestamp(d)))
    return pd.Timestamp(arr[max(0, idx - n)]), pd.Timestamp(d)

# SCREENING TOP STOCKS
def screen(screen_cache, lb_start, lb_end, cfg, top_n=5):
    scores = {}
    for t, arrays in screen_cache.items():
        eq, _ = sim_core(arrays, lb_start, lb_end, 1e8, cfg['fee_rate'], cfg['stop_loss_pct'])
        st = equity_stats(eq)
        if st is None: 
            continue
        ret, shp = st
        scores[t] = cfg['score_w_sharpe'] * shp + cfg['score_w_return'] * (ret / 100)
    return sorted(scores, key=lambda x: -scores[x])[:top_n]

# WEIGHT ALLOCATION
def inverse_vol_weights(tickers, ticker_dfs, lb_start, lb_end):
    inv = {}
    for t in tickers:
        d = ticker_dfs[t]
        dr = d[(d["Date"] >= lb_start) & (d["Date"] < lb_end)]["Close"].pct_change().dropna()
        s = dr.std()
        inv[t] = 1.0 / (s if s and s > 1e-9 else 0.05)
    tot = sum(inv.values())
    return {t: inv[t] / tot for t in tickers}

def portfolio_weights(tickers, ticker_dfs, all_dates, d, weight_scheme="equal", vol_lookback=63):
    if weight_scheme == "equal" or len(tickers) == 0:
        return {t: 1.0 / len(tickers) for t in tickers}
    vs, _ = lookback_window(all_dates, d, vol_lookback)
    return inverse_vol_weights(tickers, ticker_dfs, vs, d)

def quarterly_subreturns(eq):
    s = pd.Series(eq).dropna().sort_index()
    rets, labs = [], []
    for (y, q), g in s.groupby([s.index.year, s.index.quarter]):
        if len(g) >= 2:
            rets.append((g.iloc[-1] / g.iloc[0] - 1) * 100)
            labs.append(f"{str(y)[2:]}-Q{q}")
    return rets, labs

# PSO OPTIMIZATION
def pso_optimize(selected, trimmed, lb_start, lb_end, strat, budget, cfg):
    def objective(params):
        sh = []
        for t in selected:
            eq, _ = simulate_stock(trimmed[t], lb_start, lb_end, strat, params, 1e8, cfg, record=False)
            st = equity_stats(eq)
            sh.append((st[1] - 0.0) if st else -1.0)
        return -float(np.mean(sh))
        
    par = parametrization(strat)
    try: 
        par.random_state.seed(cfg['random_seed'])
    except Exception: 
        pass
        
    rec = ng.optimizers.PSO(parametrization=par, budget=budget).minimize(objective)
    return clean_params(strat, rec.value)

# STRATEGY RUNNER
def run_strategy(ticker_dfs, all_dates, strat, cfg, optimize=True, status_container=None):
    reb = rebalance_dates(all_dates, cfg['invest_years'], cfg['rebalance_months'])
    if not reb:
        return None
        
    last_day = pd.Timestamp(max(all_dates)) + pd.Timedelta(days=1)
    dp = default_params(strat)
    
    # Initialize cache
    screen_cache = {t: signal_arrays(df, strat, dp, cfg['trend_window']) for t, df in ticker_dfs.items()}
    
    capital = float(cfg['initial_capital'])
    hist, q_log, trades_all = [], [], []
    
    for i, d in enumerate(reb):
        q_end = reb[i + 1] if i + 1 < len(reb) else last_day
        lb_start, lb_end = lookback_window(all_dates, d, cfg['lookback_sess'])
        
        # Display progress info
        if status_container:
            status_container.write(f"🔄 **[{strat.upper()}]** Đang xử lý kỳ: `{d.date()}` (Lọc Top-{cfg['num_stocks']}...)")
            
        top = screen(screen_cache, lb_start, lb_end, cfg, cfg['num_stocks'])
        if not top:
            continue
            
        if optimize:
            if status_container:
                status_container.write(f"⚙️ **[{strat.upper()}]** Đang chạy tối ưu PSO cho kỳ: `{d.date()}`...")
            buf = lb_start - pd.Timedelta(days=160)
            trimmed = {t: ticker_dfs[t][(ticker_dfs[t]["Date"] >= buf) & (ticker_dfs[t]["Date"] < q_end)] for t in top}
            params = pso_optimize(top, trimmed, lb_start, lb_end, strat, cfg['pso_budget'], cfg)
        else:
            params = dp
            
        weights = portfolio_weights(top, ticker_dfs, all_dates, d, cfg['weight_scheme'], cfg['vol_lookback'])
        
        start_cap, stock_eq = capital, {}
        for t in top:
            eq, tr = simulate_stock(ticker_dfs[t], d, q_end, strat, params, capital * weights[t], cfg, record=True)
            if eq is not None and len(eq): 
                stock_eq[t] = eq
            for x in tr: 
                x["Ticker"] = t
            trades_all.extend(tr)
            
        if not stock_eq:
            continue
            
        port = pd.DataFrame(stock_eq).sum(axis=1)
        allocated_cash = sum(capital * weights[t] for t in stock_eq)
        idle = capital - allocated_cash
        port = port + idle
        
        for dt, val in port.items(): 
            hist.append({"Date": dt, "V": float(val)})
            
        capital = float(port.iloc[-1])
        ret = (capital / start_cap - 1) * 100
        
        q_log.append(dict(
            date=str(d.date()), 
            top=top, 
            params=params, 
            ret=ret,
            start=start_cap, 
            end=capital,
            weights={t: round(weights[t] * 100, 1) for t in top}
        ))
        
    if not hist:
        return None
        
    eq = pd.DataFrame(hist).set_index("Date")["V"].sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]
    q_returns, q_labels = quarterly_subreturns(eq)
    buys = sum(1 for x in trades_all if x["Type"] == "MUA")
    ssig = sum(1 for x in trades_all if x["Type"] == "BAN" and x.get("Reason") == "TIN HIEU")
    ssl  = sum(1 for x in trades_all if x["Type"] == "BAN" and x.get("Reason") == "STOP-LOSS")
    
    return dict(
        equity=eq, 
        metrics=metrics(eq, strat), 
        q_log=q_log, 
        q_returns=q_returns, 
        q_labels=q_labels,
        n_buy=buys, 
        n_sell_sig=ssig, 
        n_sell_sl=ssl, 
        trades=trades_all
    )

# BENCHMARK 1/N RUNNER
def buy_and_hold_100(ticker_dfs, all_dates, cfg):
    invest = sorted([pd.Timestamp(d) for d in all_dates if pd.Timestamp(d).year in cfg['invest_years']])
    if not invest:
        return None
        
    first = invest[0]
    alloc = cfg['initial_capital'] / len(ticker_dfs)
    holds = {}
    closes = {}
    
    for t, d in ticker_dfs.items():
        dd = d[d["Date"] >= first]
        bp = dd["Open"].iloc[0] if len(dd) else 0.0
        holds[t] = alloc * (1 - cfg['fee_rate']) / bp if bp > 0 else 0.0
        closes[t] = d.set_index("Date")["Close"]
        
    vals = []
    for dt in invest:
        tot = sum(holds[t] * closes[t].get(dt, np.nan) for t in ticker_dfs)
        vals.append({"Date": dt, "V": tot})
        
    eq = pd.DataFrame(vals).set_index("Date")["V"].sort_index().dropna()
    qr, qlab = quarterly_subreturns(eq)
    
    return dict(equity=eq, metrics=metrics(eq, "bnh"), q_returns=qr, q_labels=qlab)


# ==========================================================================
# 2. STREAMLIT APP LAYOUT & CONTROL FLOW
# ==========================================================================

# 2.1 Sidebar Configuration
st.sidebar.markdown("### 📁 Nguồn Dữ Liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file CSV dữ liệu HOSE", type=["csv"])

# Search path for default file
default_filename = "HOSE_2020_2023.csv"
default_file_exists = os.path.exists(default_filename)

if uploaded_file is not None:
    data_source = uploaded_file
    st.sidebar.success("Đã nhận file upload!")
elif default_file_exists:
    data_source = default_filename
    st.sidebar.info(f"Sử dụng file mặc định `{default_filename}`.")
else:
    st.sidebar.warning("Hãy tải lên file dữ liệu CSV để chạy ứng dụng.")
    st.info("💡 Để bắt đầu, hãy chuẩn bị một tệp dữ liệu CSV chứa các cột: `Date`, `Ticker`, `Open`, `Close` (hoặc các biến thể đã được điều chỉnh như `adj_open`, `adj_close`).")
    st.stop()

# Load Dataset
try:
    df_raw, has_adj = load_data(data_source)
    # Detect available years
    df_years = sorted(list(df_raw["Date"].dt.year.unique()))
except Exception as e:
    st.error(f"Lỗi khi đọc file dữ liệu: {e}")
    st.stop()

# Parameter config in Sidebar
st.sidebar.markdown("### ⚙️ Cấu Hình Backtest")

initial_capital = st.sidebar.number_input(
    "Vốn ban đầu (VND)", 
    min_value=10_000_000, 
    max_value=100_000_000_000, 
    value=1_000_000_000, 
    step=10_000_000, 
    format="%d"
)

fee_rate = st.sidebar.slider("Phí giao dịch mỗi lệnh (%)", 0.0, 1.0, 0.15, step=0.01) / 100

stop_loss_pct = st.sidebar.slider("Ngưỡng cắt lỗ (Stop Loss %)", 0, 50, 15, step=1) / 100

num_stocks = st.sidebar.number_input("Số lượng mã nắm giữ (N)", min_value=1, max_value=20, value=5)

pso_budget = st.sidebar.number_input("Số lần lặp PSO (PSO Budget)", min_value=5, max_value=200, value=50, step=5)

# Dynamic invest years selection
invest_year_options = df_years[1:] if len(df_years) > 1 else df_years
invest_years = st.sidebar.multiselect(
    "Năm đầu tư backtest", 
    options=invest_year_options, 
    default=invest_year_options
)

if not invest_years:
    st.sidebar.error("Hãy chọn ít nhất một năm đầu tư!")
    st.stop()

# Rebalance Frequency
rebalance_freq = st.sidebar.selectbox(
    "Tần suất tái cơ cấu",
    options=["Hàng năm", "Nửa năm", "Hàng quý"],
    index=0
)

freq_map = {
    "Hàng năm": [1],
    "Nửa năm": [1, 7],
    "Hàng quý": [1, 4, 7, 10]
}
rebalance_months = freq_map[rebalance_freq]

# Weight Scheme
weight_scheme = st.sidebar.selectbox(
    "Phương pháp chia tỷ trọng",
    options=["Equal Weight (Chia đều)", "Inverse Volatility (Nghịch đảo biến động)"],
    index=0
)
weight_scheme_code = "equal" if "Equal" in weight_scheme else "invvol"

# Advanced configuration parameters expandable
with st.sidebar.expander("🛠️ Tham số nâng cao"):
    lookback_sess = st.number_input("Cửa sổ lookback học (phiên)", min_value=50, max_value=500, value=252)
    vol_lookback = st.number_input("Cửa sổ tính biến động (phiên)", min_value=10, max_value=150, value=63)
    trend_window = st.number_input("Bộ lọc xu hướng SMA (phiên)", min_value=10, max_value=250, value=100)
    
    st.markdown("**Trọng số chấm điểm sàng lọc:**")
    score_w_sharpe = st.slider("Trọng số Sharpe", 0.0, 1.0, 0.6, step=0.05)
    score_w_return = 1.0 - score_w_sharpe
    st.caption(f"Trọng số Lợi nhuận tương ứng: {score_w_return:.2f}")
    
    random_seed = st.number_input("Random Seed PSO", min_value=1, max_value=1000, value=42)

# Pack everything into a config dict
cfg = {
    'initial_capital': initial_capital,
    'fee_rate': fee_rate,
    'stop_loss_pct': stop_loss_pct,
    'num_stocks': num_stocks,
    'pso_budget': pso_budget,
    'lookback_sess': lookback_sess,
    'vol_lookback': vol_lookback,
    'trend_window': trend_window,
    'score_w_sharpe': score_w_sharpe,
    'score_w_return': score_w_return,
    'invest_years': invest_years,
    'rebalance_months': rebalance_months,
    'weight_scheme': weight_scheme_code,
    'random_seed': random_seed
}


# 2.2 Execution trigger
st.markdown("---")
col_info, col_btn = st.columns([4, 1])
with col_info:
    st.markdown(f"📊 **Dữ liệu hiện tại:** Gồm **{df_raw['Ticker'].nunique()}** mã cổ phiếu. Khoảng thời gian: `{df_raw['Date'].min().date()}` đến `{df_raw['Date'].max().date()}`. Dùng giá: `{'Đã điều chỉnh (Adjusted)' if has_adj else 'Giá thô (Raw Open/Close)'}`.")
with col_btn:
    run_backtest = st.button("🚀 CHẠY BACKTEST", use_container_width=True)

# 2.3 Run Strategy Backtests
if "results" not in st.session_state:
    st.session_state["results"] = None

if run_backtest:
    st.session_state["results"] = None # Clear old results
    
    # Process dictionary of ticker dataframes
    ticker_dfs = {t: df_raw[df_raw["Ticker"] == t].copy() for t in sorted(df_raw["Ticker"].unique())}
    all_dates = build_calendar(ticker_dfs)
    
    status_box = st.status("🛠️ Đang khởi chạy backtest các chiến lược...", expanded=True)
    
    try:
        parts = {}
        # 1. Benchmark 1/N
        status_box.write("📈 Đang tính toán Benchmark 1/N (Equal Weight)...")
        parts["bnh"] = buy_and_hold_100(ticker_dfs, all_dates, cfg)
        
        # 2. RSI MACD Default (No optimization)
        status_box.write("📈 Đang chạy chiến lược RSI+MACD mặc định (không tối ưu)...")
        parts["rsi_macd_default"] = run_strategy(ticker_dfs, all_dates, "rsi_macd", cfg, optimize=False)
        
        # 3. Optimized RSI
        parts["rsi"] = run_strategy(ticker_dfs, all_dates, "rsi", cfg, optimize=True, status_container=status_box)
        
        # 4. Optimized RSI + MACD
        parts["rsi_macd"] = run_strategy(ticker_dfs, all_dates, "rsi_macd", cfg, optimize=True, status_container=status_box)
        
        # 5. Optimized Bollinger + RSI
        parts["bollinger_rsi"] = run_strategy(ticker_dfs, all_dates, "bollinger_rsi", cfg, optimize=True, status_container=status_box)
        
        status_box.update(label="✅ Backtest hoàn thành thành công!", state="complete", expanded=False)
        
        # Determine best strategy based on Sharpe ratio
        strats_to_check = ["rsi", "rsi_macd", "bollinger_rsi"]
        valid_strats = [k for k in strats_to_check if parts[k] is not None]
        
        if valid_strats:
            best_strat_name = max(valid_strats, key=lambda k: parts[k]["metrics"]["sharpe"])
        else:
            best_strat_name = "rsi_macd"
            
        st.session_state["results"] = {
            "parts": parts,
            "best": best_strat_name,
            "cfg": cfg,
            "ticker_dfs": ticker_dfs,
            "all_dates": all_dates
        }
        st.toast("Backtest đã sẵn sàng!", icon="🎉")
        
    except Exception as e:
        status_box.update(label="❌ Lỗi xảy ra trong quá trình chạy backtest!", state="error", expanded=True)
        st.error(f"Chi tiết lỗi: {e}")
        import traceback
        st.code(traceback.format_exc())

# 2.4 Display Results
if st.session_state["results"] is not None:
    res = st.session_state["results"]
    parts = res["parts"]
    best = res["best"]
    ticker_dfs = res["ticker_dfs"]
    all_dates = res["all_dates"]
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 TỔNG QUAN HIỆU SUẤT", 
        "⚙️ CHI TIẾT CHIẾN LƯỢC TỐI ƯU", 
        "📝 NHẬT KÝ GIAO DỊCH & CƠ CẤU", 
        "🔍 TRA CỨU CỔ PHIẾU CHỈ BÁO"
    ])
    
    # Names map
    NM = {
        "rsi": "RSI Tối Ưu", 
        "rsi_macd": "RSI + MACD Tối Ưu", 
        "bollinger_rsi": "RSI + Bollinger Tối Ưu", 
        "rsi_macd_default": "RSI + MACD Mặc Định",
        "bnh": "Benchmark 1/N"
    }
    
    COLOR_MAP = {
        "rsi": "#888780",
        "rsi_macd": "#185FA5",
        "bollinger_rsi": "#0F6E56",
        "rsi_macd_default": "#5DADE2",
        "bnh": "#A32D2D"
    }
    
    # ----------------------------------------------------
    # TAB 1: PERFORMANCE DASHBOARD
    # ----------------------------------------------------
    with tab1:
        st.markdown("### 🏆 Kết quả của Chiến lược tốt nhất")
        
        # Display KPI cards for the best strategy
        best_metrics = parts[best]["metrics"]
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">Chiến lược tốt nhất</div>'
                f'<div class="metric-val" style="font-size:1.4rem;">{NM[best]}</div>'
                f'<div class="best-badge">Sharpe tối đa</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
        with kpi_col2:
            val_format = f"{best_metrics['total']:+.2f}%"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">Tổng Lợi Nhuận (VND)</div>'
                f'<div class="metric-val">{best_metrics["v_end"]:,.0f}</div>'
                f'<div style="color:{"#2ecc71" if best_metrics["total"] >= 0 else "#e74c3c"}; font-weight:bold; font-size:0.95rem;">Lợi nhuận %: {val_format}</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
        with kpi_col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">Tỷ lệ Sharpe / CAGR</div>'
                f'<div class="metric-val">{best_metrics["sharpe"]:.3f}</div>'
                f'<div style="color:#f39c12; font-weight:bold; font-size:0.95rem;">CAGR: {best_metrics["cagr"]:.2f}%</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
        with kpi_col4:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">Mức Sụt Giảm Tài Sản Max</div>'
                f'<div class="metric-val">{best_metrics["mdd"]:.2f}%</div>'
                f'<div style="color:#7f8c8d; font-weight:bold; font-size:0.95rem;">Biến động năm: {best_metrics["vol"]:.2f}%</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
        # Draw Plotly Equity Chart
        st.markdown("### 📈 Biểu đồ so sánh tăng trưởng tài sản (2021-2023)")
        
        fig_equity = go.Figure()
        
        for k in ["rsi", "rsi_macd", "bollinger_rsi", "rsi_macd_default", "bnh"]:
            if parts[k] is not None:
                eq_series = parts[k]["equity"]
                # Convert to Billion VND for easier reading
                eq_billions = eq_series / 1e9
                
                fig_equity.add_trace(go.Scatter(
                    x=eq_series.index,
                    y=eq_billions,
                    name=f"{NM[k]} ({parts[k]['metrics']['total']:.1f}%)",
                    line=dict(
                        color=COLOR_MAP[k], 
                        width=3.2 if k == best else 1.8,
                        dash="dash" if k == "bnh" else "solid"
                    )
                ))
                
        # Draw Rebalance vertical lines for best strategy
        for r in parts[best]["q_log"]:
            rb_date = pd.Timestamp(r["date"])
            fig_equity.add_vline(
                x=rb_date.timestamp() * 1000, 
                line_width=0.8, 
                line_dash="dot", 
                line_color="gray"
            )
            
        fig_equity.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Thời gian",
            yaxis_title="Giá trị danh mục (Tỷ VND)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"),
            margin=dict(l=20, r=20, t=30, b=20),
            height=500
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Display Metrics Comparison Table
        st.markdown("### 📊 Bảng so sánh các chỉ số hiệu suất chi tiết")
        
        metric_rows = []
        metric_defs = [
            ("Lợi nhuận gộp (%)", "total", "{:+.3f}%"),
            ("CAGR (%)", "cagr", "{:.3f}%"),
            ("Rủi ro biến động (%)", "vol", "{:.3f}%"),
            ("Tỷ số Sharpe", "sharpe", "{:.3f}"),
            ("Tỷ số Sortino", "sortino", "{:.3f}"),
            ("Tài sản sụt giảm tối đa (%)", "mdd", "{:.3f}%"),
            ("Tỷ số Calmar", "calmar", "{:.3f}"),
            ("Tỷ lệ ngày thắng (%)", "win", "{:.3f}%")
        ]
        
        data_table = {}
        for k in ["rsi", "rsi_macd", "bollinger_rsi", "rsi_macd_default", "bnh"]:
            if parts[k] is not None:
                m_data = parts[k]["metrics"]
                col_name = NM[k]
                if k == best:
                    col_name += " 🏆"
                data_table[col_name] = []
                for label, key, fmt in metric_defs:
                    data_table[col_name].append(fmt.format(m_data[key]))
                    
        index_labels = [m[0] for m in metric_defs]
        df_comparison = pd.DataFrame(data_table, index=index_labels)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Dropdown to compare specific metric in bar chart
        st.markdown("### 📊 Trực quan hóa so sánh theo chỉ số")
        compare_metric_display = st.selectbox(
            "Chọn chỉ số để so sánh biểu đồ:",
            options=[m[0] for m in metric_defs],
            index=3 # Default to Sharpe
        )
        
        # Find key from display name
        metric_key = [m[1] for m in metric_defs if m[0] == compare_metric_display][0]
        
        compare_x = []
        compare_y = []
        compare_color = []
        
        for k in ["rsi", "rsi_macd", "bollinger_rsi", "rsi_macd_default", "bnh"]:
            if parts[k] is not None:
                compare_x.append(NM[k])
                compare_y.append(parts[k]["metrics"][metric_key])
                compare_color.append(COLOR_MAP[k])
                
        fig_bar = go.Figure(go.Bar(
            x=compare_x,
            y=compare_y,
            marker_color=compare_color,
            text=[f"{y:.3f}" for y in compare_y],
            textposition="auto",
            width=0.4
        ))
        
        fig_bar.update_layout(
            template="plotly_dark",
            title=f"So sánh chỉ số: {compare_metric_display}",
            yaxis_title=compare_metric_display,
            margin=dict(l=20, r=20, t=40, b=20),
            height=380
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: DETAILED STRATEGY ANALYSIS
    # ----------------------------------------------------
    with tab2:
        st.markdown("### 🔍 Phân tích chi tiết từng chiến lược")
        
        selected_strat_details = st.selectbox(
            "Chọn chiến lược để xem chi tiết & kiểm định:",
            options=["rsi", "rsi_macd", "bollinger_rsi"],
            format_func=lambda x: NM[x],
            index=0
        )
        
        s_data = parts[selected_strat_details]
        
        if s_data is None:
            st.error("Dữ liệu chiến lược này không tồn tại.")
        else:
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("#### 📅 Kết quả tái cơ cấu (Rebalancing Logs)")
                
                # Format the quarterly logs to display nicely
                rebalance_log = []
                for period in s_data["q_log"]:
                    rebalance_log.append({
                        "Ngày tái cơ cấu": period["date"],
                        "Top Cổ Phiếu": ", ".join(period["top"]),
                        "Tỷ trọng phân bổ (%)": ", ".join([f"{t}: {w}%" for t, w in period["weights"].items()]),
                        "Lợi nhuận kỳ (%)": f"{period['ret']:+.2f}%",
                        "Tham số PSO tối ưu": str(period["params"])
                    })
                df_rebalance_log = pd.DataFrame(rebalance_log)
                st.dataframe(df_rebalance_log, use_container_width=True, hide_index=True)
                
                # Trading statistics counters
                st.markdown("#### 📊 Thống kê lệnh giao dịch thực thi")
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Tổng lệnh mua", s_data["n_buy"])
                tc2.metric("Lệnh bán tín hiệu", s_data["n_sell_sig"])
                tc3.metric("Lệnh bán cắt lỗ (Stop-Loss)", s_data["n_sell_sl"])
                
            with col_right:
                st.markdown(f"#### 💵 Lợi nhuận danh mục theo từng quý - {NM[selected_strat_details]}")
                
                q_rets = s_data["q_returns"]
                q_labs = s_data["q_labels"]
                
                if len(q_rets) > 0:
                    q_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in q_rets]
                    fig_quarterly = go.Figure(go.Bar(
                        x=q_labs,
                        y=q_rets,
                        marker_color=q_colors,
                        text=[f"{r:+.1f}%" for r in q_rets],
                        textposition="outside",
                        width=0.5
                    ))
                    
                    fig_quarterly.update_layout(
                        template="plotly_dark",
                        yaxis_title="Lợi nhuận quý (%)",
                        xaxis_title="Quý",
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=350
                    )
                    st.plotly_chart(fig_quarterly, use_container_width=True)
                else:
                    st.info("Không có đủ dữ liệu theo quý để vẽ biểu đồ.")
                    
            st.markdown("---")
            st.markdown("### 🧪 Kiểm định giả thuyết thống kê (Statistical Hypothesis Testing)")
            st.caption("Kiểm định giả thuyết giúp chứng minh hiệu quả thực tế của chiến lược giao dịch về mặt toán học.")
            
            # Perform tests dynamically
            q_returns_best = s_data["q_returns"]
            
            # 1. T-test: Lợi nhuận OOS trung bình > 0
            test_ttest = mean_ttest(q_returns_best, 0.0)
            
            # 2. Wilcoxon: So với Default
            default_key = f"{selected_strat_details}_default" if selected_strat_details == "rsi_macd" else None
            # If rsi or bollinger, we don't have default precalculated unless we want to, so we can compare with default macd or benchmark
            compare_strat_wilcoxon = "rsi_macd_default" if parts["rsi_macd_default"] is not None else "bnh"
            
            test_vs_default = wilcoxon_greater(q_returns_best, parts[compare_strat_wilcoxon]["q_returns"])
            test_vs_bnh = wilcoxon_greater(q_returns_best, parts["bnh"]["q_returns"])
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-title">1. T-Test (Lợi nhuận quý > 0)</div>'
                    f'<div class="metric-val" style="font-size:1.5rem; color:{"#2ecc71" if test_ttest["significant"] else "#f1c40f"};">'
                    f'p = {test_ttest["pvalue"]:.5f}' if test_ttest["pvalue"] is not None else "N/A"
                    f'</div>'
                    f'<div style="font-weight:bold; font-size:0.85rem; margin-top:5px;">Kết quả: {"CÓ Ý NGHĨA" if test_ttest["significant"] else "Chưa có ý nghĩa"}</div>'
                    f'<span style="font-size:0.8rem; color:#7f8c8d;">Độ tin cậy 90% (Học kỳ OOS có lời thực sự)</span>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            with stat_col2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-title">2. Wilcoxon (Tối ưu > Mặc định MACD)</div>'
                    f'<div class="metric-val" style="font-size:1.5rem; color:{"#2ecc71" if test_vs_default["significant"] else "#f1c40f"};">'
                    f'p = {test_vs_default["pvalue"]:.5f}' if test_vs_default["pvalue"] is not None else "N/A"
                    f'</div>'
                    f'<div style="font-weight:bold; font-size:0.85rem; margin-top:5px;">Kết quả: {"CÓ Ý NGHĨA" if test_vs_default["significant"] else "Chưa có ý nghĩa"}</div>'
                    f'<span style="font-size:0.8rem; color:#7f8c8d;">Chứng minh PSO cải thiện hiệu suất rõ rệt</span>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            with stat_col3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-title">3. Wilcoxon (Chiến lược > Benchmark 1/N)</div>'
                    f'<div class="metric-val" style="font-size:1.5rem; color:{"#2ecc71" if test_vs_bnh["significant"] else "#f1c40f"};">'
                    f'p = {test_vs_bnh["pvalue"]:.5f}' if test_vs_bnh["pvalue"] is not None else "N/A"
                    f'</div>'
                    f'<div style="font-weight:bold; font-size:0.85rem; margin-top:5px;">Kết quả: {"CÓ Ý NGHĨA" if test_vs_bnh["significant"] else "Chưa có ý nghĩa"}</div>'
                    f'<span style="font-size:0.8rem; color:#7f8c8d;">Vượt trội chỉ số mua và nắm giữ thị trường</span>'
                    f'</div>', 
                    unsafe_allow_html=True
                )

    # ----------------------------------------------------
    # TAB 3: TRANSACTION LOGS
    # ----------------------------------------------------
    with tab3:
        st.markdown("### 📝 Nhật ký giao dịch và Phân bổ tỷ trọng")
        
        selected_strat_logs = st.selectbox(
            "Chọn chiến lược xem nhật ký giao dịch:",
            options=["rsi", "rsi_macd", "bollinger_rsi"],
            format_func=lambda x: NM[x],
            key="selected_strat_logs"
        )
        
        s_data_logs = parts[selected_strat_logs]
        
        if s_data_logs is None:
            st.error("Không có dữ liệu nhật ký giao dịch.")
        else:
            # Transaction log table
            st.markdown("#### 📋 Nhật ký lệnh Mua / Bán chi tiết")
            st.caption("Bảng danh sách toàn bộ các lệnh giao dịch được thực thi bởi thuật toán trong suốt giai đoạn backtest.")
            
            trades_list = []
            for t in s_data_logs["trades"]:
                trades_list.append({
                    "Ngày": pd.Timestamp(t["Date"]).strftime("%Y-%m-%d"),
                    "Mã cổ phiếu": t["Ticker"],
                    "Loại giao dịch": t["Type"],
                    "Giá khớp (VND)": f"{t['Price']:,.2f}",
                    "Lợi nhuận PnL (VND)": f"{t['PnL']:+,.0f}" if "PnL" in t else "-",
                    "Lý do": t.get("Reason", "TIN HIEU")
                })
                
            if trades_list:
                df_trades = pd.DataFrame(trades_list)
                st.dataframe(df_trades, use_container_width=True)
                
                # Download button
                csv_trades = df_trades.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 Tải xuống Nhật ký giao dịch (CSV)",
                    data=csv_trades,
                    file_name=f"trade_logs_{selected_strat_logs}.csv",
                    mime="text/csv",
                )
            else:
                st.info("Chiến lược này không thực hiện giao dịch nào trong kỳ.")
                
            # Rebalancing log table
            st.markdown("#### 🔄 Chi tiết phân bổ danh mục qua các kỳ")
            
            periods_list = []
            for period in s_data_logs["q_log"]:
                for stock in period["top"]:
                    periods_list.append({
                        "Ngày tái cơ cấu": period["date"],
                        "Mã cổ phiếu": stock,
                        "Tỷ trọng phân bổ (%)": f"{period['weights'].get(stock, 0.0)}%",
                        "Vốn đầu tư (VND)": f"{period['start'] * (period['weights'].get(stock, 0.0)/100):,.0f}",
                        "Tham số bộ lọc kỳ đó": str(period["params"])
                    })
            if periods_list:
                df_periods = pd.DataFrame(periods_list)
                st.dataframe(df_periods, use_container_width=True)
                
                csv_periods = df_periods.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 Tải xuống Danh mục phân bổ qua các kỳ (CSV)",
                    data=csv_periods,
                    file_name=f"rebalance_portfolio_{selected_strat_logs}.csv",
                    mime="text/csv",
                )

    # ----------------------------------------------------
    # TAB 4: INDIVIDUAL STOCK EXPLORER
    # ----------------------------------------------------
    with tab4:
        st.markdown("### 🔍 Trực quan hóa giá và các chỉ báo kỹ thuật của từng cổ phiếu")
        st.caption("Xem đồ thị chi tiết kèm các chỉ báo RSI, MACD, Bollinger Bands cùng với các điểm mua/bán của bất kỳ mã cổ phiếu nào.")
        
        tickers = sorted(df_raw["Ticker"].unique())
        selected_stock = st.selectbox("Chọn mã cổ phiếu muốn tra cứu:", options=tickers, index=0)
        
        # Load single stock df
        df_stock = df_raw[df_raw["Ticker"] == selected_stock].copy().sort_values("Date").reset_index(drop=True)
        
        # Check params to calculate indicators (let user adjust parameters dynamically for explorer!)
        st.markdown("#### ⚙️ Cấu hình tham số chỉ báo (Dành riêng cho màn hình tra cứu)")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            exp_rsi_win = st.number_input("Chu kỳ RSI", min_value=2, max_value=50, value=14, key="exp_rsi")
            exp_rsi_lower = st.number_input("Ngưỡng quá bán RSI", min_value=5, max_value=50, value=30, key="exp_rsi_lo")
            exp_rsi_upper = st.number_input("Ngưỡng quá mua RSI", min_value=50, max_value=95, value=70, key="exp_rsi_up")
        with exp_col2:
            exp_macd_f = st.number_input("MACD Fast line", min_value=2, max_value=40, value=12, key="exp_macd_f")
            exp_macd_s = st.number_input("MACD Slow line", min_value=15, max_value=80, value=26, key="exp_macd_s")
            exp_macd_sig = st.number_input("MACD Signal line", min_value=2, max_value=30, value=9, key="exp_macd_sig")
        with exp_col3:
            exp_bb_win = st.number_input("Chu kỳ dải Bollinger Bands", min_value=5, max_value=50, value=20, key="exp_bb_win")
            exp_bb_dev = st.number_input("Độ lệch chuẩn dải Bollinger Bands", min_value=0.5, max_value=4.0, value=2.0, key="exp_bb_dev")
            
        # Select strategy to show signals
        selected_indicator_signals = st.selectbox(
            "Chọn chiến lược hiển thị tín hiệu giao dịch trên đồ thị:",
            options=["RSI thuần", "RSI + MACD kết hợp", "RSI + Bollinger Bands kết hợp"]
        )
        
        indicator_map = {
            "RSI thuần": ("rsi", dict(rsi_window=exp_rsi_win, lower=exp_rsi_lower, upper=exp_rsi_upper)),
            "RSI + MACD kết hợp": ("rsi_macd", dict(rsi_window=exp_rsi_win, upper=exp_rsi_upper, macd_fast=exp_macd_f, macd_slow=exp_macd_s, macd_sign=exp_macd_sig)),
            "RSI + Bollinger Bands kết hợp": ("bollinger_rsi", dict(bb_window=exp_bb_win, bb_dev=exp_bb_dev, rsi_window=exp_rsi_win, lower=exp_rsi_lower, upper=exp_rsi_upper))
        }
        
        strat_key, strat_params = indicator_map[selected_indicator_signals]
        
        # Calculate indicators for plotting
        df_stock["RSI"] = rsi_wilder(df_stock["Close"], exp_rsi_win)
        df_stock["MACD"], df_stock["MACD_Signal"] = macd_lines(df_stock["Close"], exp_macd_f, exp_macd_s, exp_macd_sig)
        df_stock["BB_Upper"], df_stock["BB_Lower"] = bollinger(df_stock["Close"], exp_bb_win, exp_bb_dev)
        df_stock["SMA100"] = df_stock["Close"].rolling(trend_window, min_periods=trend_window).mean()
        
        # Generate Signals
        buy_signals, sell_signals = generate_signals(df_stock, strat_key, strat_params, trend_window)
        
        # Plot subplots (Price + Indicators)
        fig_explore = make_subplots(
            rows=3, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # 1. Main Price Plot (Close price + BB upper/lower + SMA100)
        fig_explore.add_trace(go.Scatter(
            x=df_stock["Date"], 
            y=df_stock["Close"], 
            name="Giá đóng cửa", 
            line=dict(color="#1f77b4", width=2)
        ), row=1, col=1)
        
        fig_explore.add_trace(go.Scatter(
            x=df_stock["Date"], 
            y=df_stock["SMA100"], 
            name="SMA(100)", 
            line=dict(color="#e74c3c", width=1.2, dash="dash")
        ), row=1, col=1)
        
        # Add Bollinger Bands
        fig_explore.add_trace(go.Scatter(
            x=df_stock["Date"], 
            y=df_stock["BB_Upper"], 
            name="Bollinger Band Upper", 
            line=dict(color="#2ca02c", width=0.8, dash="dot")
        ), row=1, col=1)
        
        fig_explore.add_trace(go.Scatter(
            x=df_stock["Date"], 
            y=df_stock["BB_Lower"], 
            name="Bollinger Band Lower", 
            line=dict(color="#2ca02c", width=0.8, dash="dot"),
            fill='tonexty', # fills area between upper and lower band
            fillcolor='rgba(44, 160, 44, 0.05)'
        ), row=1, col=1)
        
        # Buy signals markers on price
        buy_indices = np.nonzero(buy_signals)[0]
        if len(buy_indices) > 0:
            fig_explore.add_trace(go.Scatter(
                x=df_stock["Date"].iloc[buy_indices],
                y=df_stock["Close"].iloc[buy_indices],
                mode="markers",
                name="Tín hiệu MUA",
                marker=dict(symbol="triangle-up", color="#2ecc71", size=11, line=dict(color="white", width=0.8))
            ), row=1, col=1)
            
        # Sell signals markers on price
        sell_indices = np.nonzero(sell_signals)[0]
        if len(sell_indices) > 0:
            fig_explore.add_trace(go.Scatter(
                x=df_stock["Date"].iloc[sell_indices],
                y=df_stock["Close"].iloc[sell_indices],
                mode="markers",
                name="Tín hiệu BÁN",
                marker=dict(symbol="triangle-down", color="#e74c3c", size=11, line=dict(color="white", width=0.8))
            ), row=1, col=1)
            
        # 2. RSI Subplot
        fig_explore.add_trace(go.Scatter(
            x=df_stock["Date"], 
            y=df_stock["RSI"], 
            name="RSI", 
            line=dict(color="#9467bd", width=1.5)
        ), row=2, col=1)
        # Threshold lines for RSI
        fig_explore.add_hline(y=exp_rsi_upper, line_width=0.8, line_dash="dash", line_color="#e74c3c", row=2, col=1)
        fig_explore.add_hline(y=exp_rsi_lower, line_width=0.8, line_dash="dash", line_color="#2ecc71", row=2, col=1)
        
        # 3. MACD Subplot
        fig_explore.add_trace(go.Scatter(
            x=df_stock["Date"], 
            y=df_stock["MACD"], 
            name="MACD Line", 
            line=dict(color="#ff7f0e", width=1.2)
        ), row=3, col=1)
        
        fig_explore.add_trace(go.Scatter(
            x=df_stock["Date"], 
            y=df_stock["MACD_Signal"], 
            name="Signal Line", 
            line=dict(color="#2ca02c", width=1.2)
        ), row=3, col=1)
        
        # MACD histogram
        macd_hist = df_stock["MACD"] - df_stock["MACD_Signal"]
        hist_colors = ["#2ecc71" if h >= 0 else "#e74c3c" for h in macd_hist]
        fig_explore.add_trace(go.Bar(
            x=df_stock["Date"], 
            y=macd_hist, 
            name="MACD Histogram", 
            marker_color=hist_colors,
            opacity=0.4
        ), row=3, col=1)
        
        fig_explore.update_layout(
            template="plotly_dark",
            height=700,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
        )
        
        fig_explore.update_yaxes(title_text="Giá (VND)", row=1, col=1)
        fig_explore.update_yaxes(title_text="RSI", row=2, col=1, range=[10, 90])
        fig_explore.update_yaxes(title_text="MACD", row=3, col=1)
        
        st.plotly_chart(fig_explore, use_container_width=True)

else:
    st.info("💡 Chọn các cấu hình ở thanh bên và bấm nút **🚀 CHẠY BACKTEST** để bắt đầu phân tích dữ liệu danh mục đầu tư!")
