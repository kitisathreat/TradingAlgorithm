"""Multi-user trading simulator + model maker web app."""

import streamlit as st
import sys
import os
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

REPO_ROOT = Path(__file__).parent
ORCHESTRATOR_PATH = REPO_ROOT / "_2_Orchestrator_And_ML_Python"
sys.path.insert(0, str(ORCHESTRATOR_PATH))

st.set_page_config(
    page_title="Trading System",
    page_icon="\U0001f4c8",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from auth.session import require_login, logout_button
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

try:
    from interactive_training_app.backend.model_trainer import ModelTrainer
    MODEL_TRAINER_AVAILABLE = True
except ImportError:
    MODEL_TRAINER_AVAILABLE = False

try:
    from paper_trading.portfolio import Portfolio
    from paper_trading.broker import PaperBroker, BrokerConfig
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False

try:
    from inference.trade_suggester import TradeSuggester, InferenceParams
    SUGGESTER_AVAILABLE = True
except ImportError:
    SUGGESTER_AVAILABLE = False

try:
    from training.training_loop import TrainingLoopController
    TRAINING_LOOP_AVAILABLE = True
except ImportError:
    TRAINING_LOOP_AVAILABLE = False

try:
    from fundamental.sec_file_sourcer import SECFileSourcer
    from fundamental.fundamental_features import FundamentalFeatureExtractor
    FUNDAMENTAL_AVAILABLE = True
except ImportError:
    FUNDAMENTAL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if AUTH_AVAILABLE:
    username = require_login()
    if not username:
        st.stop()
else:
    st.warning("Auth module not available. Running in single-user demo mode.")
    username = "demo"


@st.cache_resource
def get_trainer(uname: str):
    if not MODEL_TRAINER_AVAILABLE:
        return None
    user_dir = Path("data/users") / uname
    return ModelTrainer(username=uname, user_dir=user_dir)


@st.cache_resource
def get_portfolio(uname: str):
    if not PAPER_TRADING_AVAILABLE:
        return None
    return Portfolio.from_username(uname)


def get_loop(uname: str):
    key = f"training_loop_{uname}"
    if key not in st.session_state:
        trainer = get_trainer(uname)
        if trainer is None or not TRAINING_LOOP_AVAILABLE:
            st.session_state[key] = None
        else:
            st.session_state[key] = TrainingLoopController(trainer)
    return st.session_state[key]


def get_suggester(uname: str):
    trainer = get_trainer(uname)
    if trainer is None or not SUGGESTER_AVAILABLE:
        return None
    params = st.session_state.get("inference_params", InferenceParams())
    return TradeSuggester(trainer, params)


def render_sidebar(uname: str):
    with st.sidebar:
        st.markdown(f"**Logged in as:** {uname}")
        if AUTH_AVAILABLE:
            logout_button()
        st.divider()
        st.header("\U0001f39b\ufe0f Strategy Parameters")
        params = InferenceParams(
            risk_tolerance=st.slider("Risk Tolerance", 0.0, 1.0, 0.5, 0.05,
                help="Conservative \u2194 Aggressive. Scales position size."),
            information_amount=st.slider("Information Amount", 0.0, 1.0, 1.0, 0.05,
                help="0=technical only, 1=all sources"),
            fundamental_weight=st.slider("Fundamental Weight", 0.0, 1.0, 0.3, 0.05,
                help="0=ignore SEC fundamentals"),
            confidence_threshold=st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05,
                help="HOLD if model confidence below this"),
            max_position_pct=st.slider("Max Position Size", 0.01, 0.50, 0.10, 0.01,
                help="Max % of portfolio per position"),
            stop_loss_pct=st.slider("Stop-Loss %", 0.01, 0.25, 0.05, 0.01),
            take_profit_pct=st.slider("Take-Profit %", 0.01, 0.50, 0.15, 0.01),
        ) if SUGGESTER_AVAILABLE else None
        if params:
            st.session_state["inference_params"] = params
        st.divider()
        debug_mode = st.toggle("\U0001f527 Debug Mode", value=False)
        st.session_state["debug_mode"] = debug_mode
        if debug_mode:
            st.subheader("Connection Status")
            st.write("yfinance", "\u2705" if YFINANCE_AVAILABLE else "\u274c")
            st.write("SEC / arelle", "\u2705" if FUNDAMENTAL_AVAILABLE else "\u274c")
            try:
                import trading_engine
                st.write("C++ engine", "\u2705")
            except ImportError:
                st.write("C++ engine", "\u274c")


render_sidebar(username)


def fetch_ohlcv(symbol: str, days: int = 30):
    if not YFINANCE_AVAILABLE:
        return None, {}
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days * 2)
        df = yf.Ticker(symbol).history(start=start, end=end)
        if df is None or df.empty:
            return None, {}
        df = df.tail(days)
        return df, {"current_price": float(df["Close"].iloc[-1]),
                    "volume": float(df["Volume"].iloc[-1])}
    except Exception as e:
        logger.warning("yfinance error for %s: %s", symbol, e)
        return None, {}


def candlestick_chart(df: pd.DataFrame, symbol: str):
    if not PLOTLY_AVAILABLE or df is None or df.empty:
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                  low=df["Low"], close=df["Close"], name=symbol), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color="rgba(100,100,200,0.4)"), row=2, col=1)
    fig.update_layout(height=420, showlegend=False, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


TAB_LABELS = ["\U0001f3e0 Dashboard", "\U0001f4da Train Me", "\U0001f9e0 My Model",
              "\U0001f52e Suggest Trades", "\U0001f4bc Paper Trading",
              "\U0001f3e6 Fundamentals", "\u2699\ufe0f Settings"]
tabs = st.tabs(TAB_LABELS)

with tabs[0]:
    st.title(f"Welcome back, {username}!")
    col1, col2, col3 = st.columns(3)
    trainer = get_trainer(username)
    with col1:
        st.metric("Training Examples", len(trainer.training_examples) if trainer else 0)
    with col2:
        portfolio = get_portfolio(username)
        if portfolio and YFINANCE_AVAILABLE:
            def price_lookup(sym):
                try:
                    return float(yf.Ticker(sym).fast_info["last_price"])
                except Exception:
                    return 0.0
            equity = portfolio.equity(price_lookup)
            ret = portfolio.total_return(price_lookup) * 100
            st.metric("Portfolio Equity", f"${equity:,.0f}", f"{ret:+.1f}%")
        else:
            st.metric("Portfolio", "N/A")
    with col3:
        st.metric("Trades Placed", len(portfolio.trades) if portfolio else 0)
    st.info("Use the **\U0001f4da Train Me** tab to build your model, then **\U0001f52e Suggest Trades** to get recommendations.")

with tabs[1]:
    st.header("\U0001f4da Train Me")
    loop = get_loop(username)
    if loop is None:
        st.error("Training loop not available.")
    else:
        current, required = loop.progress()
        st.progress(min(current / required, 1.0), text=f"{current}/{required} examples collected")
        counts = loop.decision_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("BUY", counts.get("BUY", 0))
        c2.metric("HOLD", counts.get("HOLD", 0))
        c3.metric("SELL", counts.get("SELL", 0))
        for warn in loop.class_balance_warnings():
            st.warning(warn)
        if st.button("\U0001f500 Get Next Stock") or "current_prompt" not in st.session_state:
            with st.spinner("Fetching stock data..."):
                prompt = loop.next_training_prompt()
                if prompt:
                    st.session_state["current_prompt"] = prompt
                else:
                    st.error("Could not fetch stock data.")
        prompt = st.session_state.get("current_prompt")
        if prompt:
            st.subheader(f"\U0001f4ca {prompt.symbol}")
            candlestick_chart(prompt.stock_df, prompt.symbol)
            reasoning = st.text_area("Your reasoning:", height=100, key="train_reasoning",
                                     placeholder="Why would you buy/sell/hold this stock?")
            b1, b2, b3 = st.columns(3)
            decision = None
            if b1.button("\U0001f7e2 BUY", use_container_width=True):
                decision = "BUY"
            if b2.button("\U0001f7e1 HOLD", use_container_width=True):
                decision = "HOLD"
            if b3.button("\U0001f534 SELL", use_container_width=True):
                decision = "SELL"
            if decision:
                if not reasoning.strip():
                    st.warning("Please add some reasoning before submitting.")
                else:
                    ok = loop.submit_decision(prompt.stock_df, decision, reasoning,
                                             symbol=prompt.symbol)
                    if ok:
                        st.success(f"\u2705 Recorded {decision} for {prompt.symbol}")
                        del st.session_state["current_prompt"]
                        st.rerun()
                    else:
                        st.error("Failed to record decision.")
        if loop.is_ready_to_train():
            st.success(f"\U0001f389 You have enough data ({current} examples)! Go to **\U0001f9e0 My Model** to train.")

with tabs[2]:
    st.header("\U0001f9e0 My Model")
    trainer = get_trainer(username)
    if trainer is None:
        st.error("ModelTrainer not available.")
    else:
        meta = getattr(trainer, "model_metadata", {}) or {}
        if meta.get("trained"):
            st.success(f"Model trained \u2014 last trained {meta.get('last_trained', 'unknown')}")
            c1, c2 = st.columns(2)
            c1.metric("Architecture", meta.get("model_type", "N/A"))
            c2.metric("Accuracy", f"{meta.get('accuracy', 0):.1%}")
        else:
            st.info("No model trained yet. Collect training data in \U0001f4da Train Me first.")
        arch = st.selectbox("Architecture", ["Simple (fast)", "Standard", "Complex (slow)"], index=1)
        arch_map = {"Simple (fast)": "simple", "Standard": "standard", "Complex (slow)": "complex"}
        if st.button("\U0001f680 Train Model"):
            loop = get_loop(username)
            if loop is None or not loop.is_ready_to_train():
                st.warning("Not enough training data yet.")
            else:
                with st.spinner("Training model..."):
                    try:
                        trainer.model_type = arch_map[arch]
                        result = trainer.train_model()
                        if result:
                            st.success("Model trained successfully!")
                            st.cache_resource.clear()
                        else:
                            st.error("Training returned no result.")
                    except Exception as e:
                        st.error(f"Training failed: {e}")

with tabs[3]:
    st.header("\U0001f52e Suggest Trades")
    if not SUGGESTER_AVAILABLE:
        st.error("TradeSuggester not available.")
    else:
        ticker_input = st.text_input("Ticker symbol", value="AAPL",
                                     key="suggest_ticker").upper().strip()
        if st.button("Get Suggestion") and ticker_input:
            with st.spinner(f"Analyzing {ticker_input}..."):
                df, info = fetch_ohlcv(ticker_input, days=30)
                if df is None:
                    st.error("Could not fetch price data.")
                else:
                    suggester = get_suggester(username)
                    if suggester is None:
                        st.error("Model not loaded. Train a model first.")
                    else:
                        fund_kpis = None
                        if FUNDAMENTAL_AVAILABLE:
                            cache_path = (Path("data/users") / username /
                                         "fundamentals_cache" / f"{ticker_input}.json")
                            if cache_path.exists():
                                try:
                                    with open(cache_path) as f:
                                        fund_kpis = json.load(f)
                                except Exception:
                                    pass
                        try:
                            suggestion = suggester.suggest(ticker_input, df, fund_kpis,
                                                           get_portfolio(username))
                            st.session_state["last_suggestion"] = {
                                "ticker": ticker_input, "suggestion": suggestion}
                        except Exception as e:
                            st.error(f"Suggestion failed: {e}")
        sug_state = st.session_state.get("last_suggestion")
        if sug_state:
            sug = sug_state["suggestion"]
            ticker = sug_state["ticker"]
            color = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "HOLD": "\U0001f7e1"}.get(sug.decision, "\u26aa")
            st.subheader(f"{color} {sug.decision} \u2014 {ticker}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{sug.confidence:.1%}")
            c2.metric("Shares", f"{sug.shares:.2f}")
            c3.metric("Stop-Loss", f"${sug.stop_loss:.2f}" if sug.stop_loss else "N/A")
            if sug.take_profit:
                st.caption(f"Take-profit: ${sug.take_profit:.2f}")
            if sug.reasoning:
                st.caption(sug.reasoning)
            if PAPER_TRADING_AVAILABLE and sug.decision in ("BUY", "SELL"):
                if st.button("Execute in Paper Trading"):
                    portfolio = get_portfolio(username)
                    if portfolio:
                        broker = PaperBroker(portfolio, config=BrokerConfig())
                        trade = broker.execute_suggestion(
                            sug, reasoning=f"Model suggestion (conf={sug.confidence:.2f})")
                        if trade:
                            portfolio.save()
                            st.success(f"Order executed: {sug.decision} {sug.shares:.2f} shares of {ticker}")
                        else:
                            st.error("Order execution failed.")

with tabs[4]:
    st.header("\U0001f4bc Paper Trading")
    if not PAPER_TRADING_AVAILABLE:
        st.error("Paper trading module not available.")
    else:
        portfolio = get_portfolio(username)
        if portfolio is None:
            st.error("Portfolio unavailable.")
        else:
            def _price(sym):
                if not YFINANCE_AVAILABLE:
                    return 0.0
                try:
                    return float(yf.Ticker(sym).fast_info["last_price"])
                except Exception:
                    return 0.0
            equity = portfolio.equity(_price)
            ret = portfolio.total_return(_price)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cash", f"${portfolio.cash:,.2f}")
            c2.metric("Equity", f"${equity:,.2f}")
            c3.metric("Total Return", f"{ret:.1%}")
            c4.metric("# Trades", len(portfolio.trades))
            if portfolio.positions:
                st.subheader("Positions")
                rows = []
                for sym, pos in portfolio.positions.items():
                    price = _price(sym)
                    rows.append({"Symbol": sym, "Shares": pos.shares,
                                 "Avg Cost": pos.avg_cost, "Current Price": price,
                                 "Market Value": pos.shares * price,
                                 "Unrealized P&L": (price - pos.avg_cost) * pos.shares})
                st.dataframe(pd.DataFrame(rows).set_index("Symbol"), use_container_width=True)
            st.subheader("Place Order")
            pc1, pc2, pc3 = st.columns(3)
            order_sym = pc1.text_input("Symbol", key="order_sym").upper().strip()
            order_side = pc2.selectbox("Side", ["BUY", "SELL"], key="order_side")
            order_shares = pc3.number_input("Shares", min_value=0.01, value=1.0,
                                            step=0.01, key="order_shares")
            if st.button("Execute Order") and order_sym:
                broker = PaperBroker(portfolio, config=BrokerConfig())
                trade = broker.execute_order(order_sym, order_side, order_shares, source="manual")
                if trade:
                    portfolio.save()
                    st.success(f"\u2705 {order_side} {order_shares:.2f} shares of {order_sym}")
                    st.rerun()
                else:
                    st.error("Order failed.")
            if portfolio.trades:
                st.subheader("Trade History")
                st.dataframe(pd.DataFrame([
                    {"Time": str(t.timestamp)[:16], "Symbol": t.symbol, "Side": t.side,
                     "Shares": t.shares, "Price": t.price, "Source": t.source}
                    for t in reversed(portfolio.trades)
                ]), use_container_width=True)
            with st.expander("\u26a0\ufe0f Reset Portfolio"):
                st.warning("This will archive your current portfolio and start fresh with $100,000.")
                if st.button("Confirm Reset"):
                    archive_dir = Path("data/users") / username / "trades" / "archive"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    portfolio.save()
                    src = Path("data/users") / username / "trades" / "portfolio.json"
                    if src.exists():
                        src.rename(archive_dir / f"portfolio_{ts}.json")
                    get_portfolio.clear()
                    st.success("Portfolio reset. Refresh the page.")

with tabs[5]:
    st.header("\U0001f3e6 Fundamentals")
    if not FUNDAMENTAL_AVAILABLE:
        st.error("Fundamental module not available (arelle/rapidfuzz may be missing).")
    else:
        f_ticker = st.text_input("Ticker", value="AAPL", key="fund_ticker").upper().strip()
        f_quarters = st.selectbox("Quarters", [4, 8, 12, 16, 20], index=1)
        if st.button("Fetch SEC Data") and f_ticker:
            progress_bar = st.progress(0, text="Starting...")
            def progress_cb(msg: str, pct: float):
                progress_bar.progress(min(pct, 1.0), text=msg)
            with st.spinner("Fetching SEC filings..."):
                try:
                    sourcer = SECFileSourcer()
                    model = sourcer.create_financial_model(f_ticker, num_quarters=f_quarters,
                                                           progress_callback=progress_cb)
                    st.session_state["fund_model"] = {"ticker": f_ticker, "model": model}
                    progress_bar.empty()
                    st.success(f"Loaded SEC data for {f_ticker}")
                except Exception as e:
                    st.error(f"SEC fetch failed: {e}")
        fund_state = st.session_state.get("fund_model")
        if fund_state:
            model = fund_state["model"]
            ticker_shown = fund_state["ticker"]
            for sheet_key, label in [("annual_income_statement", "Income Statement"),
                                      ("annual_balance_sheet", "Balance Sheet"),
                                      ("annual_cash_flow", "Cash Flow")]:
                df_sheet = model.get(sheet_key)
                if df_sheet is not None and not df_sheet.empty:
                    with st.expander(label, expanded=(sheet_key == "annual_income_statement")):
                        st.dataframe(df_sheet, use_container_width=True)
            kpis = FundamentalFeatureExtractor().extract(model, ticker_shown)
            st.subheader("KPI Summary")
            st.dataframe(pd.DataFrame([
                {"KPI": k.replace("_", " ").title(), "Value": f"{v:+.3f}",
                 "Signal": "\U0001f7e2" if v > 0.1 else ("\U0001f534" if v < -0.1 else "\U0001f7e1")}
                for k, v in kpis.items()
            ]), use_container_width=True, hide_index=True)
            cache_dir = Path("data/users") / username / "fundamentals_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            col_save, col_train = st.columns(2)
            if col_save.button("\U0001f4be Save KPIs to Cache"):
                with open(cache_dir / f"{ticker_shown}.json", "w") as f:
                    json.dump(kpis, f, indent=2)
                st.success("Saved to cache for future suggestions.")
            if col_train.checkbox("Include in Next Training Example"):
                st.session_state["pending_fundamental_kpis"] = kpis
                st.caption("7 KPI features will be added to the next training submission.")

with tabs[6]:
    st.header("\u2699\ufe0f Settings")
    if AUTH_AVAILABLE:
        with st.expander("Change Password"):
            old_pw = st.text_input("Current password", type="password", key="old_pw")
            new_pw = st.text_input("New password", type="password", key="new_pw")
            new_pw2 = st.text_input("Confirm new password", type="password", key="new_pw2")
            if st.button("Update Password"):
                if new_pw != new_pw2:
                    st.error("New passwords don't match.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        from auth.user_store import UserStore
                        import bcrypt
                        store = UserStore()
                        cfg = store.load()
                        stored_hash = cfg["credentials"]["usernames"][username]["password"].encode()
                        if bcrypt.checkpw(old_pw.encode(), stored_hash):
                            cfg["credentials"]["usernames"][username]["password"] = (
                                bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt(12)).decode())
                            store.save(cfg)
                            st.success("Password updated.")
                        else:
                            st.error("Current password is incorrect.")
                    except Exception as e:
                        st.error(f"Could not update password: {e}")
    trainer = get_trainer(username)
    if trainer and hasattr(trainer, "training_examples"):
        with st.expander("Export Training Data"):
            if st.button("Download as JSON"):
                data = json.dumps(trainer.training_examples, indent=2, default=str)
                st.download_button("\u2b07\ufe0f Download", data=data,
                                   file_name=f"{username}_training_data.json",
                                   mime="application/json")
    with st.expander("\u26a0\ufe0f Delete Account"):
        st.error("This will permanently delete your model, training data, and portfolio.")
        confirm_name = st.text_input("Type your username to confirm:", key="delete_confirm")
        if st.button("Delete My Account") and confirm_name == username:
            import shutil
            user_dir = Path("data/users") / username
            if user_dir.exists():
                shutil.rmtree(user_dir)
            if AUTH_AVAILABLE:
                try:
                    from auth.user_store import UserStore
                    store = UserStore()
                    cfg = store.load()
                    cfg["credentials"]["usernames"].pop(username, None)
                    store.save(cfg)
                except Exception:
                    pass
            st.session_state.clear()
            st.success("Account deleted. Please refresh.")

if st.session_state.get("debug_mode"):
    with st.expander("\U0001f527 Debug Panel", expanded=True):
        log_path = Path("data/users") / username / "app.log"
        if log_path.exists():
            st.code("\n".join(log_path.read_text().splitlines()[-50:]), language="text")
        else:
            st.caption("No log file yet.")
        sug = st.session_state.get("last_suggestion", {}).get("suggestion")
        if sug and hasattr(sug, "feature_snapshot") and sug.feature_snapshot:
            st.subheader("Feature Snapshot")
            st.json(sug.feature_snapshot)
