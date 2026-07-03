"""DownsideIQ dashboard (spec §15, §21) — a pure FRONTEND for the FastAPI backend.

Contains NO business logic: every panel is rendered from API responses. Set
``API_BASE`` (default http://localhost:8000) to point at the backend.
"""
from __future__ import annotations

import os
import time

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
DEC_COLOR = {"SHORT": "#e74c3c", "WATCH": "#f1c40f", "NO TRADE": "#7f8c8d"}

st.set_page_config(page_title="DownsideIQ", page_icon="📉", layout="wide")


def api_get(path: str, params: dict | None = None):
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        return {"_error": f"{r.status_code}: {r.text[:200]}"}
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"API unreachable at {API_BASE}: {exc}"}


def api_post(path: str, params: dict | None = None):
    try:
        r = httpx.post(f"{API_BASE}{path}", params=params, timeout=30)
        return r.json() if r.status_code == 200 else {"_error": f"{r.status_code}: {r.text[:200]}"}
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"API unreachable: {exc}"}


def _err(obj) -> str | None:
    return obj.get("_error") if isinstance(obj, dict) else None


# --------------------------------------------------------------------------- #
def page_system_health():
    st.header("🩺 System Health")
    h = api_get("/health")
    if _err(h):
        st.error(_err(h)); st.info("Start the backend:  `python -m src.cli run-api`"); return
    c = st.columns(4)
    c[0].metric("Status", h["status"].upper())
    c[1].metric("Database", "✓" if h["database"] else "✗")
    c[2].metric("Last signal", (h.get("last_signal_at") or "—")[:19])
    up = sum(h["providers"].values())
    c[3].metric("Providers up", f"{up}/{len(h['providers'])}")
    st.subheader("Providers")
    st.dataframe(pd.DataFrame([{"provider": k, "available": v} for k, v in h["providers"].items()],),
                 use_container_width=True)
    ms = api_get("/model/status")
    if not _err(ms) and ms.get("performance"):
        st.subheader(f"Model performance (active mode: {ms.get('active_mode')})")
        st.dataframe(pd.DataFrame(ms["performance"]), use_container_width=True, height=240)


def page_live_signal():
    st.header("📡 Live Signal")
    col = st.columns([2, 1, 1])
    ticker = col[0].text_input("Ticker", "NVDA").upper()
    mode = col[1].selectbox("Mode", ["strict", "research"])
    run = col[2].button("▶ Analyze", type="primary")
    st.caption("strict = live/capital discipline · research = exploratory (percentile-calibrated). "
               "Live trading disabled.")

    if run:
        job = api_post(f"/analyze/{ticker}", {"mode": mode})
        if _err(job):
            st.error(_err(job)); return
        with st.spinner("Running councils + risk engine…"):
            result = _poll(job["job_id"])
        if result is None:
            st.warning("Job still running — check back shortly."); return
        st.session_state["last_signal_id"] = result.get("signal_id")

    sid = st.session_state.get("last_signal_id")
    if not sid:
        st.info("Enter a ticker and click **Analyze**."); return
    _render_signal(sid)


def _poll(job_id, attempts=40):
    for _ in range(attempts):
        j = api_get(f"/jobs/{job_id}")
        if _err(j):
            return None
        if j.get("ready"):
            return j.get("result")
        time.sleep(1.0)
    return None


def _render_signal(sid: str):
    sig = api_get(f"/predictions/{sid}")
    if _err(sig):
        st.error(_err(sig)); return
    dec = sig["decision"]
    st.markdown(f"<h2 style='color:{DEC_COLOR.get(dec,'#333')}'>{dec} — {sig['ticker']} "
                f"@ {sig['timestamp']}</h2>", unsafe_allow_html=True)
    gov = sig.get("governance", {}) or {}
    scores = gov.get("scores", {})

    c = st.columns(4)
    c[0].metric("Adj P(downside)", _pct(scores.get("adjusted_p_downside")))
    c[1].metric("Adj risk score", _f(scores.get("adjusted_downside_risk_score")))
    c[2].metric("Model agreement", _f(scores.get("model_agreement_score")))
    c[3].metric("Data quality", str(sig.get("data_quality", "—")).upper())
    c2 = st.columns(4)
    c2[0].metric("Data confidence", _f(scores.get("data_confidence_score")))
    c2[1].metric("Price/vol confirm", _f(scores.get("price_volume_confirmation")))
    c2[2].metric("Expected edge", f"{_f(scores.get('expected_edge_bps'))} bps")
    c2[3].metric("Kill switch", "🔴" if sig.get("kill_switch_active") else "🟢")

    st.subheader("Decision explanation")
    st.write("**Reason:** " + (gov.get("reason") or sig.get("reason") or "—"))
    g = gov.get("gates", {})
    st.write("**Gates:** " + ", ".join(f"{k}={v}" for k, v in g.items()))

    cc = st.columns(2)
    with cc[0]:
        st.subheader("News / event intelligence")
        st.json(gov.get("news_catalysts", {}))
        dq = gov.get("data_quality", {})
        if dq.get("reasons"):
            st.warning("Degraded: " + "; ".join(dq["reasons"]))
    with cc[1]:
        st.subheader("Top drivers")
        drivers = gov.get("top_drivers") or []
        if drivers:
            st.plotly_chart(px.bar(pd.DataFrame(drivers), x="weight", y="driver",
                            orientation="h").update_layout(yaxis={"categoryorder": "total ascending"},
                                                           height=300),
                            use_container_width=True)
    sizing = gov.get("position_sizing")
    if sizing:
        st.subheader("Position sizing (paper)")
        st.json(sizing)


def page_signal_history():
    st.header("📜 Signal History (strict)")
    ticker = st.text_input("Ticker filter (blank = all)", "").upper() or None
    data = api_get("/signals/history", {"mode": "strict", "ticker": ticker, "limit": 300})
    if _err(data):
        st.error(_err(data)); return
    sigs = data.get("signals", [])
    if not sigs:
        st.info("No signals yet. Run an analysis or `python -m src.cli analyze -t NVDA`."); return
    df = pd.DataFrame(sigs)
    counts = df["decision"].value_counts()
    c = st.columns(4)
    c[0].metric("Predictions", len(df))
    for i, k in enumerate(["NO TRADE", "WATCH", "SHORT"]):
        c[i + 1].metric(k, int(counts.get(k, 0)))
    scd = df.dropna(subset=["actual_return_24h"]) if "actual_return_24h" in df else pd.DataFrame()
    if not scd.empty:
        fig = px.scatter(scd, x="adjusted_p_downside", y="actual_return_24h", color="decision",
                         color_discrete_map=DEC_COLOR, title="Predicted prob vs realised return")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[["timestamp", "decision", "adjusted_p_downside", "model_agreement_score",
                     "data_quality", "actual_return_24h", "prediction_correct"]],
                 use_container_width=True, height=320)


def page_paper_trading():
    st.header("💸 Paper Trading")
    st.warning("EXPLORATORY (research-mode) trades — demonstration of mechanics, not the strict "
               "live gate. Hypothetical, no real money.")
    data = api_get("/paper-trades", {"mode": "research"})
    if _err(data):
        st.error(_err(data)); return
    perf = data.get("performance", {})
    if perf.get("n_trades", 0) == 0:
        st.info("No paper trades yet. Run `python -m src.cli backtest -t NVDA -m research`."); return
    c = st.columns(4)
    c[0].metric("Trades", perf.get("n_trades"))
    c[1].metric("Hit rate", _pct(perf.get("hit_rate")))
    c[2].metric("Profit factor", _f(perf.get("profit_factor")))
    c[3].metric("Total P&L", f"${_f(perf.get('total_pnl'))}")
    c2 = st.columns(3)
    c2[0].metric("Sharpe", _f(perf.get("sharpe_ratio")))
    c2[1].metric("Sortino", _f(perf.get("sortino_ratio")))
    c2[2].metric("Max DD ($)", _f(perf.get("max_drawdown")))
    trades = pd.DataFrame(data.get("trades", []))
    if not trades.empty and "pnl" in trades:
        trades = trades.iloc[::-1].reset_index(drop=True)
        trades["cum_pnl"] = trades["pnl"].cumsum()
        st.plotly_chart(px.line(trades, y="cum_pnl", title="Cumulative P&L"), use_container_width=True)
        st.dataframe(trades, use_container_width=True, height=280)


def page_risk_monitor():
    st.header("🛡️ Risk Monitor")
    mode = st.selectbox("Mode", ["strict", "research"])
    d = api_get("/risk/status", {"mode": mode})
    if _err(d):
        st.error(_err(d)); return
    ks = d.get("kill_switch", {})
    st.subheader(f"Kill switch: {'🔴 ACTIVE' if ks.get('active') else '🟢 clear'}")
    if ks.get("active"):
        st.error("; ".join(ks.get("reasons", [])))
    state = ks.get("state", {})
    c = st.columns(4)
    c[0].metric("Daily P&L", _pct(state.get("daily_pnl_pct")))
    c[1].metric("Weekly P&L", _pct(state.get("weekly_pnl_pct")))
    c[2].metric("Consec. losses", state.get("consecutive_losses", 0))
    c[3].metric("Rolling accuracy", _pct(d.get("rolling_accuracy")))
    st.subheader("Data-quality distribution")
    st.json(d.get("data_quality_distribution", {}))
    st.metric("False-positive short rate", _pct(d.get("false_positive_short_rate")))


def _f(v):
    return round(float(v), 3) if isinstance(v, (int, float)) else "—"


def _pct(v):
    return f"{float(v)*100:.1f}%" if isinstance(v, (int, float)) else "—"


PAGES = {
    "System Health": page_system_health,
    "Live Signal": page_live_signal,
    "Signal History": page_signal_history,
    "Paper Trading": page_paper_trading,
    "Risk Monitor": page_risk_monitor,
}


def main():
    st.sidebar.title("📉 DownsideIQ")
    st.sidebar.caption(f"API: {API_BASE}")
    page = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.warning("Research tool — not financial advice. Live trading disabled.")
    PAGES[page]()


main()
