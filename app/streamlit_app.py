"""Streamlit interactive UI — Bloomberg terminal dark theme."""

from __future__ import annotations

import os
import sys
import tempfile
import warnings

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

from src.backtest import run_backtest, compute_factor_attribution
from src.stats import compute_stats, compute_monthly_returns, compute_drawdown_series
from src.tearsheet import generate_tearsheet
from src.universe import build_universe
from src.factors import compute_composite_score, get_quintile_portfolios, DEFAULT_WEIGHTS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Factor Model Tearsheet",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "backtest_done":     False,
    "backtest_results":  None,
    "factor_scores":     None,
    "stats":             None,
    "monthly_ret":       None,
    "drawdown":          None,
    "attribution":       None,
    "universe_tickers":  None,
    "universe_price_df": None,
    "universe_fund_df":  None,
    "universe_labels":   None,
    "last_run_meta":     None,
    "last_settings":     None,
    "run_settings":      None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG     = "#0a0e1a"
CARD   = "#111827"
CARD2  = "#1a2235"
BORDER = "#2d3748"
ACCENT = "#3b82f6"
GREEN  = "#10b981"
RED    = "#ef4444"
GOLD   = "#f59e0b"
TEXT   = "#f1f5f9"
TEXT2  = "#94a3b8"
TEXT3  = "#64748b"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0a0e1a; }
.stApp > header { background-color: #0a0e1a; }
[data-testid="stSidebar"] {
    background-color: #0d1321;
    border-right: 1px solid #2d3748;
}
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
.stMarkdown, .stText, p, span, label, div { color: #f1f5f9; }
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 28px !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #0a0e1a;
    border-bottom: 1px solid #2d3748;
    gap: 0px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #64748b;
    font-size: 13px;
    font-weight: 500;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
    background-color: transparent !important;
}
.stButton > button {
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 24px;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2563eb; }
[data-testid="stRadio"] label { color: #94a3b8 !important; font-size: 13px !important; }
[data-testid="stSlider"] label { color: #94a3b8 !important; font-size: 12px !important; }
.stSlider [data-baseweb="slider"] { margin-top: 4px; }
[data-testid="stDataFrame"] { border: 1px solid #2d3748; border-radius: 8px; overflow: hidden; }
.stDataFrame table { background: #111827 !important; color: #f1f5f9 !important; }
.stDataFrame thead tr th {
    background: #1a2235 !important;
    color: #94a3b8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    border-bottom: 1px solid #2d3748 !important;
    padding: 10px 12px !important;
}
.stDataFrame tbody tr td {
    color: #f1f5f9 !important;
    font-size: 13px !important;
    border-bottom: 1px solid #1e2a3a !important;
    padding: 8px 12px !important;
}
.stDataFrame tbody tr:hover td { background: #1a2235 !important; }
.stSuccess { background: #052e16 !important; border: 1px solid #10b981 !important; color: #10b981 !important; border-radius: 6px !important; }
.stWarning { background: #1c1007 !important; border: 1px solid #f59e0b !important; color: #f59e0b !important; border-radius: 6px !important; }
.stInfo    { background: #0c1a2e !important; border: 1px solid #3b82f6 !important; color: #94a3b8 !important; border-radius: 6px !important; }
.stError   { background: #1a0505 !important; border: 1px solid #ef4444 !important; border-radius: 6px !important; }
[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border: 1px solid #2d3748 !important;
    color: #f1f5f9 !important;
    border-radius: 6px !important;
}
hr { border-color: #2d3748 !important; }
.block-container { padding-top: 1.5rem !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def dark_chart_layout(title: str = "", height: int = 380,
                      xtitle: str = "", ytitle: str = "") -> dict:
    # NOTE: no 'legend', no 'margin' — each chart sets those independently
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color="#f1f5f9"),
            x=0, xanchor="left", pad=dict(l=4),
        ),
        height=height,
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(
            title=dict(text=xtitle, font=dict(color="#64748b", size=11)),
            tickfont=dict(color="#64748b", size=10),
            gridcolor="#1e2a3a", gridwidth=1,
            linecolor="#2d3748",
            showgrid=True, zeroline=False, tickangle=0,
        ),
        yaxis=dict(
            title=dict(text=ytitle, font=dict(color="#64748b", size=11)),
            tickfont=dict(color="#64748b", size=10),
            gridcolor="#1e2a3a", gridwidth=1,
            linecolor="#2d3748",
            showgrid=True, zeroline=False,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1a2235", bordercolor="#3b82f6",
            font=dict(color="#f1f5f9", size=11),
        ),
    )


def metric_card(label: str, value: str, positive=None) -> str:
    if positive is True:
        val_color = "#10b981"
    elif positive is False:
        val_color = "#ef4444"
    else:
        val_color = "#f1f5f9"
    return (
        f"<div style='background:#111827; border:1px solid #2d3748; "
        f"border-radius:8px; padding:16px 20px;'>"
        f"<div style='font-size:10px; font-weight:600; color:#64748b; "
        f"text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px'>{label}</div>"
        f"<div style='font-size:26px; font-weight:700; color:{val_color}; "
        f"letter-spacing:-0.5px'>{value}</div></div>"
    )


def quintile_badge(q: str) -> str:
    _colors = {
        "Q1": ("#10b981", "#052e16"),
        "Q2": ("#3b82f6", "#0c1a2e"),
        "Q3": ("#f59e0b", "#1c1007"),
        "Q4": ("#f97316", "#1c0a05"),
        "Q5": ("#ef4444", "#1a0505"),
    }
    c, bg = _colors.get(q, ("#64748b", "#1a2235"))
    return (
        f"<span style='background:{bg}; color:{c}; border:1px solid {c}; "
        f"border-radius:4px; padding:2px 8px; font-size:10px; font-weight:600'>{q}</span>"
    )


def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "momentum": "Mom", "value": "Val", "quality": "Qual",
        "low_volatility": "Low Vol", "size": "Size", "composite": "Score",
    })


def _copen() -> None:
    st.markdown(
        "<div style='background:#111827; border:1px solid #2d3748; "
        "border-radius:8px; padding:16px; margin:8px 0'>",
        unsafe_allow_html=True,
    )


def _cclose() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style='padding:8px 0 16px'>
  <div style='font-size:18px; font-weight:700; color:#f1f5f9; letter-spacing:-0.3px'>
    Factor Model</div>
  <div style='font-size:11px; color:#64748b; margin-top:3px'>
    Multi-factor long-short equity</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "<div style='font-size:10px; font-weight:600; color:#64748b; "
    "text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px'>Universe</div>",
    unsafe_allow_html=True)
market = st.sidebar.radio("", options=["us", "india", "both"],
    format_func=lambda x: {"us": "🇺🇸  US S&P 500",
                            "india": "🇮🇳  India Nifty 200",
                            "both": "🌍  Both Markets"}[x])

st.sidebar.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div style='font-size:10px; font-weight:600; color:#64748b; "
    "text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px'>Rebalance</div>",
    unsafe_allow_html=True)
rebal = st.sidebar.radio("", options=["M", "Q"],
    format_func=lambda x: {"M": "Monthly", "Q": "Quarterly"}[x])

st.sidebar.markdown(
    "<hr style='border:none; border-top:1px solid #2d3748; margin:16px 0'>",
    unsafe_allow_html=True)
st.sidebar.markdown(
    "<div style='font-size:10px; font-weight:600; color:#64748b; "
    "text-transform:uppercase; letter-spacing:.1em; margin-bottom:12px'>Factor Weights</div>",
    unsafe_allow_html=True)

w_mom  = st.sidebar.slider("Momentum", 0, 100, 30, 5)
w_val  = st.sidebar.slider("Value",    0, 100, 20, 5)
w_qual = st.sidebar.slider("Quality",  0, 100, 25, 5)
w_lvol = st.sidebar.slider("Low Vol",  0, 100, 15, 5)
w_size = st.sidebar.slider("Size",     0, 100, 10, 5)

total_w = w_mom + w_val + w_qual + w_lvol + w_size
weights_valid = (total_w == 100)

if weights_valid:
    st.sidebar.markdown(
        "<div style='font-size:11px; color:#10b981; margin:8px 0'>&#10003; Weights sum to 100%</div>",
        unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        f"<div style='font-size:11px; color:#ef4444; margin:8px 0'>"
        f"&#9888; Sum = {total_w}% (need 100%)</div>",
        unsafe_allow_html=True)

st.sidebar.markdown(
    "<hr style='border:none; border-top:1px solid #2d3748; margin:12px 0'>",
    unsafe_allow_html=True)
tx_cost = st.sidebar.slider("Transaction cost (bps)", 0, 50, 10, 5)
st.sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
run_btn = st.sidebar.button("&#9654;  Run Backtest", disabled=not weights_valid)

weights_dict = {
    "momentum":       w_mom  / 100.0,
    "value":          w_val  / 100.0,
    "quality":        w_qual / 100.0,
    "low_volatility": w_lvol / 100.0,
    "size":           w_size / 100.0,
}

# Settings fingerprint — if anything changed since last run, invalidate cache
current_settings = f"{market}_{rebal}_{tx_cost}_{w_mom}_{w_val}_{w_qual}_{w_lvol}_{w_size}"
if st.session_state.get("last_settings") != current_settings:
    st.session_state["backtest_done"]    = False
    st.session_state["backtest_results"] = None
    st.session_state["last_settings"]    = current_settings

# ---------------------------------------------------------------------------
# Main header + status line
# ---------------------------------------------------------------------------
_meta = st.session_state.get("last_run_meta")
if _meta:
    _status = (
        f"<span style='color:#10b981; margin-right:6px'>&#9679; Backtest loaded</span>"
        f"<span style='color:#1e3a5f'>|</span> "
        f"<span style='color:#64748b; margin-left:6px'>"
        f"{_meta.get('market_label','')} &nbsp;&middot;&nbsp; "
        f"{_meta.get('n_stocks','')} stocks &nbsp;&middot;&nbsp; "
        f"{_meta.get('rebalance_label','')} rebalance &nbsp;&middot;&nbsp; "
        f"{_meta.get('tx_cost','')} bps &nbsp;&middot;&nbsp; "
        f"{_meta.get('start','')} &#8594; {_meta.get('end','')}</span>"
    )
else:
    _status = "<span style='color:#334155'>No backtest loaded &mdash; configure and run from sidebar</span>"

st.markdown(f"""
<div style='margin-bottom:24px'>
  <div style='font-size:24px; font-weight:700; color:#f1f5f9; letter-spacing:-0.5px'>
    Factor Model Tearsheet</div>
  <div style='font-size:13px; color:#64748b; margin-top:4px'>
    Multi-factor long-short equity strategy &nbsp;&middot;&nbsp;
    S&P 500 + Nifty 200 &nbsp;&middot;&nbsp; Powered by Yahoo Finance</div>
  <div style='font-size:11px; margin-top:10px; padding:6px 12px;
              background:#0d1321; border-radius:4px;
              border-left:2px solid #1e3a5f; display:inline-block'>
    {_status}
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Backtest execution
# ---------------------------------------------------------------------------
if run_btn and weights_valid:
    with st.spinner("Loading universe data…"):
        try:
            tickers, price_df, fund_df, mkt_lbls = build_universe(market=market)
            st.session_state["universe_tickers"]  = tickers
            st.session_state["universe_price_df"] = price_df
            st.session_state["universe_fund_df"]  = fund_df
            st.session_state["universe_labels"]   = mkt_lbls
        except Exception as exc:
            st.error(f"Universe load failed: {exc}")
            st.stop()

    with st.spinner("Running backtest… (this may take 1–2 minutes)"):
        try:
            results = run_backtest(
                market=market,
                rebalance_freq=rebal,
                transaction_cost_bps=float(tx_cost),
                weights=weights_dict,
            )
            st.session_state["backtest_results"] = results
            st.session_state["backtest_done"]    = True
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

    with st.spinner("Computing statistics…"):
        try:
            ls_ret = results["long_short_returns"]
            lo_ret = results["long_only_returns"]
            bm_ret = results["benchmark_returns"]

            st.session_state["stats"] = {
                "long_short": compute_stats(ls_ret),
                "long_only":  compute_stats(lo_ret),
                "benchmark":  compute_stats(bm_ret),
            }
            st.session_state["monthly_ret"] = compute_monthly_returns(ls_ret)
            st.session_state["drawdown"]    = compute_drawdown_series(ls_ret)
            st.session_state["attribution"] = compute_factor_attribution(results)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cur_scores = compute_composite_score(
                    st.session_state["universe_price_df"],
                    st.session_state["universe_fund_df"],
                    weights=weights_dict,
                )
            st.session_state["factor_scores"] = cur_scores

            st.session_state["run_settings"] = {
                "market": market,
                "rebal":  rebal,
                "tx_cost": tx_cost,
                "w_mom": w_mom, "w_val": w_val, "w_qual": w_qual,
                "w_lvol": w_lvol, "w_size": w_size,
            }

            meta = results.get("metadata", {})
            st.session_state["last_run_meta"] = {
                "market_label":    {"us": "S&P 500", "india": "Nifty 200",
                                    "both": "US + India"}.get(market, market),
                "n_stocks":        meta.get("n_stocks", len(tickers)),
                "rebalance_label": {"M": "Monthly", "Q": "Quarterly"}.get(rebal, rebal),
                "tx_cost":         tx_cost,
                "start":           meta.get("start_date", ""),
                "end":             meta.get("end_date", ""),
            }
        except Exception as exc:
            st.error(f"Statistics computation failed: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_summary, tab_monthly, tab_factors, tab_rankings, tab_export = st.tabs([
    "📈 Summary",
    "📅 Monthly Returns",
    "🔬 Factor Analysis",
    "🏆 Current Rankings",
    "📄 Export PDF",
])

# ===========================================================================
# TAB 1 — SUMMARY
# ===========================================================================
with tab_summary:
    if not st.session_state["backtest_done"]:
        st.markdown("""
        <div style='background:#0c1a2e; border:1px solid #1e3a5f; border-radius:8px;
                    padding:40px; text-align:center; margin-top:40px'>
          <div style='font-size:36px; margin-bottom:16px'>📊</div>
          <div style='font-size:16px; font-weight:600; color:#f1f5f9; margin-bottom:8px'>
            No backtest loaded</div>
          <div style='font-size:13px; color:#64748b'>
            Configure universe, factor weights, and rebalance frequency in the sidebar,
            then click <b style='color:#3b82f6'>Run Backtest</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        results   = st.session_state["backtest_results"]
        all_stats = st.session_state["stats"]
        ls_stats  = all_stats["long_short"]

        ar = ls_stats.get("annual_return") or 0.0
        av = ls_stats.get("annual_vol")    or 0.0
        sr = ls_stats.get("sharpe")        or 0.0
        md = ls_stats.get("max_drawdown")  or 0.0
        wr = ls_stats.get("win_rate")      or 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(metric_card("Annual Return (L/S)", f"{ar*100:.1f}%", ar > 0),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Annual Volatility", f"{av*100:.1f}%", None),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Sharpe Ratio", f"{sr:.2f}", sr > 0.5),
                        unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Max Drawdown", f"{md*100:.1f}%", False),
                        unsafe_allow_html=True)
        with c5:
            st.markdown(metric_card("Win Rate", f"{wr*100:.1f}%", wr > 0.5),
                        unsafe_allow_html=True)

        _rs = st.session_state.get("run_settings") or {}
        _mkt_lbl = {"us": "S&P 500", "india": "Nifty 200", "both": "US+India"}.get(
            _rs.get("market", ""), _rs.get("market", ""))
        _reb_lbl = {"M": "Monthly", "Q": "Quarterly"}.get(_rs.get("rebal", ""), _rs.get("rebal", ""))
        st.markdown(
            f"<div style='font-size:11px; color:#64748b; margin:12px 0 8px'>"
            f"&#10003; Backtest complete &nbsp;&middot;&nbsp; "
            f"{_mkt_lbl} &nbsp;&middot;&nbsp; "
            f"{_reb_lbl} rebalance &nbsp;&middot;&nbsp; "
            f"{_rs.get('tx_cost', '')} bps &nbsp;&middot;&nbsp; "
            f"Mom {_rs.get('w_mom','')}% / Val {_rs.get('w_val','')}% / "
            f"Qual {_rs.get('w_qual','')}% / LVol {_rs.get('w_lvol','')}% / "
            f"Size {_rs.get('w_size','')}%"
            f"</div>",
            unsafe_allow_html=True,
        )

        # --- Cumulative returns chart ---
        ls_ret = results["long_short_returns"]
        lo_ret = results["long_only_returns"]
        bm_ret = results["benchmark_returns"]
        cum_ls = (1.0 + ls_ret).cumprod()
        cum_lo = (1.0 + lo_ret).cumprod()
        cum_bm = (1.0 + bm_ret).cumprod()

        # Time filter (replaces Plotly rangeselector)
        time_filter = st.radio(
            "", options=["6M", "1Y", "All"], index=2,
            horizontal=True, key="time_filter",
        )
        end_date = cum_ls.index[-1]
        if time_filter == "6M":
            start = end_date - relativedelta(months=6)
        elif time_filter == "1Y":
            start = end_date - relativedelta(years=1)
        else:
            start = cum_ls.index[0]
        cum_ls_f = cum_ls[cum_ls.index >= start]
        cum_lo_f = cum_lo[cum_lo.index >= start]
        cum_bm_f = cum_bm[cum_bm.index >= start]

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum_ls_f.index, y=cum_ls_f.values, name="Long-Short",
            line=dict(color="#3b82f6", width=2),
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum_lo_f.index, y=cum_lo_f.values, name="Long Only",
            line=dict(color="#10b981", width=2, dash="dash"),
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum_bm_f.index, y=cum_bm_f.values, name="Benchmark (EW)",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
        ))
        fig_cum.update_layout(
            **dark_chart_layout(
                height=380,
                xtitle="Date",
                ytitle="Indexed Return (base = 100)",
            ),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=11),
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=60, r=30, t=20, b=80),
        )
        fig_cum.update_xaxes(
            tickformat="%b %Y", tickangle=0, nticks=8,
            rangeslider_visible=False,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # --- Strategy comparison HTML table ---
        s_ls = all_stats["long_short"]
        s_lo = all_stats["long_only"]
        s_bm = all_stats["benchmark"]

        def _p(v, pos_green=True):
            if v is None:
                return "<span style='color:#334155'>—</span>"
            clr = ("#10b981" if v > 0 else "#ef4444") if pos_green else "#94a3b8"
            return f"<span style='color:{clr}'>{v*100:.1f}%</span>"

        def _f(v, threshold=0.5):
            if v is None:
                return "<span style='color:#334155'>—</span>"
            clr = "#10b981" if v > threshold else "#94a3b8"
            return f"<span style='color:{clr}'>{v:.2f}</span>"

        def _row(name, s, nc):
            return (
                f"<tr style='border-bottom:1px solid #1e2a3a'>"
                f"<td style='padding:10px 12px; color:{nc}; font-weight:600'>{name}</td>"
                f"<td style='padding:10px 12px'>{_p(s.get('annual_return'))}</td>"
                f"<td style='padding:10px 12px'>{_p(s.get('annual_vol'), False)}</td>"
                f"<td style='padding:10px 12px'>{_f(s.get('sharpe'))}</td>"
                f"<td style='padding:10px 12px; color:#ef4444'>{_p(s.get('max_drawdown'), False)}</td>"
                f"<td style='padding:10px 12px'>{_p(s.get('win_rate'))}</td>"
                f"<td style='padding:10px 12px'>{_p(s.get('best_month'))}</td>"
                f"<td style='padding:10px 12px; color:#ef4444'>{_p(s.get('worst_month'), False)}</td>"
                f"</tr>"
            )

        _th = ("style='text-align:left; padding:10px 12px; color:#64748b; font-weight:600; "
               "text-transform:uppercase; letter-spacing:.08em; "
               "border-bottom:1px solid #2d3748; font-size:11px; white-space:nowrap'")
        st.markdown(f"""
        <div style='margin-top:24px'>
          <div style='font-size:13px; font-weight:600; color:#94a3b8;
                      text-transform:uppercase; letter-spacing:.08em; margin-bottom:12px'>
            Strategy Comparison</div>
          <div style='background:#111827; border:1px solid #2d3748;
                      border-radius:8px; overflow:hidden'>
          <table style='width:100%; border-collapse:collapse; font-size:12px'>
            <thead>
              <tr>
                <th {_th}>Strategy</th>
                <th {_th}>Ann. Return</th>
                <th {_th}>Ann. Vol</th>
                <th {_th}>Sharpe</th>
                <th {_th}>Max DD</th>
                <th {_th}>Win Rate</th>
                <th {_th}>Best Mo.</th>
                <th {_th}>Worst Mo.</th>
              </tr>
            </thead>
            <tbody>
              {_row("Long-Short", s_ls, "#3b82f6")}
              {_row("Long Only",  s_lo, "#10b981")}
              {_row("Benchmark",  s_bm, "#f59e0b")}
            </tbody>
          </table>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ===========================================================================
# TAB 2 — MONTHLY RETURNS
# ===========================================================================
with tab_monthly:
    if not st.session_state["backtest_done"]:
        st.info("Run the backtest first.")
    else:
        monthly_df = st.session_state["monthly_ret"]
        drawdown   = st.session_state["drawdown"]

        st.markdown(
            "<div style='font-size:13px; font-weight:600; color:#94a3b8; "
            "text-transform:uppercase; letter-spacing:.08em; margin-bottom:12px'>"
            "Monthly Returns Heatmap (Long-Short)</div>",
            unsafe_allow_html=True)

        z_vals   = monthly_df.values * 100
        z_text   = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z_vals]
        y_labels = [str(yr) for yr in monthly_df.index]
        x_labels = list(monthly_df.columns)

        fig_heat = go.Figure(go.Heatmap(
            z=z_vals, x=x_labels, y=y_labels,
            text=z_text, texttemplate="%{text}",
            textfont=dict(size=11, color="white", family="sans-serif"),
            colorscale=[[0, "#7f1d1d"], [0.5, "#1e2a3a"], [1, "#064e3b"]],
            zmid=0,
            colorbar=dict(
                title=dict(text="Ret %", font=dict(color="#94a3b8", size=11)),
                tickfont=dict(color="#64748b", size=10),
                thickness=12, len=0.8,
            ),
        ))
        fig_heat.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            height=max(280, len(y_labels) * 80),
            xaxis=dict(side="top", tickfont=dict(color="#64748b", size=11),
                       showgrid=False, zeroline=False, linecolor="#2d3748"),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#94a3b8", size=11),
                       showgrid=False, zeroline=False, linecolor="#2d3748"),
            margin=dict(l=60, r=80, t=50, b=60),
            font=dict(color="#94a3b8", family="sans-serif"),
        )
        fig_heat.update_xaxes(showgrid=False, zeroline=False, linecolor="#2d3748")
        fig_heat.update_yaxes(showgrid=False, zeroline=False, linecolor="#2d3748")

        _copen()
        st.plotly_chart(fig_heat, use_container_width=True)
        _cclose()

        st.markdown(
            "<div style='font-size:13px; font-weight:600; color:#94a3b8; "
            "text-transform:uppercase; letter-spacing:.08em; margin:20px 0 12px'>"
            "Drawdown (Long-Short)</div>",
            unsafe_allow_html=True)

        dd = drawdown * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name="Drawdown",
            fill="tozeroy", line=dict(color="#ef4444", width=1.5),
            fillcolor="rgba(239,68,68,0.15)",
        ))
        fig_dd.add_annotation(
            x=dd.idxmin(), y=dd.min(),
            text=f"Max DD: {dd.min():.1f}%",
            font=dict(color="#ef4444", size=11),
            bgcolor="#1a0505", bordercolor="#ef4444",
            borderwidth=1, borderpad=4,
            arrowcolor="#ef4444", arrowwidth=1, ay=-40,
        )
        _dd_lay = dark_chart_layout(height=300, ytitle="Drawdown (%)")
        _dd_lay["yaxis"]["ticksuffix"] = "%"
        _dd_lay["yaxis"]["range"]      = [dd.min() * 1.1, 1]
        _dd_lay["xaxis"]["tickformat"] = "%b %Y"
        _dd_lay["xaxis"]["dtick"]      = "M3"
        _dd_lay["margin"]              = dict(l=60, r=30, t=40, b=60)
        fig_dd.update_layout(**_dd_lay)

        _copen()
        st.plotly_chart(fig_dd, use_container_width=True)
        _cclose()

# ===========================================================================
# TAB 3 — FACTOR ANALYSIS
# ===========================================================================
with tab_factors:
    if not st.session_state["backtest_done"]:
        st.info("Run the backtest first.")
    else:
        attribution = st.session_state["attribution"]
        if attribution is None or attribution.empty:
            st.warning("Factor attribution data not available.")
        else:
            factors     = attribution.index.tolist()
            q1_vals     = attribution["Q1_avg"].values * 100 if "Q1_avg" in attribution.columns else np.zeros(len(factors))
            q5_vals     = attribution["Q5_avg"].values * 100 if "Q5_avg" in attribution.columns else np.zeros(len(factors))
            spread_vals = attribution["spread"].values  * 100 if "spread"  in attribution.columns else np.zeros(len(factors))

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown(
                    "<div style='font-size:13px; font-weight:600; color:#94a3b8; "
                    "text-transform:uppercase; letter-spacing:.08em; margin-bottom:12px'>"
                    "Q1 vs Q5 Avg Score by Factor</div>",
                    unsafe_allow_html=True)

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name="Q1 (Long)", y=factors, x=q1_vals, orientation="h",
                    marker_color="#3b82f6",
                    text=[f"{v:.2f}" for v in q1_vals],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=10),
                ))
                fig_bar.add_trace(go.Bar(
                    name="Q5 (Short)", y=factors, x=q5_vals, orientation="h",
                    marker_color="#ef4444",
                    text=[f"{v:.2f}" for v in q5_vals],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=10),
                ))
                _bl = dark_chart_layout(height=350, xtitle="Avg Z-Score")
                _bl["barmode"] = "group"
                _bl["margin"] = dict(l=100, r=80, t=50, b=60)
                fig_bar.update_layout(
                    **_bl,
                    legend=dict(
                        bgcolor="rgba(0,0,0,0)", bordercolor="#2d3748", borderwidth=1,
                        font=dict(color="#94a3b8", size=11),
                        orientation="h", y=1.12, x=0,
                    ),
                )
                fig_bar.add_vline(x=0, line_color="#2d3748", line_width=1)

                _copen()
                st.plotly_chart(fig_bar, use_container_width=True)
                _cclose()

            with col_right:
                st.markdown(
                    "<div style='font-size:13px; font-weight:600; color:#94a3b8; "
                    "text-transform:uppercase; letter-spacing:.08em; margin-bottom:12px'>"
                    "Spread (Q1 &#8722; Q5) by Factor</div>",
                    unsafe_allow_html=True)

                sp_colors = ["#10b981" if v >= 0 else "#ef4444" for v in spread_vals]
                fig_sp = go.Figure(go.Bar(
                    y=factors, x=spread_vals, orientation="h",
                    marker_color=sp_colors,
                    text=[f"{v:+.2f}" for v in spread_vals],
                    textposition="outside",
                    textfont=dict(color="#f1f5f9", size=11, family="sans-serif"),
                ))
                _sl = dark_chart_layout(height=350, xtitle="Spread (Z-Score)")
                _sl["margin"] = dict(l=100, r=80, t=50, b=60)
                fig_sp.update_layout(**_sl)
                fig_sp.add_vline(x=0, line_color="#2d3748", line_width=1)

                _copen()
                st.plotly_chart(fig_sp, use_container_width=True)
                _cclose()

            # --- Summary box ---
            _pairs_html = "".join(
                f"<div style='white-space:nowrap'>"
                f"<span style='color:#94a3b8'>{f}:</span> "
                f"<span style='color:{'#10b981' if v >= 0 else '#ef4444'}; font-weight:600'>"
                f"{v:+.2f}</span></div>"
                for f, v in zip(factors, spread_vals)
            )
            _bi = int(np.argmax(spread_vals))
            _wi = int(np.argmin(spread_vals))
            st.markdown(f"""
            <div style='background:#0c1a2e; border:1px solid #1e3a5f; border-radius:8px;
                        padding:14px 18px; margin-top:16px; font-size:12px; color:#94a3b8'>
              <span style='color:#3b82f6; font-weight:600'>Factor Spread Summary</span>
              <span style='color:#334155'> &mdash; positive = factor successfully separated buckets</span>
              <div style='margin-top:10px; display:flex; gap:24px; flex-wrap:wrap'>
                {_pairs_html}
              </div>
              <div style='margin-top:10px; font-size:11px; color:#64748b'>
                Best: <span style='color:#10b981; font-weight:600'>{factors[_bi]}</span>
                ({spread_vals[_bi]:+.2f}) &nbsp;&nbsp;
                Weakest: <span style='color:#ef4444; font-weight:600'>{factors[_wi]}</span>
                ({spread_vals[_wi]:+.2f})
              </div>
            </div>
            """, unsafe_allow_html=True)

# ===========================================================================
# TAB 4 — CURRENT RANKINGS
# ===========================================================================
with tab_rankings:
    if not st.session_state["backtest_done"]:
        st.info("Run the backtest first.")
    else:
        factor_scores = st.session_state["factor_scores"]
        if factor_scores is None or factor_scores.empty:
            st.warning("Factor scores not available.")
        else:
            quintiles = get_quintile_portfolios(factor_scores)

            col_q1, col_q5 = st.columns(2)

            with col_q1:
                st.markdown("""
                <div style='display:flex; align-items:center; gap:8px; margin-bottom:12px'>
                  <div style='width:8px; height:8px; border-radius:50%;
                              background:#10b981; flex-shrink:0'></div>
                  <div style='font-size:14px; font-weight:600; color:#f1f5f9'>
                    Q1 &mdash; Long candidates</div>
                </div>
                """, unsafe_allow_html=True)
                q1_t = quintiles.get("Q1", [])
                if q1_t:
                    q1_df = (factor_scores[factor_scores.index.isin(q1_t)]
                             .sort_values("composite", ascending=False)
                             .round(2).replace("None", "—").replace({None: "—"}))
                    q1_df = _rename_cols(q1_df)
                    _sc = ["Score"] if "Score" in q1_df.columns else []
                    st.dataframe(
                        q1_df.style.background_gradient(subset=_sc, cmap="Greens", axis=0),
                        use_container_width=True, height=400)
                else:
                    st.markdown("<div style='color:#64748b'>No Q1 tickers.</div>",
                                unsafe_allow_html=True)

            with col_q5:
                st.markdown("""
                <div style='display:flex; align-items:center; gap:8px; margin-bottom:12px'>
                  <div style='width:8px; height:8px; border-radius:50%;
                              background:#ef4444; flex-shrink:0'></div>
                  <div style='font-size:14px; font-weight:600; color:#f1f5f9'>
                    Q5 &mdash; Short candidates</div>
                </div>
                """, unsafe_allow_html=True)
                q5_t = quintiles.get("Q5", [])
                if q5_t:
                    q5_df = (factor_scores[factor_scores.index.isin(q5_t)]
                             .sort_values("composite", ascending=True)
                             .round(2).replace("None", "—").replace({None: "—"}))
                    q5_df = _rename_cols(q5_df)
                    _sc = ["Score"] if "Score" in q5_df.columns else []
                    st.dataframe(
                        q5_df.style.background_gradient(subset=_sc, cmap="Reds_r", axis=0),
                        use_container_width=True, height=400)
                else:
                    st.markdown("<div style='color:#64748b'>No Q5 tickers.</div>",
                                unsafe_allow_html=True)

            # --- Full universe HTML table ---
            st.markdown("""
            <hr style='border:none; border-top:1px solid #2d3748; margin:24px 0 16px'>
            <div style='font-size:13px; font-weight:600; color:#94a3b8;
                        text-transform:uppercase; letter-spacing:.08em; margin-bottom:12px'>
              Full Universe Rankings</div>
            """, unsafe_allow_html=True)

            mkt_lbl_dict = st.session_state.get("universe_labels") or {}
            _mkt_opts = ["All"]
            if any(v == "US"    for v in mkt_lbl_dict.values()): _mkt_opts.append("US")
            if any(v == "INDIA" for v in mkt_lbl_dict.values()): _mkt_opts.append("India")
            mkt_filter = st.selectbox("Filter by Market", options=_mkt_opts, index=0)

            full_df = factor_scores.copy()
            full_df["market"] = full_df.index.map(lambda t: mkt_lbl_dict.get(t, "US"))
            if mkt_filter == "US":
                full_df = full_df[full_df["market"] == "US"]
            elif mkt_filter == "India":
                full_df = full_df[full_df["market"] == "INDIA"]
            full_df = full_df.sort_values("composite", ascending=False)

            q_map: dict[str, str] = {}
            for qi in range(1, 6):
                for tk in quintiles.get(f"Q{qi}", []):
                    q_map[tk] = f"Q{qi}"

            _col_order  = ["composite", "momentum", "value", "quality", "low_volatility", "size", "market"]
            _disp_cols  = [c for c in _col_order if c in full_df.columns]
            _disp       = full_df[_disp_cols].round(2)
            _col_labels = {
                "composite": "Score", "momentum": "Mom", "value": "Val",
                "quality": "Qual", "low_volatility": "Low Vol",
                "size": "Size", "market": "Market",
            }

            _TH = ("padding:10px 12px; color:#64748b; font-weight:600; font-size:11px; "
                   "text-transform:uppercase; letter-spacing:.05em; "
                   "border-bottom:1px solid #2d3748; text-align:left; white-space:nowrap")
            _TD = "padding:8px 12px; font-size:12px; border-bottom:1px solid #1e2a3a; color:#f1f5f9"
            _TDD = "padding:8px 12px; font-size:12px; border-bottom:1px solid #1e2a3a; color:#64748b"

            _hdr = f"<th style='{_TH}'>Rank</th><th style='{_TH}'>Ticker</th><th style='{_TH}'>Quintile</th>"
            for c in _disp_cols:
                _hdr += f"<th style='{_TH}'>{_col_labels.get(c, c)}</th>"

            _rows = ""
            for rank, (ticker, row) in enumerate(_disp.iterrows(), 1):
                q_str  = q_map.get(ticker, "—")
                badge  = quintile_badge(q_str) if q_str != "—" else "<span style='color:#334155'>—</span>"
                score  = row.get("composite", 0.0)
                sc_clr = "#10b981" if score > 0 else "#ef4444"
                cells  = f"<td style='{_TDD}'>{rank}</td>"
                cells += f"<td style='{_TD}; font-weight:600'>{ticker}</td>"
                cells += f"<td style='{_TD}'>{badge}</td>"
                for c in _disp_cols:
                    v = row.get(c, "—")
                    if c == "composite":
                        cells += f"<td style='{_TD}; color:{sc_clr}; font-weight:600'>{v}</td>"
                    elif c == "market":
                        cells += f"<td style='{_TDD}'>{v}</td>"
                    else:
                        cells += f"<td style='{_TD}'>{v}</td>"
                _rows += f"<tr>{cells}</tr>"

            st.markdown(f"""
            <div style='background:#111827; border:1px solid #2d3748; border-radius:8px;
                        overflow:auto; max-height:520px; margin-top:8px'>
              <table style='width:100%; border-collapse:collapse; font-family:sans-serif'>
                <thead style='position:sticky; top:0; z-index:1; background:#1a2235'>
                  <tr>{_hdr}</tr>
                </thead>
                <tbody>{_rows}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

# ===========================================================================
# TAB 5 — EXPORT PDF
# ===========================================================================
with tab_export:
    st.markdown(
        "<div style='font-size:13px; font-weight:600; color:#94a3b8; "
        "text-transform:uppercase; letter-spacing:.08em; margin-bottom:20px'>"
        "Export Professional Tearsheet PDF</div>",
        unsafe_allow_html=True)

    st.markdown("""
    <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:20px 0'>
      <div style='background:#111827; border:1px solid #2d3748; border-radius:8px; padding:20px'>
        <div style='font-size:10px; font-weight:600; color:#3b82f6; text-transform:uppercase;
                    letter-spacing:.1em; margin-bottom:8px'>Page 1</div>
        <div style='font-size:13px; font-weight:600; color:#f1f5f9; margin-bottom:6px'>
          Cover + Stats</div>
        <div style='font-size:12px; color:#64748b; line-height:1.6'>
          5 metric cards &middot; Cumulative returns chart</div>
      </div>
      <div style='background:#111827; border:1px solid #2d3748; border-radius:8px; padding:20px'>
        <div style='font-size:10px; font-weight:600; color:#10b981; text-transform:uppercase;
                    letter-spacing:.1em; margin-bottom:8px'>Page 2</div>
        <div style='font-size:13px; font-weight:600; color:#f1f5f9; margin-bottom:6px'>
          Monthly Returns</div>
        <div style='font-size:12px; color:#64748b; line-height:1.6'>
          Calendar heatmap &middot; Drawdown chart</div>
      </div>
      <div style='background:#111827; border:1px solid #2d3748; border-radius:8px; padding:20px'>
        <div style='font-size:10px; font-weight:600; color:#f59e0b; text-transform:uppercase;
                    letter-spacing:.1em; margin-bottom:8px'>Page 3</div>
        <div style='font-size:13px; font-weight:600; color:#f1f5f9; margin-bottom:6px'>
          Factor Analysis</div>
        <div style='font-size:12px; color:#64748b; line-height:1.6'>
          Q1 vs Q5 scores &middot; Factor spread analysis</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["backtest_done"]:
        st.markdown("""
        <div style='background:#1c1007; border:1px solid #f59e0b; border-radius:6px;
                    padding:12px 16px; font-size:13px; color:#f59e0b; margin-top:8px'>
          &#9888; Please run the backtest before generating the PDF.
        </div>
        """, unsafe_allow_html=True)
    else:
        generate_btn = st.button("📄 Generate PDF Tearsheet", type="primary")
        if generate_btn:
            with st.spinner("Rendering tearsheet PDF… (30–60 seconds)"):
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False, prefix="tearsheet_"
                    ) as tmp:
                        tmp_path = tmp.name

                    generate_tearsheet(
                        backtest_results=st.session_state["backtest_results"],
                        factor_attribution=st.session_state["attribution"],
                        output_path=tmp_path,
                    )

                    with open(tmp_path, "rb") as f:
                        pdf_bytes = f.read()
                    os.unlink(tmp_path)

                    st.markdown("""
                    <div style='background:#052e16; border:1px solid #10b981; border-radius:6px;
                                padding:12px 16px; font-size:13px; color:#10b981; margin-bottom:12px'>
                      &#10003; PDF generated &mdash; click below to download.
                    </div>
                    """, unsafe_allow_html=True)
                    st.download_button(
                        label="⬇️  Download Tearsheet PDF",
                        data=pdf_bytes,
                        file_name="factor_model_tearsheet.pdf",
                        mime="application/pdf",
                        type="primary",
                    )
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style='margin-top:48px; padding-top:16px; border-top:1px solid #1e2a3a;
            text-align:center; font-size:11px; color:#334155'>
  Factor Model Tearsheet Generator &nbsp;&middot;&nbsp;
  Multi-Factor Long-Short Equity &nbsp;&middot;&nbsp;
  Data: Yahoo Finance &nbsp;&middot;&nbsp;
  Built with Streamlit
</div>
""", unsafe_allow_html=True)
