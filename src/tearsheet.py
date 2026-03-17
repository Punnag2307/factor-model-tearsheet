"""PDF tearsheet generator — cumulative returns, heatmap, drawdown, factor exposures, stats."""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

from src.stats import compute_stats, compute_monthly_returns, compute_drawdown_series

# ---------------------------------------------------------------------------
# Theme — clean institutional light palette
# ---------------------------------------------------------------------------

BACKGROUND     = "#FFFFFF"
CARD_BG        = "#F7F9FC"
BORDER         = "#E1E8ED"
ACCENT         = "#1F3864"   # dark navy — primary
ACCENT2        = "#2E75B6"   # medium blue — secondary
GREEN          = "#1A7340"
RED            = "#C0392B"
GOLD           = "#B7860B"
TEXT           = "#1A1A1A"
TEXT_SECONDARY = "#555555"
GRID_COLOR     = "#E8ECF0"

FIGSIZE        = (16, 11)    # landscape, consistent across all pages
FIG_DPI        = 150

_FOOTER_LEFT  = "Multi-Factor Long-Short Equity Strategy  |  Data: Yahoo Finance"
_FOOTER_RIGHT = datetime.date.today().strftime("%B %d, %Y")

# ---------------------------------------------------------------------------
# Global style helpers
# ---------------------------------------------------------------------------

def _apply_chart_style(ax: plt.Axes, grid: bool = True) -> None:
    """Institutional light theme: CARD_BG face, BORDER spines, restrained grid."""
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.6)
    # labelsize=8, pad=4 on all tick axes
    ax.tick_params(axis="both", which="both",
                   colors=TEXT, labelsize=8, length=3, pad=4)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)


def _remove_top_right_spines(ax: plt.Axes) -> None:
    """Hide top and right frame lines for a cleaner chart border."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_footer(fig: plt.Figure) -> None:
    """
    Two-column footer at the bottom of every page:
      left  — strategy / data attribution (fontsize=7)
      right — generation date             (fontsize=7)
    Preceded by a thin BORDER rule at y=0.04.
    """
    fig.add_artist(Line2D(
        [0.04, 0.96], [0.04, 0.04],
        transform=fig.transFigure,
        color=BORDER, linewidth=0.7,
        solid_capstyle="butt",
    ))
    fig.text(0.04, 0.018, _FOOTER_LEFT,
             ha="left", va="bottom", fontsize=7, color=TEXT_SECONDARY)
    fig.text(0.96, 0.018, _FOOTER_RIGHT,
             ha="right", va="bottom", fontsize=7, color=TEXT_SECONDARY)


def _top_rule(fig: plt.Figure, lw: float = 6.0) -> None:
    """Thick navy accent rule at the very top of a page — GS/BlackRock style."""
    fig.add_artist(Line2D(
        [0.0, 1.0], [0.977, 0.977],
        transform=fig.transFigure,
        color=ACCENT, linewidth=lw,
        solid_capstyle="butt",
    ))


def _stat_card(
    ax: plt.Axes,
    value_str: str,
    label: str,
    value_color: str,
) -> None:
    """
    Clean rectangular KPI card:
      • white BACKGROUND, 1 px BORDER border
      • thin coloured accent stripe at top
      • large metric value (fontsize=22, bold) with transparent bbox (pad=12 pt)
      • small TEXT_SECONDARY label below (fontsize=9)
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BACKGROUND)
    ax.axis("off")

    # Outer card rectangle — sharp corners, 1 px BORDER
    ax.add_patch(Rectangle(
        (0.04, 0.04), 0.92, 0.92,
        facecolor=BACKGROUND, edgecolor=BORDER, linewidth=1.0,
        transform=ax.transAxes, clip_on=False,
    ))

    # Thin coloured accent stripe at top of card
    ax.add_patch(Rectangle(
        (0.04, 0.88), 0.92, 0.08,
        facecolor=value_color, edgecolor="none",
        transform=ax.transAxes, clip_on=False,
    ))

    # Metric value — transparent bbox gives 12 pt breathing room
    ax.text(
        0.5, 0.58, value_str,
        ha="center", va="center",
        fontsize=22, fontweight="bold", color=value_color,
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="none", edgecolor="none",
        ),
    )

    # Descriptive label
    ax.text(
        0.5, 0.22, label,
        ha="center", va="center",
        fontsize=9, color=TEXT_SECONDARY,
        transform=ax.transAxes,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(val: float | None, decimals: int = 1) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def _fmt_f(val: float | None, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def _cum100(series: pd.Series, common_idx: pd.Index) -> pd.Series:
    s = series.reindex(common_idx).fillna(0.0)
    return (1.0 + s).cumprod() * 100.0


# ---------------------------------------------------------------------------
# Page 1 — Cover + KPI cards + Cumulative returns
# ---------------------------------------------------------------------------

def _page1_cover(pdf: PdfPages, results: dict) -> None:
    meta   = results["metadata"]
    ls_ret = results["long_short_returns"]
    lo_ret = results["long_only_returns"]
    bm_ret = results["benchmark_returns"]
    stats  = compute_stats(ls_ret)

    fig = plt.figure(figsize=FIGSIZE, facecolor=BACKGROUND, dpi=FIG_DPI)
    _top_rule(fig, lw=7)

    # Three-row gridspec: [header | cards | cum-chart]
    gs = fig.add_gridspec(
        3, 5,
        height_ratios=[2.8, 2.2, 4.0],
        hspace=0.05, wspace=0.24,
        left=0.05, right=0.97, top=0.94, bottom=0.10,
    )

    # ---------------------------------------------------------------- header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(BACKGROUND)
    ax_hdr.axis("off")

    ax_hdr.text(
        0.5, 0.84, "Factor Model Tearsheet",
        ha="center", va="center",
        fontsize=28, fontweight="bold", color=ACCENT,
        transform=ax_hdr.transAxes,
    )
    ax_hdr.text(
        0.5, 0.60, "Multi-Factor Long-Short Equity Strategy",
        ha="center", va="center",
        fontsize=13, color=TEXT_SECONDARY, style="italic",
        transform=ax_hdr.transAxes,
    )
    # Thin BORDER rule separating title from metadata
    ax_hdr.axhline(y=0.42, xmin=0.03, xmax=0.97, color=BORDER, linewidth=0.8)

    # Metadata strip — bold value above, small key label below
    meta_items = [
        ("Market",    meta.get("market", "-").upper()),
        ("Rebalance", meta.get("rebalance_freq", "-")),
        ("Universe",  f"{meta.get('n_stocks', '-')} stocks"),
        ("Period",    f"{meta.get('start_date', '-')}  to  {meta.get('end_date', '-')}"),
        ("Tx Cost",   f"{meta.get('transaction_cost_bps', '-')} bps"),
    ]
    n, step = len(meta_items), 0.90 / len(meta_items)
    for i, (key, val) in enumerate(meta_items):
        x = 0.05 + i * step + step / 2
        ax_hdr.text(x, 0.22, val,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=TEXT,
                    transform=ax_hdr.transAxes)
        ax_hdr.text(x, 0.05, key,
                    ha="center", va="center",
                    fontsize=8, color=TEXT_SECONDARY,
                    transform=ax_hdr.transAxes)

    # ----------------------------------------------------------- KPI cards
    annual_ret = stats.get("annual_return")
    sharpe     = stats.get("sharpe")
    max_dd     = stats.get("max_drawdown")
    annual_vol = stats.get("annual_vol")
    win_rate   = stats.get("win_rate")

    cards = [
        (_fmt_pct(annual_ret), "Annual Return",
         GREEN if (annual_ret or 0) > 0 else RED),
        (_fmt_f(sharpe),       "Sharpe Ratio",
         GREEN if (sharpe or 0) >= 0.5 else GOLD),
        (_fmt_pct(max_dd),     "Max Drawdown",  RED),
        (_fmt_pct(annual_vol), "Annual Volatility", ACCENT),
        (_fmt_pct(win_rate),   "Win Rate",
         GREEN if (win_rate or 0) >= 0.5 else GOLD),
    ]
    for col, (val_str, label, color) in enumerate(cards):
        _stat_card(fig.add_subplot(gs[1, col]), val_str, label, color)

    # ------------------------------------------------ cumulative returns
    ax_cum = fig.add_subplot(gs[2, :])
    _apply_chart_style(ax_cum)
    _remove_top_right_spines(ax_cum)

    idx_common = (ls_ret.index
                  .intersection(lo_ret.index)
                  .intersection(bm_ret.index))
    cum_ls = _cum100(ls_ret, idx_common)
    cum_lo = _cum100(lo_ret, idx_common)
    cum_bm = _cum100(bm_ret, idx_common)

    ax_cum.plot(cum_ls.index, cum_ls.values,
                color=ACCENT2, linewidth=2.0, label="Long-Short", zorder=3)
    ax_cum.plot(cum_lo.index, cum_lo.values,
                color=GREEN, linewidth=1.7, label="Long-Only", zorder=2)
    ax_cum.plot(cum_bm.index, cum_bm.values,
                color=GOLD, linewidth=1.4, linestyle="--",
                label="Benchmark (EW)", zorder=1)

    ax_cum.axhline(y=100, color=BORDER, linewidth=0.8, linestyle=":", alpha=0.9)

    ax_cum.set_title("Cumulative Performance",
                     fontsize=13, fontweight="bold", color=TEXT, pad=14)
    ax_cum.set_ylabel("Index (base = 100)", fontsize=9, color=TEXT, labelpad=8)

    # X axis: quarterly ticks, no rotation
    ax_cum.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax_cum.xaxis.get_majorticklabels(),
             rotation=0, ha="center", color=TEXT, fontsize=8)
    ax_cum.tick_params(axis="y", labelsize=8)

    # Horizontal legend (ncol=3)
    ax_cum.legend(
        facecolor=BACKGROUND, edgecolor=BORDER,
        labelcolor=TEXT, fontsize=8, loc="upper left",
        framealpha=0.7, ncol=3,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=2.5, rect=[0.0, 0.06, 1.0, 0.97])

    _add_footer(fig)
    pdf.savefig(fig, facecolor=BACKGROUND, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2 — Monthly heatmap + Drawdown
# ---------------------------------------------------------------------------

def _page2_monthly_drawdown(pdf: PdfPages, results: dict) -> None:
    ls_ret = results["long_short_returns"]

    fig = plt.figure(figsize=FIGSIZE, facecolor=BACKGROUND, dpi=FIG_DPI)
    _top_rule(fig, lw=4)

    # Height ratios: heatmap ~58%, drawdown ~35%, gap ~7%
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[6.2, 3.8],
        hspace=0.35,
        left=0.07, right=0.96, top=0.94, bottom=0.10,
    )

    # ---------------------------------------------------------- heatmap
    ax_heat = fig.add_subplot(gs[0])
    ax_heat.set_facecolor(BACKGROUND)

    monthly_df = compute_monthly_returns(ls_ret)
    data       = monthly_df.values.astype(float)
    masked     = np.ma.masked_invalid(data)

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color=CARD_BG)

    abs_max = max(float(np.nanmax(np.abs(data))), 0.005)
    im = ax_heat.imshow(
        masked,
        cmap=cmap,
        aspect="auto",
        vmin=-abs_max,
        vmax=abs_max,
        zorder=1,
    )

    # Cell annotations — black TEXT, fontsize=8
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if np.isnan(val):
                continue
            ax_heat.text(
                c, r, f"{val:+.1%}",
                ha="center", va="center",
                fontsize=8, color=TEXT, fontweight="bold",
                zorder=3,
            )

    # White cell-separator grid (replaces default gridlines)
    ax_heat.grid(False)
    n_rows, n_cols = data.shape
    for i in range(n_rows + 1):
        ax_heat.axhline(i - 0.5, color="white", linewidth=1.5, zorder=2)
    for j in range(n_cols + 1):
        ax_heat.axvline(j - 0.5, color="white", linewidth=1.5, zorder=2)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax_heat.set_xticks(range(12))
    ax_heat.set_xticklabels(month_labels, fontsize=9, color=TEXT)
    ax_heat.set_yticks(range(len(monthly_df.index)))
    ax_heat.set_yticklabels(monthly_df.index.astype(str), fontsize=9, color=TEXT)
    ax_heat.tick_params(axis="both", length=0, pad=4)

    for spine in ax_heat.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.6)

    ax_heat.set_title("Monthly Returns - Long/Short Portfolio",
                      fontsize=13, fontweight="bold", color=TEXT, pad=14)

    # Colorbar — shrink=0.6, aspect=20
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.6, aspect=20, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=8, pad=4)
    cbar.ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)
    cbar.outline.set_edgecolor(BORDER)
    cbar.outline.set_linewidth(0.6)

    # -------------------------------------------------------- drawdown
    ax_dd = fig.add_subplot(gs[1])
    _apply_chart_style(ax_dd)
    _remove_top_right_spines(ax_dd)

    dd = compute_drawdown_series(ls_ret)

    ax_dd.fill_between(dd.index, 0, dd.values, color=RED, alpha=0.15)
    ax_dd.plot(dd.index, dd.values, color=RED, linewidth=1.5)

    # 2-week xlim buffer on each side
    if not dd.empty:
        x_min = dd.index.min() - pd.Timedelta(weeks=2)
        x_max = dd.index.max() + pd.Timedelta(weeks=2)
        ax_dd.set_xlim(x_min, x_max)

        max_dd_date = dd.idxmin()
        max_dd_val  = float(dd.min())

        ax_dd.scatter(
            [max_dd_date], [max_dd_val],
            color=RED, s=55, zorder=5,
            edgecolors=BACKGROUND, linewidths=0.8,
        )

        ofs_y = max_dd_val * 0.42
        ax_dd.annotate(
            f"Max DD: {max_dd_val:.1%}",
            xy=(max_dd_date, max_dd_val),
            xytext=(max_dd_date, ofs_y),
            ha="center", fontsize=8,
            color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
        )

    ax_dd.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    # Quarterly x-axis, no rotation
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax_dd.xaxis.get_majorticklabels(),
             rotation=0, ha="center", color=TEXT, fontsize=8)
    ax_dd.tick_params(axis="y", labelsize=8)

    ax_dd.axhline(y=0, color=BORDER, linewidth=0.7, linestyle="-")
    ax_dd.set_title("Underwater Chart (Drawdown from Peak)",
                    fontsize=13, fontweight="bold", color=TEXT, pad=14)
    ax_dd.set_ylabel("Drawdown", fontsize=9, color=TEXT, labelpad=8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=2.5, rect=[0.0, 0.06, 1.0, 0.97])

    _add_footer(fig)
    pdf.savefig(fig, facecolor=BACKGROUND, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3 — Factor attribution
# ---------------------------------------------------------------------------

def _page3_factor_analysis(
    pdf: PdfPages,
    factor_attribution: pd.DataFrame,
    results: dict,
) -> None:
    fig = plt.figure(figsize=FIGSIZE, facecolor=BACKGROUND, dpi=FIG_DPI)
    _top_rule(fig, lw=4)

    # Charts live in the upper 60 % of the figure (top=0.88, bottom=0.28).
    # The summary box is placed below via fig.add_axes() so tight_layout
    # won't disturb it.
    gs = fig.add_gridspec(
        1, 2,
        wspace=0.35,
        left=0.08, right=0.96, top=0.88, bottom=0.28,
    )

    factors = factor_attribution.index.tolist()
    q1_vals = factor_attribution["Q1_avg"].values
    q5_vals = factor_attribution["Q5_avg"].values
    spreads = factor_attribution["spread"].values
    y_pos   = np.arange(len(factors))
    bar_h   = 0.35

    def _lbl(f: str) -> str:
        return f.replace("_", " ").title()

    # ------------------------------------------ left: Q1 vs Q5 grouped bars
    ax_left = fig.add_subplot(gs[0, 0])
    _apply_chart_style(ax_left)
    _remove_top_right_spines(ax_left)

    ax_left.barh(y_pos + bar_h / 2, q1_vals, bar_h,
                 color=ACCENT2, alpha=0.88, label="Q1 (Long)")
    ax_left.barh(y_pos - bar_h / 2, q5_vals, bar_h,
                 color=RED, alpha=0.88, label="Q5 (Short)")

    ax_left.axvline(0, color=TEXT_SECONDARY, linewidth=0.7,
                    linestyle="--", alpha=0.55)
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels([_lbl(f) for f in factors],
                             fontsize=9, color=TEXT)
    ax_left.set_xlabel("Average Z-Score", fontsize=9, color=TEXT, labelpad=8)
    ax_left.tick_params(axis="x", labelsize=8)
    ax_left.set_title("Avg Factor Scores: Q1 (Long) vs Q5 (Short)",
                      fontsize=13, fontweight="bold", color=TEXT, pad=14)
    ax_left.legend(
        facecolor=BACKGROUND, edgecolor=BORDER,
        labelcolor=TEXT, fontsize=8, framealpha=0.85,
    )

    # ----------------------------------------- right: spread bars
    ax_right = fig.add_subplot(gs[0, 1])
    _apply_chart_style(ax_right)
    _remove_top_right_spines(ax_right)

    spread_colors = [GREEN if s >= 0 else RED for s in spreads]
    bars = ax_right.barh(y_pos, spreads, bar_h * 1.55,
                         color=spread_colors, alpha=0.88)

    # Value labels — fontsize=8, fontweight='normal' (not bold)
    for bar, val in zip(bars, spreads):
        ha  = "left"  if val >= 0 else "right"
        xof = 0.012   if val >= 0 else -0.012
        ax_right.text(
            val + xof,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}s",          # 's' instead of sigma — ASCII safe
            ha=ha, va="center",
            fontsize=8, color=TEXT, fontweight="normal",
        )

    ax_right.axvline(0, color=TEXT_SECONDARY, linewidth=0.7,
                     linestyle="--", alpha=0.55)
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels([_lbl(f) for f in factors],
                              fontsize=9, color=TEXT)
    ax_right.set_xlabel("Spread (Q1 - Q5) in std-dev",
                        fontsize=9, color=TEXT, labelpad=8)
    ax_right.tick_params(axis="x", labelsize=8)
    ax_right.set_title("Factor Spread (Q1 minus Q5)",
                       fontsize=13, fontweight="bold", color=TEXT, pad=14)

    # ------------------------------------------- summary box
    # Placed via add_axes so tight_layout doesn't touch it.
    # Position: [left, bottom, width, height] in figure coordinates.
    ax_summary = fig.add_axes([0.08, 0.06, 0.88, 0.16])
    ax_summary.set_facecolor(CARD_BG)
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis("off")

    # Box border
    ax_summary.add_patch(Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        facecolor=CARD_BG, edgecolor=BORDER, linewidth=1.0,
        transform=ax_summary.transAxes, clip_on=False,
    ))

    spread_dict = dict(zip(factors, spreads))
    summary_vals = "   |   ".join(
        f"{_lbl(f)}: {spread_dict[f]:+.2f}s" for f in factors
    )
    caption = (
        "Factor Spread Summary  "
        "(positive spread = factor successfully separated long from short)"
    )

    ax_summary.text(0.5, 0.72, caption,
                    ha="center", va="center",
                    fontsize=9, color=TEXT_SECONDARY, style="italic",
                    transform=ax_summary.transAxes)
    ax_summary.text(0.5, 0.30, summary_vals,
                    ha="center", va="center",
                    fontsize=10, color=TEXT, fontweight="bold",
                    transform=ax_summary.transAxes)

    # tight_layout only adjusts the gs subplots (ignores add_axes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=2.5, rect=[0.0, 0.28, 1.0, 0.97])

    _add_footer(fig)
    pdf.savefig(fig, facecolor=BACKGROUND, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_tearsheet(
    backtest_results: dict,
    factor_attribution: pd.DataFrame,
    output_path: str = "outputs/tearsheet.pdf",
) -> str:
    """
    Generate a three-page professional PDF tearsheet (light institutional theme).

    Parameters
    ----------
    backtest_results : dict
        Output of :func:`src.backtest.run_backtest`.
    factor_attribution : pd.DataFrame
        Output of :func:`src.backtest.compute_factor_attribution`.
    output_path : str
        Destination path for the generated PDF.

    Returns
    -------
    str
        Absolute path to the saved PDF.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[tearsheet] Generating PDF -> {out.resolve()}")

    # seaborn-v0_8-whitegrid as base; all key params overridden per-axes above.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    with PdfPages(str(out)) as pdf:
        info = pdf.infodict()
        info["Title"]   = "Factor Model Tearsheet"
        info["Subject"] = "Multi-Factor Long-Short Equity Strategy"
        info["Author"]  = "Factor Model Tearsheet Generator"

        _page1_cover(pdf, backtest_results)
        print("[tearsheet]   Page 1 - Cover + Stats + Cumulative Returns [OK]")

        _page2_monthly_drawdown(pdf, backtest_results)
        print("[tearsheet]   Page 2 - Monthly Heatmap + Drawdown [OK]")

        _page3_factor_analysis(pdf, factor_attribution, backtest_results)
        print("[tearsheet]   Page 3 - Factor Attribution [OK]")

    size_kb = out.stat().st_size / 1024
    print(f"[tearsheet] Done - {size_kb:.1f} KB saved to {out.resolve()}")
    return str(out.resolve())


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.backtest import run_backtest, compute_factor_attribution

    print("=" * 65)
    print("Factor Model Tearsheet Generator  (polished light theme)")
    print("=" * 65)

    print("\nStep 1/3 - Running backtest (US universe, monthly rebalance)...")
    results = run_backtest(
        market="us",
        rebalance_freq="M",
        transaction_cost_bps=10.0,
    )

    print("\nStep 2/3 - Computing factor attribution...")
    attribution = compute_factor_attribution(results)
    print(attribution.to_string())

    print("\nStep 3/3 - Rendering tearsheet PDF...")
    pdf_path = generate_tearsheet(
        backtest_results=results,
        factor_attribution=attribution,
        output_path="outputs/tearsheet.pdf",
    )

    print(f"\nTearsheet saved to {pdf_path}")

    try:
        if sys.platform == "win32":
            os.startfile(pdf_path)               # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", pdf_path])
        else:
            subprocess.Popen(["xdg-open", pdf_path])
    except Exception as exc:
        print(f"(Could not auto-open PDF: {exc})")

    print("=" * 65)
