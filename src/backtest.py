"""Long-short portfolio backtest engine — signal-to-returns simulation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.universe import build_universe
from src.factors import compute_composite_score, get_quintile_portfolios
from src.stats import compute_stats, compute_monthly_returns, compute_drawdown_series  # noqa: F401

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    market: str = "us",
    rebalance_freq: str = "M",
    top_pct: float = 0.20,
    transaction_cost_bps: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
    weights: dict | None = None,
) -> dict:
    """
    Run a long-short factor-model backtest.

    At each rebalance date the universe is scored using only data
    available up to that date (no look-ahead bias). Q1 stocks are held
    long, Q5 stocks short, in equal-weight proportions. Transaction costs
    are deducted on each rebalance proportional to portfolio turnover.

    Parameters
    ----------
    market : str
        ``"us"``, ``"india"``, or ``"both"``.
    rebalance_freq : str
        pandas offset alias — ``"M"`` (monthly), ``"Q"`` (quarterly), etc.
    top_pct : float
        Fraction of universe in long / short bucket (default 0.20 = top quintile).
        Currently informational; quintile logic is handled by
        :func:`get_quintile_portfolios`.
    transaction_cost_bps : float
        One-way cost per trade in basis points applied to new positions.
    start_date : str | None
        ``"YYYY-MM-DD"`` to trim the returned series.
    end_date : str | None
        ``"YYYY-MM-DD"`` to trim the returned series.

    Returns
    -------
    dict
        ``long_short_returns``, ``long_only_returns``, ``benchmark_returns``,
        ``factor_scores_history``, ``rebalance_dates``, ``metadata``.
    """
    # ----------------------------------------------------------------- data
    print(f"[backtest] Loading universe: market='{market}' ...")
    tickers, price_df, fundamentals_df, market_labels = build_universe(market=market)
    print(f"[backtest] {len(tickers)} tickers | {len(price_df)} trading days")

    # Daily returns (NaN for first row is expected)
    returns_df = price_df.pct_change()

    # Benchmark: equal-weight all available stocks each day
    benchmark_returns_full = returns_df.mean(axis=1)

    # --------------------------------------------------- rebalance schedule
    # 'ME' is the pandas >= 2.2 alias; fall back to 'M' for older versions
    try:
        rebalance_dates = price_df.resample("ME").last().index.tolist()
    except ValueError:
        rebalance_dates = price_df.resample("M").last().index.tolist()

    print(f"[backtest] {len(rebalance_dates)} rebalance dates "
          f"({rebalance_dates[0].date()} to {rebalance_dates[-1].date()})")

    # ------------------------------------------------- portfolio containers
    ls_daily:   list[pd.Series] = []   # long-short daily returns
    long_daily: list[pd.Series] = []   # long-only daily returns

    factor_scores_history: list[pd.DataFrame] = []
    active_rebalance_dates: list[pd.Timestamp] = []

    prev_long:  set[str] = set()
    prev_short: set[str] = set()

    tc_rate = transaction_cost_bps / 10_000.0

    # ------------------------------------------------ walk-forward loop
    for i, rd in enumerate(rebalance_dates[:-1]):
        next_rd = rebalance_dates[i + 1]

        # ---- Slice price history up to this rebalance date (no lookahead)
        price_up_to_rd = price_df.loc[:rd]

        # Need ≥ 252 rows for a valid momentum signal
        if len(price_up_to_rd) < 252:
            print(f"[backtest]   Skipping {rd.date()} — insufficient history "
                  f"({len(price_up_to_rd)} rows < 252)")
            continue

        # ---- Score the universe
        # When market="both" we score each sub-universe cross-sectionally in
        # its own currency context, then combine.  This prevents USD-priced
        # large-caps from dominating INR-priced mid-caps in a joint ranking.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if market == "both":
                    # --- split tickers by market label ---
                    avail = price_up_to_rd.columns.tolist()
                    us_cols     = [t for t in avail if market_labels.get(t) == "US"]
                    india_cols  = [t for t in avail if market_labels.get(t) == "INDIA"]

                    sub_scores_list: list[pd.DataFrame] = []
                    for sub_cols in (us_cols, india_cols):
                        if len(sub_cols) < 5:
                            continue
                        sub_price = price_up_to_rd[sub_cols]
                        sub_funds = fundamentals_df.reindex(sub_cols)
                        sub_scores = compute_composite_score(sub_price, sub_funds, weights=weights)
                        sub_scores_list.append(sub_scores)

                    if not sub_scores_list:
                        print(f"[backtest]   Skipping {rd.date()} — no scoreable sub-universe")
                        continue

                    scores = pd.concat(sub_scores_list)

                    # Long = top-20 % of US  + top-20 % of India (separately)
                    # Short= bot-20 % of US  + bot-20 % of India (separately)
                    long_tickers  = []
                    short_tickers = []
                    for sub_sc in sub_scores_list:
                        sub_q = get_quintile_portfolios(sub_sc)
                        long_tickers.extend(sub_q["Q1"])
                        short_tickers.extend(sub_q["Q5"])
                    long_tickers  = [t for t in long_tickers  if t in returns_df.columns]
                    short_tickers = [t for t in short_tickers if t in returns_df.columns]

                else:
                    scores = compute_composite_score(price_up_to_rd, fundamentals_df, weights=weights)
                    quintiles     = get_quintile_portfolios(scores)
                    long_tickers  = [t for t in quintiles["Q1"] if t in returns_df.columns]
                    short_tickers = [t for t in quintiles["Q5"] if t in returns_df.columns]

        except Exception as exc:
            print(f"[backtest]   Skipping {rd.date()} — scoring failed: {exc}")
            continue

        factor_scores_history.append(scores)
        active_rebalance_dates.append(rd)

        n_long  = len(long_tickers)
        n_short = len(short_tickers)

        if n_long == 0 or n_short == 0:
            print(f"[backtest]   Skipping {rd.date()} — empty long or short bucket")
            continue

        print(f"[backtest]   {rd.date()} | long={n_long}, short={n_short}")

        # ---- Weight vectors (aligned to full returns_df column set)
        long_w  = pd.Series(0.0, index=returns_df.columns)
        short_w = pd.Series(0.0, index=returns_df.columns)

        for t in long_tickers:
            long_w[t] = 1.0 / n_long
        for t in short_tickers:
            short_w[t] = -1.0 / n_short

        combined_w = long_w + short_w

        # ---- Holding-period returns (exclusive of rd, inclusive of next_rd)
        hold_mask  = (returns_df.index > rd) & (returns_df.index <= next_rd)
        hold_rets  = returns_df.loc[hold_mask]

        if hold_rets.empty:
            continue

        # Fill NaN with 0 so missing prices don't drop the dot product
        filled = hold_rets.fillna(0.0)

        port_ls   = filled.dot(combined_w)
        port_long = filled.dot(long_w)

        # ---- Transaction cost on new positions
        new_long  = set(long_tickers)  - prev_long
        new_short = set(short_tickers) - prev_short

        total_positions = n_long + n_short
        new_positions   = len(new_long) + len(new_short)
        turnover_frac   = new_positions / total_positions if total_positions else 0.0
        tc_cost         = turnover_frac * tc_rate

        if tc_cost > 0 and len(port_ls) > 0:
            port_ls.iloc[0]   -= tc_cost
            port_long.iloc[0] -= tc_cost * (n_long / total_positions)

        ls_daily.append(port_ls)
        long_daily.append(port_long)

        prev_long  = set(long_tickers)
        prev_short = set(short_tickers)

    # ----------------------------------------- concatenate & align
    if not ls_daily:
        raise RuntimeError("Backtest produced no output — check universe and date range.")

    long_short_returns = pd.concat(ls_daily).sort_index()
    long_only_returns  = pd.concat(long_daily).sort_index()
    benchmark_returns  = benchmark_returns_full.reindex(long_short_returns.index)

    long_short_returns.name = "long_short"
    long_only_returns.name  = "long_only"
    benchmark_returns.name  = "benchmark"

    # -------------------------------------------- optional date filter
    def _clip(s: pd.Series) -> pd.Series:
        if start_date:
            s = s[s.index >= pd.Timestamp(start_date)]
        if end_date:
            s = s[s.index <= pd.Timestamp(end_date)]
        return s

    long_short_returns = _clip(long_short_returns)
    long_only_returns  = _clip(long_only_returns)
    benchmark_returns  = _clip(benchmark_returns)

    metadata = {
        "market":               market,
        "n_stocks":             len(tickers),
        "rebalance_freq":       rebalance_freq,
        "transaction_cost_bps": transaction_cost_bps,
        "start_date":           str(long_short_returns.index[0].date()),
        "end_date":             str(long_short_returns.index[-1].date()),
        "n_rebalances":         len(active_rebalance_dates),
    }

    return {
        "long_short_returns":    long_short_returns,
        "long_only_returns":     long_only_returns,
        "benchmark_returns":     benchmark_returns,
        "factor_scores_history": factor_scores_history,
        "rebalance_dates":       active_rebalance_dates,
        "metadata":              metadata,
    }


# ---------------------------------------------------------------------------
# Factor attribution
# ---------------------------------------------------------------------------

def compute_factor_attribution(backtest_results: dict) -> pd.DataFrame:
    """
    Show which factors drove the long-short spread.

    For each factor signal, compute the average z-score of Q1 and Q5
    stocks across all rebalance dates. A large Q1 – Q5 spread means
    that factor strongly separated the long and short portfolios.

    Parameters
    ----------
    backtest_results : dict
        Output of :func:`run_backtest`.

    Returns
    -------
    pd.DataFrame
        Rows = factor names, columns = [``Q1_avg``, ``Q5_avg``, ``spread``].
    """
    history = backtest_results["factor_scores_history"]
    factor_cols = ["momentum", "value", "quality", "low_volatility", "size"]

    q1_accum: dict[str, list[float]] = {f: [] for f in factor_cols}
    q5_accum: dict[str, list[float]] = {f: [] for f in factor_cols}

    for scores_df in history:
        quintiles    = get_quintile_portfolios(scores_df)
        q1_tickers   = [t for t in quintiles["Q1"] if t in scores_df.index]
        q5_tickers   = [t for t in quintiles["Q5"] if t in scores_df.index]

        for factor in factor_cols:
            if factor not in scores_df.columns:
                continue
            if q1_tickers:
                q1_accum[factor].append(scores_df.loc[q1_tickers, factor].mean())
            if q5_tickers:
                q5_accum[factor].append(scores_df.loc[q5_tickers, factor].mean())

    rows = []
    for factor in factor_cols:
        q1_avg = float(np.nanmean(q1_accum[factor])) if q1_accum[factor] else np.nan
        q5_avg = float(np.nanmean(q5_accum[factor])) if q5_accum[factor] else np.nan
        rows.append({
            "factor":  factor,
            "Q1_avg":  round(q1_avg, 4),
            "Q5_avg":  round(q5_avg, 4),
            "spread":  round(q1_avg - q5_avg, 4),
        })

    return pd.DataFrame(rows).set_index("factor")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("Factor Model Backtest — US universe, monthly rebalance, 10 bps TC")
    print("This will take ~30 seconds — expected.")
    print("=" * 65)

    try:
        results = run_backtest(
            market="us",
            rebalance_freq="M",
            transaction_cost_bps=10.0,
        )
    except Exception as exc:
        print(f"Backtest failed: {exc}")
        sys.exit(1)

    # ---------------------------------------------------------- metadata
    print("\n--- Metadata ---")
    for k, v in results["metadata"].items():
        print(f"  {k:<25}: {v}")

    # ------------------------------------------------------ daily returns
    ls_ret = results["long_short_returns"]
    print(f"\n--- Long-Short Daily Returns (first 10 rows) ---")
    print(ls_ret.head(10).to_string())

    # ------------------------------------------------------------ stats
    print("\n--- Performance Statistics ---")
    pd.set_option("display.float_format", "{:.4f}".format)

    stats_ls   = compute_stats(ls_ret)
    stats_long = compute_stats(results["long_only_returns"])
    stats_bm   = compute_stats(results["benchmark_returns"])

    stats_table = pd.DataFrame(
        {"Long-Short": stats_ls, "Long-Only": stats_long, "Benchmark": stats_bm}
    )
    print(stats_table.to_string())

    # ------------------------------------------------- factor attribution
    print("\n--- Factor Attribution (avg z-score: Q1 vs Q5) ---")
    attr = compute_factor_attribution(results)
    print(attr.to_string())

    print("\n" + "=" * 65)
    print("Backtest complete.")
    print("=" * 65)
