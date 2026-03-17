"""Performance statistics — Sharpe, max drawdown, Calmar, win rate, monthly heatmap."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def compute_stats(returns: pd.Series) -> dict:
    """
    Compute a standard set of risk/return metrics from a daily return series.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns (e.g. 0.012 = 1.2 %).

    Returns
    -------
    dict
        Keys: annual_return, annual_vol, sharpe, max_drawdown, calmar,
              win_rate, best_month, worst_month. All values rounded to 4 d.p.
        Returns an empty dict if the cleaned series has fewer than 2 points.
    """
    r = returns.dropna()
    if len(r) < 2:
        return {}

    # ---------------------------------------------------------------- returns
    annual_return = float((1.0 + r.mean()) ** 252 - 1.0)
    annual_vol    = float(r.std(ddof=1) * np.sqrt(252))

    sharpe: float | None = (
        annual_return / annual_vol if annual_vol > 0.0 else None
    )

    # --------------------------------------------------------------- drawdown
    cum_ret     = (1.0 + r).cumprod()
    rolling_max = cum_ret.cummax()
    dd_series   = cum_ret / rolling_max - 1.0
    max_drawdown = float(dd_series.min())

    calmar: float | None = (
        annual_return / abs(max_drawdown)
        if max_drawdown != 0.0
        else None
    )

    # --------------------------------------------------------------- win rate
    win_rate = float((r > 0.0).mean())

    # --------------------------------------------------- best / worst month
    monthly = _monthly_compound(r)
    best_month  = float(monthly.max()) if not monthly.empty else None
    worst_month = float(monthly.min()) if not monthly.empty else None

    def _r4(x: float | None) -> float | None:
        """Round to 4 decimal places, preserving None."""
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(x, 4)

    return {
        "annual_return": _r4(annual_return),
        "annual_vol":    _r4(annual_vol),
        "sharpe":        _r4(sharpe),
        "max_drawdown":  _r4(max_drawdown),
        "calmar":        _r4(calmar),
        "win_rate":      _r4(win_rate),
        "best_month":    _r4(best_month),
        "worst_month":   _r4(worst_month),
    }


# ---------------------------------------------------------------------------
# Monthly returns heatmap
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    1: "Jan", 2: "Feb",  3: "Mar",  4: "Apr",
    5: "May", 6: "Jun",  7: "Jul",  8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _monthly_compound(returns: pd.Series) -> pd.Series:
    """Resample daily returns to compounded monthly returns."""
    try:
        return returns.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    except ValueError:
        # pandas < 2.2 uses 'M' instead of 'ME'
        return returns.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)


def compute_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Build a calendar heatmap of monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Daily return series with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Rows = years, columns = ["Jan" … "Dec"]. Values are compounded
        monthly returns as decimals. Missing months are NaN.
    """
    monthly = _monthly_compound(returns.dropna())

    # Build a year × month MultiIndex then unstack
    monthly.index = pd.MultiIndex.from_arrays(
        [monthly.index.year, monthly.index.month],
        names=["year", "month"],
    )
    pivot = monthly.unstack(level="month")

    # Rename numeric month columns to three-letter abbreviations
    pivot.columns = [_MONTH_NAMES.get(c, str(c)) for c in pivot.columns]

    # Ensure all 12 months are present (fill missing with NaN)
    for col in _MONTH_NAMES.values():
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[[_MONTH_NAMES[m] for m in range(1, 13)]]

    pivot.index.name = "Year"
    return pivot


# ---------------------------------------------------------------------------
# Drawdown time series
# ---------------------------------------------------------------------------

def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute the full drawdown time series.

    drawdown[t] = cumulative_return[t] / max(cumulative_return[0:t]) - 1

    Parameters
    ----------
    returns : pd.Series
        Daily return series with a DatetimeIndex.

    Returns
    -------
    pd.Series
        Drawdown series (≤ 0). Same index as *returns*.
    """
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)

    cum_ret     = (1.0 + r).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown    = cum_ret / rolling_max - 1.0
    drawdown.name = "drawdown"
    return drawdown
