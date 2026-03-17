"""Factor signal construction — Momentum, Value, Quality, Low Volatility, Size."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mstats

from src.universe import get_price_data, get_fundamentals  # noqa: F401  (re-exported for callers)

# ---------------------------------------------------------------------------
# Default factor weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum":       0.30,
    "value":          0.20,
    "quality":        0.25,
    "low_volatility": 0.15,
    "size":           0.10,
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """
    Winsorise then z-score a cross-sectional factor series.

    Steps
    -----
    1. Winsorise at the 2.5th / 97.5th percentiles to remove extreme outliers.
    2. Z-score: ``(x - mean) / std`` computed on non-NaN values only.
    3. NaN positions in the input remain NaN in the output.

    Parameters
    ----------
    series : pd.Series
        Raw factor values indexed by ticker.

    Returns
    -------
    pd.Series
        Normalised series with the same index.
    """
    if series.dropna().empty:
        return series.copy()

    # Work on a float copy so we don't mutate the caller's data
    s = series.astype(float).copy()

    # --- winsorise only the non-NaN slice ---
    valid_mask = s.notna()
    valid_vals = s[valid_mask].values

    winsorised = mstats.winsorize(valid_vals, limits=[0.025, 0.025])
    s.loc[valid_mask] = np.array(winsorised, dtype=float)

    # --- z-score ---
    mu  = s[valid_mask].mean()
    std = s[valid_mask].std(ddof=1)

    if std == 0 or np.isnan(std):
        # Degenerate case: all values identical → return zeros where valid
        s.loc[valid_mask] = 0.0
        return s

    s = (s - mu) / std
    return s


# ---------------------------------------------------------------------------
# Individual factor signals
# ---------------------------------------------------------------------------

def compute_momentum(price_df: pd.DataFrame) -> pd.Series:
    """
    12-1 month cross-sectional momentum signal.

    Uses price[-21] / price[-252] - 1 to skip the last month's
    short-term reversal effect.

    Parameters
    ----------
    price_df : pd.DataFrame
        Date × ticker close prices.

    Returns
    -------
    pd.Series
        Z-scored momentum score indexed by ticker. Named ``"momentum"``.
    """
    if len(price_df) < 252:
        raise ValueError(
            f"price_df has only {len(price_df)} rows; need at least 252 for momentum."
        )

    price_1m_ago  = price_df.iloc[-21]   # ~1 month ago
    price_12m_ago = price_df.iloc[-252]  # ~12 months ago

    raw_momentum = price_1m_ago / price_12m_ago - 1
    raw_momentum.name = "momentum"

    return cross_sectional_zscore(raw_momentum).rename("momentum")


def compute_value(fundamentals_df: pd.DataFrame) -> pd.Series:
    """
    Value signal: inverse of price-to-book ratio.

    Lower P/B → cheaper stock → higher value score.
    Missing ``priceToBook`` entries are returned as NaN.

    Parameters
    ----------
    fundamentals_df : pd.DataFrame
        Ticker-indexed fundamentals table from :func:`get_fundamentals`.

    Returns
    -------
    pd.Series
        Z-scored value score indexed by ticker. Named ``"value"``.
    """
    if "priceToBook" not in fundamentals_df.columns:
        raise KeyError("'priceToBook' column missing from fundamentals_df.")

    pb = fundamentals_df["priceToBook"].astype(float)

    # Avoid division by zero / negative P/B (data quality issue)
    pb = pb.where(pb > 0)

    raw_value = 1.0 / pb
    raw_value.name = "value"

    return cross_sectional_zscore(raw_value).rename("value")


def compute_quality(fundamentals_df: pd.DataFrame) -> pd.Series:
    """
    Quality signal: return on equity (ROE).

    Higher ROE → higher quality → higher score.

    Parameters
    ----------
    fundamentals_df : pd.DataFrame
        Ticker-indexed fundamentals table from :func:`get_fundamentals`.

    Returns
    -------
    pd.Series
        Z-scored quality score indexed by ticker. Named ``"quality"``.
    """
    if "returnOnEquity" not in fundamentals_df.columns:
        raise KeyError("'returnOnEquity' column missing from fundamentals_df.")

    raw_quality = fundamentals_df["returnOnEquity"].astype(float)
    raw_quality.name = "quality"

    return cross_sectional_zscore(raw_quality).rename("quality")


def compute_low_volatility(price_df: pd.DataFrame) -> pd.Series:
    """
    Low-volatility signal: negated 252-day annualised realised volatility.

    Volatility = std(daily returns) × √252.
    Signal is negated so that *lower* realised vol → *higher* score.

    Parameters
    ----------
    price_df : pd.DataFrame
        Date × ticker close prices.

    Returns
    -------
    pd.Series
        Z-scored low-volatility score indexed by ticker.
        Named ``"low_volatility"``.
    """
    if len(price_df) < 2:
        raise ValueError("price_df must have at least 2 rows to compute returns.")

    daily_returns = price_df.pct_change(fill_method=None).dropna(how="all")

    # Use up to 252 trading days of history
    lookback = daily_returns.iloc[-252:]
    annualised_vol = lookback.std(ddof=1) * np.sqrt(252)

    raw_low_vol = -1.0 * annualised_vol
    raw_low_vol.name = "low_volatility"

    return cross_sectional_zscore(raw_low_vol).rename("low_volatility")


def compute_size(fundamentals_df: pd.DataFrame) -> pd.Series:
    """
    Size signal: negated log of market capitalisation.

    Smaller cap → higher score (small-cap historically outperforms).
    Missing or non-positive ``marketCap`` entries are returned as NaN.

    Parameters
    ----------
    fundamentals_df : pd.DataFrame
        Ticker-indexed fundamentals table from :func:`get_fundamentals`.

    Returns
    -------
    pd.Series
        Z-scored size score indexed by ticker. Named ``"size"``.
    """
    if "marketCap" not in fundamentals_df.columns:
        raise KeyError("'marketCap' column missing from fundamentals_df.")

    mc = fundamentals_df["marketCap"].astype(float)

    # log is only valid for positive market caps
    mc = mc.where(mc > 0)

    raw_size = -1.0 * np.log(mc)
    raw_size.name = "size"

    return cross_sectional_zscore(raw_size).rename("size")


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_composite_score(
    price_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute all five factor signals and combine into a weighted composite.

    The composite is itself cross-sectionally z-scored after weighting.

    Parameters
    ----------
    price_df : pd.DataFrame
        Date × ticker close prices.
    fundamentals_df : pd.DataFrame
        Ticker-indexed fundamentals (output of :func:`get_fundamentals`).
    weights : dict, optional
        Mapping of factor name → weight. Weights need not sum to 1
        (they are normalised internally). Defaults to :data:`DEFAULT_WEIGHTS`.

    Returns
    -------
    pd.DataFrame
        Ticker-indexed DataFrame with columns:
        ``momentum``, ``value``, ``quality``, ``low_volatility``,
        ``size``, ``composite``.
        Sorted by ``composite`` descending (best candidates first).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Normalise weights so they sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # ---------------------------------------------------------------- signals
    factor_map = {
        "momentum":       compute_momentum(price_df),
        "value":          compute_value(fundamentals_df),
        "quality":        compute_quality(fundamentals_df),
        "low_volatility": compute_low_volatility(price_df),
        "size":           compute_size(fundamentals_df),
    }

    # ------------------------------------------- align on common ticker index
    # price_df tickers (columns) and fundamentals index may differ slightly
    price_tickers = set(price_df.columns)
    fund_tickers  = set(fundamentals_df.index)
    common_tickers = sorted(price_tickers & fund_tickers)

    scores = pd.DataFrame(index=common_tickers)
    for name, signal in factor_map.items():
        scores[name] = signal.reindex(common_tickers)

    # ----------------------------------------------------- weighted composite
    composite = pd.Series(0.0, index=common_tickers, name="composite")
    for name, w in weights.items():
        factor_col = scores[name]
        # NaN in any factor → that factor contributes 0 for that ticker
        # (the weight is not redistributed; composite is still normalised at end)
        composite = composite.add(factor_col.fillna(0.0) * w)

    scores["composite"] = cross_sectional_zscore(composite).rename("composite")

    return scores.sort_values("composite", ascending=False)


# ---------------------------------------------------------------------------
# Quintile portfolios
# ---------------------------------------------------------------------------

def get_quintile_portfolios(scores_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Split the universe into five equally-sized quintile buckets by
    composite score.

    Q1 = top 20 % (long candidates), Q5 = bottom 20 % (short candidates).

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of :func:`compute_composite_score` — must contain a
        ``"composite"`` column.

    Returns
    -------
    dict
        ``{"Q1": [...], "Q2": [...], "Q3": [...], "Q4": [...], "Q5": [...]}``
        where each value is a list of ticker strings.
    """
    if "composite" not in scores_df.columns:
        raise KeyError("scores_df must contain a 'composite' column.")

    sorted_tickers = (
        scores_df["composite"]
        .dropna()
        .sort_values(ascending=False)
        .index.tolist()
    )

    n = len(sorted_tickers)
    if n == 0:
        return {f"Q{q}": [] for q in range(1, 6)}

    # pd.qcut-style equal bucketing
    quintile_size = n / 5
    quintiles: dict[str, list[str]] = {}
    for q in range(1, 6):
        start = int(round((q - 1) * quintile_size))
        end   = int(round(q * quintile_size))
        quintiles[f"Q{q}"] = sorted_tickers[start:end]

    return quintiles


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.universe import build_universe

    print("=" * 65)
    print("Factor Model — loading US universe from cache")
    print("=" * 65)

    try:
        tickers, price_df, fundamentals_df, market_labels = build_universe(market="us")
    except Exception as exc:
        print(f"Failed to load universe: {exc}")
        sys.exit(1)

    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Price DF:        {price_df.shape}")
    print(f"Fundamentals DF: {fundamentals_df.shape}")

    # -------------------------------------------------------- compute signals
    print("\nComputing composite factor scores...")
    scores = compute_composite_score(price_df, fundamentals_df)

    # ---------------------------------------------------------- print results
    pd.set_option("display.float_format", "{:+.4f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)

    print("\n--- Factor Scores (sorted by composite, best to worst) ---")
    print(scores.to_string())

    # ------------------------------------------------------------ quintiles
    quintiles = get_quintile_portfolios(scores)

    print("\n--- Quintile Portfolios ---")
    for label, group in quintiles.items():
        print(f"  {label} ({len(group)} stocks): {group}")

    print("\n" + "=" * 65)
    print("Top 5 LONG  candidates (Q1):", quintiles["Q1"][:5])
    print("Bottom 5 SHORT candidates (Q5):", quintiles["Q5"][-5:])
    print("=" * 65)
