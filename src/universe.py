"""Stock universe builder — fetches and manages Nifty 200 + full S&P 500 constituents."""

from __future__ import annotations

import datetime
import glob
import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Universe constants
# ---------------------------------------------------------------------------

SP500_TICKERS: list[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    # Healthcare
    "UNH", "JNJ", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR",
    "AMGN", "GILD", "ISRG", "VRTX", "REGN", "BAX", "BSX", "EW", "MDT",
    "SYK", "ZBH",
    # Financials
    "JPM", "V", "MA", "BAC", "GS", "AXP", "USB", "TFC", "SCHW", "BK",
    "CB", "MMC", "AON", "MET", "PRU", "MSCI", "SPGI", "MCO", "ICE", "CME",
    "FICO", "MKTX",
    # Consumer staples
    "PG", "PEP", "KO", "WMT", "COST", "PM", "MDLZ", "CL", "KMB",
    # Consumer discretionary
    "HD", "MCD", "NKE", "TGT", "ROST", "TJX", "ORLY", "AZO", "TSCO",
    "LOW", "POOL",
    # Communication services
    "DIS", "NFLX", "CMCSA", "T", "QCOM",
    # Industrials
    "HON", "CAT", "DE", "ETN", "EMR", "ROK", "PH", "ITW", "CMI",
    "GWW", "FAST", "FDX", "UPS", "CSX", "NSC", "UNP",
    "LMT", "NOC", "GD", "RTX", "BA",
    "SWK", "PNR", "XYL", "ROP", "CSGP", "CPRT",
    "EXPD", "CHRW", "JBHT", "ODFL", "SAIA", "KNX", "LSTR",
    # Energy
    "CVX", "XOM", "COP", "EOG", "SLB", "HAL", "BKR",
    "OXY", "DVN", "HES", "MPC", "VLO", "MRO",
    # Utilities & Real Estate
    "NEE", "AMT", "PLD", "CCI", "EQIX", "SPG", "PSA",
    # Materials
    "LIN", "MMM",
    # Tech hardware & semiconductors
    "IBM", "INTC", "TXN", "ADBE", "CRM", "ACN", "CSCO",
    # Payments & fintech
    "PYPL", "FISV", "FIS", "GPN",
    # Analytics / data
    "VRSK", "IQV", "A", "WAT", "MTD", "BIO", "IDXX",
    # Enterprise software
    "NOW", "WDAY", "VEEV", "PAYC", "ADP", "PAYX", "BR",
    "CDNS", "SNPS", "ANSS", "KEYS", "TRMB", "FFIV", "AKAM",
    # Cybersecurity / cloud
    "ZS", "CRWD", "DDOG", "NET", "OKTA",
    # Data platforms
    "SNOW", "MDB", "TEAM", "HUBS",
    # IT services / staffing
    "CTSH", "WEX", "EPAM",
    # Transport / logistics
    "J",
]

NIFTY200_TICKERS: list[str] = [
    # Large-cap blue chips
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS",
    # Conglomerates / infra
    "ADANIENT.NS", "ADANIPORTS.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
    # Metals & mining
    "JSWSTEEL.NS", "TATASTEEL.NS", "HINDALCO.NS", "COALINDIA.NS", "VEDL.NS",
    # Financial services
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS",
    # IT mid-cap
    "TECHM.NS", "MPHASIS.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS",
    # Pharma
    "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "MAXHEALTH.NS",
    # FMCG
    "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "COLPAL.NS",
    # Beverages & tobacco
    "TATACONSUM.NS", "MCDOWELL-N.NS", "RADICO.NS", "UBL.NS",
    # Automobiles
    "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    # Power & renewables
    "TATAPOWER.NS", "ADANIGREEN.NS", "TORNTPOWER.NS", "CESC.NS",
    # Mid-cap banks / NBFCs
    "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "RBLBANK.NS",
    # New-age / platform
    "NAUKRI.NS", "ZOMATO.NS", "DELHIVERY.NS",
    # Logistics
    "IRCTC.NS", "CONCOR.NS", "BLUEDART.NS",
    # Chemicals & paints
    "PIDILITIND.NS", "BERGEPAINT.NS", "KANSAINER.NS",
    # Aviation / infra
    "INDIGO.NS", "GMRINFRA.NS", "IRB.NS",
    # Consumer durables
    "HAVELLS.NS", "VOLTAS.NS", "BLUESTAR.NS", "DIXON.NS",
    # Retail
    "TRENT.NS", "SHOPERSTOP.NS",
    # Pharma mid-cap
    "ZYDUSLIFE.NS", "TORNTPHARM.NS", "AUROPHARMA.NS", "LUPIN.NS", "ALKEM.NS",
    # NBFCs
    "CHOLAFIN.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS", "SHRIRAMFIN.NS", "LICHSGFIN.NS",
    # Capital goods
    "SIEMENS.NS", "ABB.NS", "BHEL.NS", "BEL.NS", "HAL.NS",
    # Cement
    "GRASIM.NS", "AMBUJACEM.NS", "ACC.NS", "JKCEMENT.NS", "SHREECEM.NS",
    # PSU finance
    "RECLTD.NS", "PFC.NS", "IRFC.NS",
    # Agrochemicals & speciality
    "TATACHEM.NS", "SRF.NS", "PIIND.NS", "UPL.NS", "COROMANDEL.NS",
    # Telecom infra / fibre
    "INDUSTOWER.NS", "TATACOMM.NS", "HFCL.NS", "STLTECH.NS",
    # Real estate
    "OBEROIRLTY.NS", "DLF.NS", "GODREJPROP.NS", "PRESTIGE.NS", "BRIGADE.NS",
    # PSU banks
    "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "INDIANB.NS",
    # Infrastructure
    "RVNL.NS", "IRCON.NS", "NBCC.NS", "NCC.NS",
    # Textiles
    "PAGEIND.NS", "MANYAVAR.NS", "KPRMILL.NS", "VARDHMAN.NS",
    # MNC pharma
    "PFIZER.NS", "ABBOTINDIA.NS", "SANOFI.NS", "GLAXO.NS", "NATCOPHARM.NS",
    # Oil & downstream
    "BPCL.NS", "IOC.NS", "HPCL.NS",
    # Insurance
    "MFSL.NS", "STARHEALTH.NS", "GICRE.NS",
    # QSR / food
    "JUBLFOOD.NS", "DEVYANI.NS", "WESTLIFE.NS",
    # Capital markets
    "CAMS.NS", "CDSL.NS", "BSE.NS", "MCX.NS",
    # Media
    "SUNTV.NS", "ZEEL.NS",
    # IT niche
    "CYIENT.NS", "KPITTECH.NS", "TATAELXSI.NS",
    # Auto ancillaries
    "MOTHERSON.NS", "BOSCHLTD.NS", "MINDA.NS", "SUPRAJIT.NS",
]

# Fields pulled from yfinance .info for the fundamentals table
_FUNDAMENTAL_FIELDS: list[str] = [
    "trailingPE",
    "priceToBook",
    "returnOnEquity",
    "trailingEps",
    "marketCap",
    "revenuePerShare",
]

# Project-level data directory (factor-model-tearsheet/data/)
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_CACHE_MAX_AGE_HOURS = 24


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(market: str, n_tickers: int, suffix: str) -> Path:
    """
    Return the canonical cache file path for today, embedding the ticker
    count so that a change in universe size automatically invalidates the
    old cache (no stale-ticker-count reads).
    """
    date_str = datetime.date.today().isoformat()
    return _DATA_DIR / f"universe_{market}_{n_tickers}_{date_str}_{suffix}.csv"


def _find_fresh_cache(market: str, n_tickers: int, suffix: str) -> Path | None:
    """
    Return the path to a cache file that:
      * matches the exact ticker count ``n_tickers``, AND
      * is younger than ``_CACHE_MAX_AGE_HOURS``.
    Returns None if no qualifying file exists.
    """
    # Pattern pins both the market and the expected ticker count so a cache
    # built with a different universe size is never returned.
    candidates = sorted(
        _DATA_DIR.glob(f"universe_{market}_{n_tickers}_*_{suffix}.csv"),
        reverse=True,    # newest date-string first
    )
    now = datetime.datetime.now()
    for path in candidates:
        age_h = (
            now - datetime.datetime.fromtimestamp(path.stat().st_mtime)
        ).total_seconds() / 3600
        if age_h < _CACHE_MAX_AGE_HOURS:
            return path
    return None


def _purge_stale_cache(market: str, n_tickers: int) -> None:
    """
    Delete all cache files for *market* that are either:
      * from a date other than today, OR
      * built with a different ticker count than ``n_tickers``.
    This ensures stale/mismatched caches never shadow fresh downloads.
    """
    today   = datetime.date.today().isoformat()
    n_str   = str(n_tickers)
    for pattern in (f"universe_{market}_*_prices.csv",
                    f"universe_{market}_*_fundamentals.csv"):
        for old_file in glob.glob(str(_DATA_DIR / pattern)):
            name = Path(old_file).name
            # Keep only files that match BOTH today's date AND the current ticker count
            if today in name and f"_{n_str}_" in name:
                continue
            try:
                os.remove(old_file)
                print(f"  [cache] Removed stale/mismatched file: {name}")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_price_data(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """
    Download adjusted close prices for *tickers* using yfinance.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols (duplicates are ignored).
    period : str
        yfinance period string, e.g. ``"3y"``, ``"1y"``.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame of close prices (columns = tickers).
        Columns with >30 % missing values are dropped; remaining gaps
        are forward-filled up to 5 consecutive trading days.
    """
    # Deduplicate while preserving order
    tickers = list(dict.fromkeys(tickers))
    print(f"Downloading {len(tickers)} tickers...")

    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns a MultiIndex when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices: pd.DataFrame = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Keep only tickers present in the download result
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available]

    # Drop columns with more than 30 % NaN
    missing_frac = prices.isna().mean()
    dropped = missing_frac[missing_frac > 0.30].index.tolist()
    if dropped:
        print(f"  Dropping {len(dropped)} ticker(s) with >30% missing: {dropped}")
    prices = prices.drop(columns=dropped)

    # Forward-fill gaps up to 5 consecutive trading days
    prices = prices.ffill(limit=5)

    print(f"  Price data: {prices.shape[1]} tickers x {len(prices)} dates")
    return prices


def get_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch key fundamental metrics for *tickers* via yfinance.

    Tickers are processed in batches of 20 with a 1-second sleep between
    batches to reduce the chance of rate-limit errors.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols.

    Returns
    -------
    pd.DataFrame
        Ticker-indexed DataFrame with columns:
        trailingPE, priceToBook, returnOnEquity, trailingEps,
        marketCap, revenuePerShare.
        Tickers that fail are silently skipped.
    """
    tickers = list(dict.fromkeys(tickers))   # deduplicate
    records: dict[str, dict] = {}
    n = len(tickers)
    batch_size = 20

    for batch_start in range(0, n, batch_size):
        batch = tickers[batch_start: batch_start + batch_size]

        for ticker in batch:
            try:
                info = yf.Ticker(ticker).info
                records[ticker] = {
                    field: info.get(field)
                    for field in _FUNDAMENTAL_FIELDS
                }
            except Exception as exc:          # noqa: BLE001
                print(f"  Warning: skipping {ticker} — {exc}")

        fetched = min(batch_start + batch_size, n)
        print(f"  Fetching fundamentals: {fetched}/{n} tickers...")

        # Pause between batches — skip sleep after the final batch
        if fetched < n:
            time.sleep(1)

    df = pd.DataFrame.from_dict(records, orient="index", columns=_FUNDAMENTAL_FIELDS)
    df.index.name = "ticker"
    print(f"  Fundamentals: {df.shape[0]} tickers x {df.shape[1]} fields")
    return df


def build_universe(
    market: str = "both",
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Build the stock universe for the factor model.

    Parameters
    ----------
    market : str
        One of ``"us"``, ``"india"``, or ``"both"``.

    Returns
    -------
    tuple
        ``(tickers, price_df, fundamentals_df, market_labels)``

        * **tickers** – ordered list of ticker symbols in the universe
        * **price_df** – date × ticker close-price DataFrame
        * **fundamentals_df** – ticker × fundamental-field DataFrame
        * **market_labels** – ``{ticker: "US" | "INDIA"}`` mapping
    """
    market = market.lower()
    if market not in {"india", "us", "both"}:
        raise ValueError(f"market must be 'india', 'us', or 'both'. Got: {market!r}")

    # ------------------------------------------- resolve raw ticker list
    if market == "us":
        raw_tickers = SP500_TICKERS.copy()
    elif market == "india":
        raw_tickers = NIFTY200_TICKERS.copy()
    else:
        raw_tickers = SP500_TICKERS + NIFTY200_TICKERS

    # Deduplicate while preserving order (constants contain known dupes, e.g. CAT, CVX)
    before_dedup = len(raw_tickers)
    raw_tickers  = list(dict.fromkeys(raw_tickers))
    n_tickers    = len(raw_tickers)
    if n_tickers < before_dedup:
        print(f"  After dedup: {n_tickers} unique tickers "
              f"(removed {before_dedup - n_tickers} duplicate(s))")
    else:
        print(f"  After dedup: {n_tickers} unique tickers (no duplicates found)")

    # ------------------------------------------- cache hygiene
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _purge_stale_cache(market, n_tickers)

    # ------------------------------------------- load or download
    price_cache = _find_fresh_cache(market, n_tickers, "prices")
    fund_cache  = _find_fresh_cache(market, n_tickers, "fundamentals")

    if price_cache and fund_cache:
        print(f"Loading from cache: {price_cache.name}, {fund_cache.name}")
        price_df        = pd.read_csv(price_cache, index_col=0, parse_dates=True)
        fundamentals_df = pd.read_csv(fund_cache, index_col=0)
    else:
        print(f"Downloading fresh data for {n_tickers} tickers...")
        price_df        = get_price_data(raw_tickers, period="3y")
        fundamentals_df = get_fundamentals(raw_tickers)

        price_df.to_csv(_cache_path(market, n_tickers, "prices"))
        fundamentals_df.to_csv(_cache_path(market, n_tickers, "fundamentals"))
        print(f"  Cached to {_cache_path(market, n_tickers, 'prices').name}")

    # ------------------------------------------- build market_labels
    tickers: list[str] = price_df.columns.tolist()
    market_labels: dict[str, str] = {
        t: ("INDIA" if (t.endswith(".NS") or t.endswith(".BO")) else "US")
        for t in tickers
    }

    n_us    = sum(1 for v in market_labels.values() if v == "US")
    n_india = sum(1 for v in market_labels.values() if v == "INDIA")

    print(
        f"\nUniverse: {len(tickers)} tickers  "
        f"({n_us} US, {n_india} India)"
    )
    print(f"Price DF       : {price_df.shape}  (dates x tickers)")
    print(f"Fundamentals DF: {fundamentals_df.shape}  (tickers x fields)")

    return tickers, price_df, fundamentals_df, market_labels


# ---------------------------------------------------------------------------
# __main__ — full three-market download test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use an alias so we don't shadow the module-level `import datetime`
    # that _purge_stale_cache() and other helpers depend on.
    import datetime as _dt_mod

    # ---------------------------------------------------------------- clear cache
    cleared = 0
    for _f in _DATA_DIR.glob("universe_*.csv"):
        try:
            _f.unlink()
            cleared += 1
        except OSError:
            pass
    print(f"Cleared {cleared} cache file(s) from data/\n")

    print("=" * 60)
    print("Universe builder — full download test (all three markets)")
    print("=" * 60)
    print(f"Started: {_dt_mod.datetime.now().strftime('%H:%M:%S')}")

    # ---------------------------------------------------------------- US
    print("\n=== Testing US universe ===")
    _, price_us, funds_us, labels_us = build_universe("us")
    print(f"US: {price_us.shape[1]} tickers x {price_us.shape[0]} dates")

    # ---------------------------------------------------------------- India
    print("\n=== Testing India universe ===")
    _, price_in, funds_in, labels_in = build_universe("india")
    print(f"India: {price_in.shape[1]} tickers x {price_in.shape[0]} dates")

    # ---------------------------------------------------------------- Combined
    print("\n=== Testing combined universe ===")
    _, price_both, funds_both, labels_both = build_universe("both")
    us_count = sum(1 for v in labels_both.values() if v == "US")
    in_count = sum(1 for v in labels_both.values() if v == "INDIA")
    print(f"Combined: {price_both.shape[1]} tickers ({us_count} US, {in_count} India)")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  US universe    : {price_us.shape[1]:>4} tickers  |  "
          f"{funds_us.shape[0]:>4} fundamentals rows")
    print(f"  India universe : {price_in.shape[1]:>4} tickers  |  "
          f"{funds_in.shape[0]:>4} fundamentals rows")
    print(f"  Combined       : {price_both.shape[1]:>4} tickers  |  "
          f"{funds_both.shape[0]:>4} fundamentals rows")
    print(f"\nFinished: {_dt_mod.datetime.now().strftime('%H:%M:%S')}")
