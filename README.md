# Factor Model Tearsheet Generator

A multi-factor equity ranking model for global markets (S&P 500 + Nifty 200) 
with a long-short backtest engine, factor attribution analysis, and 
automated PDF tearsheet generation.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Universe](https://img.shields.io/badge/Universe-327%20stocks-green)

## What it does

Constructs a 5-factor composite score for every stock in the universe,
runs a monthly-rebalanced long-short backtest, and generates a 3-page 
institutional-grade PDF tearsheet — updated with live market data on 
every run.

**Factors:**
- **Momentum** : 12-1 month price return (skips last month to avoid reversal)
- **Value** : inverse price-to-book ratio
- **Quality** : return on equity
- **Low Volatility** : negative 252-day realized volatility
- **Size** : negative log market cap (small-cap tilt)

Each factor is cross-sectionally z-scored and winsorized at 2.5/97.5th 
percentile. Composite score = weighted sum of z-scores.

**Strategy:** Long top quintile (Q1), short bottom quintile (Q5), 
equal-weighted within each leg, monthly rebalance with configurable 
transaction costs.

## Key findings (US S&P 500, April 2024 – March 2026)

| Metric | Long-Short | Long-Only | Benchmark (EW) |
|--------|-----------|-----------|----------------|
| Annual Return | -13.1% | +9.8% | +13.9% |
| Sharpe Ratio | -0.77 | +0.62 | +0.89 |
| Max Drawdown | -32.8% | -15.9% | -17.1% |
| Win Rate | 49.2% | 50.5% | 51.5% |

**Factor spreads (Q1 vs Q5 average z-score):**
Momentum +1.65σ | Quality +1.00σ | Low Vol +0.99σ | 
Value +0.96σ | Size +0.11σ

**Why the long-short underperforms despite strong factor spreads:**

All 4 primary factors show >0.95σ separation between long and short 
portfolios : the model is correctly identifying cheap, profitable, 
low-volatility stocks vs expensive, low-quality, high-volatility ones. 
The underperformance comes entirely from the short leg: the model 
correctly identifies NVDA, MSFT, META, AMZN as low-value/high-volatility 
and shorts them — but these stocks rallied 40-80% in 2024-25 driven by 
AI-related momentum that the value and quality factors penalize.

This is a known phenomenon called **factor crowding** : when a macro 
theme (AI capex cycle) overwhelms cross-sectional factor signals. 
The long-only portfolio (+9.8%) confirms the long-side factor selection 
works; the drag is purely from the short book.

This tension between factor signal strength and realized P&L is what 
makes the 2024-25 period an interesting case study for any systematic 
equity strategy.

## Streamlit app

Interactive dashboard with 5 tabs:
- **Summary** : stat cards, cumulative returns chart, strategy comparison table
- **Monthly Returns** : calendar heatmap + drawdown chart
- **Factor Analysis** : Q1 vs Q5 scores, factor spread attribution
- **Current Rankings** : today's top long/short candidates with factor scores
- **Export PDF** : generates the 3-page institutional tearsheet

Factor weights, rebalance frequency, and transaction costs are all 
configurable — results update on every run.

## Setup

```bash
git clone https://github.com/Punnag2307/factor-model-tearsheet
cd factor-model-tearsheet
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads ~327 stock price histories and fundamentals 
(~40 minutes). All subsequent runs load from cache instantly.

```bash
streamlit run app/streamlit_app.py
```

To generate a PDF tearsheet directly:
```bash
python src/tearsheet.py
```

## Architecture

```
src/
├── universe.py      # Stock universe — S&P 500 + Nifty 200, 
│                    # yfinance download with daily cache
├── factors.py       # Cross-sectional factor construction
│                    # with winsorization and z-scoring
├── backtest.py      # Long-short backtest engine —
│                    # no lookahead bias, realistic tx costs
├── stats.py         # Sharpe, drawdown, monthly returns
└── tearsheet.py     # 3-page matplotlib PDF generator
app/
└── streamlit_app.py # Interactive web dashboard
```

## Methodology notes

**No lookahead bias:** Factor scores at each rebalance date are computed 
using only price and fundamental data available up to that date. The 
momentum factor enforces a minimum 252-day history before scoring.

**Transaction costs:** Applied at each rebalance on new positions only 
(turnover-based). Default 10bps one-way.

**Cross-market separation:** When running both US and India universes 
together, factor scoring is done separately within each market 
(cross-sectional z-scores are market-relative, not cross-currency). 
Long-short portfolios are constructed within each market independently.

**Known limitations:**
- Universe is a curated sample (172 US, 155 India) : not full index 
  membership with proper inclusion/exclusion rules
- Fundamental data from yfinance has quality gaps for smaller Indian 
  companies : "NM" shown where unavailable
- Backtest period (2 years) is short for factor research : 
  5-10 years is standard for robust conclusions
- No benchmark hedging on the short book : 
  pure long-short, not market-neutral

## What I learned

The most interesting engineering decision was enforcing strict 
data separation between the factor scoring step and the backtest 
loop, it's easy to accidentally use future information when 
computing signals on historical data. The second interesting problem 
was cross-market normalization: you can't z-score Indian and US 
stocks together because INR/USD differences make raw fundamentals 
incomparable, but relative rankings within each market are valid.

The research finding- that factor spreads can be statistically 
strong (+1.65σ momentum spread) while the strategy loses money,
is a useful reminder that signal strength and P&L are different 
things. A factor that correctly ranks stocks doesn't automatically 
produce alpha if the macro environment systematically favors the 
short book.

