# Factor Model Tearsheet Generator

A multi-factor stock ranking model for global markets (NSE India + S&P 500) with a long-short backtest engine and professional PDF tearsheet output.

## Factors
- **Momentum** — 12-1 month price momentum
- **Value** — P/E, P/B composite
- **Quality** — ROE, debt/equity, earnings stability
- **Low Volatility** — trailing realized volatility
- **Size** — market capitalization (small-cap tilt)

## Output
A PDF tearsheet containing:
- Cumulative returns chart
- Monthly returns heatmap
- Drawdown chart
- Factor exposure bar chart
- Performance statistics table

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # fill in any required keys
```

## Usage

### Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

### CLI / Notebook
```python
from src.universe import build_universe
from src.factors import compute_factors
from src.backtest import run_backtest
from src.tearsheet import generate_tearsheet

universe = build_universe(markets=["NSE", "SP500"])
signals  = compute_factors(universe)
results  = run_backtest(signals)
generate_tearsheet(results, output_path="outputs/tearsheet.pdf")
```

## Project Structure

```
factor-model-tearsheet/
├── src/
│   ├── universe.py     # stock universe builder
│   ├── factors.py      # factor signal construction
│   ├── backtest.py     # long-short backtest engine
│   ├── tearsheet.py    # PDF tearsheet generator
│   └── stats.py        # performance statistics
├── app/
│   └── streamlit_app.py
├── data/               # cached price & fundamental data
├── outputs/            # generated PDF tearsheets
└── tests/
    └── test_factors.py
```
