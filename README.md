# rbo_colab

Daily leverage decision engine for a (base, levered) ETF pair (e.g. QQQ / QLD).
Decides what to hold for the next trading session based on the macro regime
and tactical indicators, and backtests the strategy against buy-and-hold
benchmarks.

## Files

- `decision.py` — strategy engine (live + backtest)
- `plot_backtest.py` — renders an equity-curve comparison chart
- `data/QLD_Blackouts_Tracking.csv` — macro regime (`Invest` / `Blackout period`)
  and current holding, keyed by date

## Strategies

- **Tactical** (`levered ↔ base ↔ cash`): default. De-leverages from `levered`
  to `base` when RSI-exhaustion + institutional distribution + MACD cross-down
  all line up. Re-leverages when RSI stabilizes + follow-through day + MACD
  cross-up. Goes to cash on macro `Blackout period`.
- **Macro** (`levered ↔ cash`): follows the blackout calendar only.

## Indicator helpers (`decision.py`)

| Function | Input | Output |
|---|---|---|
| `rsi(close, length=14)` | Close `pd.Series` | Wilder RSI series |
| `macd_histogram(close, 12, 26, 9)` | Close `pd.Series` | MACD histogram series |
| `distribution_days(df, window=25)` | OHLCV `DataFrame` | Rolling count of down->0.2%-on-higher-volume days |
| `follow_through_days(df)` | OHLCV `DataFrame` | Boolean series marking day-4+-off-lows surges >1.2% on higher volume |

## Main entry point

```python
from decision import decide
```

### `decide(base, levered, mode, ...)`

| Arg | Type | Default | Notes |
|---|---|---|---|
| `base` | str | `"QQQ"` | 1x ticker |
| `levered` | str | `"QLD"` | 2x ticker |
| `mode` | `"live"` \| `"backtest"` | `"live"` | dispatches |
| `period` (live) | str | `"6mo"` | Yahoo period for live mode |
| `start`, `end` (backtest) | date-like | first row of blackouts CSV / today | backtest window |
| `initial_capital` (backtest) | float | `100_000.0` | starting equity |

**Returns** a `Decision` (live) or `BacktestResult` (backtest).

### `Decision` (live mode)
Fields: `base`, `levered`, `evaluated_close`, `apply_on_open`,
`current_holding`, `macro_state`, `target_holding`, `action`, `reasoning`,
`indicators`, `warnings`. Current holding + regime are read from the most
recent row of the blackouts CSV.

### `BacktestResult` (backtest mode)
Fields: `base`, `levered`, `start_date`, `end_date`, `initial_capital`,
`curves` (dict of `CurveStats`: `final_equity`, `cagr`, `max_drawdown` for
each of `strategy_tactical`, `strategy_macro`, `buy_hold_levered`,
`buy_hold_base`), `equity_curve` (`DataFrame` with `holding_macro`,
`holding_tactical`, `strategy_macro`, `strategy_tactical`,
`buy_hold_levered`, `buy_hold_base`), `warnings`.

**Execution timing** (no look-ahead): indicators/regime evaluated on day T's
close determine the holding for day T+1. The day-T+1 return is split: the
*prior* holding owns the overnight gap (T close → T+1 open), the *new*
holding owns intraday (T+1 open → T+1 close).

## Plotter (`plot_backtest.py`)

Reads the CSV written by `decide(... mode="backtest", --out ...)` and
renders a two-panel PNG: log-y equity curves for all four strategies, plus
a linear drawdown panel below. Shaded bands mark periods the tactical
strategy held cash.

## Example

```bash
# Live decision (next session)
python3 decision.py --mode live --base QQQ --levered QLD

# Backtest + equity curve CSV
python3 decision.py --mode backtest --base QQQ --levered QLD \
    --start 2018-01-02 --end 2026-04-13 --out curve.csv

# Visualize the backtest
python3 plot_backtest.py curve.csv --out backtest.png
```

Useful flags: `--capital 100000` (starting equity), `--show` (open plot
interactively), `--json` (machine-readable output from `decision.py`).

## Data requirements

- `yfinance`, `pandas`, `numpy`, `matplotlib`
- `data/QLD_Blackouts_Tracking.csv` must cover the backtest window
  (columns: date, price, `Macro_State`, `Holdings`, ...). Missing days
  default to `Invest` with a warning.
