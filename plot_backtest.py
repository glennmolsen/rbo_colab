"""Plot a backtest equity curve produced by decision.py.

Usage:
    python3 plot_backtest.py [curve_csv] [--out PATH] [--show]

Default input: curve.csv in the repo root.
Default output: backtest.png next to the input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _max_drawdown_series(equity: pd.Series) -> pd.Series:
    peaks = equity.cummax()
    return equity / peaks - 1.0


def _shade_cash_spans(ax, df: pd.DataFrame) -> None:
    """Shade contiguous CASH stretches so blackout exits are visible."""
    in_cash = df["holding"] == "CASH"
    if not in_cash.any():
        return
    # Find runs of consecutive True
    runs = in_cash.ne(in_cash.shift()).cumsum()
    first_label_used = False
    for _, group in df[in_cash].groupby(runs[in_cash]):
        label = "In cash (blackout)" if not first_label_used else None
        ax.axvspan(group.index[0], group.index[-1], color="#f0c36d", alpha=0.25, label=label)
        first_label_used = True


def plot(csv_path: Path, out_path: Path, show: bool) -> None:
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()

    strat = df["strategy_equity"]
    bh = df["buy_hold_levered"]
    strat_dd = _max_drawdown_series(strat) * 100
    bh_dd = _max_drawdown_series(bh) * 100

    start, end = df.index[0], df.index[-1]
    years = (end - start).days / 365.25
    strat_cagr = (strat.iloc[-1] / strat.iloc[0]) ** (1 / years) - 1
    bh_cagr = (bh.iloc[-1] / bh.iloc[0]) ** (1 / years) - 1

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # --- Equity panel (log y) ---
    _shade_cash_spans(ax_eq, df)
    ax_eq.plot(strat.index, strat.values, color="#1f6feb", linewidth=1.8,
               label=f"Strategy (CAGR {strat_cagr*100:.1f}%, "
                     f"MDD {strat_dd.min():.1f}%)")
    ax_eq.plot(bh.index, bh.values, color="#d1464a", linewidth=1.4, alpha=0.85,
               label=f"Buy & hold QLD (CAGR {bh_cagr*100:.1f}%, "
                     f"MDD {bh_dd.min():.1f}%)")
    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("Equity ($, log scale)")
    ax_eq.set_title(
        f"Leverage strategy vs buy-and-hold QLD  |  "
        f"{start.date()} → {end.date()}  |  "
        f"${strat.iloc[0]:,.0f} → ${strat.iloc[-1]:,.0f}"
    )
    ax_eq.grid(True, which="both", axis="y", alpha=0.25)
    ax_eq.grid(True, which="major", axis="x", alpha=0.25)
    ax_eq.legend(loc="upper left", framealpha=0.95)

    # --- Drawdown panel ---
    ax_dd.fill_between(strat_dd.index, strat_dd.values, 0,
                       color="#1f6feb", alpha=0.35, label="Strategy")
    ax_dd.fill_between(bh_dd.index, bh_dd.values, 0,
                       color="#d1464a", alpha=0.25, label="Buy & hold QLD")
    ax_dd.axhline(0, color="black", linewidth=0.6)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(True, alpha=0.25)
    ax_dd.legend(loc="lower left", framealpha=0.95)

    # Nicer date ticks
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="curve.csv",
                        help="Path to equity curve CSV (default: curve.csv)")
    parser.add_argument("--out", help="Output image path (default: <csv>.png)")
    parser.add_argument("--show", action="store_true", help="Open the plot interactively")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out_path = Path(args.out).resolve() if args.out else csv_path.with_suffix(".png")
    plot(csv_path, out_path, args.show)


if __name__ == "__main__":
    main()
