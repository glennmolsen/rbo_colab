"""Plot a backtest equity curve produced by decision.py.

Renders four curves on a shared log-y equity axis plus a drawdown panel:
  1. Tactical strategy  (levered ↔ base ↔ cash)
  2. Macro strategy     (levered ↔ cash)
  3. Buy-and-hold levered
  4. Buy-and-hold base

Usage:
    python3 plot_backtest.py [curve_csv] [--out PATH] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


SERIES = [
    # (csv column, display label, color, linewidth, zorder)
    ("strategy_tactical", "Tactical (lev↔base↔cash)", "#1f6feb", 1.8, 5),
    ("strategy_macro",    "Macro (lev↔cash)",         "#2ca02c", 1.6, 4),
    ("buy_hold_levered",  "Buy-hold levered",         "#d1464a", 1.3, 3),
    ("buy_hold_base",     "Buy-hold base",            "#888888", 1.3, 2),
]


def _max_drawdown_series(equity: pd.Series) -> pd.Series:
    peaks = equity.cummax()
    return equity / peaks - 1.0


def _cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def _shade_cash_spans(ax, df: pd.DataFrame, col: str, color: str, label: str) -> None:
    """Shade contiguous CASH stretches for a given holding column."""
    if col not in df.columns:
        return
    in_cash = df[col] == "CASH"
    if not in_cash.any():
        return
    runs = in_cash.ne(in_cash.shift()).cumsum()
    first_label_used = False
    for _, group in df[in_cash].groupby(runs[in_cash]):
        lbl = label if not first_label_used else None
        ax.axvspan(group.index[0], group.index[-1], color=color, alpha=0.15, label=lbl)
        first_label_used = True


def plot(csv_path: Path, out_path: Path, show: bool) -> None:
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # Shade tactical cash periods (macro cash periods overlap and would clutter)
    _shade_cash_spans(ax_eq, df, "holding_tactical", "#f0c36d", "Tactical in cash")

    # Equity curves (log y)
    for col, label, color, lw, z in SERIES:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        cagr = _cagr(s)
        mdd = _max_drawdown_series(s).min() * 100
        ax_eq.plot(s.index, s.values, color=color, linewidth=lw, zorder=z,
                   label=f"{label}  (CAGR {cagr*100:.1f}%, MDD {mdd:.1f}%, "
                         f"final ${s.iloc[-1]:,.0f})")

    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("Equity ($, log scale)")
    start, end = df.index[0], df.index[-1]
    ax_eq.set_title(
        f"Strategy comparison  |  {start.date()} → {end.date()}  |  "
        f"start ${df['strategy_tactical'].iloc[0]:,.0f}"
    )
    ax_eq.grid(True, which="both", axis="y", alpha=0.25)
    ax_eq.grid(True, which="major", axis="x", alpha=0.25)
    ax_eq.legend(loc="upper left", framealpha=0.95, fontsize=9)

    # Drawdown panel
    for col, label, color, lw, z in SERIES:
        if col not in df.columns:
            continue
        dd = _max_drawdown_series(df[col].dropna()) * 100
        ax_dd.plot(dd.index, dd.values, color=color, linewidth=1.0, zorder=z)
        ax_dd.fill_between(dd.index, dd.values, 0, color=color, alpha=0.12, zorder=z)
    ax_dd.axhline(0, color="black", linewidth=0.6)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(True, alpha=0.25)

    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
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
