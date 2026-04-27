"""Daily leverage decision engine for a (base, levered) ticker pair.

Two modes:
  - live:     evaluate the most recent close and return next-session action
              for the tactical strategy (levered ↔ base ↔ cash).
  - backtest: walk a historical window day-by-day and compare two strategies
              vs two buy-and-hold benchmarks:
                * macro   : levered ↔ cash (follows blackouts only)
                * tactical: levered ↔ base ↔ cash (RSI/MACD/distribution/FTD)
                * buy-and-hold levered
                * buy-and-hold base
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parent
BLACKOUTS_CSV = REPO_ROOT / "data" / "QLD_Blackouts_Tracking.csv"


# -- Indicators ---------------------------------------------------------------

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def distribution_days(df: pd.DataFrame, window: int = 25) -> pd.Series:
    pct_change = df["Close"].pct_change()
    higher_volume = df["Volume"] > df["Volume"].shift(1)
    is_dist = (pct_change < -0.002) & higher_volume
    return is_dist.rolling(window).sum()


def follow_through_days(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].to_numpy()
    pct_change = df["Close"].pct_change().to_numpy()
    vol_up = (df["Volume"] > df["Volume"].shift(1)).to_numpy()

    ftd = np.zeros(len(df), dtype=bool)
    running_low = np.inf
    days_since_low = 0
    for i in range(len(df)):
        price = close[i]
        if price < running_low:
            running_low = price
            days_since_low = 0
        else:
            days_since_low += 1
        if days_since_low >= 4 and i > 0:
            if pct_change[i] > 0.012 and vol_up[i]:
                ftd[i] = True
    return pd.Series(ftd, index=df.index)


# -- Data helpers -------------------------------------------------------------

def _download(ticker: str, *, period: Optional[str] = None,
              start=None, end=None) -> pd.DataFrame:
    if period is not None:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
    else:
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No Yahoo Finance data for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def _load_blackouts_map(csv_path: Path) -> dict[pd.Timestamp, str]:
    df = pd.read_csv(csv_path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[date_col] = df[date_col].dt.normalize()
    return dict(zip(df[date_col], df["Macro_State"].astype(str)))


def _load_blackouts_full(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df[date_col] = df[date_col].dt.normalize()
    df = df.rename(columns={date_col: "date"})
    return df


# -- Decision rules -----------------------------------------------------------

def _apply_tactical(
    prior_holding: str,
    *,
    rsi_val: float,
    max_rsi_20: float,
    dist_count: float,
    macd_hist_today: float,
    macd_hist_prev: float,
    ftd_in_last_5: bool,
    macro_state: str,
    base: str,
    levered: str,
) -> str:
    """Tactical strategy: levered ↔ base ↔ cash."""
    if "blackout" in macro_state.lower():
        return "CASH"
    if prior_holding == "CASH":
        return levered

    macd_cross_down = macd_hist_prev >= 0 and macd_hist_today < 0
    macd_cross_up = macd_hist_prev <= 0 and macd_hist_today > 0

    if prior_holding == levered:
        if max_rsi_20 > 60 and dist_count >= 4 and macd_cross_down:
            return base
        return levered
    if prior_holding == base:
        if rsi_val > 45 and ftd_in_last_5 and macd_cross_up:
            return levered
        return base
    return prior_holding


def _apply_macro(macro_state: str, levered: str) -> str:
    """Macro-only strategy: levered ↔ cash."""
    return "CASH" if "blackout" in macro_state.lower() else levered


# -- Live mode ----------------------------------------------------------------

@dataclass
class Decision:
    base: str
    levered: str
    evaluated_close: pd.Timestamp
    apply_on_open: str
    current_holding: str
    macro_state: str
    target_holding: str
    action: str
    reasoning: list[str]
    indicators: dict
    warnings: list[str]


def _decide_live(base: str, levered: str, period: str) -> Decision:
    warnings: list[str] = []
    if period in {"1mo", "5d", "1d"}:
        warnings.append(
            f"period={period!r} is too short for RSI(14)/MACD(26)/25d distribution; "
            "indicators will be unreliable. Recommended: '6mo' or '1y'."
        )

    df = _download(base, period=period)
    if len(df) < 60:
        warnings.append(f"Only {len(df)} bars of {base} data; indicators may be noisy.")

    close = df["Close"]
    rsi_s = rsi(close, 14)
    hist_s = macd_histogram(close)
    dist_s = distribution_days(df, 25)
    ftd_s = follow_through_days(df)

    latest_close = df.index[-1]

    blackouts_df = _load_blackouts_full(BLACKOUTS_CSV)
    last_row = blackouts_df.iloc[-1]
    holding = str(last_row["Holdings"]).strip().upper()
    macro = str(last_row["Macro_State"]).strip()
    csv_as_of = last_row["date"]
    gap = len(pd.bdate_range(csv_as_of, latest_close)) - 1
    if gap > 0:
        warnings.append(
            f"Blackouts CSV last updated {csv_as_of.date()} — "
            f"{gap} trading day(s) behind latest close ({latest_close.date()})."
        )

    last_rsi = float(rsi_s.iloc[-1])
    max_rsi_20 = float(rsi_s.iloc[-20:].max())
    last_dist = float(dist_s.iloc[-1])
    last_hist = float(hist_s.iloc[-1])
    prev_hist = float(hist_s.iloc[-2])
    ftd_last_5 = bool(ftd_s.iloc[-5:].any())

    target = _apply_tactical(
        prior_holding=holding,
        rsi_val=last_rsi,
        max_rsi_20=max_rsi_20,
        dist_count=last_dist,
        macd_hist_today=last_hist,
        macd_hist_prev=prev_hist,
        ftd_in_last_5=ftd_last_5,
        macro_state=macro,
        base=base,
        levered=levered,
    )

    if target == holding:
        action = f"HOLD {holding}"
    elif target == "CASH":
        action = f"SELL {holding} / GO TO CASH"
    elif holding == "CASH":
        action = f"BUY {target}"
    else:
        action = f"SELL {holding} / BUY {target}"

    reasoning = _build_reasoning(
        holding, macro, last_rsi, max_rsi_20, last_dist,
        last_hist, prev_hist, ftd_last_5, base, levered,
    )

    indicators = {
        "rsi_14": round(last_rsi, 2),
        "max_rsi_last_20d": round(max_rsi_20, 2),
        "distribution_days_25d": int(last_dist),
        "macd_hist_today": round(last_hist, 5),
        "macd_hist_prev": round(prev_hist, 5),
        "macd_cross_down_today": prev_hist >= 0 and last_hist < 0,
        "macd_cross_up_today": prev_hist <= 0 and last_hist > 0,
        "follow_through_day_last_5d": ftd_last_5,
    }

    next_session = (latest_close + pd.tseries.offsets.BDay(1)).normalize()
    return Decision(
        base=base,
        levered=levered,
        evaluated_close=latest_close,
        apply_on_open=str(next_session.date()),
        current_holding=holding,
        macro_state=macro,
        target_holding=target,
        action=action,
        reasoning=reasoning,
        indicators=indicators,
        warnings=warnings,
    )


def _build_reasoning(holding, macro, rsi_val, max_rsi_20, dist_count,
                     hist_today, hist_prev, ftd_last_5, base, levered) -> list[str]:
    r: list[str] = []
    if "blackout" in macro.lower():
        r.append(f"Macro regime = '{macro}' → 100% cash.")
        return r
    if holding == "CASH":
        r.append(f"Macro regime = Invest and holding is CASH → default to {levered}.")
        return r
    if holding == levered:
        r.append(f"Holding {levered}. Exhaustion (max RSI last 20d > 60): "
                 f"{max_rsi_20 > 60} (max={max_rsi_20:.2f}).")
        r.append(f"Distribution (25d count ≥ 4): {dist_count >= 4} (count={int(dist_count)}).")
        r.append(f"MACD cross down today: {hist_prev >= 0 and hist_today < 0} "
                 f"(prev={hist_prev:.5f}, today={hist_today:.5f}).")
    elif holding == base:
        r.append(f"Holding {base}. Stabilization (RSI > 45): {rsi_val > 45} "
                 f"(RSI={rsi_val:.2f}).")
        r.append(f"Follow-Through Day within last 5d: {ftd_last_5}.")
        r.append(f"MACD cross up today: {hist_prev <= 0 and hist_today > 0} "
                 f"(prev={hist_prev:.5f}, today={hist_today:.5f}).")
    return r


# -- Backtest mode ------------------------------------------------------------

@dataclass
class CurveStats:
    name: str
    final_equity: float
    cagr: float
    max_drawdown: float


@dataclass
class BacktestResult:
    base: str
    levered: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    curves: dict[str, CurveStats]
    equity_curve: pd.DataFrame = field(repr=False)
    warnings: list[str] = field(default_factory=list)


def _cagr(initial: float, final: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    years = (end - start).days / 365.25
    if years <= 0 or initial <= 0:
        return float("nan")
    return (final / initial) ** (1 / years) - 1


def _max_drawdown(equity) -> float:
    arr = np.asarray(equity, dtype=float)
    if len(arr) == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    dd = arr / peaks - 1.0
    return float(dd.min())


def _curve_stats(name: str, equity: np.ndarray,
                 start: pd.Timestamp, end: pd.Timestamp) -> CurveStats:
    return CurveStats(
        name=name,
        final_equity=float(equity[-1]),
        cagr=_cagr(float(equity[0]), float(equity[-1]), start, end),
        max_drawdown=_max_drawdown(equity),
    )


def _decide_backtest(
    base: str,
    levered: str,
    start: Union[str, pd.Timestamp, None],
    end: Union[str, pd.Timestamp, None],
    initial_capital: float,
) -> BacktestResult:
    warnings: list[str] = []

    start_ts = pd.Timestamp(start).normalize() if start else pd.Timestamp("2018-01-02")
    end_ts = pd.Timestamp(end).normalize() if end else pd.Timestamp.today().normalize()

    download_start = (start_ts - pd.DateOffset(months=4)).strftime("%Y-%m-%d")
    download_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    base_df = _download(base, start=download_start, end=download_end)
    lev_df = _download(levered, start=download_start, end=download_end)

    close = base_df["Close"]
    rsi_s = rsi(close, 14)
    hist_s = macd_histogram(close)
    hist_prev_s = hist_s.shift(1)
    dist_s = distribution_days(base_df, 25)
    ftd_s = follow_through_days(base_df)
    max_rsi_20_s = rsi_s.rolling(20).max()
    ftd_in_5_s = ftd_s.rolling(5).sum() > 0
    base_ret_s = close.pct_change()
    lev_ret_s = lev_df["Close"].pct_change()

    idx_all = base_df.index.intersection(lev_df.index)
    idx = idx_all[(idx_all >= start_ts) & (idx_all <= end_ts)]
    if len(idx) < 2:
        raise ValueError(f"Backtest period too short after data alignment: {len(idx)} bar(s).")

    blackouts = _load_blackouts_map(BLACKOUTS_CSV)
    missing_blackouts = [d for d in idx if d not in blackouts]
    if missing_blackouts:
        warnings.append(
            f"{len(missing_blackouts)} backtest day(s) have no blackouts CSV entry; "
            f"defaulting those to 'Invest'. First: {missing_blackouts[0].date()}, "
            f"last: {missing_blackouts[-1].date()}."
        )

    first_macro = blackouts.get(idx[0], "Invest")
    first_holding = "CASH" if "blackout" in first_macro.lower() else levered

    holdings_macro: list[str] = [first_holding]
    holdings_tact: list[str] = [first_holding]
    eq_macro: list[float] = [float(initial_capital)]
    eq_tact: list[float] = [float(initial_capital)]

    for i in range(1, len(idx)):
        prev_d = idx[i - 1]
        cur_d = idx[i]
        macro_at_prev = blackouts.get(prev_d, "Invest")

        # Strategy 1: macro only
        new_macro = _apply_macro(macro_at_prev, levered)
        holdings_macro.append(new_macro)

        # Strategy 2: tactical
        rsi_val = float(rsi_s.loc[prev_d]) if pd.notna(rsi_s.loc[prev_d]) else 50.0
        mx20 = float(max_rsi_20_s.loc[prev_d]) if pd.notna(max_rsi_20_s.loc[prev_d]) else rsi_val
        dc = float(dist_s.loc[prev_d]) if pd.notna(dist_s.loc[prev_d]) else 0.0
        h_today = float(hist_s.loc[prev_d]) if pd.notna(hist_s.loc[prev_d]) else 0.0
        h_prev = float(hist_prev_s.loc[prev_d]) if pd.notna(hist_prev_s.loc[prev_d]) else 0.0
        ftd5 = bool(ftd_in_5_s.loc[prev_d]) if pd.notna(ftd_in_5_s.loc[prev_d]) else False

        new_tact = _apply_tactical(
            prior_holding=holdings_tact[-1],
            rsi_val=rsi_val, max_rsi_20=mx20, dist_count=dc,
            macd_hist_today=h_today, macd_hist_prev=h_prev,
            ftd_in_last_5=ftd5, macro_state=macro_at_prev,
            base=base, levered=levered,
        )
        holdings_tact.append(new_tact)

        def _ret(holding: str) -> float:
            if holding == "CASH":
                return 0.0
            if holding == base:
                return float(base_ret_s.loc[cur_d])
            return float(lev_ret_s.loc[cur_d])

        eq_macro.append(eq_macro[-1] * (1 + _ret(new_macro)))
        eq_tact.append(eq_tact[-1] * (1 + _ret(new_tact)))

    # Benchmarks
    lev_prices = lev_df["Close"].loc[idx]
    base_prices = base_df["Close"].loc[idx]
    bh_lev = (initial_capital * (lev_prices / lev_prices.iloc[0])).values
    bh_base = (initial_capital * (base_prices / base_prices.iloc[0])).values

    curve = pd.DataFrame({
        "holding_macro": holdings_macro,
        "holding_tactical": holdings_tact,
        "strategy_macro": eq_macro,
        "strategy_tactical": eq_tact,
        "buy_hold_levered": bh_lev,
        "buy_hold_base": bh_base,
    }, index=idx)

    stats = {
        "strategy_macro":     _curve_stats("Macro (lev↔cash)",      np.array(eq_macro), idx[0], idx[-1]),
        "strategy_tactical":  _curve_stats("Tactical (lev↔base↔cash)", np.array(eq_tact), idx[0], idx[-1]),
        "buy_hold_levered":   _curve_stats(f"Buy-hold {levered}",   bh_lev,  idx[0], idx[-1]),
        "buy_hold_base":      _curve_stats(f"Buy-hold {base}",      bh_base, idx[0], idx[-1]),
    }

    return BacktestResult(
        base=base, levered=levered,
        start_date=idx[0], end_date=idx[-1],
        initial_capital=float(initial_capital),
        curves=stats, equity_curve=curve, warnings=warnings,
    )


# -- Public entry point -------------------------------------------------------

def decide(
    base: str = "QQQ",
    levered: str = "QLD",
    mode: str = "live",
    *,
    period: str = "6mo",
    start: Union[str, pd.Timestamp, None] = None,
    end: Union[str, pd.Timestamp, None] = None,
    initial_capital: float = 100_000.0,
) -> Union[Decision, BacktestResult]:
    """Run the strategy on an explicit (base, levered) pair.

    mode="live"     → Decision for the next trading session (tactical strategy).
    mode="backtest" → BacktestResult comparing macro, tactical, and two
                       buy-and-hold benchmarks over [start, end].
    """
    base = base.upper()
    levered = levered.upper()
    mode = mode.lower()
    if mode == "live":
        return _decide_live(base, levered, period)
    if mode == "backtest":
        return _decide_backtest(base, levered, start, end, initial_capital)
    raise ValueError(f"Unknown mode {mode!r}. Use 'live' or 'backtest'.")


# -- Formatting ---------------------------------------------------------------

def format_decision(d: Decision) -> str:
    lines = [
        f"Ticker pair:        {d.base} (1x) / {d.levered} (2x)",
        f"Evaluated close:    {d.evaluated_close.date()}",
        f"Applies on open:    {d.apply_on_open}",
        f"Macro regime:       {d.macro_state}",
        f"Current holding:    {d.current_holding}",
        f"Target holding:     {d.target_holding}",
        f"Action:             {d.action}",
        "",
        "Indicators:",
    ]
    for k, v in d.indicators.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Reasoning:")
    for r in d.reasoning:
        lines.append(f"  - {r}")
    if d.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in d.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def format_backtest(r: BacktestResult) -> str:
    lines = [
        f"Ticker pair:        {r.base} (1x) / {r.levered} (2x)",
        f"Period:             {r.start_date.date()} → {r.end_date.date()}",
        f"Initial capital:    ${r.initial_capital:,.2f}",
        "",
        f"{'Strategy':<38} {'Final':>15} {'CAGR':>8} {'MaxDD':>8}",
        "-" * 72,
    ]
    for key in ("strategy_tactical", "strategy_macro", "buy_hold_levered", "buy_hold_base"):
        c = r.curves[key]
        lines.append(
            f"{c.name:<38} ${c.final_equity:>13,.0f} {c.cagr*100:>7.2f}% {c.max_drawdown*100:>7.2f}%"
        )
    lines.append("")
    lines.append("Time in each holding (tactical):")
    for k, v in (r.equity_curve["holding_tactical"].value_counts(normalize=True) * 100).items():
        lines.append(f"  {k}: {v:.1f}%")
    lines.append("")
    lines.append("Time in each holding (macro):")
    for k, v in (r.equity_curve["holding_macro"].value_counts(normalize=True) * 100).items():
        lines.append(f"  {k}: {v:.1f}%")
    if r.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in r.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


# -- CLI ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="QQQ", help="Unlevered (1x) ticker (default: QQQ)")
    parser.add_argument("--levered", default="QLD", help="Levered (2x) ticker (default: QLD)")
    parser.add_argument("--mode", choices=["live", "backtest"], default="live")
    parser.add_argument("--period", default="6mo", help="Yahoo period for live mode (default: 6mo)")
    parser.add_argument("--start", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Initial capital for backtest (default: 100000)")
    parser.add_argument("--out", help="Write backtest equity curve CSV to this path")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    if args.mode == "live":
        d = decide(args.base, args.levered, mode="live", period=args.period)
        if args.json:
            import json
            payload = asdict(d)
            payload["evaluated_close"] = str(d.evaluated_close.date())
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(format_decision(d))
    else:
        r = decide(
            args.base, args.levered, mode="backtest",
            start=args.start, end=args.end, initial_capital=args.capital,
        )
        if args.out:
            r.equity_curve.to_csv(args.out, index_label="date")
        if args.json:
            import json
            payload = {
                "base": r.base, "levered": r.levered,
                "start_date": str(r.start_date.date()),
                "end_date": str(r.end_date.date()),
                "initial_capital": r.initial_capital,
                "curves": {k: asdict(v) for k, v in r.curves.items()},
                "warnings": r.warnings,
            }
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(format_backtest(r))


if __name__ == "__main__":
    main()
