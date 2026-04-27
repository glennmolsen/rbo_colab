"""Daily leverage decision engine for QQQ/QLD-style pairs.

Evaluates the close of the most recent trading day and produces a
buy/sell/hold recommendation for the NEXT trading day's open, per the
algorithm specified in the project brief.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parent
BLACKOUTS_CSV = REPO_ROOT / "data" / "QLD_Blackouts_Tracking.csv"

# Base ticker -> levered equivalent. Extend as new pairs are supported.
LEVERAGE_PAIRS = {"QQQ": "QLD"}
_REVERSE_PAIRS = {v: k for k, v in LEVERAGE_PAIRS.items()}


# -- Indicators ---------------------------------------------------------------

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """14-day Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing = EMA with alpha = 1/length
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram = MACD line - signal line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def distribution_days(df: pd.DataFrame, window: int = 25) -> pd.Series:
    """Rolling count of distribution days.

    A distribution day = close down by more than 0.2% on higher volume
    than the previous trading day.
    """
    pct_change = df["Close"].pct_change()
    higher_volume = df["Volume"] > df["Volume"].shift(1)
    is_dist = (pct_change < -0.002) & higher_volume
    return is_dist.rolling(window).sum()


def follow_through_days(df: pd.DataFrame) -> pd.Series:
    """Mark follow-through days.

    Tracks the running low; on day 4+ after that low, a close up >1.2%
    on higher volume than the previous day is a follow-through day.
    A new low resets the counter.
    """
    close = df["Close"].to_numpy()
    volume = df["Volume"].to_numpy()
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


# -- Regime lookup ------------------------------------------------------------

@dataclass
class RegimeSnapshot:
    macro_state: str          # "Invest" | "Blackout period"
    current_holding: str      # "QLD" | "QQQ" | "CASH"
    as_of: pd.Timestamp
    stale_days: int           # trading days between CSV and latest market close


def load_regime(csv_path: Path, latest_market_date: pd.Timestamp) -> RegimeSnapshot:
    df = pd.read_csv(csv_path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    last = df.iloc[-1]
    as_of = pd.Timestamp(last[date_col]).normalize()

    gap = pd.bdate_range(as_of, latest_market_date)
    stale = max(0, len(gap) - 1)

    return RegimeSnapshot(
        macro_state=str(last["Macro_State"]).strip(),
        current_holding=str(last["Holdings"]).strip().upper(),
        as_of=as_of,
        stale_days=stale,
    )


# -- Decision engine ----------------------------------------------------------

@dataclass
class Decision:
    ticker: str
    base: str
    levered: str
    evaluated_close: pd.Timestamp
    apply_on_open: str
    current_holding: str
    macro_state: str
    target_holding: str
    action: str                  # e.g. "SELL QLD / BUY QQQ", "HOLD", "BUY QLD"
    reasoning: list[str]
    indicators: dict
    warnings: list[str]


def _resolve_pair(ticker: str) -> tuple[str, str]:
    t = ticker.upper()
    if t in LEVERAGE_PAIRS:
        return t, LEVERAGE_PAIRS[t]
    if t in _REVERSE_PAIRS:
        return _REVERSE_PAIRS[t], t
    raise ValueError(
        f"Unknown ticker {ticker!r}. Add it to LEVERAGE_PAIRS in decision.py."
    )


def _download(base: str, period: str) -> pd.DataFrame:
    df = yf.download(base, period=period, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {base} over {period}.")
    # yfinance can return a MultiIndex on columns when a single ticker is passed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def decide(ticker: str = "QQQ", period: str = "6mo") -> Decision:
    """Produce a buy/sell/hold recommendation for `ticker`'s pair.

    `period` is passed to yfinance (e.g. "1mo", "3mo", "6mo", "1y").
    Defaults to "6mo" so indicators have enough warm-up.
    """
    base, levered = _resolve_pair(ticker)
    warnings: list[str] = []

    if period in {"1mo", "5d", "1d"}:
        warnings.append(
            f"period={period!r} is too short for RSI(14)/MACD(26)/25-day distribution; "
            "using this will produce unreliable signals. Recommended: '6mo' or '1y'."
        )

    df = _download(base, period)
    if len(df) < 60:
        warnings.append(
            f"Only {len(df)} trading days of {base} data returned; indicators may be noisy."
        )

    close = df["Close"]
    rsi_series = rsi(close, 14)
    hist = macd_histogram(close, 12, 26, 9)
    dist_count = distribution_days(df, 25)
    ftd_series = follow_through_days(df)

    latest_close_date = df.index[-1].normalize()
    regime = load_regime(BLACKOUTS_CSV, latest_close_date)
    if regime.stale_days > 0:
        warnings.append(
            f"Blackouts CSV last updated {regime.as_of.date()} — "
            f"{regime.stale_days} trading day(s) behind latest market close "
            f"({latest_close_date.date()})."
        )

    # --- Signal evaluation (on the most recent completed bar) ---
    last_rsi = float(rsi_series.iloc[-1])
    max_rsi_20 = float(rsi_series.iloc[-20:].max())
    last_dist = float(dist_count.iloc[-1])
    last_hist = float(hist.iloc[-1])
    prev_hist = float(hist.iloc[-2])
    macd_cross_down = prev_hist >= 0 and last_hist < 0
    macd_cross_up = prev_hist <= 0 and last_hist > 0
    ftd_last_5 = bool(ftd_series.iloc[-5:].any())

    indicators = {
        "rsi_14": round(last_rsi, 2),
        "max_rsi_last_20d": round(max_rsi_20, 2),
        "distribution_days_25d": int(last_dist),
        "macd_hist_today": round(last_hist, 5),
        "macd_hist_prev": round(prev_hist, 5),
        "macd_cross_down_today": macd_cross_down,
        "macd_cross_up_today": macd_cross_up,
        "follow_through_day_last_5d": ftd_last_5,
    }

    reasoning: list[str] = []
    holding = regime.current_holding
    macro = regime.macro_state.lower()

    # Phase 1: macro gate
    if "blackout" in macro:
        target = "CASH"
        if holding == "CASH":
            action = "HOLD (already in cash — blackout regime)"
        else:
            action = f"SELL {holding} / GO TO CASH"
        reasoning.append(f"Macro regime = '{regime.macro_state}' → 100% cash.")
    else:
        # Invest regime
        if holding == "CASH":
            target = levered
            action = f"BUY {levered} (re-entering from cash under Invest regime)"
            reasoning.append(
                f"Macro regime = Invest and current holding is CASH → default to {levered}."
            )
        elif holding == levered:
            exhaustion = max_rsi_20 > 60
            distribution = last_dist >= 4
            reasoning.append(
                f"Holding {levered}. Exhaustion (max RSI last 20d > 60): {exhaustion} "
                f"(max={max_rsi_20:.2f})."
            )
            reasoning.append(
                f"Distribution (25d dist days >= 4): {distribution} "
                f"(count={int(last_dist)})."
            )
            reasoning.append(
                f"MACD histogram cross down today: {macd_cross_down} "
                f"(prev={prev_hist:.5f}, today={last_hist:.5f})."
            )
            if exhaustion and distribution and macd_cross_down:
                target = base
                action = f"SELL {levered} / BUY {base} (de-leverage)"
            else:
                target = levered
                action = f"HOLD {levered}"
        elif holding == base:
            stabilization = last_rsi > 45
            reasoning.append(
                f"Holding {base}. Stabilization (RSI > 45): {stabilization} "
                f"(RSI={last_rsi:.2f})."
            )
            reasoning.append(
                f"Follow-Through Day within last 5d: {ftd_last_5}."
            )
            reasoning.append(
                f"MACD histogram cross up today: {macd_cross_up} "
                f"(prev={prev_hist:.5f}, today={last_hist:.5f})."
            )
            if stabilization and ftd_last_5 and macd_cross_up:
                target = levered
                action = f"SELL {base} / BUY {levered} (re-leverage)"
            else:
                target = base
                action = f"HOLD {base}"
        else:
            target = holding
            action = f"HOLD {holding} (unrecognized holding; no rule applies)"
            warnings.append(
                f"Current holding {holding!r} is neither {base} nor {levered} nor CASH."
            )

    # Execution timing: decision on close of T applies to open of T+1
    next_session = (latest_close_date + pd.tseries.offsets.BDay(1)).normalize()

    return Decision(
        ticker=ticker.upper(),
        base=base,
        levered=levered,
        evaluated_close=latest_close_date,
        apply_on_open=str(next_session.date()),
        current_holding=holding,
        macro_state=regime.macro_state,
        target_holding=target,
        action=action,
        reasoning=reasoning,
        indicators=indicators,
        warnings=warnings,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="QQQ", help="Base or levered ticker (default: QQQ)")
    parser.add_argument("--period", default="6mo", help="Yahoo Finance period (default: 6mo)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    d = decide(args.ticker, args.period)
    if args.json:
        import json
        payload = asdict(d)
        payload["evaluated_close"] = str(d.evaluated_close.date())
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_decision(d))


if __name__ == "__main__":
    main()
