"""
Backtest of the Morning Scanner strategy over ~10 years of daily candles.

Rules tested exactly as the scanner flags them:
  - signal on a closed candle, enter at the NEXT candle's open
  - stop 1.5 ATR, target 3 ATR (2R), exit at close after 20 candles if neither is hit
  - if stop and target are both touched in one candle, assume the stop (conservative)
  - gaps past the stop/target fill at the open
  - one trade at a time per market, costs of 0.05R per trade deducted
"""

import pandas as pd

from report import publish
from scanner import STOP_ATR, TARGET_ATR, WATCHLIST, add_indicators, fetch

MAX_HOLD = 20
COST_R = 0.05


def run(df: pd.DataFrame) -> list:
    df = add_indicators(df)
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr, sig = df["atr"].to_numpy(), df["signal"].to_numpy()
    trades, i, n = [], 210, len(df)
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        d = 1 if sig[i] == "LONG" else -1
        entry, risk = o[i + 1], STOP_ATR * atr[i]
        stop, target = entry - d * risk, entry + d * TARGET_ATR * atr[i]
        px, j = None, i + 1
        while j < n and j <= i + MAX_HOLD:
            if d == 1:
                if l[j] <= stop:
                    px = min(o[j], stop)
                elif h[j] >= target:
                    px = max(o[j], target)
            else:
                if h[j] >= stop:
                    px = max(o[j], stop)
                elif l[j] <= target:
                    px = min(o[j], target)
            if px is not None:
                break
            j += 1
        if px is None:
            if j >= n:          # trade still open at the end of the data
                break
            j = i + MAX_HOLD
            px = c[j]
        trades.append({"date": df.index[i + 1], "side": sig[i],
                       "R": d * (px - entry) / risk - COST_R})
        i = j + 1
    return trades


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win": 0, "avg": 0, "total": 0, "pf": 0, "dd": 0}
    wins, losses = t.R[t.R > 0].sum(), -t.R[t.R <= 0].sum()
    eq = t.sort_values("date").R.cumsum()
    return {"trades": len(t), "win": (t.R > 0).mean() * 100, "avg": t.R.mean(), "total": t.R.sum(),
            "pf": wins / losses if losses else float("inf"), "dd": (eq.cummax() - eq).max()}


def row(name, s):
    return (f"| {name} | {s['trades']} | {s['win']:.0f}% | {s['avg']:+.2f}R | {s['total']:+.1f}R | "
            f"{s['pf']:.2f} | {s['dd']:.1f}R |")


def main():
    all_trades, lines = [], ["## By market", "",
                             "| Market | Trades | Win rate | Avg per trade | Total | Profit factor | Max drawdown |",
                             "|---|---:|---:|---:|---:|---:|---:|"]
    for ticker, name in WATCHLIST.items():
        try:
            t = pd.DataFrame(run(fetch(ticker, period="10y")))
            if not t.empty:
                t["market"] = name
                all_trades.append(t)
            lines.append(row(name, stats(t)))
            print(name, stats(t))
        except Exception as e:
            lines.append(f"| {name} | error: {e} | | | | | |")

    t = pd.concat(all_trades) if all_trades else pd.DataFrame(columns=["date", "side", "R", "market"])
    s = stats(t)
    lines.append(row("**All markets**", s))

    if not t.empty:
        t["year"] = pd.to_datetime(t["date"], utc=True).dt.year
        lines += ["", "## By year (all markets)", "", "| Year | Trades | Win rate | Total |", "|---|---:|---:|---:|"]
        for y, g in t.groupby("year"):
            lines.append(f"| {y} | {len(g)} | {(g.R > 0).mean() * 100:.0f}% | {g.R.sum():+.1f}R |")
        lines += ["", "## Longs vs shorts", "", "| Side | Trades | Win rate | Avg | Total |", "|---|---:|---:|---:|---:|"]
        for side, g in t.groupby("side"):
            lines.append(f"| {side} | {len(g)} | {(g.R > 0).mean() * 100:.0f}% | {g.R.mean():+.2f}R | {g.R.sum():+.1f}R |")

    lines += ["", "**How to read this:** 1R = the amount you risk per trade. Risking 1% of the account per trade, "
              "+1R is about +1%. A positive average R with a profit factor above about 1.3 over 100+ trades is "
              "the minimum worth trading. Past results do not guarantee future results.",
              "", "_Rules: next-open entry, 1.5 ATR stop, 3 ATR target, 20-candle time exit, 0.05R costs._"]
    title = f"Backtest (10y): {s['trades']} trades, {s['avg']:+.2f}R avg, PF {s['pf']:.2f}"
    publish(title, "\n".join(lines))


if __name__ == "__main__":
    main()
