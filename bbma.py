"""
BBMA research (Bollinger Bands 20/2 + LWMA 5/10 of highs & lows + EMA 50), daily candles.
Rules are fixed up front and tested with the same in-sample / out-of-sample split as research.py.

BUY setups (sells are the mirror image):
  Extreme  : LWMA5 or LWMA10 of the lows closes below the lower band (setup candle)
  Re-entry : close > EMA50, BB middle rising, a close above the upper band (CSA) within the
             previous 10 candles, and the setup candle's low dips into the LWMA5/10-low zone
             while closing above LWMA10-low
  Liquidity: (optional) the setup candle's low also sweeps below the lowest low of the prior 5 candles
  Confirmation: the NEXT candle closes above the setup candle's high and is bullish
Trade: enter at the open after confirmation; stop 0.1 ATR beyond the lowest low of setup + confirmation
(minimum 0.5 ATR); target 2R; exit at close after 20 candles otherwise; 0.05R costs.
"""

import numpy as np
import pandas as pd

from report import publish
from research import SPLIT, cell, stats
from scanner import WATCHLIST, add_indicators, fetch

COST_R = 0.05
TARGET_R = 2.0
MAX_HOLD = 20

VARIANTS = [
    # name, setup, sides, liquidity
    ("G. Extreme + confirmation (buys)", "extreme", "long", False),
    ("H. Extreme + confirmation (buys & sells)", "extreme", "both", False),
    ("I. Re-entry + confirmation (buys)", "reentry", "long", False),
    ("J. Re-entry + confirmation + liquidity sweep (buys)", "reentry", "long", True),
    ("K. Re-entry + confirmation (buys & sells)", "reentry", "both", False),
    ("Z. CONTROL: random buys, same exits", "random", "long", False),
]


def lwma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def bbma(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    df["bb_up"], df["bb_mid"], df["bb_lo"] = mid + 2 * sd, mid, mid - 2 * sd
    df["lo5"], df["lo10"] = lwma(l, 5), lwma(l, 10)
    df["hi5"], df["hi10"] = lwma(h, 5), lwma(h, 10)
    ema50 = df["ema50"]

    ext_buy = (df["lo5"] < df["bb_lo"]) | (df["lo10"] < df["bb_lo"])
    ext_sell = (df["hi5"] > df["bb_up"]) | (df["hi10"] > df["bb_up"])

    csa_up = (c > df["bb_up"]).astype(float).rolling(10).max().shift(1) > 0
    csa_dn = (c < df["bb_lo"]).astype(float).rolling(10).max().shift(1) > 0
    re_buy = ((c > ema50) & (mid > mid.shift(5)) & csa_up
              & (l <= df[["lo5", "lo10"]].max(axis=1)) & (c > df["lo10"]))
    re_sell = ((c < ema50) & (mid < mid.shift(5)) & csa_dn
               & (h >= df[["hi5", "hi10"]].min(axis=1)) & (c < df["hi10"]))

    df["sweep_lo"] = l < l.rolling(5).min().shift(1)
    df["sweep_hi"] = h > h.rolling(5).max().shift(1)
    df["conf_buy"] = (c > h.shift(1)) & (c > o)
    df["conf_sell"] = (c < l.shift(1)) & (c < o)
    df["ext_buy"], df["ext_sell"], df["re_buy"], df["re_sell"] = ext_buy, ext_sell, re_buy, re_sell
    return df


def signals(df, setup, sides, liquidity):
    if setup == "random":  # control: random buy days (fixed seed), same frequency as a typical setup
        rng = np.random.default_rng(42)
        return np.where(rng.random(len(df)) < 0.04, "LONG", "")
    buy = df[f"{'ext' if setup == 'extreme' else 're'}_buy"]
    sell = df[f"{'ext' if setup == 'extreme' else 're'}_sell"]
    if liquidity:
        buy, sell = buy & df["sweep_lo"], sell & df["sweep_hi"]
    long_sig = buy.shift(1, fill_value=False) & df["conf_buy"]      # setup yesterday, confirmed today
    short_sig = sell.shift(1, fill_value=False) & df["conf_sell"]
    if sides == "long":
        short_sig = short_sig & False
    return np.where(long_sig, "LONG", np.where(short_sig, "SHORT", ""))


def simulate(df, sig):
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr = df["atr"].to_numpy()
    trades, i, n = [], 210, len(df)
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        d = 1 if sig[i] == "LONG" else -1
        entry = o[i + 1]
        if d == 1:
            stop = min(l[i], l[i - 1]) - 0.1 * atr[i]
        else:
            stop = max(h[i], h[i - 1]) + 0.1 * atr[i]
        risk = max(d * (entry - stop), 0.5 * atr[i])
        stop = entry - d * risk
        target = entry + d * TARGET_R * risk
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
            if j >= n:
                break
            j = i + MAX_HOLD
            px = c[j]
        trades.append({"date": df.index[i + 1], "R": d * (px - entry) / risk - COST_R})
        i = j + 1
    return trades


def main():
    data = {}
    for ticker, name in WATCHLIST.items():
        try:
            data[name] = bbma(fetch(ticker, period="10y"))
        except Exception as e:
            print(name, "ERROR", e)

    results = []
    for vname, setup, sides, liq in VARIANTS:
        rows = []
        for name, df in data.items():
            for t in simulate(df, signals(df, setup, sides, liq)):
                t["market"] = name
                rows.append(t)
        t = pd.DataFrame(rows, columns=["date", "R", "market"])
        t["date"] = pd.to_datetime(t["date"], utc=True)
        results.append((vname, stats(t[t.date < SPLIT]), stats(t[t.date >= SPLIT]), t[t.date >= SPLIT]))
        print(vname, results[-1][1], results[-1][2])

    head = "| Variant | Trades | Win | Avg | PF | Total | Max DD |\n|---|---:|---:|---:|---:|---:|---:|"
    lines = ["## In-sample: 2017–2021 (used to choose)", "", head]
    lines += [f"| {v} | {cell(i)} |" for v, i, o, _ in results]
    lines += ["", "## Out-of-sample: 2022–2026 (never used to choose)", "", head]
    lines += [f"| {v} | {cell(o)} |" for v, i, o, _ in results]

    control = next(r for r in results if r[0].startswith("Z."))
    eligible = [r for r in results if not r[0].startswith("Z.") and r[1]["pf"] >= 1.2 and r[1]["n"] >= 50]
    if eligible:
        v, i, o, oos_t = max(eligible, key=lambda r: r[1]["avg"])
        beats = o["avg"] > control[2]["avg"] + 0.05
        verdict = "✅ PASSED" if (o["avg"] > 0 and o["pf"] >= 1.2 and beats) else "❌ FAILED"
        lines += ["", "## Verdict", "", f"Chosen in-sample: **{v}**.",
                  f"Out-of-sample: avg {o['avg']:+.2f}R, PF {o['pf']:.2f} over {o['n']} trades "
                  f"(random-buy control: {control[2]['avg']:+.2f}R) -> **{verdict}**.",
                  "To pass it must be profitable (PF >= 1.2) AND beat random buys by 0.05R+ per trade.",
                  "", "| Market | Trades | Avg | Total |", "|---|---:|---:|---:|"]
        for m, g in oos_t.groupby("market"):
            lines.append(f"| {m} | {len(g)} | {g.R.mean():+.2f}R | {g.R.sum():+.1f}R |")
        title = f"BBMA research: {v} {verdict} out-of-sample (PF {o['pf']:.2f})"
    else:
        lines += ["", "## Verdict", "", "❌ No BBMA variant reached profit factor 1.2 in-sample (50+ trades)."]
        title = "BBMA research: no variant passed"

    lines += ["", "Benchmark from the earlier research: **F. 50-day breakout + trailing stop**: "
              "OOS +0.31R avg, PF 1.61, 134 trades.",
              "", "_Daily candles only: Yahoo keeps just ~2 years of hourly data, too little for a fair "
              "in/out-of-sample test on H1/H4. 1R = amount risked per trade._"]
    publish(title, "\n".join(lines))


if __name__ == "__main__":
    main()
