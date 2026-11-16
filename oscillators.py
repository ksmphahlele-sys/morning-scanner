"""
RSI / Stochastic research. Same method as research.py: rules fixed up front, choose on 2017-2021,
judge on 2022-2026, and every variant must beat a RANDOM-entry control that uses the same exits.

Trend filter used by all buys: close > 200 EMA and 200 EMA rising (vs 20 candles ago).

  S1  50-day breakout + RSI(14) > 60                         -> trend exits
  S2  50-day breakout + Stoch %K > %D and %K > 50            -> trend exits
  S3  Uptrend + Stoch(14,3) %K crosses up through 20         -> trend exits
  S4  Uptrend + RSI(14) crosses up through 40                -> trend exits
  S5  Uptrend + RSI(2) < 10 (classic dip-buy)                -> dip exits
  Z-trend : random buys in the uptrend, trend exits  (control for S1-S4)
  Z-dip   : random buys in the uptrend, dip exits    (control for S5)

Trend exits: 2 ATR starting stop, 3 ATR trailing stop, 60-candle max hold.
Dip exits  : 2 ATR stop, sell at the close when close > 5-day average, or after 10 candles.
"""

import numpy as np
import pandas as pd

from report import publish
from research import SEEDS, SPLIT, cell, simulate, stats
from scanner import WATCHLIST, add_indicators, fetch

COST_R = 0.05


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    c, h, l = df["Close"], df["High"], df["Low"]
    df["rsi14"], df["rsi2"] = rsi(c, 14), rsi(c, 2)
    raw_k = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min())
    df["k"] = raw_k.rolling(3).mean()
    df["d"] = df["k"].rolling(3).mean()
    df["sma5"] = c.rolling(5).mean()
    df["uptrend"] = (c > df["ema200"]) & (df["ema200"] > df["ema200"].shift(20))
    df["breakout"] = c > h.rolling(50).max().shift(1)
    return df


def entries(df: pd.DataFrame, kind: str, seed: int = 0) -> np.ndarray:
    up = df["uptrend"]
    if kind == "S1":
        m = up & df["breakout"] & (df["rsi14"] > 60)
    elif kind == "S2":
        m = up & df["breakout"] & (df["k"] > df["d"]) & (df["k"] > 50)
    elif kind == "S3":
        m = up & (df["k"] > 20) & (df["k"].shift(1) <= 20)
    elif kind == "S4":
        m = up & (df["rsi14"] > 40) & (df["rsi14"].shift(1) <= 40)
    elif kind == "S5":
        m = up & (df["rsi2"] < 10)
    else:  # random controls
        rng = np.random.default_rng(seed)
        m = up & pd.Series(rng.random(len(df)) < 0.05, index=df.index)
    return np.where(m.fillna(False).to_numpy(), "LONG", "")


def simulate_dip(df: pd.DataFrame, sig: np.ndarray) -> list:
    o, l, c = (df[k].to_numpy() for k in ("Open", "Low", "Close"))
    atr, sma5 = df["atr"].to_numpy(), df["sma5"].to_numpy()
    trades, i, n = [], 210, len(df)
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        entry, risk = o[i + 1], 2.0 * atr[i]
        stop, px, j = entry - risk, None, i + 1
        while j < n and j <= i + 10:
            if l[j] <= stop:
                px = min(o[j], stop)
                break
            if c[j] > sma5[j]:
                px = c[j]
                break
            j += 1
        if px is None:
            if j >= n:
                break
            j = i + 10
            px = c[j]
        trades.append({"date": df.index[i + 1], "R": (px - entry) / risk - COST_R})
        i = j + 1
    return trades


VARIANTS = [
    ("S1. Breakout + RSI(14) > 60", "S1", "trend"),
    ("S2. Breakout + Stoch %K>%D, %K>50", "S2", "trend"),
    ("S3. Uptrend + Stoch %K crosses up 20", "S3", "trend"),
    ("S4. Uptrend + RSI(14) crosses up 40", "S4", "trend"),
    ("S5. Uptrend + RSI(2) < 10 dip-buy", "S5", "dip"),
    ("Z-trend. CONTROL: random uptrend buys, trend exits", "Z", "trend"),
    ("Z-dip. CONTROL: random uptrend buys, dip exits", "Z", "dip"),
]


def run(df, kind, exits, seed=0):
    sig = entries(df, kind, seed)
    if exits == "trend":
        return simulate(df, sig, 2.0, None, 3.0, 60)
    return simulate_dip(df, sig)


def main():
    data = {}
    for ticker, name in WATCHLIST.items():
        try:
            data[name] = prepare(fetch(ticker, period="10y"))
        except Exception as e:
            print(name, "ERROR", e)

    results = {}
    for vname, kind, exits in VARIANTS:
        rows = []
        for seed in (SEEDS if kind == "Z" else [0]):
            for name, df in data.items():
                for t in run(df, kind, exits, seed):
                    t["market"] = name
                    rows.append(t)
        t = pd.DataFrame(rows, columns=["date", "R", "market"])
        t["date"] = pd.to_datetime(t["date"], utc=True)
        oos = t[t.date >= SPLIT]
        results[vname] = (exits, stats(t[t.date < SPLIT]), stats(oos), oos)
        print(vname, results[vname][1], results[vname][2])

    control = {e: r for v, r in results.items() if v.startswith("Z") for e in [r[0]]}
    head = "| Variant | Trades | Win | Avg | PF | Total | Max DD |\n|---|---:|---:|---:|---:|---:|---:|"
    lines = ["## In-sample: 2017–2021 (used to choose)", "", head]
    lines += [f"| {v} | {cell(r[1])} |" for v, r in results.items()]
    lines += ["", "## Out-of-sample: 2022–2026 (never used to choose)", "", head]
    lines += [f"| {v} | {cell(r[2])} |" for v, r in results.items()]

    # Eligible: in-sample PF >= 1.2, 50+ trades, and beats its own control in-sample too.
    eligible = [(v, r) for v, r in results.items() if not v.startswith("Z") and r[1]["pf"] >= 1.2
                and r[1]["n"] >= 50 and r[1]["avg"] > control[r[0]][1]["avg"] + 0.05]
    lines += ["", "## Verdict", ""]
    if eligible:
        v, r = max(eligible, key=lambda x: x[1][1]["avg"] - control[x[1][0]][1]["avg"])
        o, bar = r[2], control[r[0]][2]["avg"] + 0.05
        sd = r[3].R.std() if len(r[3]) > 1 else float("inf")
        tscore = (o["avg"] - control[r[0]][2]["avg"]) / (sd / np.sqrt(max(o["n"], 1)))
        ok = o["avg"] > 0 and o["pf"] >= 1.2 and o["avg"] > bar and tscore >= 2
        verdict = "✅ PASSED" if ok else "❌ FAILED"
        lines += [f"Chosen in-sample: **{v}**.",
                  f"Out-of-sample: avg {o['avg']:+.2f}R, PF {o['pf']:.2f}, {o['n']} trades; must beat its random "
                  f"control by 0.05R (bar {bar:+.2f}R) and be statistically clear of luck (t-score {tscore:.1f}, "
                  f"needs 2.0+) -> **{verdict}**."]
        title = f"RSI/Stochastic research: {v} {verdict}"
    else:
        lines += ["❌ No RSI/Stochastic variant beat its random control in-sample. "
                  "The oscillators add no measurable edge on these daily markets."]
        title = "RSI/Stochastic research: no variant beat random entries"
    lines += ["", "_Control trade counts are pooled over 10 random seeds. 0.05R costs per trade. "
              "1R = amount risked per trade._"]
    publish(title, "\n".join(lines))


if __name__ == "__main__":
    main()
