"""
Strategy research: tests a FIXED list of variants (decided before seeing results) with an
in-sample / out-of-sample split, so we don't fool ourselves by curve-fitting.

  In-sample  (IS):  trades entered before 2022-01-01  -> used to pick the best variant
  Out-of-sample (OOS): trades entered from 2022-01-01 -> the honest test

Pick rule (decided up front): among variants with IS profit factor >= 1.2, choose the best IS
average R. It "passes" only if OOS average R > 0 and OOS profit factor >= 1.2.
"""

import numpy as np
import pandas as pd

from report import publish
from scanner import WATCHLIST, add_indicators, fetch

SPLIT = pd.Timestamp("2022-01-01", tz="UTC")
COST_R = 0.05

VARIANTS = [
    # name, entry, sides, stop_atr, target_atr, trail_atr, max_hold, slope_filter
    ("A. Original (long+short, 2R target)", "pullback", "both", 1.5, 3.0, None, 20, False),
    ("B. Longs only", "pullback", "long", 1.5, 3.0, None, 20, False),
    ("C. Longs, wider 2 ATR stop, 2R", "pullback", "long", 2.0, 4.0, None, 30, False),
    ("D. Longs, 3 ATR trailing stop", "pullback", "long", 2.0, None, 3.0, 60, False),
    ("E. Longs, trailing + rising 200 EMA", "pullback", "long", 2.0, None, 3.0, 60, True),
    ("F. 50-day breakout, trailing (classic trend)", "breakout", "long", 2.0, None, 3.0, 60, True),
    ("Z1. CONTROL: random buys, F exits", "random", "long", 2.0, None, 3.0, 60, False),
    ("Z2. CONTROL: random buys above rising 200 EMA, F exits", "random_trend", "long", 2.0, None, 3.0, 60, True),
]
SEEDS = range(1, 11)  # controls are averaged over 10 random seeds


def signals(df: pd.DataFrame, entry: str, sides: str, slope: bool, seed: int = 0) -> np.ndarray:
    if entry.startswith("random"):
        rng = np.random.default_rng(seed)
        pick = rng.random(len(df)) < 0.05
        if entry == "random_trend":
            pick &= (df["Close"] > df["ema200"]).to_numpy()
        sig = np.where(pick, "LONG", "")
    elif entry == "pullback":
        sig = df["signal"].to_numpy().copy()
    else:
        hh = df["High"].rolling(50).max().shift(1)
        sig = np.where((df["Close"] > hh) & (df["Close"] > df["ema200"]), "LONG", "")
    if sides == "long":
        sig = np.where(sig == "LONG", "LONG", "")
    if slope:
        rising = (df["ema200"] > df["ema200"].shift(20)).to_numpy()
        falling = (df["ema200"] < df["ema200"].shift(20)).to_numpy()
        sig = np.where((sig == "LONG") & rising, "LONG", np.where((sig == "SHORT") & falling, "SHORT", ""))
    return sig


def simulate(df, sig, stop_atr, target_atr, trail_atr, max_hold):
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr = df["atr"].to_numpy()
    trades, i, n = [], 210, len(df)
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        d = 1 if sig[i] == "LONG" else -1
        entry, risk = o[i + 1], stop_atr * atr[i]
        stop = entry - d * risk
        target = entry + d * target_atr * atr[i] if target_atr else None
        best, px, j = entry, None, i + 1
        while j < n and j <= i + max_hold:
            if d == 1:
                if l[j] <= stop:
                    px = min(o[j], stop)
                elif target is not None and h[j] >= target:
                    px = max(o[j], target)
            else:
                if h[j] >= stop:
                    px = max(o[j], stop)
                elif target is not None and l[j] <= target:
                    px = min(o[j], target)
            if px is not None:
                break
            if trail_atr:  # move the stop only after the candle closes (no look-ahead)
                best = max(best, h[j]) if d == 1 else min(best, l[j])
                trail = best - d * trail_atr * atr[i]
                stop = max(stop, trail) if d == 1 else min(stop, trail)
            j += 1
        if px is None:
            if j >= n:
                break
            j = i + max_hold
            px = c[j]
        trades.append({"date": df.index[i + 1], "R": d * (px - entry) / risk - COST_R})
        i = j + 1
    return trades


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0, "win": 0.0, "avg": 0.0, "total": 0.0, "pf": 0.0, "dd": 0.0}
    w, ls = t.R[t.R > 0].sum(), -t.R[t.R <= 0].sum()
    eq = t.sort_values("date").R.cumsum()
    return {"n": len(t), "win": (t.R > 0).mean() * 100, "avg": t.R.mean(), "total": t.R.sum(),
            "pf": w / ls if ls else 99.0, "dd": float((eq.cummax() - eq).max())}


def cell(s):
    return f"{s['n']} | {s['win']:.0f}% | {s['avg']:+.2f}R | {s['pf']:.2f} | {s['total']:+.0f}R | {s['dd']:.0f}R"


def main():
    data = {}
    for ticker, name in WATCHLIST.items():
        try:
            data[name] = add_indicators(fetch(ticker, period="10y"))
        except Exception as e:
            print(name, "ERROR", e)

    results = []
    for (vname, entry, sides, s_atr, t_atr, tr_atr, hold, slope) in VARIANTS:
        rows = []
        for seed in (SEEDS if entry.startswith("random") else [0]):
            for name, df in data.items():
                for t in simulate(df, signals(df, entry, sides, slope, seed), s_atr, t_atr, tr_atr, hold):
                    t["market"] = name
                    rows.append(t)
        t = pd.DataFrame(rows, columns=["date", "R", "market"])
        t["date"] = pd.to_datetime(t["date"], utc=True)
        is_t, oos_t = t[t.date < SPLIT], t[t.date >= SPLIT]
        results.append((vname, stats(is_t), stats(oos_t), oos_t))
        print(vname, stats(is_t), stats(oos_t))

    head = "| Variant | Trades | Win | Avg | PF | Total | Max DD |\n|---|---:|---:|---:|---:|---:|---:|"
    lines = ["## In-sample: 2017–2021 (used to choose)", "", head]
    lines += [f"| {v} | {cell(i)} |" for v, i, o, _ in results]
    lines += ["", "## Out-of-sample: 2022–2026 (never used to choose)", "", head]
    lines += [f"| {v} | {cell(o)} |" for v, i, o, _ in results]

    controls = [r for r in results if r[0].startswith("Z")]
    bar = max(c[2]["avg"] for c in controls) + 0.05
    eligible = [r for r in results if not r[0].startswith("Z") and r[1]["pf"] >= 1.2 and r[1]["n"] >= 50]
    if eligible:
        best = max(eligible, key=lambda r: r[1]["avg"])
        v, i, o, oos_t = best
        passed = o["avg"] > 0 and o["pf"] >= 1.2 and o["avg"] > bar
        verdict = "✅ PASSED" if passed else "❌ FAILED"
        lines += ["", "## Verdict", "", f"Chosen in-sample: **{v}**.",
                  f"Out-of-sample: avg {o['avg']:+.2f}R, PF {o['pf']:.2f} over {o['n']} trades -> **{verdict}**.",
                  f"To pass it must also beat the best random control by 0.05R+ per trade (bar: {bar:+.2f}R). "
                  "Control trade counts are pooled over 10 seeds."]
        if not oos_t.empty:
            lines += ["", "Out-of-sample by market:", "", "| Market | Trades | Avg | Total |", "|---|---:|---:|---:|"]
            for m, g in oos_t.groupby("market"):
                lines.append(f"| {m} | {len(g)} | {g.R.mean():+.2f}R | {g.R.sum():+.1f}R |")
        title = f"Strategy research: {v} {verdict} out-of-sample (PF {o['pf']:.2f})"
    else:
        lines += ["", "## Verdict", "", "❌ No variant reached profit factor 1.2 in-sample. None are worth trading."]
        title = "Strategy research: no variant passed"

    lines += ["", "_Daily candles, next-open entries, gaps filled at the open, stop assumed first when stop and "
              "target touch the same candle, 0.05R costs per trade. 1R = amount risked per trade._"]
    publish(title, "\n".join(lines))


if __name__ == "__main__":
    main()
