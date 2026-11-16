"""
Walk-forward parameter optimisation of the trend strategy (breakout + trailing stop).

Grid (108 combinations):
  breakout lookback : 20, 50, 100 days
  starting stop     : 1.5, 2, 3 ATR
  trailing stop     : 2, 3, 4, 5 ATR
  max hold          : 40, 60, 120 days
All buys require close > 200 EMA and a rising 200 EMA.

Walk-forward (never test on data used to choose):
  Window 1: optimise on 2017-2019 -> test 2020-2021
  Window 2: optimise on 2017-2021 -> test 2022-2026
Objective: highest average R per trade (minimum 40 trades in the optimisation period).

Honesty checks reported:
  - the chosen settings' unseen results vs. a random-entry control using the SAME settings
  - the median unseen result of ALL 108 combos (does optimising beat just picking any?)
  - rank correlation between optimisation-period and unseen results across all combos
    (near 0 = the "best" settings were mostly fitting noise)
"""

import itertools

import numpy as np
import pandas as pd

from report import publish
from research import SEEDS, simulate
from scanner import WATCHLIST, add_indicators, fetch

COST_NOTE = "0.05R costs per trade"
GRID = list(itertools.product([20, 50, 100], [1.5, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0], [40, 60, 120]))
WINDOWS = [
    ("2017-01-01", "2020-01-01", "2022-01-01"),
    ("2017-01-01", "2022-01-01", "2100-01-01"),
]


def sigs(df, lookback, seed=None):
    up = (df["Close"] > df["ema200"]) & (df["ema200"] > df["ema200"].shift(20))
    if seed is None:
        m = up & (df["Close"] > df["High"].rolling(lookback).max().shift(1))
    else:
        rng = np.random.default_rng(seed)
        m = up & pd.Series(rng.random(len(df)) < 0.05, index=df.index)
    return np.where(m.fillna(False).to_numpy(), "LONG", "")


def trades_for(data, params, seed=None):
    lb, stop, trail, hold = params
    rows = []
    for name, df in data.items():
        for t in simulate(df, sigs(df, lb, seed), stop, None, trail, hold):
            t["market"] = name
            rows.append(t)
    t = pd.DataFrame(rows, columns=["date", "R", "market"])
    t["date"] = pd.to_datetime(t["date"], utc=True)
    return t


def avg(t):
    return t.R.mean() if len(t) else float("nan")


def label(p):
    return f"{p[0]}-day breakout, {p[1]} ATR stop, {p[2]} ATR trail, {p[3]}-day max"


def main():
    data = {}
    for ticker, name in WATCHLIST.items():
        try:
            data[name] = add_indicators(fetch(ticker, period="10y"))
        except Exception as e:
            print(name, "ERROR", e)

    all_trades = {p: trades_for(data, p) for p in GRID}
    lines, oos_chosen, oos_control = [], [], []

    for k, (start, split, end) in enumerate(WINDOWS, 1):
        s, m, e = (pd.Timestamp(x, tz="UTC") for x in (start, split, end))
        rows = []
        for p, t in all_trades.items():
            ins, out = t[(t.date >= s) & (t.date < m)], t[(t.date >= m) & (t.date < e)]
            rows.append({"p": p, "is_n": len(ins), "is_avg": avg(ins), "oos_n": len(out), "oos_avg": avg(out),
                         "oos_t": out})
        g = pd.DataFrame(rows)
        ok = g[g.is_n >= 40]
        best = ok.loc[ok.is_avg.idxmax()]
        ctrl = pd.concat([trades_for(data, best.p, seed) for seed in SEEDS])
        ctrl = ctrl[(ctrl.date >= m) & (ctrl.date < e)]
        corr = g[["is_avg", "oos_avg"]].rank().corr().iloc[0, 1]
        top10 = ok.nlargest(10, "is_avg")
        oos_chosen.append(best.oos_t)
        oos_control.append(ctrl)
        lines += [f"## Window {k}: optimise {start[:4]}–{int(split[:4]) - 1}, test {split[:4]}–"
                  f"{'2026' if end.startswith('2100') else int(end[:4]) - 1}", "",
                  f"- **Chosen settings:** {label(best.p)}",
                  f"- Optimisation period: {best.is_avg:+.2f}R avg over {best.is_n} trades",
                  f"- **Unseen period: {best.oos_avg:+.2f}R avg over {best.oos_n} trades**",
                  f"- Random entries with the same settings (control): {avg(ctrl):+.2f}R avg",
                  f"- Median unseen result of all 108 settings: {g.oos_avg.median():+.2f}R",
                  f"- Top-10 optimised settings, average unseen result: {top10.oos_avg.mean():+.2f}R",
                  f"- Rank correlation optimisation vs unseen across all settings: {corr:+.2f} "
                  "(near 0 = past best doesn't predict future best)", ""]

    oos = pd.concat(oos_chosen)
    ctrl = pd.concat(oos_control)
    sd = oos.R.std()
    tscore = (avg(oos) - avg(ctrl)) / (sd / np.sqrt(len(oos))) if len(oos) > 1 else 0.0
    wins, losses = oos.R[oos.R > 0].sum(), -oos.R[oos.R <= 0].sum()
    pf = wins / losses if losses else 99.0
    passed = avg(oos) > 0 and pf >= 1.2 and tscore >= 2
    verdict = "✅ PASSED" if passed else "❌ FAILED"
    lines += ["## Combined walk-forward result (all unseen periods)", "",
              f"- Optimised strategy: **{avg(oos):+.2f}R avg, PF {pf:.2f}, {len(oos)} trades**",
              f"- Random-entry control, same settings: {avg(ctrl):+.2f}R avg",
              f"- Edge over random: t-score {tscore:.1f} (needs 2.0+)",
              f"- Verdict: **{verdict}**", "",
              f"_Daily candles, next-open entries, gaps filled at the open, {COST_NOTE}. "
              "1R = amount risked per trade. Control pooled over 10 random seeds._"]
    title = f"Walk-forward optimisation: {verdict} ({avg(oos):+.2f}R vs random {avg(ctrl):+.2f}R, t={tscore:.1f})"
    publish(title, "\n".join(lines))


if __name__ == "__main__":
    main()
