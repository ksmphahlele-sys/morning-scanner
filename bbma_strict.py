"""
STRICT BBMA (Oma Ally) backtest.

Indicators (every timeframe): Bollinger Bands 20/2 on close, LWMA 5 & 10 of HIGHS and of LOWS, EMA 50.

BUY structure (sells are the exact mirror):
  1. EXTREME : LWMA5-low or LWMA10-low closes OUTSIDE (below) the lower band.
  2. MHV     : 2-10 candles after the extreme, price tests the lower band again (low <= lower band)
               but FAILS to close outside it, and LWMA5-low stays inside the band (lost volume).
  3. CSAK    : within 10 candles of MHV, a bullish candle that opens below and closes above the
               middle band AND closes above LWMA5-high and LWMA10-high (direction-change candle).
  4. RE-ENTRY: within 10 candles of CSAK (or of a CSM = close outside the upper band), the FIRST
               candle whose low retraces into the LWMA5/10-low zone while closing above LWMA10-low
               and above the middle band.
  5. CONFIRM : the NEXT candle is bullish and closes above the re-entry candle's high.
  Higher-timeframe (HTF) direction: HTF close above HTF middle band and HTF middle band rising
  (daily setups use the WEEKLY chart; H4 setups use the DAILY chart; completed HTF candles only).
  Liquidity (optional): the re-entry candle also sweeps below the lowest low of the prior 5 candles.

Trade: enter at the open after confirmation. Stop 0.1 ATR beyond the lowest low of the re-entry +
confirmation candles (minimum 0.5 ATR). Target = the opposite (upper) band at the signal, at least 1R
("band" target), or 2R. Max hold 20 candles. 0.05R costs per trade.

Every variant is compared with RANDOM entries that use the same HTF filter and the same exits.
Daily: 10 years, choose on 2017-2021, judge on 2022-2026. H4: only ~2 years of hourly data exist
for free, so H4 is a single-period check against random.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from report import publish
from research import SEEDS, SPLIT, cell, stats
from scanner import WATCHLIST, add_indicators, fetch

COST_R = 0.05
MAX_HOLD = 20


def lwma(s, n):
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def bbma_ind(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    out = pd.DataFrame({"bb_up": mid + 2 * sd, "bb_mid": mid, "bb_lo": mid - 2 * sd,
                        "lo5": lwma(l, 5), "lo10": lwma(l, 10), "hi5": lwma(h, 5), "hi10": lwma(h, 10),
                        "ema50x": c.ewm(span=50, adjust=False).mean()}, index=df.index)
    return out


def htf_direction(df, rule, shift):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    w = df[["Open", "High", "Low", "Close"]].resample(rule).agg(agg).dropna()
    ind = bbma_ind(w)
    up = (w["Close"] > ind["bb_mid"]) & (ind["bb_mid"] > ind["bb_mid"].shift(3))
    dn = (w["Close"] < ind["bb_mid"]) & (ind["bb_mid"] < ind["bb_mid"].shift(3))
    d = pd.DataFrame({"htf_up": up, "htf_dn": dn})
    if shift:
        d = d.shift(1)
    return d.reindex(df.index, method="ffill").fillna(False).astype(bool)


def prepare(df, htf_rule, htf_shift):
    df = add_indicators(df)
    df = df.join(bbma_ind(df))
    df = df.join(htf_direction(df, htf_rule, htf_shift))
    return df


def structure(df, side):
    """Return (diamond_setup, reentry_setup, extreme_mhv_setup) boolean arrays for one side."""
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    up, mid, lo = df["bb_up"].to_numpy(), df["bb_mid"].to_numpy(), df["bb_lo"].to_numpy()
    lo5, lo10, hi5, hi10 = (df[k].to_numpy() for k in ("lo5", "lo10", "hi5", "hi10"))
    n = len(df)
    if side == "buy":
        ext = (lo5 < lo) | (lo10 < lo)
        mhv = (l <= lo) & (c > lo) & (lo5 >= lo)
        csak = (c > o) & (o < mid) & (c > mid) & (c > hi5) & (c > hi10)
        csm = c > up
        zone = (l <= np.maximum(lo5, lo10)) & (c > lo10) & (c > mid)
    else:
        ext = (hi5 > up) | (hi10 > up)
        mhv = (h >= up) & (c < up) & (hi5 <= up)
        csak = (c < o) & (o > mid) & (c < mid) & (c < lo5) & (c < lo10)
        csm = c < lo
        zone = (h >= np.minimum(hi5, hi10)) & (c < hi10) & (c < mid)

    diamond, reentry, extmhv = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    ext_i = mhv_i = csak_i = None      # diamond chain
    trig_i = None                      # last CSAK/CSM for plain re-entry
    for t in range(n):
        if np.isnan(up[t]) or np.isnan(lo10[t]):
            continue
        # --- diamond chain: Extreme -> MHV -> CSAK -> first Re-entry
        if ext[t]:
            ext_i, mhv_i, csak_i = t, None, None
        elif ext_i is not None and mhv_i is None and 2 <= t - ext_i <= 10 and mhv[t]:
            mhv_i = t
            extmhv[t] = True
        elif mhv_i is not None and csak_i is None and t - mhv_i <= 10 and csak[t]:
            csak_i = t
        elif csak_i is not None and t - csak_i <= 10 and zone[t]:
            diamond[t] = True
            ext_i = mhv_i = csak_i = None
        if ext_i is not None and mhv_i is None and t - ext_i > 10:
            ext_i = None
        if mhv_i is not None and csak_i is None and t - mhv_i > 10:
            ext_i = mhv_i = None
        if csak_i is not None and t - csak_i > 10:
            ext_i = mhv_i = csak_i = None
        # --- plain re-entry after CSAK or CSM (first touch only)
        if csak[t] or csm[t]:
            trig_i = t
        elif trig_i is not None and t - trig_i <= 10 and zone[t]:
            reentry[t] = True
            trig_i = None
        elif trig_i is not None and t - trig_i > 10:
            trig_i = None
    return diamond, reentry, extmhv


def signals(df, kind, htf, liquidity, seed=None):
    """Signal on the CONFIRMATION candle. Returns (sig, stop_ref, band_target)."""
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    n = len(df)
    sig = np.array([""] * n, dtype=object)
    stop_ref, band = np.full(n, np.nan), np.full(n, np.nan)
    for side in ("buy", "sell"):
        if seed is not None:
            rng = np.random.default_rng(seed + (0 if side == "buy" else 1000))
            setup = rng.random(n) < 0.02
        else:
            d, r, e = structure(df, side)
            setup = {"diamond": d, "reentry": r, "extreme": e}[kind]
        filt = df["htf_up" if side == "buy" else "htf_dn"].to_numpy() if htf else np.ones(n, bool)
        if liquidity:
            if side == "buy":
                sweep = (df["Low"] < df["Low"].rolling(5).min().shift(1)).to_numpy()
            else:
                sweep = (df["High"] > df["High"].rolling(5).max().shift(1)).to_numpy()
            setup = setup & sweep
        for t in np.flatnonzero(setup[:-1]):
            k = t + 1  # confirmation candle
            if side == "buy":
                ok = c[k] > o[k] and c[k] > h[t]
            else:
                ok = c[k] < o[k] and c[k] < l[t]
            if seed is not None:
                ok = True
            if ok and filt[k] and not sig[k]:
                sig[k] = "LONG" if side == "buy" else "SHORT"
                stop_ref[k] = min(l[t], l[k]) if side == "buy" else max(h[t], h[k])
                if kind == "extreme":   # extreme trades target the middle band (TPW)
                    band[k] = df["bb_mid"].iat[k]
                else:
                    band[k] = df["bb_up"].iat[k] if side == "buy" else df["bb_lo"].iat[k]
    return sig, stop_ref, band


def simulate(df, sig, stop_ref, band, target_mode):
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr = df["atr"].to_numpy()
    trades, i, n = [], 210, len(df)
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        d = 1 if sig[i] == "LONG" else -1
        entry = o[i + 1]
        ref = stop_ref[i] if not np.isnan(stop_ref[i]) else entry - d * 1.5 * atr[i]
        risk = max(d * (entry - (ref - d * 0.1 * atr[i])), 0.5 * atr[i])
        stop = entry - d * risk
        if target_mode == "band" and not np.isnan(band[i]):
            target = band[i] if d * (band[i] - entry) >= risk else entry + d * risk
        else:
            target = entry + d * 2 * risk
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


VARIANTS = [
    # name, setup, htf filter, liquidity, target
    ("V1. Diamond: Extreme→MHV→CSAK→Re-entry + confirm, band TP", "diamond", False, False, "band"),
    ("V2. V1 + higher-timeframe direction", "diamond", True, False, "band"),
    ("V3. V2 with 2R target", "diamond", True, False, "2r"),
    ("V4. V2 + liquidity sweep", "diamond", True, True, "band"),
    ("V5. Re-entry after CSAK/CSM + confirm + HTF, band TP", "reentry", True, False, "band"),
    ("V6. Re-entry + confirm + HTF + liquidity, band TP", "reentry", True, True, "band"),
    ("V7. Extreme→MHV + confirm + HTF, mid-band TP (TPW)", "extreme", True, False, "band"),
]


def run_set(data, variants, split):
    results = []
    for vname, kind, htf, liq, tgt in variants:
        rows, crow = [], []
        for name, df in data.items():
            s, sr, b = signals(df, kind, htf, liq)
            rows += [dict(t, market=name) for t in simulate(df, s, sr, b, tgt)]
            for seed in SEEDS:
                s, sr, b = signals(df, kind, htf, False, seed=seed)
                crow += [dict(t, market=name) for t in simulate(df, s, sr, b, tgt)]
        t = pd.DataFrame(rows, columns=["date", "R", "market"])
        ct = pd.DataFrame(crow, columns=["date", "R", "market"])
        for x in (t, ct):
            x["date"] = pd.to_datetime(x["date"], utc=True)
        if split is not None:
            results.append((vname, t[t.date < split], t[t.date >= split], ct[ct.date < split], ct[ct.date >= split]))
        else:
            results.append((vname, t, t, ct, ct))
        print(vname, stats(results[-1][1]), stats(results[-1][2]))
    return results


def tscore(t, ct):
    if len(t) < 2 or len(ct) < 2:
        return 0.0
    return (t.R.mean() - ct.R.mean()) / (t.R.std() / np.sqrt(len(t)))


def main():
    daily = {}
    for ticker, name in WATCHLIST.items():
        try:
            daily[name] = prepare(fetch(ticker, period="10y"), "W-FRI", False)
        except Exception as e:
            print(name, "daily ERROR", e)

    res = run_set(daily, VARIANTS, SPLIT)
    head = ("| Variant | Trades | Win | Avg | PF | Total | Max DD | Random control avg |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|")
    lines = ["# Daily setups, weekly direction (10 years)", "",
             "## 2017–2021 (used to choose)", "", head]
    for v, a, b, ca, cb in res:
        lines.append(f"| {v} | {cell(stats(a))} | {ca.R.mean() if len(ca) else 0:+.2f}R |")
    lines += ["", "## 2022–2026 (unseen)", "", head]
    for v, a, b, ca, cb in res:
        lines.append(f"| {v} | {cell(stats(b))} | {cb.R.mean() if len(cb) else 0:+.2f}R |")

    elig = [r for r in res if len(r[1]) >= 30 and stats(r[1])["pf"] >= 1.2
            and r[1].R.mean() > r[3].R.mean() + 0.05]
    lines += ["", "## Verdict (daily)", ""]
    if elig:
        v, a, b, ca, cb = max(elig, key=lambda r: r[1].R.mean() - r[3].R.mean())
        sb, ts = stats(b), tscore(b, cb)
        ok = sb["avg"] > 0 and sb["pf"] >= 1.2 and sb["avg"] > cb.R.mean() + 0.05 and ts >= 2 and sb["n"] >= 30
        verdict = "✅ PASSED" if ok else "❌ FAILED"
        lines += [f"Chosen on 2017–2021: **{v}**.",
                  f"2022–2026: {sb['avg']:+.2f}R avg, PF {sb['pf']:.2f}, {sb['n']} trades vs random "
                  f"{cb.R.mean():+.2f}R; t-score {ts:.1f} (needs 2.0+, and 30+ trades) -> **{verdict}**."]
    else:
        verdict = "❌ FAILED"
        lines += ["❌ No strict-BBMA variant beat random entries (same filter, same exits) on 2017–2021 "
                  "with 30+ trades and PF 1.2+."]

    # ---- H4 check (about 2 years only)
    h4 = {}
    for ticker, name in WATCHLIST.items():
        try:
            raw = yf.Ticker(ticker).history(period="730d", interval="1h", auto_adjust=False)
            raw = raw.dropna(subset=["Open", "High", "Low", "Close"]).iloc[:-1]
            agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            bars = raw[["Open", "High", "Low", "Close"]].resample("4h").agg(agg).dropna()
            if len(bars) > 400:
                h4[name] = prepare(bars, "1D", True)
        except Exception as e:
            print(name, "H4 ERROR", e)
    if h4:
        hres = run_set(h4, [x for x in VARIANTS if x[2]], None)
        lines += ["", "# H4 setups, daily direction (≈2 years, single period)", "", head]
        for v, a, b, ca, cb in hres:
            lines.append(f"| {v} | {cell(stats(a))} | {ca.R.mean() if len(ca) else 0:+.2f}R |")
        lines += ["", "H4 t-scores vs random: " + ", ".join(
            f"{v.split('.')[0]} {tscore(a, ca):.1f}" for v, a, b, ca, cb in hres)]
        lines += [f"", f"_H4 markets with data: {', '.join(h4)}. Two years is too short to choose AND "
                  "confirm on separate data, so treat H4 as a hint only._"]

    lines += ["", "_Daily: entries at next open, gaps filled at the open, stop assumed first if stop and "
              "target touch the same candle, 0.05R costs, max 20 candles. Random control: random setups "
              "with the same HTF filter and exits, pooled over 10 seeds. 1R = amount risked._"]
    publish(f"Strict BBMA research: {verdict} (daily, weekly direction) + H4 check", "\n".join(lines))


if __name__ == "__main__":
    main()
