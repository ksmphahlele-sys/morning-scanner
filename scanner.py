"""
Morning Scanner - Gold & Indices, Trend + Pullback setups (daily candles).

LONG  : close > 200 EMA and 50 EMA > 200 EMA (uptrend)
        AND in the previous 10 candles price stretched >= 1 ATR above the 20 EMA (impulse)
        AND the last candle's low came within 0.5 ATR of the 20 or 50 EMA (pullback)
        AND it still closed above the 50 EMA (trend intact)
SHORT : the mirror image in a downtrend.
Trade plan: enter next open, stop 1.5 ATR, target 3 ATR (2R).
"""

import os

import numpy as np
import pandas as pd
import yfinance as yf

from report import build_report, publish

# ---- Edit your watchlist here: "Yahoo ticker": "Display name" ----
WATCHLIST = {
    "GC=F": "Gold (XAU/USD futures)",
    "SI=F": "Silver",
    "^NDX": "NASDAQ 100",
    "^GSPC": "S&P 500",
    "^DJI": "US30 (Dow Jones)",
    "^GDAXI": "DAX 40",
    "^FTSE": "FTSE 100",
    "^J200.JO": "JSE Top 40",
}

PULLBACK_ATR = 0.5   # how close price must come to the 20/50 EMA
EXTEND_ATR = 1.0     # how far price must have stretched from the 20 EMA first
LOOKBACK = 10        # candles to look back for that stretch
STOP_ATR = 1.5
TARGET_ATR = 3.0


def fetch(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    # Drop today's unfinished candle so signals only use closed candles.
    now = pd.Timestamp.now(tz=df.index.tz)
    return df[df.index.normalize() < now.normalize()]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    stretch_up = ((h - df["ema20"]) >= EXTEND_ATR * df["atr"]).astype(float)
    stretch_dn = ((df["ema20"] - l) >= EXTEND_ATR * df["atr"]).astype(float)
    ext_up = stretch_up.rolling(LOOKBACK).max().shift(1).fillna(0) > 0
    ext_dn = stretch_dn.rolling(LOOKBACK).max().shift(1).fillna(0) > 0

    up = (c > df["ema200"]) & (df["ema50"] > df["ema200"])
    dn = (c < df["ema200"]) & (df["ema50"] < df["ema200"])
    near = PULLBACK_ATR * df["atr"]
    touch_up = ((l - df["ema20"]) <= near) | ((l - df["ema50"]) <= near)
    touch_dn = ((df["ema20"] - h) <= near) | ((df["ema50"] - h) <= near)

    df["trend"] = np.where(up, "Up", np.where(dn, "Down", "Sideways"))
    long_sig = up & ext_up & touch_up & (c > df["ema50"])
    short_sig = dn & ext_dn & touch_dn & (c < df["ema50"])
    df["signal"] = np.where(long_sig, "LONG", np.where(short_sig, "SHORT", ""))
    return df


def latest(df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    c, a, sig = row["Close"], row["atr"], row["signal"] or None
    stop = target = None
    if sig == "LONG":
        stop, target = c - STOP_ATR * a, c + TARGET_ATR * a
    elif sig == "SHORT":
        stop, target = c + STOP_ATR * a, c - TARGET_ATR * a
    return {
        "date": df.index[-1].strftime("%Y-%m-%d"), "close": c, "ema20": row["ema20"],
        "ema50": row["ema50"], "ema200": row["ema200"], "atr": a, "trend": row["trend"],
        "signal": sig, "entry": c, "stop": stop, "target": target,
        "dist20_atr": (c - row["ema20"]) / a if a else 0.0,
    }


def main():
    results, errors = [], []
    for ticker, name in WATCHLIST.items():
        try:
            df = fetch(ticker)
            if len(df) < 210:
                raise ValueError("not enough history")
            r = latest(add_indicators(df))
            r["name"] = name
            results.append(r)
            print(f"{name:26} {r['date']} {r['trend']:9} close={r['close']:,.2f} signal={r['signal']}")
        except Exception as e:  # keep going if one market fails
            errors.append(name)
            print(f"{name}: ERROR {e}")

    title, body = build_report(results, errors)
    publish(title, body)


if __name__ == "__main__":
    main()
