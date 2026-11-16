"""
Morning Scanner - Gold & Indices, Trend + Pullback setups.

Runs every weekday morning on GitHub Actions and emails you a summary.

Setup logic (daily candles):
  LONG  : close > 200 EMA and 50 EMA > 200 EMA (uptrend)
          AND yesterday's low came within 0.5 ATR of the 20 or 50 EMA (pullback)
          AND close is still above the 50 EMA (trend not broken)
  SHORT : the mirror image in a downtrend.

Reference levels (not advice): entry = last close,
stop = 1.5 ATR away, target = 3 ATR away (2R).
"""

import os

import pandas as pd
import yfinance as yf

from report import build_email, send_email

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

PULLBACK_ATR = 0.5   # how close (in ATRs) price must come to the 20/50 EMA
STOP_ATR = 1.5
TARGET_ATR = 3.0


def fetch(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=False)
    return df.dropna(subset=["Close"])


def analyse(df: pd.DataFrame) -> dict:
    close, high, low = df["Close"], df["High"], df["Low"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    c, h, l = close.iloc[-1], high.iloc[-1], low.iloc[-1]
    e20, e50, e200, a = ema20.iloc[-1], ema50.iloc[-1], ema200.iloc[-1], atr.iloc[-1]

    uptrend = c > e200 and e50 > e200
    downtrend = c < e200 and e50 < e200

    # Require a real impulse first: within the last 10 candles price stretched
    # at least 1 ATR away from the 20 EMA, so we only catch genuine pullbacks.
    extended_up = ((high - ema20) >= atr).iloc[-11:-1].any()
    extended_down = ((ema20 - low) >= atr).iloc[-11:-1].any()

    signal, entry, stop, target = None, c, None, None
    if uptrend:
        touched = (l - e20) <= PULLBACK_ATR * a or (l - e50) <= PULLBACK_ATR * a
        if extended_up and touched and c > e50:
            signal = "LONG"
            stop, target = c - STOP_ATR * a, c + TARGET_ATR * a
    elif downtrend:
        touched = (e20 - h) <= PULLBACK_ATR * a or (e50 - h) <= PULLBACK_ATR * a
        if extended_down and touched and c < e50:
            signal = "SHORT"
            stop, target = c + STOP_ATR * a, c - TARGET_ATR * a

    trend = "Up" if uptrend else "Down" if downtrend else "Sideways"
    return {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": c, "ema20": e20, "ema50": e50, "ema200": e200, "atr": a,
        "trend": trend, "signal": signal, "entry": entry, "stop": stop, "target": target,
        "dist20_atr": (c - e20) / a if a else 0.0,
    }


def main():
    results, errors = [], []
    for ticker, name in WATCHLIST.items():
        try:
            df = fetch(ticker)
            if len(df) < 210:
                raise ValueError("not enough history")
            r = analyse(df)
            r["name"] = name
            results.append(r)
            print(f"{name:28} {r['trend']:9} close={r['close']:,.2f}  signal={r['signal']}")
        except Exception as e:  # keep going if one market fails
            errors.append(name)
            print(f"{name}: ERROR {e}")

    subject, html = build_email(results, errors)
    print("\n" + subject)

    if os.environ.get("DRY_RUN") == "1":
        with open("preview.html", "w") as f:
            f.write(html)
        print("DRY_RUN: email saved to preview.html, not sent.")
    else:
        send_email(subject, html)
        print("Email sent.")


if __name__ == "__main__":
    main()
