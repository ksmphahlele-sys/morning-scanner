"""Builds the Morning Scan report and posts it as a GitHub issue (GitHub emails it to you)."""

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone

SAST = timezone(timedelta(hours=2))


def fmt(x):
    return "-" if x is None else f"{x:,.2f}"


def build_report(results: list, errors: list) -> tuple:
    signals = [r for r in results if r["signal"]]
    today = datetime.now(SAST).strftime("%a %d %b %Y")
    title = f"Morning Scan {today}: {len(signals)} setup(s)" if signals else f"Morning Scan {today}: no setups"

    lines = []
    if signals:
        lines += ["## Setups today", "",
                  "| Market | Signal | Entry (approx.) | Stop | Target (2R) |",
                  "|---|---|---:|---:|---:|"]
        for r in signals:
            icon = "🟢" if r["signal"] == "LONG" else "🔴"
            lines.append(f"| {r['name']} | {icon} **{r['signal']}** | {fmt(r['entry'])} | "
                         f"{fmt(r['stop'])} | {fmt(r['target'])} |")
        lines += ["", "Entry is the last close; in practice you enter at the next open and keep "
                  "the stop 1.5 ATR and target 3 ATR from your actual fill.", ""]
    else:
        lines += ["## No setups today", "", "No trend + pullback setups. Patience is a position.", ""]

    lines += ["## Full watchlist", "",
              "| Market | Candle | Trend | Close | 20 EMA | 50 EMA | 200 EMA | ATR | Dist. 20 EMA (ATR) |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['date']} | {r['trend']} | {fmt(r['close'])} | {fmt(r['ema20'])} | "
                     f"{fmt(r['ema50'])} | {fmt(r['ema200'])} | {fmt(r['atr'])} | {r['dist20_atr']:+.1f} |")
    if errors:
        lines += ["", "⚠️ Could not fetch: " + ", ".join(errors)]
    lines += ["", "_Based on the last completed daily candle. Reference levels, not financial advice. "
              "Risk about 1% of your account per trade._"]
    return title, "\n".join(lines)


def publish(title: str, body: str):
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER")
    if owner:
        body += f"\n\ncc @{owner}"
    if os.environ.get("DRY_RUN") == "1" or not os.environ.get("GITHUB_TOKEN"):
        print(f"\n# {title}\n\n{body}")
        return
    repo = os.environ["GITHUB_REPOSITORY"]
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({"title": title, "body": body}).encode(),
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        print("Posted:", json.load(resp)["html_url"])
