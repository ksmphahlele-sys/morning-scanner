"""Builds and sends the Morning Scan email."""

import os
import smtplib
import ssl
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SAST = timezone(timedelta(hours=2))


def fmt(x):
    return "-" if x is None else f"{x:,.2f}"


def build_email(results: list, errors: list) -> tuple:
    signals = [r for r in results if r["signal"]]
    today = datetime.now(SAST).strftime("%a %d %b %Y")
    subject = f"Morning Scan {today}: {len(signals)} setup(s)" if signals else f"Morning Scan {today}: no setups"

    td = 'style="padding:6px 10px;border-bottom:1px solid #ddd;text-align:right"'
    tdl = 'style="padding:6px 10px;border-bottom:1px solid #ddd;text-align:left"'
    th = 'style="padding:6px 10px;background:#1f2937;color:#fff;text-align:right"'
    thl = 'style="padding:6px 10px;background:#1f2937;color:#fff;text-align:left"'

    html = [f"<h2 style='font-family:Arial'>Morning Scan - {today}</h2>"]

    if signals:
        html.append("<h3 style='font-family:Arial'>Setups today</h3>")
        html.append(f"<table style='border-collapse:collapse;font-family:Arial;font-size:14px'><tr>"
                    f"<th {thl}>Market</th><th {thl}>Signal</th><th {th}>Entry</th>"
                    f"<th {th}>Stop</th><th {th}>Target (2R)</th></tr>")
        for r in signals:
            colour = "#059669" if r["signal"] == "LONG" else "#dc2626"
            html.append(f"<tr><td {tdl}>{r['name']}</td>"
                        f"<td {tdl}><b style='color:{colour}'>{r['signal']}</b></td>"
                        f"<td {td}>{fmt(r['entry'])}</td><td {td}>{fmt(r['stop'])}</td>"
                        f"<td {td}>{fmt(r['target'])}</td></tr>")
        html.append("</table>")
    else:
        html.append("<p style='font-family:Arial'>No trend + pullback setups today. Patience is a position.</p>")

    html.append("<h3 style='font-family:Arial'>Full watchlist</h3>")
    html.append(f"<table style='border-collapse:collapse;font-family:Arial;font-size:14px'><tr>"
                f"<th {thl}>Market</th><th {thl}>Trend</th><th {th}>Close</th>"
                f"<th {th}>20 EMA</th><th {th}>50 EMA</th><th {th}>200 EMA</th>"
                f"<th {th}>ATR</th><th {th}>Dist. 20 EMA (ATR)</th></tr>")
    for r in results:
        html.append(f"<tr><td {tdl}>{r['name']}</td><td {tdl}>{r['trend']}</td>"
                    f"<td {td}>{fmt(r['close'])}</td><td {td}>{fmt(r['ema20'])}</td>"
                    f"<td {td}>{fmt(r['ema50'])}</td><td {td}>{fmt(r['ema200'])}</td>"
                    f"<td {td}>{fmt(r['atr'])}</td><td {td}>{r['dist20_atr']:+.1f}</td></tr>")
    html.append("</table>")

    if errors:
        html.append("<p style='font-family:Arial;color:#b45309'>Could not fetch: " + ", ".join(errors) + "</p>")

    html.append("<p style='font-family:Arial;font-size:12px;color:#6b7280'>Based on the last completed daily candle. "
                "Levels are reference points, not financial advice. Check the chart and size positions to your risk "
                "(e.g. 1% of account per trade).</p>")
    return subject, "\n".join(html)


def send_email(subject: str, html: str):
    sender = os.environ["EMAIL_ADDRESS"]
    password = os.environ["EMAIL_APP_PASSWORD"]
    recipient = os.environ.get("EMAIL_TO") or sender

    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, sender, recipient
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient.split(","), msg.as_string())
