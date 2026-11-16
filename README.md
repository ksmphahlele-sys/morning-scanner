# Morning Scanner

Trend + pullback setups on gold and the major indices, delivered every weekday at about 07:00 SAST.
No passwords: each scan is posted as an **issue** in this repo, and GitHub emails it to you.

- `scanner.py`: watchlist and strategy rules
- `report.py`: builds the report and posts it
- `backtest.py`: tests the same rules on 10 years of data
- `.github/workflows/morning-scan.yml`: the daily schedule
- `.github/workflows/backtest.yml`: run the backtest on demand (Actions > Backtest > Run workflow)

## Changing it
- Markets: edit `WATCHLIST` in `scanner.py` (Yahoo Finance tickers).
- Time: edit the `cron` line (UTC, so `0 5` = 07:00 SAST). GitHub can start scheduled runs 15 to 30 minutes late.
- Not getting emails? Check github.com/settings/notifications: Email must be ticked for "Participating, @mentions".

Reference levels, not financial advice. Risk about 1% of the account per trade.
