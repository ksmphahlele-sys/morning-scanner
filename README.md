# Morning Scanner

Emails trend + pullback setups on gold and the major indices at 07:00 SAST, Monday to Friday.

- `scanner.py`: watchlist and setup logic
- `report.py`: builds and sends the email
- `.github/workflows/morning-scan.yml`: the schedule

## Finish setup
1. Create a Gmail app password at myaccount.google.com/apppasswords (2-Step Verification must be on).
2. Settings > Secrets and variables > Actions > New repository secret. Add `EMAIL_ADDRESS`, `EMAIL_APP_PASSWORD` and `EMAIL_TO`.
3. Actions > Morning Scan > Run workflow to test.

## Changing it
- Markets: edit `WATCHLIST` in `scanner.py` (Yahoo Finance tickers).
- Time: edit the `cron` line (UTC, so `0 5` = 07:00 SAST). GitHub can start scheduled runs 15 to 30 minutes late.

Levels are reference points, not financial advice. Never put passwords in the code.
