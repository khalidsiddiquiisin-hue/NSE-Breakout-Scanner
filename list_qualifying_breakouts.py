"""
List all fully-qualifying breakouts across the cached universe.
Runs entirely offline against data/*.csv -- no network calls, takes seconds.
"""
import sys
import glob
import os
sys.path.insert(0, ".")
import pandas as pd
from resistance_breakout_scanner import detect_resistance_breakouts, MIN_RESISTANCE_YEARS, MIN_DELIVERY_PCT

"""
List qualifying breakouts across the cached universe.
Runs entirely offline against data/*.csv -- no network calls, takes seconds.

By default shows only RECENT breakouts (last RECENT_DAYS days) -- old historical
matches (years ago) aren't actionable for trading, so showing them every time just
creates confusion about what's actually new. Full history is still saved to
qualifying_breakouts_full.csv for reference/backtesting if needed.

Set env var RECENT_DAYS to change the window (e.g. RECENT_DAYS=90), or
RECENT_DAYS=0 to show full history instead.
"""
import sys
import glob
import os
from datetime import date, timedelta
sys.path.insert(0, ".")
import pandas as pd
from resistance_breakout_scanner import detect_resistance_breakouts, MIN_RESISTANCE_YEARS, MIN_DELIVERY_PCT

RECENT_DAYS = int(os.environ.get("RECENT_DAYS", "30"))

results = []
csv_files = sorted(glob.glob("data/*.csv"))
print(f"Checking {len(csv_files)} cached symbol(s)...")

for path in csv_files:
    sym = os.path.basename(path).replace(".csv", "")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"  [skip] {sym}: could not read ({e})")
        continue

    events = detect_resistance_breakouts(df)
    if events.empty:
        continue

    qualifying = events[events["all_conditions_met"] == True]
    for _, row in qualifying.iterrows():
        results.append({"symbol": sym, **row.to_dict()})

if results:
    out_full = pd.DataFrame(results).sort_values("breakout_date", ascending=False)
    out_full.to_csv("qualifying_breakouts_full.csv", index=False)

    if RECENT_DAYS > 0:
        cutoff = date.today() - timedelta(days=RECENT_DAYS)
        out = out_full[pd.to_datetime(out_full["breakout_date"]).dt.date >= cutoff]
        window_label = f"last {RECENT_DAYS} days (since {cutoff})"
    else:
        out = out_full
        window_label = "full history"

    print(f"\n=== {len(out)} fully-qualifying breakout(s) in {window_label} "
          f"({len(out_full)} total in full history -- see qualifying_breakouts_full.csv) ===\n")

    if not out.empty:
        cols = ["symbol", "breakout_date", "breakout_close", "resistance_age_years",
                "volume_ratio", "delivery_pct", "prior_resistance", "resistance_set_on"]
        print(out[cols].to_string(index=False))
        out.to_csv("qualifying_breakouts.csv", index=False)
        print(f"\nSaved to qualifying_breakouts.csv (recent) "
              f"and qualifying_breakouts_full.csv (all-time)")
    else:
        print(f"No breakouts in the {window_label} window. "
              f"({len(out_full)} exist further back -- see qualifying_breakouts_full.csv)")
else:
    print(f"\n=== 0 fully-qualifying breakout(s) found "
          f"(resistance >= {MIN_RESISTANCE_YEARS}yr, volume >= 1.4x, delivery% >= {MIN_DELIVERY_PCT}) ===")
    print("None found.")
