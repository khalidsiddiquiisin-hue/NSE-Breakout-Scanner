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

print(f"\n=== {len(results)} fully-qualifying breakout(s) found "
      f"(resistance >= {MIN_RESISTANCE_YEARS}yr, volume >= 1.4x, delivery% >= {MIN_DELIVERY_PCT}) ===\n")

if results:
    out = pd.DataFrame(results).sort_values("breakout_date", ascending=False)
    cols = ["symbol", "breakout_date", "breakout_close", "resistance_age_years",
            "volume_ratio", "delivery_pct", "prior_resistance", "resistance_set_on"]
    print(out[cols].to_string(index=False))
    out.to_csv("qualifying_breakouts.csv", index=False)
    print(f"\nSaved to qualifying_breakouts.csv")
else:
    print("None found.")
