"""Quick check: does fetch_index_constituents() actually work? No historical
scanning here, just the constituent-list fetch, so this should take seconds."""
import sys
sys.path.insert(0, ".")
from resistance_breakout_scanner import fetch_index_constituents, _nse_session

sess = _nse_session()
symbols = fetch_index_constituents("NIFTY500", session=sess)
print(f"\nTotal symbols: {len(symbols)}")
print(f"First 10: {symbols[:10]}")
print(f"Last 10: {symbols[-10:]}")
print(f"Any duplicates: {len(symbols) != len(set(symbols))}")
