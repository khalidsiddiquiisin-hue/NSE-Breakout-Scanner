"""
NSE Multi-Year Resistance Breakout Scanner — v0.5
======================================================
Changelog:
  v0.5 - Added support for NSE's LEGACY pre-2019 bhavcopy zip archive format
         (archives.nseindia.com), confirmed reachable back to at least 2010.
         fetch_history() now automatically routes to legacy vs modern format
         based on date (cutover: 23-Aug-2019). This fixes the ~7-year data
         ceiling from v0.4 -- resistance-age checks can now see much further
         back. Legacy format has no delivery%, which is fine since delivery%
         is only ever checked on the (recent) breakout day.
  v0.4 - Confirmed via GitHub Actions test: NSE's bhavcopy archive (static files)
         is reachable, but the interactive historical API is blocked by Akamai
         bot-mitigation on cloud IPs (Oracle VM included). Dropped the historical
         API entirely.
  v0.3 - Dropped Yahoo/yfinance entirely (Oracle Cloud IPs get 429-rate-limited by
         Yahoo's edge).
  v0.2 - Added Phase 2: delivery% confirmation via NSE bhavcopy, checked only
         on the breakout day itself (per Khalid's call).
  v0.1 - Phase 1: core price+volume breakout detection logic.
======================================================

Rules being encoded (from Khalid's spec):
  1. Resistance = the stock's running ALL-TIME HIGH, and that ATH must have stood
     unbroken for at least MIN_RESISTANCE_YEARS (default 5) before being broken.
     (No upper cap — the older the resistance, the better.)
  2. Breakout day = close crosses above that long-standing ATH.
  3. Volume spike: breakout-day volume >= VOLUME_SPIKE_MULT x the average volume of
     the VOLUME_LOOKBACK trading days immediately before the breakout day.
  4. Delivery % >= MIN_DELIVERY_PCT on breakout day, checked only on the breakout day.
  5. Support/resistance zone = the consolidation range the stock was pinned inside
     for those 5+ years, useful for the chart / watchlist note.
  6. Discord alert fires once ALL confirmed conditions are true.               -> PHASE 3

Run this file directly to backtest one or more symbols and sanity-check the logic.
Data source: NSE's own bhavcopy archive (nsearchives.nseindia.com) — one file per
trading day, covering every listed equity's OHLCV + delivery% at once.

IMPORTANT: run this on GitHub Actions (or any non-cloud-VM network), not the
Oracle VM directly — NSE's Akamai bot-mitigation resets connections from many
cloud provider IP ranges including Oracle Cloud's.

Requires: pip install pandas numpy requests
NOTE: This sandbox has no network access to NSE. Run this via GitHub Actions or
any machine with normal internet access to actually fetch data.
"""

import os
import time
from datetime import datetime, timedelta, date
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None  # only needed for fetch_history() and fetch_nse_delivery_bhavcopy()

# ---------------- Config (tweak here) ----------------
MIN_RESISTANCE_YEARS = 4.5
VOLUME_LOOKBACK = 20          # trading days
VOLUME_SPIKE_MULT = 1.4       # "40% more than last volumes"
MIN_DELIVERY_PCT = 40.0       # Phase 2


_NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _nse_session() -> "requests.Session":
    """NSE blocks naive requests; needs a warmed-up session (cookies from homepage first)."""
    if requests is None:
        raise RuntimeError("requests not installed. Run: pip install requests --break-system-packages")
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    s.get("https://www.nseindia.com", timeout=10)  # sets cookies needed for the API endpoints
    time.sleep(0.5)
    return s


def fetch_history(symbol: str, years: int = 20, session=None, cache_dir: str = "data",
                   retry_failed: bool = True) -> pd.DataFrame:
    """
    Build daily OHLCV (+ delivery%) history for an NSE symbol from NSE's own
    bhavcopy archive — one file per trading day, each covering every symbol.
    (Not from the interactive historical API — that's blocked by Akamai
    bot-mitigation on most cloud/VM IPs; the static archive files are not.)

    CACHING: if cache_dir is set (default "data"), this loads any previously
    saved history for this symbol from {cache_dir}/{SYMBOL}.csv, fetches ONLY
    the trading days not already cached, and writes the merged result back.
    On a fresh symbol this is just as slow as before; on a repeat run it's
    close to instant. The GitHub Actions workflow commits this directory back
    to the repo after each run so the cache persists between runs.

    Symbol should be the bare NSE symbol, e.g. 'TATASTEEL' (no .NS suffix).
    """
    import os

    sym = symbol.upper().replace(".NS", "")
    sess = session or _nse_session()

    end = datetime.today().date()
    start = end - timedelta(days=years * 365)

    cache_path = None
    nodata_path = None
    cached_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "DeliveryPct"])
    cached_nodata_dates = set()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{sym}.csv")
        nodata_path = os.path.join(cache_dir, f"{sym}_nodata.txt")
        if os.path.exists(cache_path):
            cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"  [{sym}] cache hit: {len(cached_df)} rows already saved, "
                  f"{cached_df.index.min().date()} to {cached_df.index.max().date()}")
        if os.path.exists(nodata_path):
            with open(nodata_path) as f:
                cached_nodata_dates = {datetime.strptime(line.strip(), "%Y-%m-%d").date()
                                        for line in f if line.strip()}
            print(f"  [{sym}] {len(cached_nodata_dates)} confirmed-no-data day(s) also cached "
                  f"(e.g. pre-listing) -- will skip re-checking these")

    trading_days = [start + timedelta(days=i) for i in range((end - start).days + 1)
                     if (start + timedelta(days=i)).weekday() < 5]

    cached_dates = set(cached_df.index.date) if not cached_df.empty else set()
    already_known = cached_dates | cached_nodata_dates
    missing_days = [d for d in trading_days if d not in already_known]

    print(f"  [{sym}] {len(missing_days)} day(s) to fetch, "
          f"{len(trading_days) - len(missing_days)} already cached/known")

    rows = []
    new_nodata_dates = []
    failed_days = []
    n_ok, n_no_symbol, n_holiday = 0, 0, 0

    for d in missing_days:
        is_holiday = False
        try:
            bhav = fetch_bhavcopy_any_era(d, session=sess)
        except NoDataForDate:
            new_nodata_dates.append(d)  # confirmed holiday -- cache permanently, never retry
            n_holiday += 1
            is_holiday = True
            bhav = None
        except Exception:
            bhav = None

        if is_holiday:
            pass
        elif bhav is None:
            failed_days.append(d)
        elif sym in bhav.index:
            r = bhav.loc[sym]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            rows.append({"date": d, "Open": r["OPEN"], "High": r["HIGH"], "Low": r["LOW"],
                         "Close": r["CLOSE"], "Volume": r["VOLUME"], "DeliveryPct": r["DELIV_PER"]})
            n_ok += 1
        else:
            n_no_symbol += 1
            new_nodata_dates.append(d)  # file fetched fine, symbol confirmed absent -- cache this fact
        time.sleep(0.25)

    # Second pass: retry genuinely-failed days once (NOT confirmed holidays -- those
    # are already cached above and correctly excluded from this retry), in case it
    # was a temporary block that's since lifted. Skippable via retry_failed=False --
    # for large multi-symbol batches, a fresh session + cooldown per symbol (just to
    # chase 1-2 residual days, often just today's not-yet-published file) adds fixed
    # overhead that multiplies badly at scale for near-zero benefit.
    n_recovered = 0
    still_failed = list(failed_days)
    if failed_days and retry_failed:
        print(f"  [{sym}] retrying {len(failed_days)} failed day(s)...")
        still_failed = []
        for d in failed_days:
            bhav = None
            try:
                bhav = fetch_bhavcopy_any_era(d, session=sess)  # reuse main session, no re-warmup
            except NoDataForDate:
                new_nodata_dates.append(d)
                n_holiday += 1
                continue
            except Exception:
                pass
            if bhav is not None:
                n_recovered += 1
                if sym in bhav.index:
                    r = bhav.loc[sym]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    rows.append({"date": d, "Open": r["OPEN"], "High": r["HIGH"], "Low": r["LOW"],
                                 "Close": r["CLOSE"], "Volume": r["VOLUME"], "DeliveryPct": r["DELIV_PER"]})
                    n_ok += 1
                else:
                    n_no_symbol += 1
                    new_nodata_dates.append(d)
            else:
                still_failed.append(d)
            time.sleep(0.25)

    n_fail = len(still_failed)
    print(f"  [{sym}] fetch summary: {n_ok} new days OK, {n_holiday} confirmed holiday(s) cached, "
          f"{n_fail} still-failed after retry ({n_recovered} recovered), "
          f"{n_no_symbol} days symbol not found in bhavcopy")

    if rows:
        new_df = pd.DataFrame(rows).set_index("date").sort_index()
        new_df.index = pd.to_datetime(new_df.index)
        combined = pd.concat([cached_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = cached_df

    if cache_path is not None and rows:
        combined.to_csv(cache_path)
        print(f"  [{sym}] cache updated: {len(combined)} total rows saved to {cache_path}")

    if nodata_path is not None and new_nodata_dates:
        all_nodata = cached_nodata_dates | set(new_nodata_dates)
        with open(nodata_path, "w") as f:
            f.write("\n".join(d.strftime("%Y-%m-%d") for d in sorted(all_nodata)))
        print(f"  [{sym}] no-data cache updated: {len(all_nodata)} confirmed-absent day(s) saved to {nodata_path}")

    combined = combined.dropna(subset=["Close", "Volume"])
    if not combined.empty:
        print(f"  [{sym}] usable history: {len(combined)} rows, {combined.index.min()} to {combined.index.max()}")
    return combined


def detect_resistance_breakouts(
    df: pd.DataFrame,
    min_resistance_years: float = MIN_RESISTANCE_YEARS,
    volume_lookback: int = VOLUME_LOOKBACK,
    volume_spike_mult: float = VOLUME_SPIKE_MULT,
    min_delivery_pct: float = MIN_DELIVERY_PCT,
) -> pd.DataFrame:
    """
    Walk the price series day by day, tracking the running ATH and the date it was set.
    Flag a day as a qualifying breakout if:
      - close > running ATH (a genuine new all-time high)
      - that ATH had stood unbroken for >= min_resistance_years
      - volume on breakout day >= volume_spike_mult x avg volume of the prior
        `volume_lookback` days (excluding the breakout day itself)
      - delivery% on breakout day >= min_delivery_pct (pulled straight from the
        DeliveryPct column, sourced from the same bhavcopy fetch as price/volume)

    Returns one row per qualifying breakout event, including the consolidation
    zone (support/resistance range) leading into it.
    """
    df = df.sort_index().copy()
    dates = df.index
    closes = df["Close"].values
    volumes = df["Volume"].values
    lows = df["Low"].values
    highs = df["High"].values
    deliv = df["DeliveryPct"].values if "DeliveryPct" in df.columns else [None] * len(df)

    ath = -np.inf
    ath_date = None
    ath_idx = None
    results = []

    for i, date in enumerate(dates):
        close = closes[i]
        volume = volumes[i]

        if ath > -np.inf:
            resistance_age_years = (date - ath_date).days / 365.25

            if close > ath and resistance_age_years >= min_resistance_years:
                if i >= volume_lookback:
                    avg_vol = volumes[i - volume_lookback:i].mean()
                    vol_ratio = volume / avg_vol if avg_vol > 0 else 0.0

                    if vol_ratio >= volume_spike_mult:
                        zone_lo = lows[ath_idx:i].min() if i > ath_idx else lows[i]
                        zone_hi = ath  # the resistance itself is the top of the zone
                        deliv_pct = deliv[i]
                        deliv_ok = (deliv_pct is not None and not pd.isna(deliv_pct)
                                    and deliv_pct >= min_delivery_pct)
                        results.append({
                            "breakout_date": date.date() if hasattr(date, "date") else date,
                            "breakout_close": round(float(close), 2),
                            "prior_resistance": round(float(ath), 2),
                            "resistance_set_on": ath_date.date() if hasattr(ath_date, "date") else ath_date,
                            "resistance_age_years": round(resistance_age_years, 1),
                            "breakout_volume": int(volume),
                            "avg_volume_20d": int(avg_vol),
                            "volume_ratio": round(float(vol_ratio), 2),
                            "zone_support": round(float(zone_lo), 2),
                            "zone_resistance": round(float(zone_hi), 2),
                            "delivery_pct": None if deliv_pct is None or pd.isna(deliv_pct) else round(float(deliv_pct), 2),
                            "delivery_ok": bool(deliv_ok),
                            "all_conditions_met": bool(deliv_ok),
                        })

        # update running ATH AFTER evaluating today (ATH known as of "yesterday" when checking breakout)
        if close > ath:
            ath = close
            ath_date = date
            ath_idx = i

    return pd.DataFrame(results)


def fetch_index_constituents(index: str = "NIFTY500", session=None) -> list:
    """
    Fetch NSE's official constituent list for a given index, e.g. NIFTY500, NIFTY50,
    NIFTYMIDCAP150. Returns a list of bare symbols (no .NS suffix).
    Source: archives.nseindia.com's published index CSVs (same reachable-from-GitHub-
    Actions domain as the bhavcopy archives -- NOT the blocked interactive API).
    """
    slug_map = {
        "NIFTY500": "ind_nifty500list.csv",
        "NIFTY50": "ind_nifty50list.csv",
        "NIFTY100": "ind_nifty100list.csv",
        "NIFTY200": "ind_nifty200list.csv",
        "NIFTYMIDCAP150": "ind_niftymidcap150list.csv",
        "NIFTYSMALLCAP250": "ind_niftysmallcap250list.csv",
        "NIFTYTOTALMARKET": "ind_niftytotalmarket_list.csv",
    }
    filename = slug_map.get(index.upper().replace(" ", ""))
    if filename is None:
        raise ValueError(f"Unknown index '{index}'. Known: {list(slug_map)}")

    url = f"https://archives.nseindia.com/content/indices/{filename}"
    sess = session or _nse_session()
    resp = sess.get(url, timeout=15)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    symbol_col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
    if symbol_col is None:
        raise ValueError(f"Could not find a Symbol column in {url}. Columns: {list(df.columns)}")

    symbols = df[symbol_col].str.strip().str.upper().tolist()
    print(f"Fetched {len(symbols)} constituents for {index}")
    return symbols


def scan_symbol(symbol: str, years_history: int = 20, session=None, retry_failed: bool = True, **kwargs) -> pd.DataFrame:
    """
    Fetch history (with delivery%) + detect qualifying breakouts, in one call.
    This is now the complete Phase 1+2 pipeline — delivery% is already included
    since it comes from the same bhavcopy fetch as price/volume.
    """
    df = fetch_history(symbol, years=years_history, session=session, retry_failed=retry_failed)
    return detect_resistance_breakouts(df, **kwargs)


# ---------------- Phase 2: Delivery % (NSE bhavcopy) ----------------

_delivery_cache: Dict[str, pd.DataFrame] = {}

# Confirmed holidays (HTTP 404 -- no bhavcopy exists at all that day) are GLOBAL:
# every symbol shares the same trading calendar, so this must be shared across ALL
# symbols in a run (in-memory) AND persisted across runs (disk), not per-symbol.
# This was a real bug in earlier versions: holidays were only cached per-symbol,
# so every one of 500 symbols independently re-discovered every holiday over the
# network -- ~500 x 830 wasted requests, enough to blow past GitHub Actions' 6hr cap.
_known_holidays: set = set()
_holidays_loaded_from = None


def _load_global_holidays(cache_dir: str = "data"):
    """Load previously-confirmed holidays from disk into the in-memory set. Call once
    at the start of a run (before processing any symbols)."""
    global _holidays_loaded_from
    if not cache_dir:
        return
    path = os.path.join(cache_dir, "_holidays.txt")
    if os.path.exists(path):
        with open(path) as f:
            dates = {datetime.strptime(line.strip(), "%Y-%m-%d").date()
                      for line in f if line.strip()}
        _known_holidays.update(dates)
        print(f"Loaded {len(dates)} known holiday(s) from {path}")
    _holidays_loaded_from = cache_dir


def _save_global_holidays(cache_dir: str = "data"):
    """Persist the in-memory holiday set back to disk. Call once at the end of a run."""
    if not cache_dir or not _known_holidays:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "_holidays.txt")
    with open(path, "w") as f:
        f.write("\n".join(d.strftime("%Y-%m-%d") for d in sorted(_known_holidays)))
    print(f"Saved {len(_known_holidays)} known holiday(s) to {path}")


class NoDataForDate(Exception):
    """Raised when NSE confirms (via HTTP 404) that no bhavcopy exists for a date --
    almost always a market holiday. Safe to cache permanently, unlike other failures
    (network blips, Akamai blocks) which might succeed on a later attempt."""
    pass


def fetch_nse_delivery_bhavcopy(date, session=None) -> pd.DataFrame:
    """
    Fetch NSE's full bhavcopy for a single trading date — this ONE file contains
    OHLCV + delivery% for every NSE-listed equity that day. This is our sole data
    source now (the interactive historical API is blocked by Akamai bot-mitigation
    on most cloud IPs; this static archive file is not).

    date: datetime.date, datetime.datetime, or 'DD-MM-YYYY' string.
    Returns a DataFrame indexed by symbol with normalized columns:
    OPEN, HIGH, LOW, CLOSE, VOLUME, DELIV_PER.
    Cached per date within a run/process since many symbols share the same day's file.
    Raises NoDataForDate on a confirmed HTTP 404 (holiday); other errors raise normally.
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%d-%m-%Y").date()
    elif isinstance(date, datetime):
        date = date.date()

    if date in _known_holidays:
        raise NoDataForDate(f"Known holiday: {date} (no network call needed)")

    cache_key = date.strftime("%d%m%Y")
    if cache_key in _delivery_cache:
        return _delivery_cache[cache_key]

    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{cache_key}.csv"
    sess = session or _nse_session()
    resp = sess.get(url, timeout=15)
    if resp.status_code == 404:
        _known_holidays.add(date)
        raise NoDataForDate(f"No bhavcopy for {date} (likely a holiday)")
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    df["SYMBOL"] = df["SYMBOL"].str.strip()
    df["SERIES"] = df["SERIES"].str.strip()
    df = df[df["SERIES"] == "EQ"]  # equity series only, drop BE/SM/etc.

    def _find_col(*candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_map = {
        "OPEN": _find_col("OPEN_PRICE", "OPEN"),
        "HIGH": _find_col("HIGH_PRICE", "HIGH"),
        "LOW": _find_col("LOW_PRICE", "LOW"),
        "CLOSE": _find_col("CLOSE_PRICE", "CLOSE"),
        "VOLUME": _find_col("TTL_TRD_QNTY", "TOTTRDQTY", "TOT_TRD_QTY"),
        "DELIV_PER": _find_col("DELIV_PER", "DELIV_PER "),
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(f"Bhavcopy for {date} missing expected columns {missing}. "
                          f"Actual columns: {list(df.columns)}")

    for std_name, actual_col in col_map.items():
        df[std_name] = pd.to_numeric(df[actual_col], errors="coerce")

    df = df.set_index("SYMBOL")[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "DELIV_PER"]]

    _delivery_cache[cache_key] = df
    return df


def fetch_legacy_bhavcopy(date, session=None) -> pd.DataFrame:
    """
    Fetch NSE's LEGACY bhavcopy zip archive (pre-~2019 era format) for a single
    trading date. Confirmed reachable back to at least 2010 (NSE's public docs
    suggest this format goes back to 1994). Contains OHLCV but NOT delivery% --
    that's fine, since delivery% is only ever checked on the (recent) breakout
    day, which will always fall within the modern format's coverage.

    Returns a DataFrame indexed by symbol with columns OPEN, HIGH, LOW, CLOSE, VOLUME, DELIV_PER (NaN).
    """
    import zipfile
    from io import BytesIO

    if isinstance(date, str):
        date = datetime.strptime(date, "%d-%m-%Y").date()
    elif isinstance(date, datetime):
        date = date.date()

    if date in _known_holidays:
        raise NoDataForDate(f"Known holiday: {date} (no network call needed)")

    cache_key = "LEGACY_" + date.strftime("%d%m%Y")
    if cache_key in _delivery_cache:
        return _delivery_cache[cache_key]

    month = date.strftime("%b").upper()
    url = (f"https://archives.nseindia.com/content/historical/EQUITIES/"
           f"{date.year}/{month}/cm{date.strftime('%d')}{month}{date.year}bhav.csv.zip")

    sess = session or _nse_session()
    resp = sess.get(url, timeout=15)
    if resp.status_code == 404:
        _known_holidays.add(date)
        raise NoDataForDate(f"No legacy bhavcopy for {date} (likely a holiday)")
    resp.raise_for_status()

    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    df.columns = [c.strip() for c in df.columns]
    df["SYMBOL"] = df["SYMBOL"].str.strip()
    df["SERIES"] = df["SERIES"].str.strip()
    df = df[df["SERIES"] == "EQ"]

    df["OPEN"] = pd.to_numeric(df["OPEN"], errors="coerce")
    df["HIGH"] = pd.to_numeric(df["HIGH"], errors="coerce")
    df["LOW"] = pd.to_numeric(df["LOW"], errors="coerce")
    df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
    df["VOLUME"] = pd.to_numeric(df["TOTTRDQTY"], errors="coerce")
    df["DELIV_PER"] = np.nan  # not available in this legacy format

    df = df.set_index("SYMBOL")[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "DELIV_PER"]]

    _delivery_cache[cache_key] = df
    return df


# Dates on/after this use the modern sec_bhavdata_full format (has delivery%).
# Dates before it fall back to the legacy zip format (no delivery%, not needed
# for old dates -- see fetch_legacy_bhavcopy docstring).
MODERN_FORMAT_CUTOVER = date(2019, 8, 23)


def fetch_bhavcopy_any_era(d, session=None) -> pd.DataFrame:
    """Routes to the modern or legacy bhavcopy fetcher depending on date."""
    if isinstance(d, str):
        d = datetime.strptime(d, "%d-%m-%Y").date()
    elif isinstance(d, datetime):
        d = d.date()

    if d >= MODERN_FORMAT_CUTOVER:
        return fetch_nse_delivery_bhavcopy(d, session=session)
    else:
        return fetch_legacy_bhavcopy(d, session=session)


def confirm_delivery(events: pd.DataFrame, symbol: str, session=None,
                      min_delivery_pct: float = MIN_DELIVERY_PCT) -> pd.DataFrame:
    """
    DEPRECATED as of v0.4 — delivery% is now included directly in scan_symbol()'s
    output since fetch_history() already pulls it from the same bhavcopy fetch as
    price/volume. Kept only for backward compatibility; just returns events unchanged.
    """
    return events


def scan_symbol_full(symbol: str, years_history: int = 20, **kwargs) -> pd.DataFrame:
    """Alias for scan_symbol() — kept for backward compatibility. Delivery% is
    already included in scan_symbol()'s output as of v0.4."""
    return scan_symbol(symbol, years_history=years_history, **kwargs)


if __name__ == "__main__":
    import os
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # ensure prints stream out in real time in CI logs
    # Usage: python3 resistance_breakout_scanner.py SYMBOL1 SYMBOL2 ...
    # Or set env vars SCANNER_SYMBOLS="SYM1,SYM2" and SCANNER_YEARS=3 to override
    # without editing this file (used by the GitHub Actions workflow_dispatch inputs).
    # Or set SCANNER_UNIVERSE="NIFTY500" to scan an entire index's constituents,
    # fetched automatically from NSE (takes priority over SCANNER_SYMBOLS if both set).
    sess = _nse_session()

    env_universe = os.environ.get("SCANNER_UNIVERSE", "").strip()
    env_symbols = os.environ.get("SCANNER_SYMBOLS", "").strip()
    if env_universe:
        symbols = fetch_index_constituents(env_universe, session=sess)
        offset = int(os.environ.get("SCANNER_OFFSET", "0") or "0")
        limit_raw = os.environ.get("SCANNER_LIMIT", "").strip()
        limit = int(limit_raw) if limit_raw else None
        if offset or limit:
            end_idx = (offset + limit) if limit else None
            symbols = symbols[offset:end_idx]
            print(f"Sliced to offset={offset}, limit={limit}: {len(symbols)} symbol(s) "
                  f"(index {offset} to {offset + len(symbols) - 1})")
    elif env_symbols:
        symbols = [s.strip().upper() for s in env_symbols.split(",") if s.strip()]
    else:
        symbols = sys.argv[1:] or ["RELIANCE", "TCS", "INFY"]

    years = int(os.environ.get("SCANNER_YEARS", "20"))

    print(f"Scanning {len(symbols)} symbol(s), {years} year(s) history each")
    if len(symbols) <= 20:
        print(f"Symbols: {symbols}")

    _load_global_holidays()

    # For large batches, skip the per-symbol retry pass -- chasing 1-2 residual
    # failed days (often just today's not-yet-published bhavcopy) isn't worth the
    # fixed overhead multiplied across hundreds of symbols. Small manual batches
    # still get the thorough retry, since completeness matters more there.
    retry_failed = len(symbols) <= 20
    if not retry_failed:
        print(f"Large batch ({len(symbols)} symbols) -- disabling per-symbol retry "
              f"pass to avoid multiplying fixed overhead; a handful of residual "
              f"failed days per symbol is an acceptable tradeoff at this scale.")

    n_qualifying_total = 0
    for i, sym in enumerate(symbols):
        print(f"\nScanning {sym} ...")
        try:
            events = scan_symbol(sym, years_history=years, session=sess, retry_failed=retry_failed)
        except Exception as e:
            print(f"  [error] {sym}: {e}")
            continue

        if events.empty:
            print("  No qualifying price+volume breakouts found with current thresholds.")
        else:
            print(events.to_string(index=False))
            n_fully_confirmed = int(events["all_conditions_met"].sum())
            n_qualifying_total += n_fully_confirmed
            print(f"  -> {n_fully_confirmed}/{len(events)} breakout(s) also cleared delivery% >= {MIN_DELIVERY_PCT}")

        # Save holiday knowledge periodically, not just at the end -- protects against
        # losing it if a very long multi-symbol run gets killed by a timeout partway.
        if (i + 1) % 20 == 0:
            _save_global_holidays()

    _save_global_holidays()

    if len(symbols) > 1:
        print(f"\n=== SUMMARY: {n_qualifying_total} fully-qualifying breakout(s) "
              f"across {len(symbols)} symbol(s) ===")
