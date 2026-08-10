"""
NSE Multi-Year Resistance Breakout Scanner — v0.4
======================================================
Changelog:
  v0.4 - Confirmed via GitHub Actions test: NSE's bhavcopy archive (static files)
         is reachable, but the interactive historical API is blocked by Akamai
         bot-mitigation on cloud IPs (Oracle VM included). Dropped the historical
         API entirely — price/volume/delivery% now all come from the same daily
         bhavcopy file. This also means this scanner should run on GitHub Actions,
         not the Oracle VM, going forward.
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

import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None  # only needed for fetch_history() and fetch_nse_delivery_bhavcopy()

# ---------------- Config (tweak here) ----------------
MIN_RESISTANCE_YEARS = 5
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


def fetch_history(symbol: str, years: int = 20, session=None, cache_dir: str = None) -> pd.DataFrame:
    """
    Build daily OHLCV (+ delivery%) history for an NSE symbol from NSE's own
    bhavcopy archive — one file per trading day, each covering every symbol.
    (Not from the interactive historical API — that's blocked by Akamai
    bot-mitigation on most cloud/VM IPs; the static archive files are not.)

    Symbol should be the bare NSE symbol, e.g. 'TATASTEEL' (no .NS suffix).
    NOTE: this makes ~1 request per trading day in the window (~250/year), since
    each bhavcopy file only covers one day. For Phase 3 (many symbols), call
    fetch_nse_delivery_bhavcopy(date) once per day yourself and slice out each
    symbol, rather than calling this per-symbol (which would re-fetch the same
    daily files redundantly).
    """
    sym = symbol.upper().replace(".NS", "")
    sess = session or _nse_session()

    end = datetime.today().date()
    start = end - timedelta(days=years * 365)

    rows = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # skip Sat/Sun; holidays will just 404/error and get skipped
            try:
                bhav = fetch_nse_delivery_bhavcopy(d, session=sess)
                if sym in bhav.index:
                    r = bhav.loc[sym]
                    if isinstance(r, pd.DataFrame):  # guard against duplicate index
                        r = r.iloc[0]
                    rows.append({
                        "date": d, "Open": r["OPEN"], "High": r["HIGH"], "Low": r["LOW"],
                        "Close": r["CLOSE"], "Volume": r["VOLUME"], "DeliveryPct": r["DELIV_PER"],
                    })
            except Exception:
                pass  # holiday (404) or transient issue -- skip this day
            time.sleep(0.25)  # be polite to NSE's archive
        d += timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "DeliveryPct"])

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df = df.dropna(subset=["Close", "Volume"])
    return df


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


def scan_symbol(symbol: str, years_history: int = 20, session=None, **kwargs) -> pd.DataFrame:
    """
    Fetch history (with delivery%) + detect qualifying breakouts, in one call.
    This is now the complete Phase 1+2 pipeline — delivery% is already included
    since it comes from the same bhavcopy fetch as price/volume.
    """
    df = fetch_history(symbol, years=years_history, session=session)
    return detect_resistance_breakouts(df, **kwargs)


# ---------------- Phase 2: Delivery % (NSE bhavcopy) ----------------

_delivery_cache: Dict[str, pd.DataFrame] = {}


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
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%d-%m-%Y").date()
    elif isinstance(date, datetime):
        date = date.date()

    cache_key = date.strftime("%d%m%Y")
    if cache_key in _delivery_cache:
        return _delivery_cache[cache_key]

    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{cache_key}.csv"
    sess = session or _nse_session()
    resp = sess.get(url, timeout=15)
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
    import sys
    # Usage: python3 resistance_breakout_scanner.py SYMBOL1 SYMBOL2 ...
    symbols = sys.argv[1:] or ["RELIANCE", "TCS", "INFY"]  # placeholder list, swap for your own
    sess = _nse_session()

    for sym in symbols:
        print(f"\nScanning {sym} ...")
        try:
            events = scan_symbol(sym, years_history=20, session=sess)
        except Exception as e:
            print(f"  [error] {sym}: {e}")
            continue

        if events.empty:
            print("  No qualifying price+volume breakouts found with current thresholds.")
        else:
            print(events.to_string(index=False))
            n_fully_confirmed = int(events["all_conditions_met"].sum())
            print(f"  -> {n_fully_confirmed}/{len(events)} breakout(s) also cleared delivery% >= {MIN_DELIVERY_PCT}")