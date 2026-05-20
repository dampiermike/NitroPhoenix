#!/usr/bin/env python3
"""
Daily end-of-day Yahoo Finance fetch for NitroPhoenix.

Downloads the last 5 trading days for 16 tickers and merges any new rows into the
corresponding history CSVs under data/csv/history/.

Two file format families are handled:
  • legacy single-header (SQQQ, ^VIX): "Date,Adj Close,Close,High,Low,Open,Volume"
    with M/D/YY dates.
  • Yahoo 3-row header (14 macro series): "Price,Adj Close,Close,...";
    "Ticker,<SYM>,...";  "Date,,,,,,"; then YYYY-MM-DD data rows.

A date-stamped snapshot of each ticker is written to data/csv/daily/.

Usage:
    python3 fetch_yahoo_daily.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


# ── Paths ─────────────────────────────────────────────────────────────────────

def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / 'data' / 'csv' / 'history').is_dir():
            return p
    return start.parent


SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
HIST_DIR     = PROJECT_ROOT / 'data' / 'csv' / 'history'
DAILY_DIR    = PROJECT_ROOT / 'data' / 'csv' / 'daily'
DAILY_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_DATE_FMT = '%-m/%-d/%y'  # M/D/YY for sqqq-from-yahoo, vix-from-yahoo
MACRO_DATE_FMT  = '%Y-%m-%d'    # YYYY-MM-DD for the 14 macro files


# ── Ticker → file mapping ────────────────────────────────────────────────────
# Legacy format (Date,Adj Close,Close,High,Low,Open,Volume — single header, M/D/YY)
LEGACY_TICKERS = {
    'SQQQ': 'sqqq-from-yahoo.csv',
    '^VIX': 'vix-from-yahoo.csv',
}

# Macro format (3-row Yahoo header, YYYY-MM-DD)
MACRO_TICKERS = {
    '^TNX':     '^TNX_historical_data.csv',
    '^IRX':     '^IRX_historical_data.csv',
    'TLT':      'TLT_historical_data.csv',
    'DX-Y.NYB': 'DX-Y.NYB_historical_data.csv',
    'GLD':      'GLD_historical_data.csv',
    'SPY':      'SPY_historical_data.csv',
    'SMH':      'SMH_historical_data.csv',
    'IWM':      'IWM_historical_data.csv',
    'HYG':      'HYG_historical_data.csv',
    'LQD':      'LQD_historical_data.csv',
    'VWEHX':    'VWEHX_historical_data.csv',
    '^VVIX':    '^VVIX_historical_data.csv',
    '^VXN':     '^VXN_historical_data.csv',
    '^VIX9D':   '^VIX9D_historical_data.csv',
}


# ── Yahoo download ───────────────────────────────────────────────────────────

def download_ticker(symbol: str) -> pd.DataFrame:
    """Last 5 trading days from Yahoo; flat columns Date,Adj Close,Close,High,Low,Open,Volume."""
    raw = yf.download(symbol, period='5d', auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    raw = raw.reset_index()
    raw['Date'] = pd.to_datetime(raw['Date'])
    if 'Adj Close' not in raw.columns:
        raw['Adj Close'] = raw['Close']
    cols = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    raw = raw[[c for c in cols if c in raw.columns]].copy()
    return raw


# ── Legacy format helpers ─────────────────────────────────────────────────────

def _load_legacy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def _save_legacy(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out['Date'] = out['Date'].dt.strftime(LEGACY_DATE_FMT)
    out.to_csv(path, index=False)


def _merge_legacy(history: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    existing = set(history['Date'])
    fresh = new[~new['Date'].isin(existing)].copy()
    if fresh.empty:
        return history, 0
    merged = pd.concat([history, fresh], ignore_index=True).sort_values('Date').reset_index(drop=True)
    return merged, len(fresh)


# ── Macro format helpers ──────────────────────────────────────────────────────

def _load_macro(path: Path) -> tuple[list[str], pd.DataFrame]:
    """Return (raw_header_lines, data_df).  Preserves the 3-row Yahoo header verbatim."""
    with path.open('r') as f:
        header_lines = [f.readline() for _ in range(3)]
    data = pd.read_csv(
        path, skiprows=3, header=None,
        names=['Date', 'AdjClose', 'Close', 'High', 'Low', 'Open', 'Volume'],
    )
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date']).reset_index(drop=True)
    return header_lines, data


def _save_macro(header_lines: list[str], df: pd.DataFrame, path: Path) -> None:
    """Write header verbatim, then data rows with YYYY-MM-DD dates."""
    out = df.copy()
    out['Date'] = out['Date'].dt.strftime(MACRO_DATE_FMT)
    # Match the original column order: Date, AdjClose, Close, High, Low, Open, Volume
    out = out[['Date', 'AdjClose', 'Close', 'High', 'Low', 'Open', 'Volume']]
    with path.open('w') as f:
        for line in header_lines:
            f.write(line if line.endswith('\n') else line + '\n')
        out.to_csv(f, index=False, header=False)


def _merge_macro(history: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """new uses 'Adj Close' (with space); history uses 'AdjClose'.  Normalize first."""
    fresh = new.rename(columns={'Adj Close': 'AdjClose'}).copy()
    existing = set(history['Date'])
    fresh = fresh[~fresh['Date'].isin(existing)]
    if fresh.empty:
        return history, 0
    merged = pd.concat([history, fresh[history.columns.tolist()]], ignore_index=True) \
               .sort_values('Date').reset_index(drop=True)
    return merged, len(fresh)


# ── Snapshot helper ───────────────────────────────────────────────────────────

def _save_snapshot(symbol: str, df_new: pd.DataFrame, today_str: str, fmt: str) -> Path:
    safe_sym = symbol.replace('^', '').replace('.', '_')
    daily_path = DAILY_DIR / f"{safe_sym.lower()}_{today_str}.csv"
    snap = df_new.copy()
    snap['Date'] = snap['Date'].dt.strftime(fmt)
    snap.to_csv(daily_path, index=False)
    return daily_path


# ── Per-ticker drivers ────────────────────────────────────────────────────────

def update_legacy(symbol: str, filename: str, today_str: str) -> int:
    hist_path = HIST_DIR / filename
    df_new = download_ticker(symbol)
    if df_new.empty:
        print(f"  ⚠️  No data for {symbol}; skipping.")
        return 0
    print(f"  Downloaded {len(df_new)} rows "
          f"({df_new['Date'].min().date()} → {df_new['Date'].max().date()})")
    snap_path = _save_snapshot(symbol, df_new, today_str, LEGACY_DATE_FMT)
    print(f"  Snapshot → {snap_path.name}")
    if not hist_path.is_file():
        print(f"  ⚠️  Missing history file: {hist_path}; skipping merge.")
        return 0
    df_hist = _load_legacy(hist_path)
    df_merged, added = _merge_legacy(df_hist, df_new)
    if added == 0:
        print(f"  History already up to date for {filename}.")
    else:
        _save_legacy(df_merged, hist_path)
        print(f"  ✅ Added {added} row(s) to {filename}  "
              f"(now {len(df_merged)} rows, latest {df_merged['Date'].max().date()})")
    return added


def update_macro(symbol: str, filename: str, today_str: str) -> int:
    hist_path = HIST_DIR / filename
    df_new = download_ticker(symbol)
    if df_new.empty:
        print(f"  ⚠️  No data for {symbol}; skipping.")
        return 0
    print(f"  Downloaded {len(df_new)} rows "
          f"({df_new['Date'].min().date()} → {df_new['Date'].max().date()})")
    snap_path = _save_snapshot(symbol, df_new, today_str, MACRO_DATE_FMT)
    print(f"  Snapshot → {snap_path.name}")
    if not hist_path.is_file():
        print(f"  ⚠️  Missing history file: {hist_path}; skipping merge.")
        return 0
    header_lines, df_hist = _load_macro(hist_path)
    df_merged, added = _merge_macro(df_hist, df_new)
    if added == 0:
        print(f"  History already up to date for {filename}.")
    else:
        _save_macro(header_lines, df_merged, hist_path)
        print(f"  ✅ Added {added} row(s) to {filename}  "
              f"(now {len(df_merged)} rows, latest {df_merged['Date'].max().date()})")
    return added


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    print(f"fetch_yahoo_daily  {today_str}")
    print(f"project root: {PROJECT_ROOT}\n")

    total_added = 0
    failures: list[str] = []

    print("── Legacy format (SQQQ, ^VIX) ─────────────────────────────")
    for symbol, fn in LEGACY_TICKERS.items():
        print(f"\n── {symbol} → {fn}")
        try:
            total_added += update_legacy(symbol, fn, today_str)
        except Exception as e:
            print(f"  ❌ {symbol} failed: {e}")
            failures.append(symbol)

    print("\n── Macro format (14 Yahoo CSVs) ─────────────────────────")
    for symbol, fn in MACRO_TICKERS.items():
        print(f"\n── {symbol} → {fn}")
        try:
            total_added += update_macro(symbol, fn, today_str)
        except Exception as e:
            print(f"  ❌ {symbol} failed: {e}")
            failures.append(symbol)

    print(f"\nDone.  Rows added across all files: {total_added}.")
    if failures:
        print(f"⚠️  Failures: {', '.join(failures)}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
