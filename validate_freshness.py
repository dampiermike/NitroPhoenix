#!/usr/bin/env python3
"""
Freshness assertion for daily history files.

Checks that every live CSV under data/csv/history/ ends on the expected trading
day.  Exits 0 if all current, 1 (with details) if any file is stale.

Designed to be called from run_daily.sh after the fetchers complete, so that a
stale file fires the existing failure-email trap.

Expected trading day is resolved in this order:
  1. --expected-date YYYY-MM-DD CLI override
  2. yfinance SPY's last close (most reliable — accounts for NYSE holidays)
  3. Last weekday on or before today (fallback, NO holiday awareness)

Per-file tolerances live in FILE_TOL.  Static files (never updated) are skipped.

Usage:
    python3 validate_freshness.py
    python3 validate_freshness.py --expected-date 2026-05-19
    python3 validate_freshness.py --tolerance-days 1     # global slack
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ── Project root discovery (works in both Nitro and NitroPhoenix layouts) ─────

def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "csv" / "history").is_dir():
            return p
    return start.parent


SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
HIST_DIR     = PROJECT_ROOT / "data" / "csv" / "history"


# ── Per-file tolerance config ────────────────────────────────────────────────

# Default: every live file must be current (0 trading days behind).
DEFAULT_TOLERANCE = 0

# Per-file overrides: tolerance in NYSE trading days.  Add files here that have
# known publication lag (e.g. mutual funds posted next day).
FILE_TOL: dict[str, int] = {
    "VWEHX_historical_data.csv": 1,   # Vanguard mutual fund — Yahoo posts T+1
}

# Files that never update — skip the check.
STATIC_FILES: set[str] = {
    "synthetic-tqqq-ohlc-1999-2010.csv",
    "synthetic_TQQQ_close_2000_2010.csv",
}


# ── CSV parsers (handles both legacy single-header and Yahoo 3-row formats) ──

def _last_date(path: Path) -> pd.Timestamp | None:
    """Return the most recent date in the file, or None on parse failure."""
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            first_line = f.readline()
    except OSError:
        return None

    try:
        if first_line.startswith("Date,"):
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
        elif first_line.startswith("Price,"):
            # Yahoo 3-row header (the 14 macro files in NitroPhoenix)
            df = pd.read_csv(
                path, skiprows=3, header=None,
                names=["Date", "AdjClose", "Close", "High", "Low", "Open", "Volume"],
            )
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            return None
    except Exception:
        return None

    df = df.dropna(subset=["Date"])
    if df.empty:
        return None
    return df["Date"].max()


# ── Expected-trading-day resolver ────────────────────────────────────────────

def _spy_last_close() -> pd.Timestamp | None:
    """Use SPY via yfinance as the canonical 'what's the most recent NYSE
    trading day' — handles holidays accurately. Returns None if yfinance fails."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        spy = yf.download("SPY", period="7d", auto_adjust=False, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = [c[0] for c in spy.columns]
        spy = spy.reset_index()
        if spy.empty:
            return None
        return pd.to_datetime(spy["Date"].max())
    except Exception:
        return None


def _last_weekday(ref: date | None = None) -> pd.Timestamp:
    d = ref or date.today()
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return pd.Timestamp(d)


def expected_trading_day(cli_override: str | None = None) -> tuple[pd.Timestamp, str]:
    """Resolve the expected trading day.  Returns (timestamp, source_label)."""
    if cli_override:
        return pd.Timestamp(cli_override), "CLI override"
    spy = _spy_last_close()
    if spy is not None:
        return spy, "SPY (yfinance)"
    return _last_weekday(), "last weekday (fallback, no holiday awareness)"


# ── NYSE trading-day counter (using vix-from-yahoo as anchor when present) ───

def _trading_day_anchor() -> list[pd.Timestamp] | None:
    """Try to load vix-from-yahoo.csv as the canonical NYSE trading-day list."""
    vix_path = HIST_DIR / "vix-from-yahoo.csv"
    if not vix_path.exists():
        return None
    try:
        v = pd.read_csv(vix_path, encoding="utf-8-sig")
        v["Date"] = pd.to_datetime(v["Date"], format="%m/%d/%y", errors="coerce")
        return sorted(v["Date"].dropna())
    except Exception:
        return None


def trading_days_behind(last: pd.Timestamp, expected: pd.Timestamp,
                        anchor: list[pd.Timestamp] | None) -> int:
    if last >= expected:
        return 0
    if anchor:
        # Count anchor dates strictly between last and expected, plus expected
        # itself (which may not yet be in the anchor if the anchor is also stale).
        anchor_in_range = [d for d in anchor if last < d <= expected]
        if expected > (anchor[-1] if anchor else pd.Timestamp(0)) \
                and expected not in anchor_in_range:
            anchor_in_range.append(expected)
        if anchor_in_range:
            return len(anchor_in_range)
    # Fallback: weekday count (overcounts holidays — conservative)
    cur = last + pd.Timedelta(days=1)
    n = 0
    while cur <= expected:
        if cur.weekday() < 5:
            n += 1
        cur += pd.Timedelta(days=1)
    return n


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--expected-date",
                    help="Override expected trading day (YYYY-MM-DD).")
    ap.add_argument("--tolerance-days", type=int, default=DEFAULT_TOLERANCE,
                    help=f"Default tolerance (NYSE trading days). Default: {DEFAULT_TOLERANCE}")
    args = ap.parse_args()

    expected, source = expected_trading_day(args.expected_date)
    print(f"validate_freshness")
    print(f"  project root:  {PROJECT_ROOT}")
    print(f"  expected TD:   {expected.date()}  (source: {source})")

    if not HIST_DIR.is_dir():
        print(f"  ❌ history dir missing: {HIST_DIR}", file=sys.stderr)
        return 1

    files = sorted(HIST_DIR.glob("*.csv"))
    if not files:
        print(f"  ❌ no CSV files in {HIST_DIR}", file=sys.stderr)
        return 1

    anchor = _trading_day_anchor()

    width = max(len(f.name) for f in files) + 2
    print(f"\n  {'file':<{width}} {'last bar':<12} {'TD behind':>10} {'tol':>5}   status")
    print(f"  {'-' * (width + 36)}")

    fails: list[tuple[str, str]] = []
    skipped: int = 0
    for path in files:
        if path.name in STATIC_FILES:
            skipped += 1
            continue
        last = _last_date(path)
        if last is None:
            print(f"  {path.name:<{width}} {'(parse err)':<12} {'?':>10} {'?':>5}   ❌")
            fails.append((path.name, "parse failure"))
            continue

        behind = trading_days_behind(last, expected, anchor)
        tol = FILE_TOL.get(path.name, args.tolerance_days)
        ok = behind <= tol
        status = "✓" if ok else "❌"
        print(f"  {path.name:<{width}} {last.date()!s:<12} {behind:>10} {tol:>5}   {status}")
        if not ok:
            fails.append((path.name, f"{behind} TD behind (tolerance {tol})"))

    live_count = len(files) - skipped
    if fails:
        print(f"\n  ❌ FRESHNESS FAIL — {len(fails)} of {live_count} live file(s) stale:")
        for name, why in fails:
            print(f"    {name}: {why}")
        sys.stdout.flush()
        return 1

    print(f"\n  ✅ all {live_count} live files current "
          f"({skipped} static file(s) skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
