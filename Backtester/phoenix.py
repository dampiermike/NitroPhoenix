"""
Dampier Nitro++ Phoenix backtest engine.

Built strictly from Dampier_Nitro_PlusPlus_Phoenix_Build_Guide.docx.
v14 trend engine + R21 regime-conditional sizing + Capitulation Bounce.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# Resolve relative to this file so the engine works wherever the project is
# checked out (Backtester/ sits one level below the project root).
DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "csv", "history"))

BACKTEST_START = pd.Timestamp("2000-01-03")
BACKTEST_END = pd.Timestamp("2026-04-01")
SQQQ_CUTOFF = pd.Timestamp("2010-02-11")
TQQQ_STITCH_DATE = pd.Timestamp("2010-03-31")


# ----------------------------------------------------------------------------
# Step 1 — Load and merge
# ----------------------------------------------------------------------------

def _read_qqq_vv():
    path = os.path.join(DATA_DIR, "qqq-from-vv.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"Open": "QQQ_Open", "High": "QQQ_High",
                            "Low": "QQQ_Low", "Close": "QQQ_Close",
                            "Volume": "QQQ_Volume", "RT": "QQQ_RT"})
    return df


def _read_tqqq_vv():
    path = os.path.join(DATA_DIR, "tqqq-from-vv.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"Open": "TQQQ_open", "High": "TQQQ_high",
                            "Low": "TQQQ_low", "Close": "TQQQ_close",
                            "RT": "TQQQ_rt"})
    return df[["Date", "TQQQ_open", "TQQQ_high", "TQQQ_low", "TQQQ_close", "TQQQ_rt"]]


def _read_tqqq_synth():
    path = os.path.join(DATA_DIR, "synthetic-tqqq-ohlc-1999-2010.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"Open": "TQQQ_open", "High": "TQQQ_high",
                            "Low": "TQQQ_low", "Close": "TQQQ_close",
                            "RT_v6": "TQQQ_rt"})
    return df[["Date", "TQQQ_open", "TQQQ_high", "TQQQ_low", "TQQQ_close", "TQQQ_rt"]]


def _read_vv_views():
    path = os.path.join(DATA_DIR, "vectorvest-views-w3place-precision.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"VVC-RT": "RT", "BS Ratio": "BSR"})
    df["BSR"] = pd.to_numeric(df["BSR"], errors="coerce")
    return df[["Date", "Trend", "RT", "MTI", "BSR"]]


def _read_vix():
    path = os.path.join(DATA_DIR, "vix-from-yahoo.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"Close": "VIX"})
    return df[["Date", "VIX"]]


def _read_sqqq():
    path = os.path.join(DATA_DIR, "sqqq-from-yahoo.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _stitch_tqqq():
    synth = _read_tqqq_synth()
    real = _read_tqqq_vv()
    # Real wins at boundary: synth for Date < 2010-03-31, real for Date >= 2010-03-31
    synth_part = synth[synth["Date"] < TQQQ_STITCH_DATE]
    real_part = real[real["Date"] >= TQQQ_STITCH_DATE]
    out = pd.concat([synth_part, real_part], ignore_index=True)
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def _compute_tqqq_atr_full(tqqq: pd.DataFrame) -> pd.DataFrame:
    """Simple 10-bar ATR% computed on full stitched TQQQ series (pre-2000 included)."""
    high = tqqq["TQQQ_high"].values
    low = tqqq["TQQQ_low"].values
    close = tqqq["TQQQ_close"].values
    n = len(tqqq)
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr_pct = np.full(n, np.nan)
    for i in range(9, n):
        atr_pct[i] = np.mean(tr[i - 9:i + 1]) / close[i] * 100
    tqqq = tqqq.copy()
    tqqq["TQQQ_atr"] = atr_pct
    return tqqq


def _compute_atr14(df: pd.DataFrame) -> np.ndarray:
    """Wilder ATR14 on QQQ in dollars, computed after the date filter."""
    high = df["QQQ_High"].values
    low = df["QQQ_Low"].values
    close = df["QQQ_Close"].values
    n = len(df)
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    atr[13] = np.mean(tr[0:14])
    for i in range(14, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    return atr


def _compute_cc(trend: pd.Series) -> np.ndarray:
    cc = np.empty(len(trend), dtype=object)
    cur = "C/Up"  # initial; first concrete C/Up / C/Dn overrides
    for i, t in enumerate(trend):
        if t == "C/Up":
            cur = "C/Up"
        elif t == "C/Dn":
            cur = "C/Dn"
        cc[i] = cur
    return cc


def build_step1(end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build the merged engine DataFrame.

    end_date: cap the backtest window at this date.  Defaults to BACKTEST_END
              (2026-04-01) so the historical fingerprints reproduce byte-exact.
              Pass a later date for live daily operation.
    """
    qqq = _read_qqq_vv()
    tqqq = _stitch_tqqq()
    tqqq = _compute_tqqq_atr_full(tqqq)  # full-series ATR before merge

    vv = _read_vv_views()
    vix = _read_vix()
    sqqq = _read_sqqq()

    # LEFT-join TQQQ onto QQQ (preserves all QQQ rows incl. 5 NYSE-closed placeholders)
    df = qqq.merge(tqqq, on="Date", how="left")
    df = df.merge(vv, on="Date", how="left")
    df = df.merge(vix, on="Date", how="left")

    # Return columns computed on FULL merged df (before window filter), so row 0
    # after filter has prior trading day's close available.
    df = df.sort_values("Date").reset_index(drop=True)

    df["c2c_QQQ"] = df["QQQ_Close"].pct_change()
    df["o2c_QQQ"] = df["QQQ_Close"] / df["QQQ_Open"] - 1
    df["pc2o_QQQ"] = df["QQQ_Open"] / df["QQQ_Close"].shift(1) - 1

    # PSQ baseline (pre-2010): 1× inverse QQQ; c2c carries borrow cost.
    df["o2c_PSQ"] = -1 * df["o2c_QQQ"]
    df["pc2o_PSQ"] = -1 * df["pc2o_QQQ"]
    df["c2c_PSQ"] = -1 * df["c2c_QQQ"] - 0.0095 / 252

    # Real SQQQ at 1/3 position for post-2010-02-11
    sqqq_adj_factor = sqqq["Adj Close"] / sqqq["Close"]
    sqqq_open_adj = sqqq["Open"] * sqqq_adj_factor
    sqqq_close_adj = sqqq["Adj Close"]
    o2c_SQQQ3 = (sqqq_close_adj / sqqq_open_adj - 1) / 3
    c2c_SQQQ3 = sqqq_close_adj.pct_change() / 3
    pc2o_SQQQ3 = (sqqq_open_adj / sqqq_close_adj.shift(1) - 1) / 3
    sqqq_returns = pd.DataFrame({"Date": sqqq["Date"],
                                 "o2c_SQQQ3": o2c_SQQQ3,
                                 "c2c_SQQQ3": c2c_SQQQ3,
                                 "pc2o_SQQQ3": pc2o_SQQQ3})
    df = df.merge(sqqq_returns, on="Date", how="left")
    mask = df["Date"] >= SQQQ_CUTOFF
    df.loc[mask, "o2c_PSQ"] = df.loc[mask, "o2c_SQQQ3"]
    df.loc[mask, "c2c_PSQ"] = df.loc[mask, "c2c_SQQQ3"]
    df.loc[mask, "pc2o_PSQ"] = df.loc[mask, "pc2o_SQQQ3"]
    df = df.drop(columns=["o2c_SQQQ3", "c2c_SQQQ3", "pc2o_SQQQ3"])

    # TQQQ returns
    df["o2c_TQQQ"] = df["TQQQ_close"] / df["TQQQ_open"] - 1
    df["c2c_TQQQ"] = df["TQQQ_close"].pct_change(fill_method=None)
    df["pc2o_TQQQ"] = df["TQQQ_open"] / df["TQQQ_close"].shift(1) - 1

    # Apply explicit backtest window AFTER computing returns.
    window_end = end_date if end_date is not None else BACKTEST_END
    df = df[(df["Date"] >= BACKTEST_START) & (df["Date"] <= window_end)].reset_index(drop=True)

    # ATR14 (Wilder, on QQQ, dollars) computed on filtered df.
    df["ATR14"] = _compute_atr14(df)

    # cc (carry-forward Confirmed Call) from Trend
    df["cc"] = _compute_cc(df["Trend"])

    # qqq_5d_ret for Step 8 Cap trigger
    df["qqq_5d_ret"] = df["QQQ_Close"] / df["QQQ_Close"].shift(5) - 1

    # DEW_Signal: computed on the raw QQQ series (Jul 1999 onward) for warmup, then merged.
    df = _attach_dew_signal(df)

    return df


# ----------------------------------------------------------------------------
# Step 2 — DEW Oscillator
# ----------------------------------------------------------------------------

def _attach_dew_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Compute DEW state machine on full raw QQQ series, left-merge onto df."""
    raw = _read_qqq_vv()  # 1999-07-01 onward
    close = raw["QQQ_Close"].values
    n = len(close)

    # D = (Close - Close.rolling(20).mean().shift(11)).round(2)
    sma20 = pd.Series(close).rolling(20).mean().shift(11).values
    d_arr = np.round(close - sma20, 2)

    # EMA10 and bands
    ema10 = pd.Series(close).ewm(span=10, adjust=False).mean().values
    e_upper = ema10 * 1.06
    e_lower = ema10 * 0.94

    # W: weighted MA with weights 1..30 (newest=30), divisor=465
    weights = np.arange(1, 31, dtype=float)
    w_arr = np.full(n, np.nan)
    for i in range(29, n):
        w_arr[i] = np.sum(close[i - 29:i + 1] * weights) / 465.0

    state = "Normal"
    last_fired = None
    sig = np.array([""] * n, dtype=object)
    for i in range(41, n):
        c = close[i]
        d_v = d_arr[i]
        w_v = w_arr[i]
        if np.isnan(d_v) or np.isnan(w_v):
            continue

        if state == "Normal":
            cand = "Buy" if (c > w_v and d_v > 0) else ("Sell" if (c < w_v and d_v < 0) else None)
        else:  # Primed → OR logic
            cand = "Buy" if (c > w_v or d_v > 0) else ("Sell" if (c < w_v or d_v < 0) else None)

        if cand == last_fired:
            cand = None

        if cand:
            sig[i] = cand
            last_fired = cand
            state = "Normal"

        if c > e_upper[i] or c < e_lower[i]:
            state = "Primed"

    raw_sig = pd.DataFrame({"Date": raw["Date"], "DEW_Signal": sig})
    df = df.merge(raw_sig, on="Date", how="left")
    df["DEW_Signal"] = df["DEW_Signal"].fillna("")
    return df


# ----------------------------------------------------------------------------
# Step 1 verification
# ----------------------------------------------------------------------------

def verify_step1(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Step 1 — Load and Merge verification")
    print("=" * 70)
    checks = []

    rows = df.shape[0]
    checks.append(("df.shape[0]", rows, 6606, rows == 6606))

    atr14_13 = round(float(df.loc[13, "ATR14"]), 4)
    checks.append(("ATR14[13]", atr14_13, 4.8201, atr14_13 == 4.8201))

    vix_covid = float(df.loc[df["Date"] == "2020-03-16", "VIX"].iloc[0])
    checks.append(("VIX 2020-03-16", round(vix_covid, 2), 82.69, round(vix_covid, 2) == 82.69))

    tqqq_atr_first = round(float(df["TQQQ_atr"].dropna().iloc[0]), 4)
    checks.append(("TQQQ_atr first non-NaN", tqqq_atr_first, 7.9588, tqqq_atr_first == 7.9588))

    cc_head = df["cc"].head(3).tolist()
    checks.append(("cc.head(3)", cc_head, ["C/Up", "C/Up", "C/Up"], cc_head == ["C/Up", "C/Up", "C/Up"]))

    def _spot(date, exp_rt, exp_mti, exp_bsr):
        row = df.loc[df["Date"] == date].iloc[0]
        got = (round(float(row["RT"]), 2), round(float(row["MTI"]), 2), round(float(row["BSR"]), 2))
        exp = (exp_rt, exp_mti, exp_bsr)
        return got, exp, got == exp

    g, e, ok = _spot("2005-07-06", 1.08, 1.43, 2.49)
    checks.append(("Spot 2005-07-06 (RT,MTI,BSR)", g, e, ok))

    g, e, ok = _spot("2015-09-24", 0.85, 0.63, 0.16)
    checks.append(("Spot 2015-09-24 (RT,MTI,BSR)", g, e, ok))

    width_label = 36
    width_val = 22
    print(f"{'Metric':<{width_label}}{'Got':<{width_val}}{'Target':<{width_val}}Match?")
    print("─" * (width_label + 2 * width_val + 6))
    for label, got, target, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"{label:<{width_label}}{str(got):<{width_val}}{str(target):<{width_val}}{mark}")
    print()

    failed = [c for c in checks if not c[-1]]
    if failed:
        raise AssertionError(f"Step 1 verification failed: {len(failed)} of {len(checks)} checks")


def verify_step2(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Step 2 — DEW Oscillator verification")
    print("=" * 70)
    checks = []

    n_buy = int((df["DEW_Signal"] == "Buy").sum())
    checks.append(("DEW Buy count", n_buy, 194, n_buy == 194))

    n_sell = int((df["DEW_Signal"] == "Sell").sum())
    checks.append(("DEW Sell count", n_sell, 195, n_sell == 195))

    n_empty = int((df["DEW_Signal"] == "").sum())
    checks.append(("DEW empty count", n_empty, 6217, n_empty == 6217))

    # First 3 signals
    fired = df[df["DEW_Signal"].isin(["Buy", "Sell"])].head(3)
    expected = [
        ("2000-01-28", "Sell", "C/Up"),
        ("2000-02-01", "Buy", "C/Up"),
        ("2000-04-03", "Sell", "C/Dn"),
    ]
    for (date_exp, sig_exp, cc_exp), (_, row) in zip(expected, fired.iterrows()):
        date_got = row["Date"].strftime("%Y-%m-%d")
        sig_got = row["DEW_Signal"]
        cc_got = row["cc"]
        ok = (date_got == date_exp and sig_got == sig_exp and cc_got == cc_exp)
        checks.append((f"First signal {date_exp}", f"{date_got} {sig_got} {cc_got}",
                       f"{date_exp} {sig_exp} {cc_exp}", ok))

    width_label = 36
    width_val = 32
    print(f"{'Metric':<{width_label}}{'Got':<{width_val}}{'Target':<{width_val}}Match?")
    print("─" * (width_label + 2 * width_val + 6))
    for label, got, target, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"{label:<{width_label}}{str(got):<{width_val}}{str(target):<{width_val}}{mark}")
    print()

    failed = [c for c in checks if not c[-1]]
    if failed:
        raise AssertionError(f"Step 2 verification failed: {len(failed)} of {len(checks)} checks")


# ----------------------------------------------------------------------------
# Performance metrics
# ----------------------------------------------------------------------------

def compute_perf(dates: pd.Series, dret: np.ndarray, initial: float = 100_000.0) -> dict:
    eq = initial * np.cumprod(1.0 + dret)
    final_eq = float(eq[-1])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr = (final_eq / initial) ** (1.0 / years) - 1.0
    if np.std(dret, ddof=1) > 0:
        sharpe = float(np.mean(dret) / np.std(dret, ddof=1) * np.sqrt(252))
    else:
        sharpe = float("nan")
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min())
    return {"final": final_eq, "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "eq": eq}


# ----------------------------------------------------------------------------
# Step 3 — Isolated trade type functions
# ----------------------------------------------------------------------------

def step3a_qqq_long(df: pd.DataFrame) -> tuple[list, np.ndarray]:
    """QQQ long, isolated. Entry: DEW Buy and last_dew!='Buy'. Exits: TP +5%, ATR stop, C/Dn flip, DEW Sell."""
    n = len(df)
    Open = df["QQQ_Open"].values
    Close = df["QQQ_Close"].values
    ATR = df["ATR14"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values
    Date = df["Date"].values
    o2c = df["o2c_QQQ"].values
    c2c = df["c2c_QQQ"].values
    pc2o = df["pc2o_QQQ"].values

    dret = np.zeros(n)
    trades = []
    in_trade = False
    entry_idx = None
    cum_mult = 1.0
    stop_level = None
    last_dew = None
    prev_cc = cc[0]

    for i in range(n):
        # CC transitions reset last_dew (regime change)
        if i > 0 and cc[i] != prev_cc:
            last_dew = None
        prev_cc = cc[i]

        # Phase B — exit if in trade
        if in_trade:
            d = o2c[i] if i == entry_idx else c2c[i]
            cum_mult *= (1.0 + d)

            exit_reason = None
            # Priority: TP > ATR stop > C/Dn flip > DEW Sell
            if cum_mult - 1.0 >= 0.05:
                exit_reason = "TP"
            elif Close[i] <= stop_level:
                exit_reason = "ATR"
            elif Trend[i] == "C/Dn" and cc[i - 1] == "C/Up" and i != entry_idx:
                exit_reason = "Flip"
            elif DEW[i] == "Sell":
                exit_reason = "DEW"

            if exit_reason is not None and i + 1 < n:
                exit_fill_idx = i + 1
                dret[i] = (1.0 + d) * (1.0 + pc2o[i + 1]) - 1.0
                trade_ret = cum_mult * (1.0 + pc2o[i + 1]) - 1.0
                trades.append({
                    "entry_fill": Date[entry_idx],
                    "exit_fill": Date[exit_fill_idx],
                    "instrument": "QQQ",
                    "ret": trade_ret,
                    "reason": exit_reason,
                })
                in_trade = False
                entry_idx = None
                cum_mult = 1.0
                stop_level = None
                if exit_reason == "DEW":
                    last_dew = "Sell"
                # On TP/ATR/Flip: last_dew unchanged (per spec for combined; in isolation it's same)
                # In isolation, reset last_dew=None on every exit (per 3d-style note? No—3a doesn't say that).
                # 3a spec doesn't explicitly reset last_dew. Per "last_dew update rules" table:
                #   QQQ exits via TP/ATR — unchanged
                #   QQQ exits via Flip — flip_blocks (same bar) — does NOT change last_dew
                # So keep current behavior.
                continue
            else:
                dret[i] = d
                continue

        # Phase C — entry
        if DEW[i] == "Buy" and last_dew != "Buy" and i + 1 < n:
            in_trade = True
            entry_idx = i + 1
            cum_mult = 1.0
            atr_mult = 2.0 if cc[i] == "C/Dn" else 1.0
            stop_level = Open[i + 1] - atr_mult * ATR[i]
            last_dew = "Buy"
            # No dret on signal bar — fill happens at i+1's open.

    return trades, dret


def step3b_psq_short(df: pd.DataFrame) -> tuple[list, np.ndarray]:
    """PSQ short. Entry: DEW Sell and last_dew!='Sell' and cc=='C/Dn'."""
    n = len(df)
    Open = df["QQQ_Open"].values
    Close = df["QQQ_Close"].values
    ATR = df["ATR14"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values
    Date = df["Date"].values
    o2c = df["o2c_PSQ"].values
    c2c = df["c2c_PSQ"].values
    pc2o = df["pc2o_PSQ"].values

    dret = np.zeros(n)
    trades = []
    in_trade = False
    entry_idx = None
    cum_mult = 1.0
    stop_level = None
    last_dew = None
    prev_cc = cc[0]

    for i in range(n):
        if i > 0 and cc[i] != prev_cc:
            last_dew = None
        prev_cc = cc[i]

        if in_trade:
            d = o2c[i] if i == entry_idx else c2c[i]
            cum_mult *= (1.0 + d)

            exit_reason = None
            if cum_mult - 1.0 >= 0.05:
                exit_reason = "TP"
            elif Close[i] >= stop_level:  # PSQ stop fires when QQQ rallies through it
                exit_reason = "ATR"
            elif Trend[i] == "C/Up" and cc[i - 1] == "C/Dn" and i != entry_idx:
                exit_reason = "Flip"
            elif DEW[i] == "Buy":
                exit_reason = "DEW"

            if exit_reason is not None and i + 1 < n:
                exit_fill_idx = i + 1
                dret[i] = (1.0 + d) * (1.0 + pc2o[i + 1]) - 1.0
                trade_ret = cum_mult * (1.0 + pc2o[i + 1]) - 1.0
                trades.append({
                    "entry_fill": Date[entry_idx],
                    "exit_fill": Date[exit_fill_idx],
                    "instrument": "PSQ",
                    "ret": trade_ret,
                    "reason": exit_reason,
                })
                in_trade = False
                entry_idx = None
                cum_mult = 1.0
                stop_level = None
                if exit_reason == "DEW":
                    last_dew = "Buy"
                continue
            else:
                dret[i] = d
                continue

        # Phase C — entry
        if DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn" and i + 1 < n:
            in_trade = True
            entry_idx = i + 1
            cum_mult = 1.0
            stop_level = Open[i + 1] + 2.0 * ATR[i]
            last_dew = "Sell"

    return trades, dret


def step3c_cup_tqqq(df: pd.DataFrame) -> tuple[list, np.ndarray]:
    """C/Up TQQQ, isolated. cup_entered resets on every exit in this isolation."""
    n = len(df)
    cc = df["cc"].values
    Trend = df["Trend"].values
    DEW = df["DEW_Signal"].values
    Date = df["Date"].values

    TQQQ_open = df["TQQQ_open"].values
    TQQQ_close = df["TQQQ_close"].values
    TQQQ_rt = df["TQQQ_rt"].values
    TQQQ_atr = df["TQQQ_atr"].values
    BSR = df["BSR"].values
    MTI = df["MTI"].values
    VIX = df["VIX"].values

    o2c = df["o2c_TQQQ"].values
    c2c = df["c2c_TQQQ"].values
    pc2o = df["pc2o_TQQQ"].values

    dret = np.zeros(n)
    trades = []
    in_trade = False
    entry_idx = None
    cum_mult = 1.0
    tqqq_target = None
    tqqq_stop = None
    be_moved = False
    cup_entered = False
    prev_cc = cc[0]

    exit_counts = {"TP": 0, "Stop": 0, "MTI": 0, "MaxHold": 0, "Regime": 0, "DEW": 0, "Signal": 0}

    for i in range(n):
        if i > 0 and cc[i] != prev_cc:
            # In isolated 3c, cup_entered does reset on new C/Up; spec says reset on every exit too.
            pass
        prev_cc = cc[i]

        if in_trade:
            d = o2c[i] if i == entry_idx else c2c[i]
            cum_mult *= (1.0 + d)

            # Breakeven ratchet at +9%
            if not be_moved and cum_mult - 1.0 >= 0.09:
                tqqq_stop = TQQQ_open[entry_idx]
                be_moved = True

            # Exit priority: MTI > TP > Stop > MaxHold > Regime > DEW Sell
            exit_reason = None
            if i != entry_idx and not np.isnan(MTI[i]) and MTI[i] < 0.75:
                exit_reason = "MTI"
            elif not np.isnan(TQQQ_close[i]) and TQQQ_close[i] >= tqqq_target:
                exit_reason = "TP"
            elif not np.isnan(TQQQ_close[i]) and TQQQ_close[i] <= tqqq_stop:
                exit_reason = "Stop"
            elif (i - entry_idx) >= 70:
                exit_reason = "MaxHold"
            elif cc[i] == "C/Dn":
                exit_reason = "Regime"
            elif DEW[i] == "Sell":
                exit_reason = "DEW"

            if exit_reason is not None and i + 1 < n:
                exit_fill_idx = i + 1
                dret[i] = (1.0 + d) * (1.0 + pc2o[i + 1]) - 1.0
                trade_ret = cum_mult * (1.0 + pc2o[i + 1]) - 1.0
                trades.append({
                    "entry_fill": Date[entry_idx],
                    "exit_fill": Date[exit_fill_idx],
                    "instrument": "TQQQ_cup",
                    "ret": trade_ret,
                    "reason": exit_reason,
                })
                exit_counts[exit_reason] = exit_counts.get(exit_reason, 0) + 1
                in_trade = False
                entry_idx = None
                cum_mult = 1.0
                tqqq_target = None
                tqqq_stop = None
                be_moved = False
                cup_entered = False  # isolation: reset on every exit
                continue
            else:
                dret[i] = d
                continue

        # Phase C — entry (only one TQQQ entry per C/Up "session" via cup_entered guard)
        if (cc[i] == "C/Up"
                and not np.isnan(TQQQ_rt[i]) and TQQQ_rt[i] < 1.40
                and not np.isnan(BSR[i]) and BSR[i] > 1.05
                and not np.isnan(TQQQ_atr[i]) and TQQQ_atr[i] < 7.0
                and not np.isnan(VIX[i]) and VIX[i] < 30
                and not cup_entered
                and i + 1 < n
                and not np.isnan(TQQQ_open[i + 1])):
            in_trade = True
            entry_idx = i + 1
            cum_mult = 1.0
            tqqq_target = TQQQ_open[i + 1] * 1.50
            tqqq_stop = TQQQ_open[i + 1] * 0.94
            be_moved = False
            cup_entered = True

    return trades, dret, exit_counts


def step3d_cdn_tqqq(df: pd.DataFrame) -> tuple[list, np.ndarray, dict]:
    """C/Dn TQQQ, isolated. Reset last_dew=None on every exit so cdn_f can re-fire in the same C/Dn run."""
    n = len(df)
    cc = df["cc"].values
    Trend = df["Trend"].values
    DEW = df["DEW_Signal"].values
    Date = df["Date"].values

    TQQQ_open = df["TQQQ_open"].values
    TQQQ_close = df["TQQQ_close"].values
    RT = df["RT"].values
    MTI = df["MTI"].values

    o2c = df["o2c_TQQQ"].values
    c2c = df["c2c_TQQQ"].values
    pc2o = df["pc2o_TQQQ"].values

    dret = np.zeros(n)
    trades = []
    in_trade = False
    entry_idx = None
    cum_mult = 1.0
    tqqq_target = None
    tqqq_stop = None
    be_moved = False
    last_dew = None
    prev_cc = cc[0]

    exit_counts = {"TP": 0, "Stop": 0, "MTI": 0, "MaxHold": 0, "DEW": 0, "Flip": 0}

    for i in range(n):
        if i > 0 and cc[i] != prev_cc:
            last_dew = None
        prev_cc = cc[i]

        if in_trade:
            d = o2c[i] if i == entry_idx else c2c[i]
            cum_mult *= (1.0 + d)

            if not be_moved and cum_mult - 1.0 >= 0.09:
                tqqq_stop = TQQQ_open[entry_idx]
                be_moved = True

            exit_reason = None
            # Priority: MTI > TP > Stop > MaxHold > DEW Sell > C/Up flip
            # The spec note: in 3d, C/Up flip has NO entry-bar guard.
            if i != entry_idx and not np.isnan(MTI[i]) and MTI[i] < 0.75:
                exit_reason = "MTI"
            elif not np.isnan(TQQQ_close[i]) and TQQQ_close[i] >= tqqq_target:
                exit_reason = "TP"
            elif not np.isnan(TQQQ_close[i]) and TQQQ_close[i] <= tqqq_stop:
                exit_reason = "Stop"
            elif (i - entry_idx) >= 70:
                exit_reason = "MaxHold"
            elif DEW[i] == "Sell":
                exit_reason = "DEW"
            elif Trend[i] == "C/Up" and cc[i - 1] == "C/Dn":  # no i != entry_idx guard
                exit_reason = "Flip"

            if exit_reason is not None and i + 1 < n:
                exit_fill_idx = i + 1
                dret[i] = (1.0 + d) * (1.0 + pc2o[i + 1]) - 1.0
                trade_ret = cum_mult * (1.0 + pc2o[i + 1]) - 1.0
                trades.append({
                    "entry_fill": Date[entry_idx],
                    "exit_fill": Date[exit_fill_idx],
                    "instrument": "TQQQ_cdn",
                    "ret": trade_ret,
                    "reason": exit_reason,
                })
                exit_counts[exit_reason] = exit_counts.get(exit_reason, 0) + 1
                in_trade = False
                entry_idx = None
                cum_mult = 1.0
                tqqq_target = None
                tqqq_stop = None
                be_moved = False
                last_dew = None  # isolation: reset on every exit
                continue
            else:
                dret[i] = d
                continue

        # Phase C entry
        if (cc[i] == "C/Dn"
                and DEW[i] == "Buy" and last_dew != "Buy"
                and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                and not np.isnan(MTI[i]) and MTI[i] < 1.00
                and i + 1 < n
                and not np.isnan(TQQQ_open[i + 1])):
            in_trade = True
            entry_idx = i + 1
            cum_mult = 1.0
            tqqq_target = TQQQ_open[i + 1] * 1.30
            tqqq_stop = TQQQ_open[i + 1] * 0.94
            be_moved = False
            last_dew = "Buy"

    return trades, dret, exit_counts


# ----------------------------------------------------------------------------
# Step 6 — Macro signals and 8-vote bearish gate
# ----------------------------------------------------------------------------

SERIES_FILES = {
    "TNX": "^TNX_historical_data.csv",
    "IRX": "^IRX_historical_data.csv",
    "TLT": "TLT_historical_data.csv",
    "DXY": "DX-Y.NYB_historical_data.csv",
    "GLD": "GLD_historical_data.csv",
    "SPY": "SPY_historical_data.csv",
    "SMH": "SMH_historical_data.csv",
    "IWM": "IWM_historical_data.csv",
    "HYG": "HYG_historical_data.csv",
    "LQD": "LQD_historical_data.csv",
    "VWEHX": "VWEHX_historical_data.csv",
    "VVIX": "^VVIX_historical_data.csv",
    "VXN": "^VXN_historical_data.csv",
    "VIX9D": "^VIX9D_historical_data.csv",
}


def _load_yahoo(filename: str, name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    raw = pd.read_csv(path, skiprows=3, header=None,
                      names=["Date", "AdjClose", "Close", "High", "Low", "Open", "Volume"])
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).reset_index(drop=True)
    raw = raw.sort_values("Date").reset_index(drop=True)
    return raw[["Date", "Close"]].rename(columns={"Close": f"{name}_close"})


def attach_macro_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Load 14 Yahoo CSVs, apply .shift(1) for causal lag, merge + ffill(5)."""
    df = df.copy()
    for name, fn in SERIES_FILES.items():
        s = _load_yahoo(fn, name)
        s[f"ext_{name}"] = s[f"{name}_close"].shift(1)  # causal lag
        df = df.merge(s[["Date", f"ext_{name}"]], on="Date", how="left")
    ext_cols = [c for c in df.columns if c.startswith("ext_")]
    df[ext_cols] = df[ext_cols].ffill(limit=5)
    return df


def _zscore(arr: np.ndarray, w: int, mp: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return ((s - s.rolling(w, min_periods=mp).mean())
            / s.rolling(w, min_periods=mp).std()).values


def build_step6_signals(df: pd.DataFrame) -> dict:
    """Compute the 8 source signals used by the bearish gate."""
    sigs = {}
    sigs["QQQ_over_SPY_z60"] = _zscore(df["QQQ_Close"].values / df["ext_SPY"].values, w=60, mp=30)
    sigs["VIX_5d_chg"] = pd.Series(df["VIX"].values).pct_change(5, fill_method=None).values
    sigs["VWEHX_20d_ret"] = pd.Series(df["ext_VWEHX"].values).pct_change(20, fill_method=None).values
    iwm_20 = pd.Series(df["ext_IWM"].values).pct_change(20, fill_method=None).values
    spy_20 = pd.Series(df["ext_SPY"].values).pct_change(20, fill_method=None).values
    sigs["IWM_minus_SPY_20d"] = iwm_20 - spy_20
    sigs["VXN_over_VIX"] = df["ext_VXN"].values / df["VIX"].values
    sigs["SMH_over_QQQ_z60"] = _zscore(df["ext_SMH"].values / df["QQQ_Close"].values, w=60, mp=30)
    sigs["TNX_minus_IRX"] = df["ext_TNX"].values - df["ext_IRX"].values
    sigs["TLT_20d_ret"] = pd.Series(df["ext_TLT"].values).pct_change(20, fill_method=None).values
    return sigs


BLK = [
    ("QQQ_over_SPY_z60", "bot", 0.20),
    ("VIX_5d_chg", "top", 0.80),
    ("VWEHX_20d_ret", "top", 0.80),
    ("IWM_minus_SPY_20d", "top", 0.80),
    ("VXN_over_VIX", "bot", 0.20),
    ("SMH_over_QQQ_z60", "top", 0.80),
    ("TNX_minus_IRX", "top", 0.80),
    ("TLT_20d_ret", "bot", 0.20),
]


def compute_bearish_gate(df: pd.DataFrame, sigs: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    votes = np.zeros(n, dtype=int)
    for sig_name, side, q in BLK:
        sig = sigs[sig_name]
        s = pd.Series(sig)
        qv = s.rolling(252, min_periods=126).quantile(q).values
        if side == "top":
            fires = (sig >= qv) & ~np.isnan(sig) & ~np.isnan(qv)
        else:
            fires = (sig <= qv) & ~np.isnan(sig) & ~np.isnan(qv)
        votes += fires.astype(int)
    scale = np.where(votes >= 5, 0.0, 1.0)
    return votes, scale


def verify_step6(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("=" * 70)
    print("Step 6 — 8-vote bearish gate")
    print("=" * 70)
    df = attach_macro_signals(df)
    sigs = build_step6_signals(df)
    votes, scale = compute_bearish_gate(df, sigs)
    df["scale_5plus_to_00"] = scale

    checks = []
    fired = int((votes >= 5).sum())
    checks.append(("(votes >= 5).sum()", fired, 172, fired == 172))
    n_blocked = int(np.sum(scale == 0.0))
    checks.append(("scale==0 count", n_blocked, 172, n_blocked == 172))
    mean_scale = round(float(np.mean(scale)), 4)
    checks.append(("scale.mean()", mean_scale, 0.9740, mean_scale == 0.9740))

    def _spot(date, exp_fired):
        idx = df[df["Date"] == date].index[0]
        got = bool(votes[idx] >= 5)
        return got, exp_fired, got == exp_fired

    g, e, ok = _spot("2008-10-09", True)
    checks.append(("2008-10-09 fired (GFC)", g, e, ok))
    g, e, ok = _spot("2020-03-16", True)
    checks.append(("2020-03-16 fired (COVID)", g, e, ok))
    g, e, ok = _spot("2022-06-13", True)
    checks.append(("2022-06-13 fired (2022)", g, e, ok))
    g, e, ok = _spot("2017-08-10", False)
    checks.append(("2017-08-10 not fired (calm)", g, e, ok))

    for r in checks:
        _print_perf_row(*r, width_label=28, width_val=16)
    print()

    failed = [c for c in checks if not c[-1]]
    if failed:
        raise AssertionError(f"Step 6 verification failed: {len(failed)} of {len(checks)} checks")
    return df, sigs


# ----------------------------------------------------------------------------
# Step 7 — R21 sizing layers
# ----------------------------------------------------------------------------

SIGNS_13 = [
    ("QQQ_over_SPY_z60", +1),
    ("VIX_5d_chg", -1),
    ("VWEHX_20d_ret", -1),
    ("IWM_minus_SPY_20d", -1),
    ("VXN_over_VIX", +1),
    ("SMH_over_QQQ_z60", -1),
    ("TNX_minus_IRX", -1),
    ("TLT_20d_ret", +1),
    ("IWM_over_QQQ_z60", -1),
    ("VXN_over_VIX_5d_chg", +1),
    ("VIX9D_over_VIX", +1),
    ("VIX_pctrank_252", -1),
    ("SMH_dd_60", +1),
]


def build_13_signals(df: pd.DataFrame, sigs8: dict) -> dict:
    """Extend the 8 step-6 signals with the 5 new ones to make 13."""
    sigs = dict(sigs8)
    sigs["IWM_over_QQQ_z60"] = _zscore(df["ext_IWM"].values / df["QQQ_Close"].values, w=60, mp=30)
    sigs["VXN_over_VIX_5d_chg"] = pd.Series(sigs["VXN_over_VIX"]).pct_change(5, fill_method=None).values
    sigs["VIX9D_over_VIX"] = df["ext_VIX9D"].values / df["VIX"].values
    sigs["VIX_pctrank_252"] = pd.Series(df["VIX"].values).rolling(
        252, min_periods=126).rank(pct=True).values
    smh_peak = pd.Series(df["ext_SMH"].values).rolling(60, min_periods=1).max()
    sigs["SMH_dd_60"] = df["ext_SMH"].values / smh_peak.values - 1
    return sigs


def build_combo_r21(df: pd.DataFrame, sigs13: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (combo_R21_winsor175_asym, mean_score, zs_matrix)."""
    n = len(df)
    zs = np.full((n, 13), np.nan)
    for k, (name, sign) in enumerate(SIGNS_13):
        s = pd.Series(sigs13[name], dtype=float)
        z = ((s - s.rolling(252, min_periods=126).mean())
             / s.rolling(252, min_periods=126).std()).values
        zs[:, k] = sign * z
    clipped = np.clip(zs, -1.75, 1.75)
    with np.errstate(invalid="ignore"):
        mean_score = np.nanmean(clipped, axis=1)  # divisor = non-NaN count

    cup_hi, cdn_hi, cs = 1.5, 4.0, 0.4
    cc = df["cc"].values
    in_cup = (cc == "C/Up")
    in_cdn = (cc == "C/Dn")
    sc = np.clip(mean_score, 0.0, cs)
    cup_scaler = np.where(np.isnan(mean_score), 1.0, 1.0 + (cup_hi - 1.0) * (sc / cs))
    cdn_scaler = np.where(np.isnan(mean_score), 1.0, 1.0 + (cdn_hi - 1.0) * (sc / cs))
    regime_scaler = np.where(in_cup, cup_scaler, np.where(in_cdn, cdn_scaler, 1.0))

    scale_5plus = df["scale_5plus_to_00"].values
    combo = scale_5plus * regime_scaler
    return combo, mean_score, zs


def build_realized_vol(df: pd.DataFrame) -> np.ndarray:
    """20-bar realized vol of c2c_QQQ, annualized (ddof=1)."""
    c2c = pd.Series(df["c2c_QQQ"].values)
    return (c2c.rolling(20).std() * np.sqrt(252)).values


def verify_step7_columns(df: pd.DataFrame, sigs8: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build combo column and realized_vol; verify the per-bar fingerprints."""
    print("=" * 70)
    print("Step 7 — column fingerprints (Checkpoints 7b and 7c)")
    print("=" * 70)
    sigs13 = build_13_signals(df, sigs8)
    combo, mean_score, zs = build_combo_r21(df, sigs13)
    realized_vol = build_realized_vol(df)
    df = df.copy()
    df["combo_R21_winsor175_asym"] = combo
    df["realized_vol_20"] = realized_vol
    df["mean_score"] = mean_score

    # Checkpoint 7b — combo values at named dates
    spot_combo = [
        ("2000-12-29", 1.000000),
        ("2008-10-09", 0.000000),
        ("2020-03-16", 0.000000),
        ("2022-06-13", 0.000000),
        ("2026-03-31", 4.000000),
        ("2003-08-12", 1.500000),
        ("2004-04-05", 1.019420),
        ("2005-06-07", 1.296307),
        ("2017-08-10", 1.000000),
    ]
    print("Checkpoint 7b — combo values:")
    for date, target in spot_combo:
        idx = df.index[df["Date"] == date][0]
        got = round(float(combo[idx]), 6)
        ok = (got == target)
        mark = "✓" if ok else "✗"
        print(f"  {date}  got={got:<10}  target={target:<10}  {mark}")

    # mean_score warmup (counts z-scored signals, not raw signals)
    print("\nmean_score warmup:")
    first_partial = int(np.argmax(~np.isnan(mean_score)))
    all13_mask = (~np.isnan(zs)).all(axis=1)
    first_all = int(np.argmax(all13_mask))
    nan_count = int(np.isnan(mean_score).sum())
    all13_count = int(all13_mask.sum())
    partial_count = int(len(mean_score) - nan_count - all13_count)

    rows = [
        ("First mean_score non-NaN idx", first_partial, 125, first_partial == 125),
        ("Date at first non-NaN", str(df.loc[first_partial, "Date"].date()),
         "2000-06-30", str(df.loc[first_partial, "Date"].date()) == "2000-06-30"),
        ("First all-13 non-NaN idx", first_all, 2898, first_all == 2898),
        ("Date at first all-13", str(df.loc[first_all, "Date"].date()),
         "2011-07-05", str(df.loc[first_all, "Date"].date()) == "2011-07-05"),
        ("Bars mean_score NaN", nan_count, 125, nan_count == 125),
        ("Bars all 13 non-NaN", all13_count, 3708, all13_count == 3708),
        ("Bars 1-12 partial", partial_count, 2773, partial_count == 2773),
    ]
    for r in rows:
        _print_perf_row(*r, width_label=30, width_val=18)

    # Checkpoint 7c — realized_vol values
    spot_vol = [
        ("2000-12-29", 0.782407),
        ("2008-10-09", 0.507197),
        ("2017-08-10", 0.101390),
        ("2020-03-16", 0.790158),
        ("2022-06-13", 0.388680),
        ("2026-03-31", 0.228706),
    ]
    print("\nCheckpoint 7c — realized_vol_20 values:")
    for date, target in spot_vol:
        idx = df.index[df["Date"] == date][0]
        got = round(float(realized_vol[idx]), 6)
        ok = (got == target)
        mark = "✓" if ok else "✗"
        print(f"  {date}  got={got:<10}  target={target:<10}  {mark}")
    print()

    return df, combo, realized_vol


# ----------------------------------------------------------------------------
# Step 4 / 5 — Combined engine
# ----------------------------------------------------------------------------

def run_combined_engine(df: pd.DataFrame, *,
                        scenario: str = "4b",
                        pyramid: bool = False,
                        r21_combo: np.ndarray | None = None,
                        realized_vol: np.ndarray | None = None,
                        dd_brake: bool = False,
                        enable_cap: bool = False,
                        stop_at: int | None = None,
                        return_state: bool = False):
    """
    Combined v14 trend engine + optional R21 sizing layers.

    scenario: "4a" = no filters / no preempt / no VIX gate.
              "4b" = all v14 features (filters, preemption, VIX gate).
              "5"  = same rules as 4b; pyramid sizing for TQQQ enabled.
              "7"  = Step 7 R21 sizing layers active (pyramid required).
    pyramid:     apply Step 5 pyramid sizing.
    r21_combo:   length-n combo_R21_winsor175_asym scaler. None -> 1.0 each bar.
    realized_vol: length-n realized_vol_20 (annualized). None -> vol-scaler off.
    dd_brake:    apply Step 7a TQQQ drawdown brake (× 0.70 when eq/peak<0.75).
    stop_at:     if set, process bars 0..stop_at (inclusive) and stop.  Default
                 processes the full df.  Used by the daily signal generator to
                 capture engine state at end of bar (today-1) before previewing
                 the last bar's would-be action.
    return_state: if True, return (trades, dret, state_dict).  state_dict captures
                 the engine's loop variables at the end of the processed range,
                 enabling last-bar preview.
    """
    n = len(df)
    Date = df["Date"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values

    Open_q = df["QQQ_Open"].values
    Close_q = df["QQQ_Close"].values
    ATR = df["ATR14"].values
    RT = df["RT"].values
    BSR = df["BSR"].values
    MTI = df["MTI"].values
    VIX = df["VIX"].values

    TQQQ_open = df["TQQQ_open"].values
    TQQQ_close = df["TQQQ_close"].values
    TQQQ_rt = df["TQQQ_rt"].values
    TQQQ_atr = df["TQQQ_atr"].values

    o2c_q = df["o2c_QQQ"].values
    c2c_q = df["c2c_QQQ"].values
    pc2o_q = df["pc2o_QQQ"].values
    o2c_p = df["o2c_PSQ"].values
    c2c_p = df["c2c_PSQ"].values
    pc2o_p = df["pc2o_PSQ"].values
    o2c_t = df["o2c_TQQQ"].values
    c2c_t = df["c2c_TQQQ"].values
    pc2o_t = df["pc2o_TQQQ"].values

    apply_filters = scenario != "4a"
    apply_preempt = scenario != "4a"
    apply_vix = scenario != "4a"

    in_trade = False
    instrument = None
    tqqq_variant = None  # "cup" or "cdn"
    entry_idx = None
    cum_mult = 1.0
    stop_qqq = None
    tqqq_target = None
    tqqq_stop = None
    be_moved = False
    last_dew = None
    cup_entered = False
    cdn_active = False
    pyramid_on = False
    just_latched = False
    tqqq_F = 1.0  # R21 entry-time base size, frozen per trade

    # Step 8 Cap state
    cap_target = None
    cap_stop = None
    last_cap_signal_idx = -10 ** 9
    qqq_5d_ret = df["qqq_5d_ret"].values

    # R21 equity tracking (always on so dd brake can read prior-bar eq when enabled)
    eq_running = 100_000.0
    peak_eq = 100_000.0

    dret = np.zeros(n)
    trades = []

    def compute_F(i: int) -> float:
        """F = combo[i] × dd_brake_factor × vol_scaler. Read at signal bar i."""
        f = 1.0
        if r21_combo is not None and not np.isnan(r21_combo[i]):
            f *= float(r21_combo[i])
        if dd_brake:
            # eq_running and peak_eq hold values AT CLOSE of bar i-1 right now
            if peak_eq > 0 and (eq_running / peak_eq) < 0.75:
                f *= 0.70
        if realized_vol is not None and not np.isnan(realized_vol[i]) and realized_vol[i] > 0:
            vs = 0.24 / realized_vol[i]
            vs = max(0.30, min(1.20, vs))
            f *= vs
        return f

    def _isnan(x):
        return isinstance(x, float) and np.isnan(x)

    def qqq_filter_blocks(i: int) -> bool:
        if not apply_filters:
            return False
        cond = False
        if not np.isnan(RT[i]) and 0.85 <= RT[i] < 0.90:
            cond = True
        if not np.isnan(BSR[i]) and BSR[i] > 1.50:
            cond = True
        if not np.isnan(MTI[i]) and MTI[i] > 1.15:
            cond = True
        return cond

    def psq_filter_blocks(i: int) -> bool:
        if not apply_filters:
            return False
        if not np.isnan(RT[i]) and RT[i] > 0.95 and not np.isnan(MTI[i]) and MTI[i] > 0.95:
            return True
        return False

    def record_trade(entry_i, exit_i, instr, ret, reason):
        trades.append({
            "entry_fill": Date[entry_i],
            "exit_fill": Date[exit_i],
            "instrument": instr,
            "ret": ret,
            "reason": reason,
        })

    last_idx = n - 1 if stop_at is None else min(stop_at, n - 1)
    for i in range(last_idx + 1):
        # Roll eq forward with the prior bar's dret so eq_running == eq[i-1]
        # throughout bar i's processing (this is the value the DD brake reads).
        if i > 0:
            eq_running *= (1.0 + dret[i - 1])
            if eq_running > peak_eq:
                peak_eq = eq_running

        fell_through = False
        flip_blocks = False

        # Phase A — CC transitions
        if i > 0:
            if Trend[i] == "C/Up" and cc[i - 1] != "C/Up":
                last_dew = None
                cup_entered = False
            if Trend[i] == "C/Dn" and cc[i - 1] != "C/Dn":
                last_dew = None

        # Signal flags
        cup_f = (cc[i] == "C/Up"
                 and not np.isnan(TQQQ_rt[i]) and TQQQ_rt[i] < 1.40
                 and not np.isnan(BSR[i]) and BSR[i] > 1.05
                 and not np.isnan(TQQQ_atr[i]) and TQQQ_atr[i] < 7.0
                 and not np.isnan(VIX[i]) and VIX[i] < 30)
        cdn_f_base = (cc[i] == "C/Dn"
                      and DEW[i] == "Buy" and last_dew != "Buy"
                      and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                      and not np.isnan(MTI[i]) and MTI[i] < 1.00)
        if apply_vix:
            vix_too_high = not np.isnan(VIX[i]) and VIX[i] > 35
        else:
            vix_too_high = False
        cdn_f = cdn_f_base and not vix_too_high

        # 4b — Preempts (run BEFORE Phase B)
        if apply_preempt and in_trade and i + 1 < n:
            # 1a: C/Up TQQQ preempts open QQQ/PSQ
            if instrument in ("QQQ", "PSQ") and cup_f and not cup_entered:
                # exit current trade
                if instrument == "QQQ":
                    d = o2c_q[i] if i == entry_idx else c2c_q[i]
                    cum_mult *= (1.0 + d)
                    exit_dret = (1.0 + d) * (1.0 + pc2o_q[i + 1]) - 1.0
                    trade_ret = cum_mult * (1.0 + pc2o_q[i + 1]) - 1.0
                    dret[i] = exit_dret
                    record_trade(entry_idx, i + 1, "QQQ", trade_ret, "Preempt")
                else:  # PSQ
                    d = o2c_p[i] if i == entry_idx else c2c_p[i]
                    cum_mult *= (1.0 + d)
                    exit_dret = (1.0 + d) * (1.0 + pc2o_p[i + 1]) - 1.0
                    trade_ret = cum_mult * (1.0 + pc2o_p[i + 1]) - 1.0
                    dret[i] = exit_dret
                    record_trade(entry_idx, i + 1, "PSQ", trade_ret, "Preempt")
                # Enter TQQQ C/Up at same i+1 open
                in_trade = True
                instrument = "TQQQ"
                tqqq_variant = "cup"
                entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = TQQQ_open[i + 1] * 1.50
                tqqq_stop = TQQQ_open[i + 1] * 0.94
                be_moved = False
                cup_entered = True
                cdn_active = False
                pyramid_on = False
                just_latched = False
                last_dew = None
                tqqq_F = compute_F(i)
                continue

            # 1b: C/Dn switch while in QQQ
            if (instrument == "QQQ" and not cdn_active
                    and cc[i] == "C/Dn"
                    and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                    and not np.isnan(MTI[i]) and MTI[i] < 0.95
                    and not vix_too_high):
                d = o2c_q[i] if i == entry_idx else c2c_q[i]
                cum_mult *= (1.0 + d)
                exit_dret = (1.0 + d) * (1.0 + pc2o_q[i + 1]) - 1.0
                trade_ret = cum_mult * (1.0 + pc2o_q[i + 1]) - 1.0
                dret[i] = exit_dret
                record_trade(entry_idx, i + 1, "QQQ", trade_ret, "Preempt")
                # Enter TQQQ C/Dn at same open
                in_trade = True
                instrument = "TQQQ"
                tqqq_variant = "cdn"
                entry_idx = i + 1
                cum_mult = 1.0
                tqqq_target = TQQQ_open[i + 1] * 1.30
                tqqq_stop = TQQQ_open[i + 1] * 0.94
                be_moved = False
                cdn_active = True
                pyramid_on = False
                just_latched = False
                last_dew = None
                tqqq_F = compute_F(i)
                continue

        # Compute "would v14 want to enter this bar" — used by Cap preempt.
        v14_wants_entry = False
        if enable_cap:
            v14_wants_entry = (
                (cup_f and not cup_entered)
                or (cdn_f and last_dew != "Buy")
                or (DEW[i] == "Buy" and last_dew != "Buy"
                    and not vix_too_high and not qqq_filter_blocks(i))
                or (DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn"
                    and not vix_too_high and not psq_filter_blocks(i))
            )

        # Phase B — Exits
        if in_trade:
            exit_reason = None
            exit_inst = instrument

            if instrument == "Cap":
                # Cap trades QQQ at 1.0× — no pyramid, no R21 sizing.
                d = o2c_q[i] if i == entry_idx else c2c_q[i]
                cum_mult *= (1.0 + d)

                # Exit priority: TP > Stop > MaxHold > Preempt
                if not np.isnan(Close_q[i]) and Close_q[i] >= cap_target:
                    exit_reason = "TP"
                elif not np.isnan(Close_q[i]) and Close_q[i] <= cap_stop:
                    exit_reason = "Stop"
                elif (i - entry_idx) >= 15:
                    exit_reason = "MaxHold"
                elif v14_wants_entry:
                    exit_reason = "Preempt"

                if exit_reason is not None and i + 1 < n:
                    exit_dret = (1.0 + d) * (1.0 + pc2o_q[i + 1]) - 1.0
                    trade_ret = cum_mult * (1.0 + pc2o_q[i + 1]) - 1.0
                    dret[i] = exit_dret
                    record_trade(entry_idx, i + 1, "Cap", trade_ret, exit_reason)
                    in_trade = False
                    instrument = None
                    entry_idx = None
                    cum_mult = 1.0
                    cap_target = None
                    cap_stop = None
                    # ALL Cap exits fall through to Phase C (spec 8d).
                    # last_cap_signal_idx is preserved (cooldown clock).
                    # fall through (do NOT continue)
                else:
                    dret[i] = d
                    continue

            elif instrument == "TQQQ":
                # Pick return columns for TQQQ
                d_raw = o2c_t[i] if i == entry_idx else c2c_t[i]
                cum_mult *= (1.0 + d_raw)

                # Breakeven ratchet at +9%
                if not be_moved and cum_mult - 1.0 >= 0.09:
                    tqqq_stop = TQQQ_open[entry_idx]
                    be_moved = True

                # Exit priority for TQQQ:
                # 1 MTI (not on entry bar) > 2 TP > 3 Stop > 4 MaxHold > 9 Regime
                if i != entry_idx and not np.isnan(MTI[i]) and MTI[i] < 0.75:
                    exit_reason = "MTI"
                elif not np.isnan(TQQQ_close[i]) and TQQQ_close[i] >= tqqq_target:
                    exit_reason = "TP"
                elif not np.isnan(TQQQ_close[i]) and TQQQ_close[i] <= tqqq_stop:
                    exit_reason = "Stop"
                elif (i - entry_idx) >= 70:
                    exit_reason = "MaxHold"
                else:
                    # Regime depends on variant
                    if tqqq_variant == "cup" and cc[i] == "C/Dn":
                        exit_reason = "Regime"
                    elif tqqq_variant == "cdn":
                        if DEW[i] == "Sell":
                            exit_reason = "Regime"
                        elif Trend[i] == "C/Up" and cc[i - 1] == "C/Dn":
                            exit_reason = "Regime"

                if exit_reason is not None and i + 1 < n:
                    F = tqqq_F
                    # === Compute dret with optional pyramid sizing + F ===
                    if pyramid and pyramid_on:
                        if just_latched:
                            # Exit ON the first post-latch bar itself (rare)
                            # F-inside, piece-by-piece per spec.
                            exit_dret = ((1.0 + F * pc2o_t[i])
                                         * (1.0 + F * 1.25 * o2c_t[i])
                                         * (1.0 + F * 1.25 * pc2o_t[i + 1])) - 1.0
                            just_latched = False
                        else:
                            # Exit (post-latch). F-OUTSIDE per spec.
                            exit_dret = (((1.0 + 1.25 * d_raw)
                                          * (1.0 + 1.25 * pc2o_t[i + 1])) - 1.0) * F
                    else:
                        # Exit (no pyramid). F-OUTSIDE per spec.
                        exit_dret = ((1.0 + d_raw) * (1.0 + pc2o_t[i + 1]) - 1.0) * F
                    trade_ret = cum_mult * (1.0 + pc2o_t[i + 1]) - 1.0

                    dret[i] = exit_dret
                    record_trade(entry_idx, i + 1, "TQQQ", trade_ret, exit_reason)
                    in_trade = False
                    instrument = None
                    tqqq_variant = None
                    entry_idx = None
                    cum_mult = 1.0
                    tqqq_target = None
                    tqqq_stop = None
                    be_moved = False
                    last_dew = None  # reset after TQQQ exit
                    cdn_active = False
                    pyramid_on = False
                    just_latched = False
                    # cup_entered NOT reset (only on new C/Up transition)
                    continue
                else:
                    F = tqqq_F
                    # Mid-trade dret for TQQQ (F-inside per spec)
                    if pyramid:
                        if just_latched:
                            # First post-latch split bar (mid-trade)
                            dret[i] = (1.0 + F * pc2o_t[i]) * (1.0 + F * 1.25 * o2c_t[i]) - 1.0
                            just_latched = False
                        else:
                            pmp = 1.25 if pyramid_on else 1.0
                            dret[i] = F * pmp * d_raw
                        # Latch fires at end of mid-trade bar
                        if cum_mult - 1.0 >= 0.10 and not pyramid_on:
                            pyramid_on = True
                            just_latched = True
                    else:
                        dret[i] = F * d_raw
                    continue

            elif instrument == "QQQ":
                d_raw = o2c_q[i] if i == entry_idx else c2c_q[i]
                cum_mult *= (1.0 + d_raw)

                # Priority: TP (5) > ATR (6) > C/Dn flip (10) > DEW Sell (12)
                if cum_mult - 1.0 >= 0.05:
                    exit_reason = "TP"
                elif Close_q[i] <= stop_qqq:
                    exit_reason = "ATR"
                elif Trend[i] == "C/Dn" and cc[i - 1] == "C/Up" and i != entry_idx:
                    exit_reason = "Flip"
                elif DEW[i] == "Sell":
                    exit_reason = "DEW"

                if exit_reason is not None and i + 1 < n:
                    exit_dret = (1.0 + d_raw) * (1.0 + pc2o_q[i + 1]) - 1.0
                    trade_ret = cum_mult * (1.0 + pc2o_q[i + 1]) - 1.0
                    dret[i] = exit_dret
                    record_trade(entry_idx, i + 1, "QQQ", trade_ret, exit_reason)
                    in_trade = False
                    instrument = None
                    entry_idx = None
                    cum_mult = 1.0
                    stop_qqq = None
                    if exit_reason in ("TP", "ATR"):
                        # last_dew unchanged → continue
                        continue
                    elif exit_reason == "Flip":
                        flip_blocks = True
                        fell_through = True
                        # fall through to Phase C
                    elif exit_reason == "DEW":
                        last_dew = "Sell"
                        fell_through = True
                        # fall through to Phase C
                else:
                    dret[i] = d_raw
                    continue

            elif instrument == "PSQ":
                d_raw = o2c_p[i] if i == entry_idx else c2c_p[i]
                cum_mult *= (1.0 + d_raw)

                if cum_mult - 1.0 >= 0.05:
                    exit_reason = "TP"
                elif Close_q[i] >= stop_qqq:
                    exit_reason = "ATR"
                elif Trend[i] == "C/Up" and cc[i - 1] == "C/Dn" and i != entry_idx:
                    exit_reason = "Flip"
                elif DEW[i] == "Buy":
                    exit_reason = "DEW"

                if exit_reason is not None and i + 1 < n:
                    exit_dret = (1.0 + d_raw) * (1.0 + pc2o_p[i + 1]) - 1.0
                    trade_ret = cum_mult * (1.0 + pc2o_p[i + 1]) - 1.0
                    dret[i] = exit_dret
                    record_trade(entry_idx, i + 1, "PSQ", trade_ret, exit_reason)
                    in_trade = False
                    instrument = None
                    entry_idx = None
                    cum_mult = 1.0
                    stop_qqq = None
                    if exit_reason in ("TP", "ATR"):
                        continue
                    elif exit_reason == "Flip":
                        flip_blocks = True
                        fell_through = True
                    elif exit_reason == "DEW":
                        last_dew = "Buy"
                        fell_through = True
                else:
                    dret[i] = d_raw
                    continue

        # Phase C — Entries (only reached if not in_trade after exits/fall-through)
        if in_trade:
            continue

        # Entry priority: cup_f, cdn_f, QQQ, PSQ
        if cup_f and not cup_entered and not fell_through and i + 1 < n \
                and not np.isnan(TQQQ_open[i + 1]):
            in_trade = True
            instrument = "TQQQ"
            tqqq_variant = "cup"
            entry_idx = i + 1
            cum_mult = 1.0
            tqqq_target = TQQQ_open[i + 1] * 1.50
            tqqq_stop = TQQQ_open[i + 1] * 0.94
            be_moved = False
            cup_entered = True
            cdn_active = False
            pyramid_on = False
            just_latched = False
            tqqq_F = compute_F(i)
            continue

        if cdn_f and last_dew != "Buy" and not fell_through and i + 1 < n \
                and not np.isnan(TQQQ_open[i + 1]):
            in_trade = True
            instrument = "TQQQ"
            tqqq_variant = "cdn"
            entry_idx = i + 1
            cum_mult = 1.0
            tqqq_target = TQQQ_open[i + 1] * 1.30
            tqqq_stop = TQQQ_open[i + 1] * 0.94
            be_moved = False
            cdn_active = True
            pyramid_on = False
            just_latched = False
            tqqq_F = compute_F(i)
            continue

        # QQQ DEW Buy
        if (DEW[i] == "Buy" and last_dew != "Buy"
                and not vix_too_high
                and not qqq_filter_blocks(i)
                and i + 1 < n):
            in_trade = True
            instrument = "QQQ"
            entry_idx = i + 1
            cum_mult = 1.0
            atr_mult = 2.0 if cc[i] == "C/Dn" else 1.0
            stop_qqq = Open_q[i + 1] - atr_mult * ATR[i]
            last_dew = "Buy"
            continue

        # PSQ DEW Sell
        if (DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn"
                and not vix_too_high
                and not psq_filter_blocks(i)
                and i + 1 < n):
            in_trade = True
            instrument = "PSQ"
            entry_idx = i + 1
            cum_mult = 1.0
            stop_qqq = Open_q[i + 1] + 2.0 * ATR[i]
            last_dew = "Sell"
            continue

        # Phase D — Capitulation Bounce (lowest priority)
        if (enable_cap
                and not in_trade
                and not flip_blocks
                and (i - last_cap_signal_idx) >= 5
                and i + 1 < n
                and not np.isnan(qqq_5d_ret[i])
                and qqq_5d_ret[i] <= -0.10):
            in_trade = True
            instrument = "Cap"
            entry_idx = i + 1
            cum_mult = 1.0
            cap_target = Open_q[i + 1] * 1.05
            cap_stop = Open_q[i + 1] * 0.96
            last_cap_signal_idx = i
            # Cap never sets last_dew, cup_entered, cdn_active.

    if return_state:
        # Apply the last processed bar's dret so eq_running reflects end-of-bar.
        if last_idx >= 0:
            eq_running *= (1.0 + dret[last_idx])
            if eq_running > peak_eq:
                peak_eq = eq_running
        state = {
            "last_idx":            last_idx,
            "in_trade":            in_trade,
            "instrument":          instrument,
            "tqqq_variant":        tqqq_variant,
            "entry_idx":           entry_idx,
            "cum_mult":            cum_mult,
            "stop_qqq":            stop_qqq,
            "tqqq_target":         tqqq_target,
            "tqqq_stop":           tqqq_stop,
            "be_moved":            be_moved,
            "last_dew":            last_dew,
            "cup_entered":         cup_entered,
            "cdn_active":          cdn_active,
            "pyramid_on":          pyramid_on,
            "just_latched":        just_latched,
            "tqqq_F":              tqqq_F,
            "cap_target":          cap_target,
            "cap_stop":            cap_stop,
            "last_cap_signal_idx": last_cap_signal_idx,
            "eq_running":          eq_running,
            "peak_eq":             peak_eq,
        }
        return trades, dret, state

    return trades, dret


def verify_step4a(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Step 4a — Combined engine, no filters (SOFT checkpoint)")
    print("=" * 70)
    trades, dret = run_combined_engine(df, scenario="4a")
    perf = compute_perf(df["Date"], dret)
    q = sum(1 for t in trades if t["instrument"] == "QQQ")
    p = sum(1 for t in trades if t["instrument"] == "PSQ")
    tq = sum(1 for t in trades if t["instrument"] == "TQQQ")
    print(f"Trades: {len(trades)} (QQQ={q}, PSQ={p}, TQQQ={tq})")
    print(f"Final: ${perf['final']:,.0f}  CAGR: {perf['cagr']*100:.2f}%  "
          f"Sharpe: {perf['sharpe']:.2f}  Max DD: {perf['max_dd']*100:.2f}%")
    print("Target (soft, ±3 trades): 228 (QQQ=108, PSQ=50, TQQQ=70)  $52,458,205")

    print("\nFirst 5 trades (must match byte-exact):")
    expected = [
        ("2000-02-02", "2000-02-07", "QQQ", 0.054201, "TP"),
        ("2000-04-04", "2000-04-12", "PSQ", 0.053210, "TP"),
        ("2000-05-02", "2000-05-03", "QQQ", -0.056653, "DEW"),
        ("2000-05-17", "2000-05-19", "QQQ", -0.061625, "DEW"),
        ("2000-05-31", "2000-06-01", "QQQ", 0.008512, "DEW"),
    ]
    for (exp_e, exp_x, exp_i, exp_r, exp_reason), tr in zip(expected, trades[:5]):
        e = pd.Timestamp(tr["entry_fill"]).strftime("%Y-%m-%d")
        x = pd.Timestamp(tr["exit_fill"]).strftime("%Y-%m-%d")
        r = round(tr["ret"], 4)
        ok = (e == exp_e and x == exp_x and tr["instrument"] == exp_i
              and r == round(exp_r, 4) and tr["reason"] == exp_reason)
        mark = "✓" if ok else "✗"
        print(f"  {e} → {x}  {tr['instrument']:<4} {tr['ret']*100:+.4f}%  {tr['reason']:<6} "
              f"[exp {exp_e}→{exp_x} {exp_i} {exp_r*100:+.4f}% {exp_reason}] {mark}")
    print()


def verify_step4b(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Step 4b — Combined engine with filters/preemption/VIX gate")
    print("=" * 70)
    trades, dret = run_combined_engine(df, scenario="4b")
    perf = compute_perf(df["Date"], dret)
    q = sum(1 for t in trades if t["instrument"] == "QQQ")
    p = sum(1 for t in trades if t["instrument"] == "PSQ")
    tq = sum(1 for t in trades if t["instrument"] == "TQQQ")

    targets = [
        ("Trades", len(trades), 187, len(trades) == 187),
        ("QQQ", q, 60, q == 60),
        ("PSQ", p, 40, p == 40),
        ("TQQQ", tq, 87, tq == 87),
        ("Final", round(perf["final"]), 517526341, abs(round(perf["final"]) - 517526341) <= 1),
        ("CAGR", round(perf["cagr"] * 100, 2), 38.52, round(perf["cagr"] * 100, 2) == 38.52),
        ("Sharpe", round(perf["sharpe"], 2), 1.45, round(perf["sharpe"], 2) == 1.45),
        ("Max DD", round(perf["max_dd"] * 100, 2), -29.32, round(perf["max_dd"] * 100, 2) == -29.32),
    ]
    for r in targets:
        _print_perf_row(*r)
    print()


def verify_step5(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Step 5 — v14 complete with pyramid sizing")
    print("=" * 70)
    trades, dret = run_combined_engine(df, scenario="5", pyramid=True)
    perf = compute_perf(df["Date"], dret)
    q = sum(1 for t in trades if t["instrument"] == "QQQ")
    p = sum(1 for t in trades if t["instrument"] == "PSQ")
    tq = sum(1 for t in trades if t["instrument"] == "TQQQ")

    targets = [
        ("Trades", len(trades), 187, len(trades) == 187),
        ("QQQ", q, 60, q == 60),
        ("PSQ", p, 40, p == 40),
        ("TQQQ", tq, 87, tq == 87),
        ("Final", round(perf["final"]), 1525489743,
         abs(round(perf["final"]) - 1525489743) <= 1),
        ("CAGR", round(perf["cagr"] * 100, 2), 44.35, round(perf["cagr"] * 100, 2) == 44.35),
        ("Sharpe", round(perf["sharpe"], 2), 1.45, round(perf["sharpe"], 2) == 1.45),
        ("Max DD", round(perf["max_dd"] * 100, 2), -33.18, round(perf["max_dd"] * 100, 2) == -33.18),
    ]
    for r in targets:
        _print_perf_row(*r)
    print()


def verify_step7_engine(df: pd.DataFrame, combo: np.ndarray, realized_vol: np.ndarray) -> None:
    print("=" * 70)
    print("Step 7 — Engine sub-configs (Checkpoint 7f)")
    print("=" * 70)

    # Smoke test: combo all 1.0, no dd, no vol -> must equal v14 byte-exact
    ones = np.ones(len(df))
    trades, dret = run_combined_engine(df, scenario="7", pyramid=True,
                                       r21_combo=ones, realized_vol=None, dd_brake=False)
    perf = compute_perf(df["Date"], dret)
    smoke_ok = abs(round(perf["final"]) - 1_525_489_743) <= 1
    _print_perf_row("Smoke (combo=1, no dd/vol)", round(perf["final"]), 1_525_489_743, smoke_ok,
                    width_label=32, width_val=18)

    # v14 baseline via scenario="5" already verified; reaffirm here.
    trades_v14, dret_v14 = run_combined_engine(df, scenario="5", pyramid=True)
    perf_v14 = compute_perf(df["Date"], dret_v14)
    v14_ok = abs(round(perf_v14["final"]) - 1_525_489_743) <= 1
    _print_perf_row("v14 baseline", round(perf_v14["final"]), 1_525_489_743, v14_ok,
                    width_label=32, width_val=18)

    # combo only
    trades_c, dret_c = run_combined_engine(df, scenario="7", pyramid=True,
                                            r21_combo=combo, realized_vol=None, dd_brake=False)
    perf_c = compute_perf(df["Date"], dret_c)
    combo_ok = abs(round(perf_c["final"]) - 83_380_574_777) <= 1
    _print_perf_row("combo only", round(perf_c["final"]), 83_380_574_777, combo_ok,
                    width_label=32, width_val=18)

    # all three layers
    trades_all, dret_all = run_combined_engine(df, scenario="7", pyramid=True,
                                                r21_combo=combo, realized_vol=realized_vol,
                                                dd_brake=True)
    perf_all = compute_perf(df["Date"], dret_all)
    all_ok = abs(round(perf_all["final"]) - 89_556_942_109) <= 1
    _print_perf_row("all three (full R21)", round(perf_all["final"]), 89_556_942_109, all_ok,
                    width_label=32, width_val=18)

    print(f"\nFull R21 perf: CAGR {perf_all['cagr']*100:.2f}%  Sharpe {perf_all['sharpe']:.4f}  "
          f"Max DD {perf_all['max_dd']*100:.2f}%")
    qc = sum(1 for t in trades_all if t["instrument"] == "QQQ" and t["reason"] != "Preempt")
    qp = sum(1 for t in trades_all if t["instrument"] == "QQQ" and t["reason"] == "Preempt")
    pp = sum(1 for t in trades_all if t["instrument"] == "PSQ" and t["reason"] != "Preempt")
    pq = sum(1 for t in trades_all if t["instrument"] == "PSQ" and t["reason"] == "Preempt")
    tq = sum(1 for t in trades_all if t["instrument"] == "TQQQ")
    total_non_preempt = sum(1 for t in trades_all if t["reason"] != "Preempt")
    print(f"Trades (non-preempt): {total_non_preempt}  (QQQ={qc} PSQ={pp} TQQQ={tq})")
    print(f"Preempts logged: QQQ→TQQQ={qp}  PSQ→TQQQ={pq}")
    print()


def verify_step8(df: pd.DataFrame, combo: np.ndarray, realized_vol: np.ndarray) -> None:
    print("=" * 70)
    print("Step 8 — Capitulation Bounce + R21 = Phoenix")
    print("=" * 70)

    # Run full Phoenix
    trades, dret = run_combined_engine(df, scenario="7", pyramid=True,
                                        r21_combo=combo, realized_vol=realized_vol,
                                        dd_brake=True, enable_cap=True)
    perf = compute_perf(df["Date"], dret)
    cap_trades = [t for t in trades if t["instrument"] == "Cap"]
    v14_trades = [t for t in trades if t["instrument"] != "Cap"]

    cap_exit_counts = {"TP": 0, "Stop": 0, "MaxHold": 0, "Preempt": 0}
    cap_wins = 0
    for t in cap_trades:
        cap_exit_counts[t["reason"]] = cap_exit_counts.get(t["reason"], 0) + 1
        if t["ret"] > 0:
            cap_wins += 1
    win_rate = cap_wins / len(cap_trades) if cap_trades else 0

    checks = [
        ("Cap trades", len(cap_trades), 37, len(cap_trades) == 37),
        ("Cap TP", cap_exit_counts.get("TP", 0), 26, cap_exit_counts.get("TP", 0) == 26),
        ("Cap Stop", cap_exit_counts.get("Stop", 0), 9, cap_exit_counts.get("Stop", 0) == 9),
        ("Cap MaxHold", cap_exit_counts.get("MaxHold", 0), 2, cap_exit_counts.get("MaxHold", 0) == 2),
        ("Cap Preempt", cap_exit_counts.get("Preempt", 0), 0, cap_exit_counts.get("Preempt", 0) == 0),
        ("Cap win rate", round(win_rate * 100, 1), 75.7, round(win_rate * 100, 1) == 75.7),
        ("v14 trades preserved", len(v14_trades), 187, len(v14_trades) == 187),
        ("Total trades", len(trades), 224, len(trades) == 224),
        ("Final equity", round(perf["final"]), 363_899_029_740,
         abs(round(perf["final"]) - 363_899_029_740) <= 1),
        ("CAGR %", round(perf["cagr"] * 100, 2), 77.83, round(perf["cagr"] * 100, 2) == 77.83),
        ("Sharpe", round(perf["sharpe"], 2), 1.59, round(perf["sharpe"], 2) == 1.59),
        ("Max DD %", round(perf["max_dd"] * 100, 2), -37.26, round(perf["max_dd"] * 100, 2) == -37.26),
    ]
    for r in checks:
        _print_perf_row(*r, width_label=28, width_val=18)

    print("\nFirst 5 Cap trades:")
    expected_first5 = [
        ("2000-01-31", "2000-02-02", 0.074236, "TP"),
        ("2000-04-14", "2000-04-17", -0.093501, "Stop"),
        ("2000-05-11", "2000-05-16", 0.087613, "TP"),
        ("2000-05-24", "2000-05-25", 0.070000, "TP"),
        ("2000-07-31", "2000-08-08", 0.050648, "TP"),
    ]
    for (exp_e, exp_x, exp_r, exp_reason), tr in zip(expected_first5, cap_trades[:5]):
        e = pd.Timestamp(tr["entry_fill"]).strftime("%Y-%m-%d")
        x = pd.Timestamp(tr["exit_fill"]).strftime("%Y-%m-%d")
        r = round(tr["ret"], 4)
        ok = (e == exp_e and x == exp_x and r == round(exp_r, 4) and tr["reason"] == exp_reason)
        mark = "✓" if ok else "✗"
        print(f"  {e} → {x}  {tr['ret']*100:+.4f}%  {tr['reason']:<8} "
              f"[exp {exp_e}→{exp_x} {exp_r*100:+.4f}% {exp_reason}] {mark}")

    print("\nLast 5 Cap trades:")
    expected_last5 = [
        ("2020-03-17", "2020-03-27", 0.065310, "TP"),
        ("2020-09-11", "2020-10-05", 0.007327, "MaxHold"),
        ("2022-05-12", "2022-05-16", 0.044182, "TP"),
        ("2022-06-14", "2022-06-27", 0.066052, "TP"),
        ("2025-04-09", "2025-04-10", 0.091417, "TP"),
    ]
    for (exp_e, exp_x, exp_r, exp_reason), tr in zip(expected_last5, cap_trades[-5:]):
        e = pd.Timestamp(tr["entry_fill"]).strftime("%Y-%m-%d")
        x = pd.Timestamp(tr["exit_fill"]).strftime("%Y-%m-%d")
        r = round(tr["ret"], 4)
        ok = (e == exp_e and x == exp_x and r == round(exp_r, 4) and tr["reason"] == exp_reason)
        mark = "✓" if ok else "✗"
        print(f"  {e} → {x}  {tr['ret']*100:+.4f}%  {tr['reason']:<8} "
              f"[exp {exp_e}→{exp_x} {exp_r*100:+.4f}% {exp_reason}] {mark}")
    print()

    failed = [c for c in checks if not c[-1]]
    if failed:
        print(f"NOTE: {len(failed)} Step 8 check(s) failed:")
        for c in failed:
            print(f"  {c[0]}: got {c[1]} vs target {c[2]}")


def _print_perf_row(name: str, got, target, ok: bool, width_label=22, width_val=18) -> None:
    mark = "✓" if ok else "✗"
    print(f"{name:<{width_label}}{str(got):<{width_val}}{str(target):<{width_val}}{mark}")


def verify_step3a(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Step 3a — QQQ long isolated")
    print("=" * 70)
    trades, dret = step3a_qqq_long(df)
    perf = compute_perf(df["Date"], dret)
    targets = {"trades": 162, "final": 295205, "cagr": 0.0421, "sharpe": 0.50, "max_dd": -0.2289}
    rows = [
        ("Trades", len(trades), targets["trades"], len(trades) == targets["trades"]),
        ("Final", round(perf["final"]), targets["final"],
         abs(round(perf["final"]) - targets["final"]) <= 1),
        ("CAGR", round(perf["cagr"], 4), targets["cagr"],
         round(perf["cagr"], 4) == targets["cagr"]),
        ("Sharpe", round(perf["sharpe"], 2), targets["sharpe"],
         round(perf["sharpe"], 2) == targets["sharpe"]),
        ("Max DD", round(perf["max_dd"], 4), targets["max_dd"],
         round(perf["max_dd"], 4) == targets["max_dd"]),
    ]
    for r in rows:
        _print_perf_row(*r)
    print()
    print("First 5 trades:")
    expected5 = [
        ("2000-02-02", "2000-02-07", 0.054201, "TP"),
        ("2000-05-02", "2000-05-03", -0.056653, "DEW"),
        ("2000-05-17", "2000-05-19", -0.061625, "DEW"),
        ("2000-05-31", "2000-06-01", 0.008512, "DEW"),
        ("2000-06-05", "2000-06-20", 0.065041, "TP"),
    ]
    for (exp_e, exp_x, exp_r, exp_reason), tr in zip(expected5, trades[:5]):
        e = pd.Timestamp(tr["entry_fill"]).strftime("%Y-%m-%d")
        x = pd.Timestamp(tr["exit_fill"]).strftime("%Y-%m-%d")
        r = round(tr["ret"], 4)
        ok = (e == exp_e and x == exp_x and r == round(exp_r, 4) and tr["reason"] == exp_reason)
        mark = "✓" if ok else "✗"
        print(f"  {e} → {x}  {tr['ret']*100:+.4f}%  {tr['reason']:<7}  [exp {exp_e}→{exp_x} {exp_r*100:+.4f}% {exp_reason}] {mark}")
    print()


def verify_step3b(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Step 3b — PSQ short isolated")
    print("=" * 70)
    trades, dret = step3b_psq_short(df)
    perf = compute_perf(df["Date"], dret)
    targets = {"trades": 75, "final": 127666, "cagr": 0.0094, "sharpe": 0.16, "max_dd": -0.2724}
    rows = [
        ("Trades", len(trades), targets["trades"], len(trades) == targets["trades"]),
        ("Final", round(perf["final"]), targets["final"],
         abs(round(perf["final"]) - targets["final"]) <= 1),
        ("CAGR", round(perf["cagr"], 4), targets["cagr"],
         round(perf["cagr"], 4) == targets["cagr"]),
        ("Sharpe", round(perf["sharpe"], 2), targets["sharpe"],
         round(perf["sharpe"], 2) == targets["sharpe"]),
        ("Max DD", round(perf["max_dd"], 4), targets["max_dd"],
         round(perf["max_dd"], 4) == targets["max_dd"]),
    ]
    for r in rows:
        _print_perf_row(*r)
    print()


def verify_step3c(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Step 3c — C/Up TQQQ isolated")
    print("=" * 70)
    trades, dret, exits = step3c_cup_tqqq(df)
    perf = compute_perf(df["Date"], dret)
    targets = {"trades": 121, "final": 4600652, "cagr": 0.1571, "sharpe": 0.71, "max_dd": -0.4853}
    rows = [
        ("Trades", len(trades), targets["trades"], len(trades) == targets["trades"]),
        ("Final", round(perf["final"]), targets["final"],
         abs(round(perf["final"]) - targets["final"]) <= 1),
        ("CAGR", round(perf["cagr"], 4), targets["cagr"],
         round(perf["cagr"], 4) == targets["cagr"]),
        ("Sharpe", round(perf["sharpe"], 2), targets["sharpe"],
         round(perf["sharpe"], 2) == targets["sharpe"]),
        ("Max DD", round(perf["max_dd"], 4), targets["max_dd"],
         round(perf["max_dd"], 4) == targets["max_dd"]),
    ]
    for r in rows:
        _print_perf_row(*r)
    print(f"Exit breakdown: Signal={exits.get('DEW',0)} Stop={exits.get('Stop',0)} "
          f"Regime={exits.get('Regime',0)} TP={exits.get('TP',0)} MTI={exits.get('MTI',0)} "
          f"MaxHold={exits.get('MaxHold',0)}")
    print("Target:         Signal=53     Stop=31     Regime=27     TP=8        MTI=2")
    print()


def verify_step3d(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Step 3d — C/Dn TQQQ isolated")
    print("=" * 70)
    trades, dret, exits = step3d_cdn_tqqq(df)
    perf = compute_perf(df["Date"], dret)
    targets = {"trades": 27, "final": 355494, "cagr": 0.0495, "sharpe": 0.61, "max_dd": -0.1605}
    rows = [
        ("Trades", len(trades), targets["trades"], len(trades) == targets["trades"]),
        ("Final", round(perf["final"]), targets["final"],
         abs(round(perf["final"]) - targets["final"]) <= 1),
        ("CAGR", round(perf["cagr"], 4), targets["cagr"],
         round(perf["cagr"], 4) == targets["cagr"]),
        ("Sharpe", round(perf["sharpe"], 2), targets["sharpe"],
         round(perf["sharpe"], 2) == targets["sharpe"]),
        ("Max DD", round(perf["max_dd"], 4), targets["max_dd"],
         round(perf["max_dd"], 4) == targets["max_dd"]),
    ]
    for r in rows:
        _print_perf_row(*r)
    print(f"Exit breakdown: Flip={exits.get('Flip',0)} DEW={exits.get('DEW',0)} "
          f"MTI={exits.get('MTI',0)} Stop={exits.get('Stop',0)} TP={exits.get('TP',0)} "
          f"MaxHold={exits.get('MaxHold',0)}")
    print("Target:         Flip=16      DEW=4       MTI=3       Stop=3       TP=1")
    print()


if __name__ == "__main__":
    df = build_step1()
    verify_step1(df)
    print(f"Step 1 OK — {df.shape[0]} rows, {df['Date'].min().date()} to {df['Date'].max().date()}\n")
    verify_step2(df)
    print("Step 2 OK\n")
    verify_step3a(df)
    verify_step3b(df)
    verify_step3c(df)
    verify_step3d(df)
    verify_step4a(df)
    verify_step4b(df)
    verify_step5(df)
    df, sigs = verify_step6(df)
    df, combo, realized_vol = verify_step7_columns(df, sigs)
    verify_step7_engine(df, combo, realized_vol)
    verify_step8(df, combo, realized_vol)
