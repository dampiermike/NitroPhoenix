#!/usr/bin/env python3
"""
Dampier Nitro++ Phoenix — Daily Signal Generator.

Replays the full production engine (Steps 1–8) up to today's close, then previews
what action will fill at tomorrow's open.

Live mapping:
    TQQQ : long TQQQ at F × pyramid_mult of equity
    QQQ  : long QQQ at full equity
    PSQ  : short QQQ via SQQQ at 1/3 position (post 2010-02-11)
    Cap  : long QQQ at 1.0× equity (Capitulation Bounce)

Output: console report + email + iMessage/SMS, mirroring Nitro v16.
"""
from __future__ import annotations

import os
import sys
import smtplib
import subprocess
from datetime import date, datetime
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
# phoenix.py lives in Backtester/ alongside this script
sys.path.insert(0, str(_HERE / "Backtester"))

from phoenix import (  # noqa: E402
    build_step1, attach_macro_signals, build_step6_signals, compute_bearish_gate,
    build_13_signals, build_combo_r21, build_realized_vol,
    run_combined_engine, compute_perf,
)


# ── Notification config ───────────────────────────────────────────────────────
GMAIL_USER  = os.environ.get("GOOGLE_EMAIL", "dampiermike@gmail.com")
GMAIL_PASS  = os.environ.get("GOOGLE_APP_PASSWORD", "")
TO_EMAIL    = os.environ.get("PHOENIX_TO_EMAIL", GMAIL_USER).split(",")
SMS_NUMBERS = [n.strip() for n in os.environ.get("PHOENIX_SMS_NUMBERS", "").split(",") if n.strip()]
# iMessage-only? Forced SMS numbers go through Continuity (paired iPhone SMS relay).
SMS_FORCE   = set(n.strip() for n in os.environ.get("PHOENIX_SMS_FORCE", "").split(",") if n.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Last-bar preview
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_last_bar(df: pd.DataFrame, state: dict, combo: np.ndarray,
                      realized_vol: np.ndarray, qqq_5d_ret: np.ndarray) -> dict:
    """Preview the engine's would-be action on the last bar (today)."""
    n = len(df)
    i = n - 1

    Date = df["Date"].values
    Trend = df["Trend"].values
    cc = df["cc"].values
    DEW = df["DEW_Signal"].values
    Close_q = df["QQQ_Close"].values
    Open_q = df["QQQ_Open"].values
    ATR = df["ATR14"].values
    RT = df["RT"].values
    BSR = df["BSR"].values
    MTI = df["MTI"].values
    VIX = df["VIX"].values
    TQQQ_close = df["TQQQ_close"].values
    TQQQ_open = df["TQQQ_open"].values
    TQQQ_rt = df["TQQQ_rt"].values
    TQQQ_atr = df["TQQQ_atr"].values

    o2c_q = df["o2c_QQQ"].values
    c2c_q = df["c2c_QQQ"].values
    o2c_p = df["o2c_PSQ"].values
    c2c_p = df["c2c_PSQ"].values
    o2c_t = df["o2c_TQQQ"].values
    c2c_t = df["c2c_TQQQ"].values

    # Unpack engine state from end of bar (today - 1).
    in_trade = state["in_trade"]
    instrument = state["instrument"]
    tqqq_variant = state["tqqq_variant"]
    entry_idx = state["entry_idx"]
    cum_mult = state["cum_mult"]
    stop_qqq = state["stop_qqq"]
    tqqq_target = state["tqqq_target"]
    tqqq_stop = state["tqqq_stop"]
    be_moved = state["be_moved"]
    last_dew = state["last_dew"]
    cup_entered = state["cup_entered"]
    cdn_active = state["cdn_active"]
    pyramid_on = state["pyramid_on"]
    tqqq_F = state["tqqq_F"]
    cap_target = state["cap_target"]
    cap_stop = state["cap_stop"]
    last_cap_signal_idx = state["last_cap_signal_idx"]
    eq_running = state["eq_running"]
    peak_eq = state["peak_eq"]

    # Phase A on bar i: apply CC transitions (mirrors engine).
    if cc[i] == "C/Up" and (i > 0 and cc[i - 1] != "C/Up"):
        last_dew = None
        cup_entered = False
    if cc[i] == "C/Dn" and (i > 0 and cc[i - 1] != "C/Dn"):
        last_dew = None

    # Today's signal flags (4b filter version — production system).
    vix_too_high = (not np.isnan(VIX[i])) and VIX[i] > 35
    cup_f = (cc[i] == "C/Up"
             and not np.isnan(TQQQ_rt[i]) and TQQQ_rt[i] < 1.40
             and not np.isnan(BSR[i]) and BSR[i] > 1.05
             and not np.isnan(TQQQ_atr[i]) and TQQQ_atr[i] < 7.0
             and not np.isnan(VIX[i]) and VIX[i] < 30)
    cdn_f = (cc[i] == "C/Dn"
             and DEW[i] == "Buy" and last_dew != "Buy"
             and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
             and not np.isnan(MTI[i]) and MTI[i] < 1.00
             and not vix_too_high)

    def qqq_blocked():
        return ((not np.isnan(RT[i]) and 0.85 <= RT[i] < 0.90)
                or (not np.isnan(BSR[i]) and BSR[i] > 1.50)
                or (not np.isnan(MTI[i]) and MTI[i] > 1.15))

    def psq_blocked():
        return (not np.isnan(RT[i]) and RT[i] > 0.95
                and not np.isnan(MTI[i]) and MTI[i] > 0.95)

    def compute_F_preview() -> float:
        f = 1.0
        if combo is not None and not np.isnan(combo[i]):
            f *= float(combo[i])
        if peak_eq > 0 and (eq_running / peak_eq) < 0.75:
            f *= 0.70
        if realized_vol is not None and not np.isnan(realized_vol[i]) and realized_vol[i] > 0:
            f *= max(0.30, min(1.20, 0.24 / realized_vol[i]))
        return f

    pending: dict = {
        "action": "HOLD",
        "reason": "",
        "exit_type": None,
        "entry_inst": None,
        "block_reasons": [],
        "today_mult": None,
        "cup_f": cup_f,
        "cdn_f": cdn_f,
        "vix_too_high": vix_too_high,
        "F_preview": None,
        "eq_running": eq_running,
        "peak_eq": peak_eq,
        "dd_now": (eq_running / peak_eq - 1.0) if peak_eq > 0 else 0.0,
    }

    # ─── In a trade: check exits / preempts ───
    if in_trade:
        # Preempts (run before Phase B)
        if instrument in ("QQQ", "PSQ") and cup_f and not cup_entered:
            pending.update(action="PREEMPT_1A",
                           exit_type=f"Preempt → TQQQ (cup) from {instrument}",
                           entry_inst="TQQQ",
                           F_preview=compute_F_preview(),
                           reason=(f"cup_f active while in {instrument}: "
                                   f"SELL {instrument}, BUY TQQQ at tomorrow's open."))
            return pending

        if (instrument == "QQQ" and not cdn_active
                and cc[i] == "C/Dn"
                and not np.isnan(RT[i]) and 0.95 <= RT[i] < 1.00
                and not np.isnan(MTI[i]) and MTI[i] < 0.95
                and not vix_too_high):
            pending.update(action="PREEMPT_1B",
                           exit_type="Preempt → TQQQ (cdn) from QQQ",
                           entry_inst="TQQQ",
                           F_preview=compute_F_preview(),
                           reason=("C/Dn switch while in QQQ: "
                                   "SELL QQQ, BUY TQQQ (C/Dn variant) at tomorrow's open."))
            return pending

        # Accrue today's return for preview (raw, no F/pyramid scaling)
        if instrument == "TQQQ":
            d = o2c_t[i] if i == entry_idx else c2c_t[i]
        elif instrument == "QQQ":
            d = o2c_q[i] if i == entry_idx else c2c_q[i]
        elif instrument == "PSQ":
            d = o2c_p[i] if i == entry_idx else c2c_p[i]
        else:  # Cap
            d = o2c_q[i] if i == entry_idx else c2c_q[i]
        today_mult = cum_mult * (1.0 + d)
        pending["today_mult"] = today_mult

        # Effective stop for TQQQ (preview breakeven ratchet)
        eff_stop = tqqq_stop
        if instrument == "TQQQ" and not be_moved and today_mult - 1.0 >= 0.09:
            eff_stop = max(eff_stop or 0.0, float(TQQQ_open[entry_idx]))
            pending["breakeven_preview"] = True

        # v14 exit priority on the current trade
        if instrument == "TQQQ":
            if i != entry_idx and not np.isnan(MTI[i]) and MTI[i] < 0.75:
                pending.update(action="EXIT", exit_type="MTI",
                               reason=f"MTI={MTI[i]:.2f} < 0.75 — exit TQQQ at tomorrow's open.")
            elif not np.isnan(TQQQ_close[i]) and TQQQ_close[i] >= tqqq_target:
                pending.update(action="EXIT", exit_type="TP",
                               reason=f"TQQQ close {TQQQ_close[i]:.2f} ≥ target "
                                      f"{tqqq_target:.2f} — exit at tomorrow's open.")
            elif not np.isnan(TQQQ_close[i]) and TQQQ_close[i] <= eff_stop:
                lbl = "Stop (breakeven)" if pending.get("breakeven_preview") or be_moved else "Stop"
                pending.update(action="EXIT", exit_type=lbl,
                               reason=f"TQQQ close {TQQQ_close[i]:.2f} ≤ stop "
                                      f"{eff_stop:.2f} — exit at tomorrow's open.")
            elif (i - entry_idx) >= 70:
                pending.update(action="EXIT", exit_type="MaxHold",
                               reason=f"Held {i-entry_idx} bars ≥ 70 — exit at tomorrow's open.")
            else:
                if tqqq_variant == "cup" and cc[i] == "C/Dn":
                    pending.update(action="EXIT", exit_type="Regime",
                                   reason="cc flipped to C/Dn — exit C/Up TQQQ at tomorrow's open.")
                elif tqqq_variant == "cdn":
                    if DEW[i] == "Sell":
                        pending.update(action="EXIT", exit_type="Regime",
                                       reason="C/Dn TQQQ DEW Sell — exit at tomorrow's open.")
                    elif Trend[i] == "C/Up" and i > 0 and cc[i - 1] == "C/Dn":
                        pending.update(action="EXIT", exit_type="Regime",
                                       reason="C/Dn TQQQ flipped C/Up — exit at tomorrow's open.")

        elif instrument == "QQQ":
            if today_mult - 1.0 >= 0.05:
                pending.update(action="EXIT", exit_type="TP",
                               reason=f"QQQ cum {(today_mult-1)*100:.2f}% ≥ 5% — "
                                      f"exit at tomorrow's open.")
            elif Close_q[i] <= stop_qqq:
                pending.update(action="EXIT", exit_type="ATR",
                               reason=f"QQQ close {Close_q[i]:.2f} ≤ ATR stop "
                                      f"{stop_qqq:.2f} — exit at tomorrow's open.")
            elif i != entry_idx and Trend[i] == "C/Dn" and i > 0 and cc[i - 1] == "C/Up":
                pending.update(action="EXIT", exit_type="Flip",
                               reason="C/Dn flip while long QQQ — exit at tomorrow's open.")
            elif DEW[i] == "Sell":
                pending.update(action="EXIT", exit_type="DEW",
                               reason="DEW Sell while long QQQ — exit at tomorrow's open.")

        elif instrument == "PSQ":
            if today_mult - 1.0 >= 0.05:
                pending.update(action="EXIT", exit_type="TP",
                               reason=f"PSQ cum {(today_mult-1)*100:.2f}% ≥ 5% — "
                                      f"cover SQQQ at tomorrow's open.")
            elif Close_q[i] >= stop_qqq:
                pending.update(action="EXIT", exit_type="ATR",
                               reason=f"QQQ close {Close_q[i]:.2f} ≥ ATR stop "
                                      f"{stop_qqq:.2f} — cover SQQQ at tomorrow's open.")
            elif i != entry_idx and Trend[i] == "C/Up" and i > 0 and cc[i - 1] == "C/Dn":
                pending.update(action="EXIT", exit_type="Flip",
                               reason="C/Up flip while short PSQ — cover at tomorrow's open.")
            elif DEW[i] == "Buy":
                pending.update(action="EXIT", exit_type="DEW",
                               reason="DEW Buy while short PSQ — cover at tomorrow's open.")

        elif instrument == "Cap":
            if Close_q[i] >= cap_target:
                pending.update(action="EXIT", exit_type="TP",
                               reason=f"QQQ close {Close_q[i]:.2f} ≥ cap_target "
                                      f"{cap_target:.2f} — exit Cap at tomorrow's open.")
            elif Close_q[i] <= cap_stop:
                pending.update(action="EXIT", exit_type="Stop",
                               reason=f"QQQ close {Close_q[i]:.2f} ≤ cap_stop "
                                      f"{cap_stop:.2f} — exit Cap at tomorrow's open.")
            elif (i - entry_idx) >= 15:
                pending.update(action="EXIT", exit_type="MaxHold",
                               reason=f"Cap held {i-entry_idx} bars ≥ 15 — exit at tomorrow's open.")
            # Cap preempt-by-v14 would be checked here, but if v14 wants to enter
            # tomorrow we'd see it in the Phase C entry block below.

        if pending["action"] == "HOLD":
            pending["reason"] = f"No exit condition met. Holding {instrument}."

    # ─── Not in trade: check entries (v14 priority order) ───
    if not pending.get("action") in ("PREEMPT_1A", "PREEMPT_1B") and (
            not in_trade or pending["action"] == "EXIT"):
        # If exiting today, v14 entries CAN fire same bar via fall-through.
        if cup_f and not cup_entered:
            pending["next_entry"] = {
                "instrument": "TQQQ", "variant": "cup",
                "F_preview": compute_F_preview(),
                "reason": "C/Up TQQQ (cup_f) — BUY TQQQ at tomorrow's open.",
            }
        elif cdn_f and last_dew != "Buy":
            pending["next_entry"] = {
                "instrument": "TQQQ", "variant": "cdn",
                "F_preview": compute_F_preview(),
                "reason": "C/Dn TQQQ (cdn_f) — BUY TQQQ at tomorrow's open.",
            }
        elif DEW[i] == "Buy" and last_dew != "Buy":
            reasons = []
            if not np.isnan(RT[i]) and 0.85 <= RT[i] < 0.90: reasons.append(f"RT={RT[i]:.2f}∈[0.85,0.90)")
            if not np.isnan(BSR[i]) and BSR[i] > 1.50:        reasons.append(f"BSR={BSR[i]:.2f}>1.50")
            if not np.isnan(MTI[i]) and MTI[i] > 1.15:        reasons.append(f"MTI={MTI[i]:.2f}>1.15")
            if vix_too_high:                                  reasons.append(f"VIX={VIX[i]:.2f}>35")
            if reasons:
                pending["next_entry"] = {"instrument": "QQQ", "blocked": True,
                                         "reasons": reasons,
                                         "reason": f"DEW Buy → QQQ blocked: {', '.join(reasons)}"}
            else:
                pending["next_entry"] = {"instrument": "QQQ",
                                         "reason": "DEW Buy — BUY QQQ at tomorrow's open."}
        elif DEW[i] == "Sell" and last_dew != "Sell" and cc[i] == "C/Dn":
            reasons = []
            if psq_blocked():
                reasons.append(f"RT={RT[i]:.2f}>0.95 AND MTI={MTI[i]:.2f}>0.95")
            if vix_too_high:
                reasons.append(f"VIX={VIX[i]:.2f}>35")
            if reasons:
                pending["next_entry"] = {"instrument": "PSQ", "blocked": True,
                                         "reasons": reasons,
                                         "reason": f"DEW Sell → PSQ blocked: {', '.join(reasons)}"}
            else:
                pending["next_entry"] = {"instrument": "PSQ",
                                         "reason": ("DEW Sell in C/Dn — SHORT QQQ via SQQQ "
                                                    "(1/3 position) at tomorrow's open.")}
        else:
            # Cap entry check
            if (not in_trade or pending["action"] == "EXIT") \
                    and (i - last_cap_signal_idx) >= 5 \
                    and not np.isnan(qqq_5d_ret[i]) and qqq_5d_ret[i] <= -0.10:
                pending["next_entry"] = {
                    "instrument": "Cap",
                    "reason": (f"QQQ 5-day return {qqq_5d_ret[i]*100:.2f}% ≤ -10% — "
                               f"BUY QQQ (Cap, 1.0×) at tomorrow's open."),
                }

        if pending["action"] == "HOLD" and "next_entry" not in pending:
            pending["reason"] = "No signal. Flat."

    return pending


# ──────────────────────────────────────────────────────────────────────────────
# Report formatter
# ──────────────────────────────────────────────────────────────────────────────

def format_report(df: pd.DataFrame, state: dict, pending: dict, trades: list,
                  perf: dict, combo: np.ndarray, realized_vol: np.ndarray) -> str:
    i = len(df) - 1
    today = df.loc[i, "Date"]
    row = df.iloc[i]
    lines = []

    W = 70
    def rule(ch="─"): return ch * W
    def hdr(s, ch="═"): return f"\n{rule(ch)}\n  {s}\n{rule(ch)}"

    lines.append(hdr(f"Phoenix Daily Signal — {today.strftime('%Y-%m-%d')} (last bar)"))

    # Today's market data snapshot
    lines.append(hdr("Today's market data", "─"))
    lines.append(f"  QQQ Close = {row['QQQ_Close']:.2f}   Trend = {row['Trend']}   cc = {row['cc']}")
    rt_s   = f"{row['RT']:.2f}"   if not pd.isna(row['RT'])   else "—"
    mti_s  = f"{row['MTI']:.2f}"  if not pd.isna(row['MTI'])  else "—"
    bsr_s  = f"{row['BSR']:.2f}"  if not pd.isna(row['BSR'])  else "—"
    vix_s  = f"{row['VIX']:.2f}"  if not pd.isna(row['VIX'])  else "—"
    dew_s  = row['DEW_Signal'] if row['DEW_Signal'] else "(none)"
    lines.append(f"  RT = {rt_s}   MTI = {mti_s}   BSR = {bsr_s}   VIX = {vix_s}")
    lines.append(f"  DEW signal today = {dew_s}")
    if not pd.isna(row.get('TQQQ_rt', np.nan)):
        lines.append(f"  TQQQ_rt = {row['TQQQ_rt']:.2f}   TQQQ_atr = {row['TQQQ_atr']:.2f}   "
                     f"TQQQ_close = {row['TQQQ_close']:.2f}")
    qqq5 = (row['QQQ_Close'] / df['QQQ_Close'].iloc[i - 5] - 1) if i >= 5 else float("nan")
    if not np.isnan(qqq5):
        lines.append(f"  qqq_5d_ret = {qqq5*100:+.2f}%")

    # Macro / R21 state
    lines.append(hdr("R21 macro state", "─"))
    combo_v = combo[i] if not np.isnan(combo[i]) else float("nan")
    vol_v   = realized_vol[i] if not np.isnan(realized_vol[i]) else float("nan")
    vol_sc  = max(0.30, min(1.20, 0.24 / vol_v)) if not np.isnan(vol_v) and vol_v > 0 else 1.0
    lines.append(f"  combo (regime scaler)  = {combo_v:.4f}")
    lines.append(f"  realized_vol_20 (ann.) = {vol_v:.4f}   vol_scaler = {vol_sc:.4f}")
    dd_pct = pending["dd_now"] * 100
    brake_status = "FIRING (×0.70)" if pending["dd_now"] < -0.25 else "off"
    lines.append(f"  Portfolio eq = ${pending['eq_running']:,.0f}  "
                 f"peak = ${pending['peak_eq']:,.0f}  dd = {dd_pct:+.2f}%  "
                 f"DD brake = {brake_status}")

    # Current position
    lines.append(hdr("Current position", "─"))
    if state["in_trade"]:
        entry_date = df.loc[state["entry_idx"], "Date"].strftime('%Y-%m-%d')
        bars = i - state["entry_idx"]
        s = (f"  {state['instrument']}")
        if state["instrument"] == "TQQQ":
            s += f" ({state['tqqq_variant']})  F={state['tqqq_F']:.4f}"
            if state["pyramid_on"]:
                s += "  pyramid=1.25×"
        lines.append(s)
        lines.append(f"  Entry fill: {entry_date}  ({bars} bars held)")
        if state["instrument"] == "TQQQ":
            lines.append(f"  Target = {state['tqqq_target']:.2f}   Stop = {state['tqqq_stop']:.2f}"
                         f"   be_moved = {state['be_moved']}")
        elif state["instrument"] in ("QQQ", "PSQ"):
            lines.append(f"  ATR stop = {state['stop_qqq']:.2f}")
        elif state["instrument"] == "Cap":
            lines.append(f"  Cap target = {state['cap_target']:.2f}   "
                         f"Cap stop = {state['cap_stop']:.2f}")
        if pending["today_mult"] is not None:
            lines.append(f"  Today's cum_mult = {pending['today_mult']:.4f}  "
                         f"({(pending['today_mult']-1)*100:+.2f}%)")
    else:
        lines.append("  Flat — no open position.")
        lines.append(f"  last_dew = {state['last_dew']}   cup_entered = {state['cup_entered']}   "
                     f"cdn_active = {state['cdn_active']}")
        if state["last_cap_signal_idx"] > -10**8:
            cd = i - state["last_cap_signal_idx"]
            lines.append(f"  Cap cooldown: {cd} bars since last signal "
                         f"({'cleared' if cd >= 5 else 'ACTIVE'})")

    # Tomorrow's action
    lines.append(hdr("ACTION FOR TOMORROW'S OPEN", "═"))
    if pending["action"] == "EXIT":
        lines.append(f"  >>> EXIT  ({pending['exit_type']})")
        lines.append(f"      {pending['reason']}")
        if "next_entry" in pending:
            ne = pending["next_entry"]
            if ne.get("blocked"):
                lines.append(f"      ↳ Subsequent v14 entry candidate: {ne['instrument']} — "
                             f"BLOCKED ({', '.join(ne['reasons'])})")
            else:
                lines.append(f"      ↳ Same-bar entry: {ne['instrument']}  "
                             f"({ne.get('reason','')})")
    elif pending["action"] in ("PREEMPT_1A", "PREEMPT_1B"):
        lines.append(f"  >>> {pending['action'].replace('_',' ')}")
        lines.append(f"      {pending['reason']}")
        if pending.get("F_preview") is not None:
            lines.append(f"      TQQQ F = {pending['F_preview']:.4f}")
    elif "next_entry" in pending:
        ne = pending["next_entry"]
        if ne.get("blocked"):
            lines.append(f"  >>> SIGNAL BLOCKED — {ne['reason']}")
        else:
            lines.append(f"  >>> ENTER {ne['instrument']}")
            lines.append(f"      {ne['reason']}")
            if ne.get("F_preview") is not None:
                lines.append(f"      TQQQ F = {ne['F_preview']:.4f}")
    else:
        lines.append(f"  >>> {pending['action']}")
        lines.append(f"      {pending['reason']}")

    # Backtest summary
    lines.append(hdr("Backtest reconciliation (through bar N-2)", "─"))
    lines.append(f"  Replay through {df.loc[state['last_idx'], 'Date'].strftime('%Y-%m-%d')}: "
                 f"{len(trades)} trades  eq = ${perf['final']:,.0f}")
    lines.append(f"  CAGR {perf['cagr']*100:.2f}%   Sharpe {perf['sharpe']:.2f}   "
                 f"Max DD {perf['max_dd']*100:.2f}%")
    if trades:
        last = trades[-1]
        lines.append(f"  Most recent trade: {pd.Timestamp(last['entry_fill']).date()} → "
                     f"{pd.Timestamp(last['exit_fill']).date()}  {last['instrument']}  "
                     f"{last['ret']*100:+.4f}%  ({last['reason']})")

    lines.append(rule("═"))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Subject / SMS summary
# ──────────────────────────────────────────────────────────────────────────────

def build_subject(state: dict, pending: dict, today_str: str) -> str:
    if pending["action"] == "EXIT":
        verb = f"EXIT {state['instrument']} ({pending['exit_type']})"
        if "next_entry" in pending and not pending["next_entry"].get("blocked"):
            verb += f" → {pending['next_entry']['instrument']}"
    elif pending["action"].startswith("PREEMPT"):
        verb = f"PREEMPT {state['instrument']} → TQQQ"
    elif "next_entry" in pending:
        ne = pending["next_entry"]
        verb = ("BLOCKED " if ne.get("blocked") else "ENTER ") + ne["instrument"]
    else:
        verb = f"HOLD {state['instrument']}" if state["in_trade"] else "FLAT"
    return f"Phoenix {today_str}: {verb}"


def build_sms(state: dict, pending: dict, today_str: str) -> str:
    sub = build_subject(state, pending, today_str)
    extra = pending.get("reason", "")
    return f"{sub}\n{extra}" if extra else sub


# ──────────────────────────────────────────────────────────────────────────────
# Email / iMessage
# ──────────────────────────────────────────────────────────────────────────────

def send_email(subject: str, body: str) -> None:
    if not GMAIL_PASS or not GMAIL_USER:
        print("(skipping email: GOOGLE_EMAIL/APP_PASSWORD not set)")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = ", ".join(TO_EMAIL)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        print(f"email sent to {TO_EMAIL}")
    except Exception as e:
        print(f"email send failed: {e}", file=sys.stderr)


def send_imessage(numbers: list[str], body: str) -> None:
    """Send via Continuity (paired iPhone). macOS only."""
    for num in numbers:
        service = "SMS" if num in SMS_FORCE else "iMessage"
        script = f'''
on run argv
    set targetService to "{service}"
    set targetNumber to "{num}"
    set msgBody to (item 1 of argv)
    tell application "Messages"
        set targetBuddy to participant targetNumber of (first service whose service type is {{targetService}})
        send msgBody to targetBuddy
    end tell
end run
'''
        try:
            subprocess.run(["osascript", "-e", script, body],
                           check=True, capture_output=True)
            print(f"iMessage sent to {num} ({service})")
        except subprocess.CalledProcessError as e:
            print(f"iMessage to {num} failed: {e.stderr.decode().strip()}", file=sys.stderr)
        except Exception as e:
            print(f"iMessage to {num} failed: {e}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    today = pd.Timestamp(date.today())

    # Build the full pipeline.  Cap the date window at today (or further out if
    # the CSVs happen to contain future-dated rows from a stale feed).
    print(f"Phoenix daily signal — {today.date()}", flush=True)
    print("Loading data and computing macro signals...", flush=True)
    df = build_step1(end_date=today)
    df = attach_macro_signals(df)
    sigs8 = build_step6_signals(df)
    votes, scale = compute_bearish_gate(df, sigs8)
    df["scale_5plus_to_00"] = scale
    sigs13 = build_13_signals(df, sigs8)
    combo, _mean_score, _zs = build_combo_r21(df, sigs13)
    realized_vol = build_realized_vol(df)

    n = len(df)
    if n < 2:
        print("Not enough data to evaluate.", file=sys.stderr)
        return 1

    print(f"df: {n} rows, {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}", flush=True)

    # Replay through bar n-2 (so engine state reflects close of yesterday from
    # the perspective of today, the last bar).
    trades, dret, state = run_combined_engine(
        df, scenario="7", pyramid=True,
        r21_combo=combo, realized_vol=realized_vol, dd_brake=True, enable_cap=True,
        stop_at=n - 2, return_state=True,
    )
    # Pad dret to full length for the perf metric (we leave the last bar at 0).
    perf = compute_perf(df["Date"].iloc[:n - 1], dret[:n - 1])

    qqq_5d_ret = df["qqq_5d_ret"].values
    pending = evaluate_last_bar(df, state, combo, realized_vol, qqq_5d_ret)

    report = format_report(df, state, pending, trades, perf, combo, realized_vol)
    print(report)

    today_str = df["Date"].iloc[-1].strftime("%Y-%m-%d")
    subject = build_subject(state, pending, today_str)
    sms = build_sms(state, pending, today_str)

    send_email(subject, report)
    if SMS_NUMBERS:
        send_imessage(SMS_NUMBERS, sms)

    return 0


if __name__ == "__main__":
    sys.exit(main())
