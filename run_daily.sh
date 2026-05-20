#!/bin/bash
# Dampier Nitro++ Phoenix — Daily runner.
# 1. Fetch Yahoo data for 14 macros + SQQQ + VIX
# 2. Fetch VectorVest data (QQQ, TQQQ OHLC + timing table)
# 3. Sync 6 shared files from the sibling Nitro project (Nitro = source of truth)
# 4. Freshness check (aborts run if any file stale)
# 5. Generate today's signal (replays engine + previews tomorrow's open)
# 6. Commit & push data files
# Logs to logs/daily-YYYY-MM-DD_HHMMSS.log, emails heartbeat or failure.

set -eo pipefail

# Load credentials: VECTORVEST_EMAIL/PASSWORD, GOOGLE_EMAIL/APP_PASSWORD,
# STOCKS_DIR, GIT_TOKEN, PHOENIX_SMS_NUMBERS (optional).
source "$HOME/.bash_profile"

PROJECT_SUBDIR="${PHOENIX_PROJECT_SUBDIR:-NitroPhoenix}"
NITRO_SUBDIR="${PHOENIX_NITRO_SUBDIR:-Nitro}"

if [ -z "$STOCKS_DIR" ]; then
    echo "ERROR: STOCKS_DIR is not set. Add 'export STOCKS_DIR=...' to ~/.bash_profile." >&2
    exit 1
fi

ROOT_DIR="$STOCKS_DIR/$PROJECT_SUBDIR"
if [ ! -d "$ROOT_DIR" ]; then
    echo "ERROR: project directory does not exist: $ROOT_DIR" >&2
    exit 1
fi

# Sibling Nitro project — source of truth for the 6 shared history files.
NITRO_DIR="$STOCKS_DIR/$NITRO_SUBDIR"

cd "$ROOT_DIR"

# Sanity check: Backtester/phoenix.py must exist (imported by phoenix_daily_signal.py)
if [ ! -f "$ROOT_DIR/Backtester/phoenix.py" ]; then
    echo "ERROR: Backtester/phoenix.py not found at $ROOT_DIR/Backtester/" >&2
    exit 1
fi

# Prefer the project venv if present; fall back to system python3.
if [ -x "$ROOT_DIR/.venv/bin/python3" ]; then
    PY="$ROOT_DIR/.venv/bin/python3"
else
    PY="$(command -v python3)"
fi

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date '+%Y-%m-%d_%H%M%S')"
LOG_FILE="$LOG_DIR/daily-${TS}.log"

# ── Email alert helper ────────────────────────────────────────────────────────
send_alert() {
    local subject="$1"
    local body_file="$2"
    "$PY" -c '
import os, sys, smtplib
from email.mime.text import MIMEText
user = os.environ.get("GOOGLE_EMAIL", "")
pw   = os.environ.get("GOOGLE_APP_PASSWORD", "")
if not (user and pw):
    print("send_alert: GOOGLE_EMAIL/GOOGLE_APP_PASSWORD not set — skipping", file=sys.stderr)
    sys.exit(0)
subject = sys.argv[1]
with open(sys.argv[2], "r") as f:
    body = f.read()
msg = MIMEText(body)
msg["Subject"] = subject
msg["From"]    = user
msg["To"]      = user
try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.sendmail(user, [user], msg.as_string())
    print(f"alert sent: {subject}")
except Exception as e:
    print(f"alert email failed: {e}", file=sys.stderr)
' "$subject" "$body_file"
}

# ── Git publish helper ────────────────────────────────────────────────────────
git_publish() {
    if [ -z "$GIT_TOKEN" ];    then echo "WARN: GIT_TOKEN not set — skipping git push";   return 1; fi
    if [ -z "$GOOGLE_EMAIL" ]; then echo "WARN: GOOGLE_EMAIL not set — skipping git push"; return 1; fi

    git add data/csv/daily data/csv/history || return 1
    if git diff --cached --quiet; then
        echo "No data changes to commit."
        return 0
    fi
    git -c user.email="$GOOGLE_EMAIL" -c user.name="$GOOGLE_EMAIL" \
        commit -m "Daily data update — $(date '+%Y-%m-%d')" || return 1

    local encoded_user="${GOOGLE_EMAIL//@/%40}"
    local push_url="https://${encoded_user}:${GIT_TOKEN}@github.com/dampiermike/NitroPhoenix.git"
    git -c credential.helper= push "$push_url" HEAD:main || return 1
    echo "Pushed to origin/main."
}

# ── Nitro shared-file sync helper ─────────────────────────────────────────────
# Six history files also live in the sibling Nitro project. Nitro runs its own
# daily fetch and is the source of truth for them, so after our own fetch steps
# we overwrite our copies with Nitro's (whenever Nitro has the file).
NITRO_SHARED_FILES=(
    qqq-from-vv.csv
    sqqq-from-yahoo.csv
    synthetic-tqqq-ohlc-1999-2010.csv
    tqqq-from-vv.csv
    vectorvest-views-w3place-precision.csv
    vix-from-yahoo.csv
)

sync_from_nitro() {
    local src_dir="$NITRO_DIR/data/csv/history"
    local dst_dir="$ROOT_DIR/data/csv/history"
    if [ ! -d "$src_dir" ]; then
        echo "WARN: Nitro history dir not found ($src_dir) — skipping sync."
        return 0
    fi
    mkdir -p "$dst_dir"
    local copied=0 missing=0
    for f in "${NITRO_SHARED_FILES[@]}"; do
        if [ -f "$src_dir/$f" ]; then
            if cp -p "$src_dir/$f" "$dst_dir/$f"; then
                echo "  copied  $f"
                copied=$((copied + 1))
            else
                echo "  WARN: failed to copy $f"
            fi
        else
            echo "  skip    $f (not present in Nitro — keeping existing copy)"
            missing=$((missing + 1))
        fi
    done
    echo "Nitro sync: $copied copied, $missing not present in Nitro."
}

# ── Failure trap ──────────────────────────────────────────────────────────────
on_failure() {
    local rc=$?
    local alert_body="/tmp/${PROJECT_SUBDIR}_fail_$$.txt"
    {
        echo "Phoenix daily run FAILED"
        echo "Exit code: $rc"
        echo "Timestamp: $(date)"
        echo "Log file:  $LOG_FILE"
        echo ""
        echo "─── Last 50 log lines ───"
        tail -n 50 "$LOG_FILE" 2>/dev/null || echo "(no log available)"
    } > "$alert_body"
    send_alert "❌ Phoenix daily FAILED (rc=$rc)" "$alert_body"
    rm -f "$alert_body"
    exit $rc
}
trap on_failure ERR

# ── Pipeline ──────────────────────────────────────────────────────────────────
{
    echo "=========================================="
    echo "  Phoenix Daily Run — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  python: $PY"
    echo "  root:   $ROOT_DIR"
    echo "=========================================="

    echo
    echo "── Step 1/6: Yahoo daily fetch (14 macros + SQQQ + VIX) ──"
    "$PY" fetch_yahoo_daily.py

    echo
    echo "── Step 2/6: VectorVest daily fetch (QQQ, TQQQ, timing) ──"
    "$PY" fetch_vv_daily.py

    echo
    echo "── Step 3/6: Sync shared files from Nitro (Nitro = source of truth) ──"
    sync_from_nitro

    echo
    echo "── Step 4/6: Freshness check (aborts run if any file stale) ──"
    "$PY" validate_freshness.py

    echo
    echo "── Step 5/6: Phoenix daily signal ──"
    "$PY" phoenix_daily_signal.py

    echo
    echo "── Step 6/6: Commit & push data ──"
    git_publish || echo "WARN: git commit/push failed (continuing)"

    echo
    echo "=========================================="
    echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
} 2>&1 | tee "$LOG_FILE"

# ── Heartbeat on success ──────────────────────────────────────────────────────
alert_body="/tmp/${PROJECT_SUBDIR}_ok_$$.txt"
{
    echo "Phoenix daily run completed successfully."
    echo "Timestamp: $(date)"
    echo "Log file:  $LOG_FILE"
    echo ""
    echo "─── Last 40 log lines ───"
    tail -n 40 "$LOG_FILE"
} > "$alert_body"
send_alert "✅ Phoenix daily OK" "$alert_body"
rm -f "$alert_body"
