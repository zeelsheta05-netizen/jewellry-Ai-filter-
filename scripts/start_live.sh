#!/bin/bash
# Run Design Finder live: the app on this Mac plus a Cloudflare quick tunnel,
# so phones (whose camera needs https) and other computers can open it.
#
#     scripts/start_live.sh          # Ctrl+C stops everything
#
# Prints the public link. It changes on every start (not when the app is
# restarted after a crash: the tunnel keeps running, so the link stays); the app's "On phone"
# button always shows the current one as a QR code. The Mac is kept awake
# while this runs (sleep would disconnect the dataset drive), but closing the
# lid still sleeps it unless it is on power with a display attached.
# With the dataset in S3 (JEWEL_STORAGE=s3://...) the SSD is not needed at all.
set -u
cd "$(dirname "$0")/.." || exit 1
PORT="${PORT:-8765}"
LOG=data/tunnel.log

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use: stop the other server first (Ctrl+C in its terminal)." >&2
  exit 1
fi
# the dataset storage (S3 bucket, or the SSD before the move): JEWEL_STORAGE in .env
.venv/bin/python -c "from jewelsearch import storage as s; st = s.get(); st.ready() or print(f'Warning: {st.label()} is not available; full-size photos, 3D videos and try-on photos will not load.')"

cleanup() { kill "$APP" "$TUNNEL" "$AWAKE" 2>/dev/null; wait 2>/dev/null; echo "Stopped."; }
trap cleanup EXIT
trap 'exit 0' INT TERM

start_app() { .venv/bin/uvicorn jewelsearch.server:app --host 127.0.0.1 --port "$PORT" & APP=$!; STARTED=$(date +%s); }

caffeinate -ims -w $$ & AWAKE=$!
start_app
./.tools/cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$LOG" 2>&1 & TUNNEL=$!

URL=""
for _ in $(seq 60); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1)
  [ -n "$URL" ] && break
  sleep 1
done
if [ -n "$URL" ]; then
  echo
  echo "  Design Finder is live:  $URL"
  echo "  On this Mac:            http://localhost:$PORT"
  echo "  On a phone: press \"On phone\" (top right) and scan the QR code."
  echo
else
  echo "The tunnel didn't start (see $LOG). The app still runs on http://localhost:$PORT" >&2
fi

# run until the tunnel stops. An app that stops on its own is started again behind the same
# tunnel, so the link keeps working (pages say "try again in a minute" while it loads);
# one that stops 3 times in a row within 2 minutes of starting is broken, not unlucky.
QUICK=0
while kill -0 "$TUNNEL" 2>/dev/null; do
  if ! kill -0 "$APP" 2>/dev/null; then
    wait "$APP"; CODE=$?
    if [ $(( $(date +%s) - STARTED )) -lt 120 ]; then QUICK=$((QUICK + 1)); else QUICK=0; fi
    if [ "$QUICK" -ge 3 ]; then
      echo "The app keeps stopping right after it starts (exit $CODE); not starting it again." >&2
      exit 1
    fi
    echo "=== $(date '+%F %T') the app stopped (exit $CODE); starting it again, same link: $URL ===" >&2
    start_app
  fi
  sleep 2
done
echo "The tunnel stopped (tunnel log: $LOG)." >&2
