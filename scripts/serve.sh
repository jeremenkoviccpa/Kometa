#!/bin/sh
# Server entry point (Dockerfile CMD). Settings come from environment variables:
#   AT_API_OWNER_TOKEN   required, 32+ characters: the hub's access token (never printed)
#   AT_DEMO_BROKER       sim (default: simulated gold, no account) or oanda (practice account; also needs
#                        AT_ENV=paper, AT_OANDA_TOKEN, AT_OANDA_ACCOUNT and the owner-signed paper limits)
#   AT_STATE_DIR         persistent state, default /data/kometa (mount a volume at /data)
#   PORT                 set by the host (Railway), default 8080
set -eu
BROKER="${AT_DEMO_BROKER:-sim}"
STATE="${AT_STATE_DIR:-/data/kometa}"
PORT="${PORT:-8080}"
mkdir -p "$STATE"
if [ "$BROKER" = "sim" ]; then
  # the simulated market restarts from scratch each time (--fresh); a broker account's state never does
  exec at demo run --broker sim --host 0.0.0.0 --port "$PORT" --var "$STATE/sim" --fresh \
    --speed "${AT_DEMO_SPEED:-60}" --catchup-days "${AT_DEMO_CATCHUP_DAYS:-10}" \
    --run-days "${AT_DEMO_RUN_DAYS:-365}"
fi
exec at demo run --broker "$BROKER" --host 0.0.0.0 --port "$PORT" --var "$STATE/$BROKER"
