#!/bin/bash
# stress_once.sh - Realistic edge-like stress scenario using Docker + stress-ng + Prometheus snapshot

set -euo pipefail

# Explicit paths (cron/launchd safe)
DOCKER="/usr/local/bin/docker"
CURL="/usr/bin/curl"
JQ="/opt/homebrew/bin/jq"   # adjust if jq is installed elsewhere (run: which jq)

DOCKER_NETWORK=${DOCKER_NETWORK:-aether-net}
PROM_URL=${PROM_URL:-http://localhost:9090}

LOG_DIR="$HOME/Documents/capstone01/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stress-$(date +%Y%m%d).log"
echo "[$(date)] starting stress session..." >> "$LOG_FILE"

# ----------------------------
# Helpers
# ----------------------------
rand() { echo $(( $1 + RANDOM % ($2 - $1 + 1) )); }
ts() { date +"[%F %T]"; }

snapshot() {
  cpu=$($CURL -s "${PROM_URL}/api/v1/query" \
    --data-urlencode 'query=100 - (avg by(instance)(irate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)' \
    | $JQ -r '.data.result[0].value[1]' 2>/dev/null || echo NA)

  mem=$($CURL -s "${PROM_URL}/api/v1/query" \
    --data-urlencode 'query=(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100' \
    | $JQ -r '.data.result[0].value[1]' 2>/dev/null || echo NA)

  load=$($CURL -s "${PROM_URL}/api/v1/query" \
    --data-urlencode 'query=node_load1' \
    | $JQ -r '.data.result[0].value[1]' 2>/dev/null || echo NA)

  io=$($CURL -s "${PROM_URL}/api/v1/query" \
    --data-urlencode 'query=avg by(instance)(irate(node_disk_io_time_seconds_total[1m])) * 100' \
    | $JQ -r '.data.result[0].value[1]' 2>/dev/null || echo NA)

  echo "[$(ts)] [metrics] cpu=${cpu}% mem=${mem}% load1=${load} io_busy=${io}%" | tee -a "$LOG_FILE"
}

# ----------------------------
# Stress profile runner
# ----------------------------
run_profile() {
  local profile=$1
  case $profile in
    light)
      CPU=$(rand 1 2); VM=1; MEM=256M; IO=1; HDD=1; DUR=$(rand 15 30) ;;
    medium)
      CPU=$(rand 3 4); VM=2; MEM=512M; IO=2; HDD=2; DUR=$(rand 20 45) ;;
    heavy)
      CPU=$(rand 5 6); VM=3; MEM=1G;   IO=3; HDD=3; DUR=$(rand 30 60) ;;
  esac

  echo "$(ts) [START] profile=$profile cpu=$CPU vm=$VM mem=$MEM io=$IO hdd=$HDD dur=${DUR}s" \
    | tee -a "$LOG_FILE"

  $DOCKER run --rm --network "$DOCKER_NETWORK" alpine sh -c \
    "apk add --no-cache stress-ng >/dev/null && \
     stress-ng --cpu $CPU --vm $VM --vm-bytes $MEM --io $IO --hdd $HDD --timeout ${DUR}s" \
     >> "$LOG_FILE" 2>&1

  snapshot
  echo "$(ts) [END] profile=$profile" | tee -a "$LOG_FILE"
}

# ----------------------------
# Main session
# ----------------------------
BURSTS=$(rand 1 3)  # number of bursts in this run
echo "$(ts) [session] bursts=$BURSTS" | tee -a "$LOG_FILE"

for i in $(seq 1 $BURSTS); do
  profile=$(printf "light\nmedium\nheavy\n" | sort -R | head -n1)
  run_profile "$profile"

  if [[ $i -lt $BURSTS ]]; then
    COOLDOWN=$(rand 10 30)
    echo "$(ts) [cooldown] sleeping ${COOLDOWN}s before next burst..." | tee -a "$LOG_FILE"
    sleep $COOLDOWN
  fi
done

echo "$(ts) [session finished]" | tee -a "$LOG_FILE"