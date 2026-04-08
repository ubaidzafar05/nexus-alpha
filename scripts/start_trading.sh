#!/usr/bin/env bash
# Start Nexus-Alpha full free-stack for safe paper trading
# Usage: ./scripts/start_trading.sh [--env .env] [--interval 3600]
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ENV_FILE=${1:-.env}
RETRAIN_INTERVAL=${2:-3600}

cd "$REPO_ROOT"

if [ ! -f "$ENV_FILE" ]; then
  echo ".env not found at $ENV_FILE. Copy .env.sample and fill it first."
  exit 1
fi

echo "Starting docker-compose services (timescaledb, redis, kafka, ollama)"
docker-compose up -d timescaledb redis kafka ollama

echo "Waiting for services to settle..."
sleep 6

echo "Starting retrain watcher (background)"
# Load .env safely (skip comments and empty lines); avoid brittle xargs usage
# Launch the retrain watcher in a subshell with env vars exported
nohup bash -lc "(set -a; while IFS= read -r line || [ -n \"$line\" ]; do \
  # skip comments and blank lines
  [[ -z \"$line\" ]] && continue; \
  [[ \"$line\" =~ ^[[:space:]]*# ]] && continue; \
  export \"$line\"; \
  done < \"$ENV_FILE\"; set +a; exec python3 infra/self_healing/retrain_watcher.py --interval \"$RETRAIN_INTERVAL\")" &>/tmp/retrain-watcher.log &
echo $! > /tmp/retrain-watcher.pid

echo "Starting Nexus in PAPER mode (background). Logs: /tmp/nexus-paper.log"
# Run Nexus CLI paper mode — uses env file loaded safely like above
nohup bash -lc "(set -a; while IFS= read -r line || [ -n \"$line\" ]; do \
  [[ -z \"$line\" ]] && continue; \
  [[ \"$line\" =~ ^[[:space:]]*# ]] && continue; \
  export \"$line\"; \
  done < \"$ENV_FILE\"; set +a; exec python3 -m nexus_alpha.cli paper --min-signal-confidence 0.35 --max-position-age-minutes 15)" &>/tmp/nexus-paper.log &
echo $! > /tmp/nexus-paper.pid

sleep 1

echo "Started. PIDs:" 
echo "  retrain_watcher: $(cat /tmp/retrain-watcher.pid)"
echo "  nexus_paper:     $(cat /tmp/nexus-paper.pid)"

echo "Tail logs with: tail -f /tmp/nexus-paper.log /tmp/retrain-watcher.log"
