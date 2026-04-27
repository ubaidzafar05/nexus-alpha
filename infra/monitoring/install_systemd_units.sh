#!/usr/bin/env bash
set -euo pipefail
# This script prints sudo commands to install systemd unit templates.
# It does NOT run them automatically to avoid requiring sudo in CI.
REPO_ROOT="/Users/azt/Desktop/Python/nexus-alpha"
UNIT_DIR="$REPO_ROOT/infra/monitoring/systemd"

cat <<'EOF'
To install systemd units, run the following commands as a privileged user (copy/paste):

sudo cp "$UNIT_DIR"/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now telegram_forwarder.service metrics_server.service ollama_monitor.service post_eval_watcher.service docker-services.service
# Watch logs for services
sudo journalctl -u telegram_forwarder -f &
sudo journalctl -u metrics_server -f &
EOF
