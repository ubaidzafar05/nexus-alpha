#!/bin/bash
# NEXUS-ULTRA v9 Background Runner
# Ensures the God-Loop persists after terminal disconnect.

set -e

# Configuration
LOG_FILE="logs/nexus_production.log"
PID_FILE="data/nexus.pid"

mkdir -p logs data

# Prevent duplicate runs
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "NEXUS is already running with PID $PID. Stop it first."
        exit 1
    fi
fi

echo "Starting NEXUS-ULTRA v9 God-Loop in background..."
export PYTHONPATH=.
nohup python3 nexus_alpha/cli.py run > "$LOG_FILE" 2>&1 &
NEW_PID=$!

echo $NEW_PID > "$PID_FILE"
echo "NEXUS launched securely. PID: $NEW_PID"
echo "Monitor logs with: tail -f $LOG_FILE"
