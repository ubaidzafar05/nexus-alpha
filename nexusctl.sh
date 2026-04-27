#!/bin/bash

# Nexus-Alpha Operational Controller
# Tiered service management to protect 16GB RAM constraints.

PROJECT_NAME="nexus-alpha"
DOCKER_BIN="/usr/local/bin/docker"

# Core Services: Essential for trading loop
CORE_SERVICES="timescaledb redis kafka ollama qdrant nexus"

# Full Stack includes: Monitoring (Grafana, Prom, Loki) & ML (MLFlow, Qdrant, Crawl4ai, Freqtrade)
ALL_SERVICES="timescaledb redis kafka ollama nexus crawl4ai qdrant mlflow freqtrade prometheus grafana loki"

usage() {
    echo "Usage: $0 {start|start-full|stop|restart|status|logs}"
    echo "  start       - Only start the 6 CORE services (Safe for 16GB RAM)"
    echo "  start-full  - Start all 13 services (High RAM usage)"
    echo "  stop        - Stop all containers"
    echo "  restart     - Restart core services"
    echo "  status      - Show status of all services"
    echo "  logs        - Follow logs for the main nexus agent"
}

case "$1" in
    start)
        echo "🚀 Starting Nexus-Alpha in CORE mode (RAM Safe)..."
        $DOCKER_BIN compose up -d $CORE_SERVICES
        ;;
    start-full)
        echo "🔥 Starting FULL Nexus-Alpha stack (Warning: High RAM usage)..."
        $DOCKER_BIN compose up -d
        ;;
    stop)
        echo "🛑 Stopping all Nexus-Alpha services..."
        $DOCKER_BIN compose down
        ;;
    restart)
        $0 stop
        $0 start
        ;;
    status)
        $DOCKER_BIN compose ps
        ;;
    logs)
        $DOCKER_BIN compose logs -f nexus
        ;;
    *)
        usage
        exit 1
        ;;
esac
