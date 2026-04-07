NEXUS-ALPHA Self-Healing Watchdog

This small utility monitors core docker-compose services and restarts them when
unhealthy. It's designed to be lightweight and run from the host (or as a tiny
container).

Usage
------

Install (no deps beyond project): ensure Python 3.11+ and docker/docker-compose
are available.

Run once:

  PYTHONPATH=. python infra/self_healing/watchdog.py

Recommended deployment
----------------------

As a systemd service (example):

  /etc/systemd/system/nexus-watchdog.service

  [Unit]
  Description=Nexus Alpha Watchdog
  After=docker.service

  [Service]
  WorkingDirectory=/path/to/nexus-alpha
  Environment=ENV_FILE=/path/to/nexus-alpha/.env
  ExecStart=/usr/bin/python3 -m infra.self_healing.watchdog
  Restart=always

  [Install]
  WantedBy=multi-user.target

Configuration
-------------

Control behavior via environment variables in .env or systemd unit:

  WATCH_SERVICES - comma separated list (default: timescaledb,kafka,ollama,redis,nexus)
  MONITOR_INTERVAL - seconds between checks (default 60)
  MIN_RESTART_INTERVAL - minimum seconds between restarts for same service (default 300)

Alerts
------

If TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are configured (in .env or env), the
watchdog will send system health alerts via Telegram using the project's
TelegramAlerts client.
