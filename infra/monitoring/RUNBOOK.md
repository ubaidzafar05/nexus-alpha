Runbook — Nexus Alpha Monitoring & Safe-Retrain

Overview
- This runbook explains how to deploy and operate the monitoring components: Prometheus metrics server, Ollama monitor, Alertmanager -> Telegram forwarder, and safe_retrain watchers.

Prereqs
- Host with docker and systemd (or bare python env)
- Rotate any leaked keys before proceeding
- Ensure user 'azt' exists (or edit service files to use your user)

Deploy (recommended - docker compose)
1. Build and start containers
   - cd infra/monitoring
   - docker compose up -d --build
2. Verify containers running
   - docker ps | grep nexus
3. Import Grafana dashboard (if Grafana running)
   - export GRAFANA_URL=...; export GRAFANA_API_KEY=...
   - python3 import_grafana_dashboard.py

Deploy (systemd, non-container)
1. Copy unit files (requires sudo)
   - sudo cp infra/monitoring/systemd/*.service /etc/systemd/system/
   - sudo systemctl daemon-reload
   - sudo systemctl enable --now telegram_forwarder.service metrics_server.service ollama_monitor.service post_eval_watcher.service
2. Follow logs
   - sudo journalctl -u telegram_forwarder -f
   - sudo journalctl -u metrics_server -f

Environment & validation
- Load conservative defaults:
  - source infra/monitoring/ollama_env_defaults.env
  - python3 infra/monitoring/validate_env.py

Testing alerts
- Simulate an Alertmanager webhook:
  curl -XPOST -H "Content-Type: application/json" -d '{"alerts":[{"startsAt":"now","labels":{"alertname":"TEST_ALERT"},"annotations":{"summary":"test alert"}}]}' http://127.0.0.1:5001/

Rollback / Troubleshooting
- If promotion or retrain goes wrong:
  - Stop post_eval_watcher.service
  - Revert to previous checkpoint (data/checkpoints/*)
  - Rotate keys and restart services

Contact
- Ops: your-team@example.com

