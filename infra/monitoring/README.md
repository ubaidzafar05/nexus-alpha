Monitoring & Alerts — Nexus Alpha

This folder contains Prometheus/Grafana helper files and an Alertmanager -> Telegram forwarder.

Quick checklist (recommended, minimal):
- Run Prometheus with prometheus_scrape.yml configured to scrape:
  - metrics_server.py (http://localhost:8000/metrics)
  - ollama_prometheus_exporter.py (http://localhost:8001/metrics)
- Run alertmanager with alertmanager.yml and point it to the local forwarder at http://127.0.0.1:5001/
- Start the forwarder: TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python3 alertmanager/telegram_forwarder.py
- Import Grafana dashboard: GRAFANA_URL=... GRAFANA_API_KEY=... python3 import_grafana_dashboard.py

Required secrets (rotate immediately if leaked):
- TELEGRAM_BOT_TOKEN (free)
- TELEGRAM_CHAT_ID (free)
- GRAFANA_API_KEY (created in Grafana; free if self-hosted)
- Any exchange API keys (Binance) — rotate if exposed

Notes on costs and alternatives:
- Ollama (local) is free/open-source to run locally. Avoid cloud LLMs unless needed.
- Prometheus + Grafana are free (self-hosted). Grafana Cloud has paid tiers.
- Telegram is free for alerts.
- Binance API is free to use but requires account and may have rate limits.
- CoinGecko provides a free public API tier (rate-limited). Alternatives: CryptoCompare, Nomics (may be paid).

Security:
- Do NOT commit secrets. Use environment variables or a secrets manager.
- Rotate any keys that were posted in chat.

If desired, next steps will:
- Add a systemd unit for the forwarder and metrics server
- Add GitHub Actions job to run the safe-retrain smoke check
- Harden OLLAMA defaults in .env and add a process watchdog (systemd) for Ollama

