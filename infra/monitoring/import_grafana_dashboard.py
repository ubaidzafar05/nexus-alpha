#!/usr/bin/env python3
"""
Import grafana dashboard JSON into Grafana using API token.
Env vars required:
  GRAFANA_URL - base url (e.g. http://localhost:3000)
  GRAFANA_API_KEY - admin API key
Place grafana_nexus_dashboard.json beside this script.
"""
import os
import json
import requests

GRAFANA_URL = os.environ.get('GRAFANA_URL')
API_KEY = os.environ.get('GRAFANA_API_KEY')
if not GRAFANA_URL or not API_KEY:
    raise SystemExit('GRAFANA_URL and GRAFANA_API_KEY must be set in environment')

with open('grafana_nexus_dashboard.json', 'r') as f:
    dashboard = json.load(f)

payload = {"dashboard": dashboard, "overwrite": True}
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
resp = requests.post(f"{GRAFANA_URL}/api/dashboards/db", json=payload, headers=headers, timeout=30)
if resp.ok:
    print('Dashboard imported/updated')
else:
    print('Failed to import dashboard', resp.status_code, resp.text)
    raise SystemExit(1)
