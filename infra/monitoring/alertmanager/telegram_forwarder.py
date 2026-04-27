#!/usr/bin/env python3
"""
Small Alertmanager webhook forwarder that posts concise alerts to Telegram.
Requires: pip install requests flask
Environment variables:
  TELEGRAM_BOT_TOKEN - bot token (do NOT commit)
  TELEGRAM_CHAT_ID  - chat id (or user id)
Run: TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python3 telegram_forwarder.py
"""
import os
import json
import logging
from flask import Flask, request, jsonify
import requests

log = logging.getLogger("telegram_forwarder")
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if not BOT_TOKEN or not CHAT_ID:
    log.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing; forwarding will fail until set.")

app = Flask(__name__)

TELEGRAM_SEND_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


def _format_alert(payload: dict) -> str:
    alerts = payload.get("alerts", [])
    lines = []
    for a in alerts[:6]:
        labels = a.get("labels", {})
        status = a.get("status", "fired")
        starts = a.get("startsAt", "?")
        ends = a.get("endsAt", "?")
        summary = a.get("annotations", {}).get("summary") or a.get("annotations", {}).get("description") or ''
        ln = f"[{status.upper()}] {labels.get('alertname', '')}: {summary}"
        lines.append(ln)
    if len(alerts) > 6:
        lines.append(f"(+{len(alerts)-6} more alerts)")
    text = "\n".join(lines)
    # Truncate to Telegram-friendly length
    return (text[:3800] + "...") if len(text) > 3800 else text


@app.route("/", methods=["POST"])
def webhook():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        text = _format_alert(payload)
        if not text:
            return jsonify({"ok": True, "msg": "empty payload"}), 200
        if not BOT_TOKEN or not CHAT_ID:
            log.error("Missing TELEGRAM env; cannot send message")
            return jsonify({"ok": False, "error": "missing_telegram_env"}), 500
        data = {"chat_id": CHAT_ID, "text": text}
        resp = requests.post(TELEGRAM_SEND_URL, json=data, timeout=10)
        if not resp.ok:
            log.error("Telegram send failed %s %s", resp.status_code, resp.text)
            return jsonify({"ok": False, "status": resp.status_code, "text": resp.text}), 500
        return jsonify({"ok": True}), 200
    except Exception as e:
        log.exception("error handling webhook")
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('ALERT_FORWARDER_PORT', 5001)))
