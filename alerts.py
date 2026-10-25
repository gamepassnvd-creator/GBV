"""Alerts: notify humans when new high/critical content appears (webhook / email).

Configured via env vars only — no secrets ever stored in the database:
  GBV_WEBHOOK_URL   → JSON POST {"text": "..."} (Slack/Discord-compatible webhook)
  GBV_ALERT_EMAIL   → recipient list, comma separated
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS  (STARTTLS)
Both are optional; with neither configured nothing is sent and the audit log notes it.
"""
from __future__ import annotations

import json
import os
from email.message import EmailMessage
from smtplib import SMTP, SMTPException

import requests

import storage


def _env_rcpt() -> list[str]:
    return [a.strip() for a in os.environ.get("GBV_ALERT_EMAIL", "").split(",") if a.strip()]


def send(text: str) -> dict:
    done = {}
    url = os.environ.get("GBV_WEBHOOK_URL", "").strip()
    if url:
        try:
            r = requests.post(url, data=json.dumps({"text": text}),
                              headers={"Content-Type": "application/json"}, timeout=30)
            done["webhook"] = r.status_code
        except Exception as e:
            done["webhook"] = f"error: {e}"
    rcpt = _env_rcpt()
    if rcpt and os.environ.get("SMTP_HOST"):
        try:
            msg = EmailMessage()
            msg["To"] = ", ".join(rcpt)
            msg["Subject"] = "fb_gbv_monitor alert"
            msg.set_content(text)
            with SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587")), timeout=60) as s:
                s.starttls()
                u, p = os.environ.get("SMTP_USER", ""), os.environ.get("SMTP_PASS", "")
                if u:
                    s.login(u, p)
                s.send_message(msg)
            done["email"] = "sent"
        except (SMTPException, OSError, ValueError) as e:
            done["email"] = f"error: {e}"
    if not done:
        done["note"] = "no alerts configured (set GBV_WEBHOOK_URL and/or GBV_ALERT_EMAIL+SMTP_*)"
    storage.audit("monitor", "alert", text[:300])
    return done
