"""
deliver.py — Pharma Regulatory Pulse
Sends the formatted HTML newsletter to all subscribers.

Supports two delivery providers:
  1. Resend  — preferred (3,000/month free tier)
  2. SMTP    — fallback (any SMTP server)

Rate limiting: max 2 emails/second to stay within Resend free tier limits.
Per-recipient error isolation: failures are logged but do not abort the run.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Rate limit: Resend free tier allows ~100/day; 2/sec is a safe burst rate
SEND_INTERVAL_SECONDS = 0.5   # 2 emails per second


# ---------------------------------------------------------------------------
# Subscriber list loader
# ---------------------------------------------------------------------------

def _load_subscribers(csv_path: str) -> list[dict[str, str]]:
    """
    Load subscribers from a CSV file with columns: email, name.

    Returns
    -------
    list[dict]
        Each dict has 'email' and 'name' keys.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Subscribers CSV not found: %s — sending to no recipients", csv_path)
        return []

    subscribers: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            email = row.get("email", "").strip()
            name = row.get("name", "").strip()
            if email and "@" in email:
                subscribers.append({"email": email, "name": name})
            else:
                logger.warning("Skipping invalid subscriber row: %s", row)

    logger.info("Loaded %d subscribers from %s", len(subscribers), csv_path)
    return subscribers


# ---------------------------------------------------------------------------
# Plain-text strip
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Strip HTML tags to produce a plain-text alternative."""
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Resend provider
# ---------------------------------------------------------------------------

def _send_via_resend(
    html: str,
    plain_text: str,
    subscriber: dict[str, str],
    subject: str,
    email_config: dict,
) -> bool:
    """
    Send a single email via Resend SDK.

    Returns True on success, False on failure.
    """
    try:
        import resend  # lazy import — only required if using Resend
    except ImportError:
        logger.error("'resend' package not installed. Run: pip install resend")
        return False

    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logger.error("RESEND_API_KEY not set")
        return False

    resend.api_key = api_key

    try:
        params: Any = {
            "from": email_config.get("from", "newsletter@yourdomain.com"),
            "to": [subscriber["email"]],
            "subject": subject,
            "html": html,
            "text": plain_text,
            "reply_to": email_config.get("reply_to", ""),
        }
        resend.Emails.send(params)
        return True
    except Exception as exc:
        logger.error(
            "Resend error for %s: %s", subscriber["email"], exc
        )
        return False


# ---------------------------------------------------------------------------
# SMTP provider
# ---------------------------------------------------------------------------

def _send_via_smtp(
    html: str,
    plain_text: str,
    subscriber: dict[str, str],
    subject: str,
    email_config: dict,
) -> bool:
    """
    Send a single email via SMTP.

    Reads connection settings from environment variables:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD

    Returns True on success, False on failure.
    """
    smtp_host = os.getenv("SMTP_HOST") or email_config.get("smtp_host", "")
    smtp_port = int(os.getenv("SMTP_PORT") or email_config.get("smtp_port", 587))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    from_addr = email_config.get("from", smtp_user)

    if not smtp_host:
        logger.error("SMTP_HOST not configured")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = subscriber["email"]
        msg["Reply-To"] = email_config.get("reply_to", from_addr)

        msg.attach(MIMEText(plain_text, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.sendmail(from_addr, subscriber["email"], msg.as_string())

        return True
    except Exception as exc:
        logger.error(
            "SMTP error for %s: %s", subscriber["email"], exc
        )
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def send_newsletter(
    html: str,
    newsletter_md: str,
    email_config: dict,
) -> dict[str, int]:
    """
    Deliver the newsletter to all subscribers listed in the CSV.

    Parameters
    ----------
    html : str
        Rendered HTML email body from format_email.py.
    newsletter_md : str
        Markdown version used as plain-text fallback.
    email_config : dict
        The 'email' section of config.yaml.

    Returns
    -------
    dict
        {'success': N, 'failed': N, 'total': N}
    """
    csv_path = email_config.get("recipients_csv", "subscribers.csv")
    subscribers = _load_subscribers(csv_path)

    if not subscribers:
        logger.warning("No subscribers — skipping delivery")
        return {"success": 0, "failed": 0, "total": 0}

    # Build subject
    from datetime import date
    date_str = date.today().strftime("%B %d, %Y")
    subject_template = email_config.get(
        "subject_template", "Pharma Regulatory Pulse — Week of {date}"
    )
    subject = subject_template.format(date=date_str)

    plain_text = _strip_html(html)
    provider = email_config.get("provider", "resend").lower()
    success_count = 0
    failure_count = 0

    for subscriber in subscribers:
        if provider == "resend":
            ok = _send_via_resend(html, plain_text, subscriber, subject, email_config)
        else:
            ok = _send_via_smtp(html, plain_text, subscriber, subject, email_config)

        if ok:
            success_count += 1
            logger.info("Sent to: %s", subscriber["email"])
        else:
            failure_count += 1
            logger.error("Failed: %s", subscriber["email"])

        # Rate limiting
        time.sleep(SEND_INTERVAL_SECONDS)

    total = len(subscribers)
    logger.info(
        "Delivered to %d/%d recipients (%d failures)",
        success_count, total, failure_count,
    )
    return {"success": success_count, "failed": failure_count, "total": total}
