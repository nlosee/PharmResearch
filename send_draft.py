"""
send_draft.py — Pharma Regulatory Pulse
Phase 2 of the human-approval workflow.

Reads an approved draft markdown file from drafts/YYYY-MM-DD.md,
formats it as HTML, and sends it to subscribers.

This is the delivery step that runs only after human review and approval.
Usage:
  python send_draft.py drafts/2025-03-24.md
  python send_draft.py drafts/2025-03-24.md --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Send an approved newsletter draft")
    parser.add_argument("draft_path", help="Path to the draft markdown file (e.g. drafts/2025-03-24.md)")
    parser.add_argument("--dry-run", action="store_true", help="Format HTML but do not send")
    args = parser.parse_args()

    draft_path = Path(args.draft_path)
    if not draft_path.exists():
        logger.error("Draft file not found: %s", draft_path)
        sys.exit(1)

    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found")
        sys.exit(1)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    newsletter_config = config.get("newsletter", {})
    email_config = config.get("email", {})

    # Read the approved draft
    newsletter_md = draft_path.read_text(encoding="utf-8")

    # Strip the auto-added title line if present (added by save_draft)
    newsletter_md = re.sub(r"^# .+\n\n", "", newsletter_md, count=1)

    # Infer date from filename (YYYY-MM-DD.md)
    try:
        date_str = date.fromisoformat(draft_path.stem).strftime("%B %d, %Y")
    except ValueError:
        date_str = date.today().strftime("%B %d, %Y")

    logger.info("Sending approved draft: %s (date: %s)", draft_path, date_str)

    # Format as HTML
    from format_email import format_newsletter_html
    html = format_newsletter_html(newsletter_md, newsletter_config, date_str)
    logger.info("HTML formatted: %d characters", len(html))

    if args.dry_run:
        logger.info("DRY RUN — printing HTML, not sending")
        print(html)
        return

    # Send
    from deliver import send_newsletter
    result = send_newsletter(html, newsletter_md, email_config)
    logger.info(
        "Delivery complete: %d/%d sent (%d failures)",
        result["success"], result["total"], result["failed"],
    )

    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
