"""
fetch_sheets.py — Pharma Regulatory Pulse
Authenticates with a Google Service Account to securely download
the "Subscribers" and "Social_Roster" worksheets from a private Google Sheet.

Reads the worksheet ID from config.yaml and the Service Account
JSON from the GOOGLE_CREDENTIALS environment variable.
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUBSCRIBERS_CSV_PATH = Path("subscribers.csv")
SOCIAL_ROSTER_PATH = Path("social_roster.json")

def load_config() -> dict:
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        sys.exit(1)


def _write_legacy_subscribers_csv(raw_csv: str) -> bool:
    raw_csv = (raw_csv or "").strip()
    if not raw_csv:
        return False

    SUBSCRIBERS_CSV_PATH.write_text(raw_csv + "\n", encoding="utf-8")
    logger.warning(
        "Falling back to SUBSCRIBERS_CSV secret and writing %s",
        SUBSCRIBERS_CSV_PATH,
    )
    return True


def _ensure_placeholder_outputs() -> None:
    if not SOCIAL_ROSTER_PATH.exists():
        SOCIAL_ROSTER_PATH.write_text("[]\n", encoding="utf-8")

    if not SUBSCRIBERS_CSV_PATH.exists():
        SUBSCRIBERS_CSV_PATH.write_text("email,name\n", encoding="utf-8")


def _filter_non_empty(records: list[dict], preferred_keys: tuple[str, ...]) -> list[dict]:
    filtered: list[dict] = []
    for row in records:
        for key in preferred_keys:
            if str(row.get(key, "")).strip():
                filtered.append(row)
                break
    return filtered

def main():
    config = load_config()
    sheet_id = config.get("social_listening", {}).get("google_sheet_id")

    credentials_json = os.getenv("GOOGLE_CREDENTIALS")
    legacy_subscribers_csv = os.getenv("SUBSCRIBERS_CSV", "")

    if not sheet_id:
        logger.warning("No google_sheet_id found in config.yaml.")
        if _write_legacy_subscribers_csv(legacy_subscribers_csv):
            _ensure_placeholder_outputs()
            return
        logger.error("No Google Sheet configured and no SUBSCRIBERS_CSV fallback available.")
        sys.exit(1)

    if not credentials_json:
        logger.warning(
            "GOOGLE_CREDENTIALS environment variable is missing. "
            "Trying legacy SUBSCRIBERS_CSV fallback."
        )
        if _write_legacy_subscribers_csv(legacy_subscribers_csv):
            _ensure_placeholder_outputs()
            return
        logger.error("GOOGLE_CREDENTIALS is missing and no SUBSCRIBERS_CSV fallback is available.")
        sys.exit(1)

    try:
        import gspread
    except ImportError as e:
        logger.error(f"gspread is not installed: {e}")
        if _write_legacy_subscribers_csv(legacy_subscribers_csv):
            _ensure_placeholder_outputs()
            return
        sys.exit(1)

    try:
        creds_dict = json.loads(credentials_json)
        gc = gspread.service_account_from_dict(creds_dict)
        logger.info("Successfully authenticated with Google Service Account.")
    except Exception as e:
        logger.error(f"Failed to authenticate with GOOGLE_CREDENTIALS: {e}")
        if _write_legacy_subscribers_csv(legacy_subscribers_csv):
            _ensure_placeholder_outputs()
            return
        sys.exit(1)

    try:
        workbook = gc.open_by_key(sheet_id)
        logger.info(f"Successfully opened workbook: {workbook.title}")
    except Exception as e:
        logger.error(f"Failed to open Google Sheet by ID '{sheet_id}'. "
                     f"Exception type: {type(e).__name__}. Detail: {repr(e)}")
        logger.error("Common causes: (1) Google Sheets API not enabled in Cloud Console, "
                     "(2) Google Drive API not enabled, (3) sheet not shared with service account email.")
        if _write_legacy_subscribers_csv(legacy_subscribers_csv):
            _ensure_placeholder_outputs()
            return
        sys.exit(1)

    wrote_subscribers = False

    # 1. Fetch Subscribers
    try:
        subscribers_sheet = workbook.worksheet("Subscribers")
        records = subscribers_sheet.get_all_records()
        records = _filter_non_empty(records, ("email", "Email", "EMAIL"))

        if records:
            normalized_records = [
                {
                    "email": str(r.get("email") or r.get("Email") or r.get("EMAIL") or "").strip(),
                    "name": str(r.get("name") or r.get("Name") or r.get("NAME") or "").strip(),
                }
                for r in records
            ]
            normalized_records = [r for r in normalized_records if r["email"]]

            with SUBSCRIBERS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["email", "name"])
                writer.writeheader()
                writer.writerows(normalized_records)
            wrote_subscribers = True
            logger.info("Downloaded %d subscribers to %s", len(normalized_records), SUBSCRIBERS_CSV_PATH)
        else:
            logger.warning("Subscribers sheet is empty or only contains blank rows.")
    except gspread.exceptions.WorksheetNotFound:
        logger.warning(f"Worksheet named 'Subscribers' not found in workbook '{workbook.title}'. Skipping.")

    if not wrote_subscribers:
        if not _write_legacy_subscribers_csv(legacy_subscribers_csv):
            logger.warning("No subscribers were fetched and no SUBSCRIBERS_CSV fallback was available.")

    # 2. Fetch Social Roster
    try:
        social_sheet = workbook.worksheet("Social_Roster")
        records = social_sheet.get_all_records()
        records = _filter_non_empty(records, ("handle", "Handle/Hashtag", "Handle", "hashtag"))

        if records:
            with SOCIAL_ROSTER_PATH.open("w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            logger.info("Downloaded %d social roster accounts to %s", len(records), SOCIAL_ROSTER_PATH)
        else:
            logger.warning("Social_Roster sheet is empty or only contains blank rows.")
    except gspread.exceptions.WorksheetNotFound:
        logger.warning(f"Worksheet named 'Social_Roster' not found in workbook '{workbook.title}'. Skipping.")

    _ensure_placeholder_outputs()

if __name__ == "__main__":
    main()
