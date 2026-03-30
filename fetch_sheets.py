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

import gspread
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_config() -> dict:
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        sys.exit(1)

def main():
    config = load_config()
    sheet_id = config.get("social_listening", {}).get("google_sheet_id")
    
    if not sheet_id or sheet_id == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms":
        logger.error("No valid google_sheet_id found in config.yaml. Please set it to your real Spreadsheet ID.")
        sys.exit(1)

    credentials_json = os.getenv("GOOGLE_CREDENTIALS")
    if not credentials_json:
        logger.error("GOOGLE_CREDENTIALS environment variable is missing. "
                     "Create a Google Cloud Service Account, export the JSON key, "
                     "and store it in GitHub Secrets.")
        sys.exit(1)

    try:
        creds_dict = json.loads(credentials_json)
        gc = gspread.service_account_from_dict(creds_dict)
        logger.info("Successfully authenticated with Google Service Account.")
    except Exception as e:
        logger.error(f"Failed to authenticate with GOOGLE_CREDENTIALS: {e}")
        sys.exit(1)

    try:
        workbook = gc.open_by_key(sheet_id)
        logger.info(f"Successfully opened workbook: {workbook.title}")
    except Exception as e:
        logger.error(f"Failed to open Google Sheet by ID '{sheet_id}'. "
                     f"Did you share the sheet with the Service Account email address? Error: {e}")
        sys.exit(1)

    # 1. Fetch Subscribers
    try:
        subscribers_sheet = workbook.worksheet("Subscribers")
        records = subscribers_sheet.get_all_records()
        # Filter out empty rows (where email is blank)
        records = [r for r in records if r.get("email", "").strip()]
        
        if records:
            with open("subscribers.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            logger.info(f"Downloaded {len(records)} subscribers to subscribers.csv")
        else:
            logger.warning("Subscribers sheet is empty or only contains blank rows.")
    except gspread.exceptions.WorksheetNotFound:
        logger.warning(f"Worksheet named 'Subscribers' not found in workbook '{workbook.title}'. Skipping.")

    # 2. Fetch Social Roster
    try:
        social_sheet = workbook.worksheet("Social_Roster")
        records = social_sheet.get_all_records()
        # Filter out empty rows (where handle is blank)
        records = [r for r in records if r.get("handle", r.get("Handle/Hashtag", "")).strip()]
        
        if records:
            with open("social_roster.json", "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            logger.info(f"Downloaded {len(records)} social roster accounts to social_roster.json")
        else:
            logger.warning("Social_Roster sheet is empty or only contains blank rows.")
    except gspread.exceptions.WorksheetNotFound:
        logger.warning(f"Worksheet named 'Social_Roster' not found in workbook '{workbook.title}'. Skipping.")

if __name__ == "__main__":
    main()
