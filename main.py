"""
main.py — Pharma Regulatory Pulse
Full pipeline orchestrator: research → dedup → synthesize → format → deliver → archive.

Usage:
  python main.py                  # Full run: generate and send
  python main.py --dry-run        # Generate and print HTML to stdout, no email sent
  python main.py --draft-only     # Generate and save markdown draft, no format/send
  python main.py --no-pubmed      # Skip PubMed (useful if NCBI quota exceeded)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging setup — structured timestamps on all stages
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        logger.error("config.yaml not found at: %s", config_path.absolute())
        sys.exit(1)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Draft archival
# ---------------------------------------------------------------------------

def save_draft(newsletter_md: str, title: str) -> Path:
    """
    Save the newsletter markdown to drafts/YYYY-MM-DD.md for audit trail.

    Returns
    -------
    Path
        Path to the saved draft file.
    """
    drafts_dir = Path("drafts")
    drafts_dir.mkdir(exist_ok=True)
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    draft_path = drafts_dir / f"{date_str}.md"

    with draft_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title} — {date_str}\n\n")
        f.write(newsletter_md)

    logger.info("Draft saved: %s", draft_path)
    return draft_path


def save_archive(articles: list[dict], date_str: str | None = None) -> Path:
    """
    Save the raw enriched articles to archive/YYYY-MM-DD.json.
    This builds the long-term Knowledge Graph corpus before any articles are dropped.
    """
    archive_dir = Path("archive")
    archive_dir.mkdir(exist_ok=True)
    if not date_str:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    
    archive_path = archive_dir / f"{date_str}.json"
    
    with archive_path.open("w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
        
    return archive_path


def save_ingest_bundle(articles: list[dict], archive_path: Path, date_str: str) -> Path | None:
    """Persist a KB ingest bundle alongside the newsletter archive artifacts.
    Skipped silently when the knowledge_base module is not installed (e.g. on GitHub Actions)."""
    try:
        from knowledge_base.ingest.bundle import build_ingest_bundle_items, emit_ingest_bundle
    except ImportError:
        return None

    bundle_dir = Path("artifacts") / "knowledge_base"
    bundle_path = bundle_dir / f"{date_str}.ingest_bundle.jsonl"
    items = build_ingest_bundle_items(
        articles,
        archive_path=str(archive_path),
        run_date=date_str,
    )
    emit_ingest_bundle(bundle_path, items)
    return bundle_path


def update_history(articles: list[dict]) -> None:
    """
    Save the URLs of the final published articles to published_history.json.
    This allows dedup.py to prevent these articles from being included again next week.
    """
    history_path = Path("published_history.json")
    history: list[str] = []
    
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            pass

    # Normalize URLs before saving to ensure exact matches
    import urllib.parse
    import re
    
    new_urls = []
    for art in articles:
        url = art.get("url", "").strip().lower()
        if not url:
            continue
        try:
            parsed = urllib.parse.urlparse(url)
            netloc = re.sub(r"^www\.", "", parsed.netloc)
            path = parsed.path.rstrip("/")
            clean_url = f"{parsed.scheme}://{netloc}{path}"
            new_urls.append(clean_url)
        except Exception:
            new_urls.append(url)
            
    # Append unique new URLs
    for url in new_urls:
        if url not in history:
            history.append(url)
            
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    
    logger.info("Updated published_history.json with %d total URLs", len(history))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full Pharma Regulatory Pulse pipeline."""
    # Parse CLI flags
    parser = argparse.ArgumentParser(description="Pharma Regulatory Pulse Newsletter Agent")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate newsletter and print HTML to stdout without sending emails",
    )
    parser.add_argument(
        "--draft-only",
        action="store_true",
        help="Generate and save markdown draft only (no HTML formatting or sending)",
    )
    parser.add_argument(
        "--no-pubmed",
        action="store_true",
        help="Skip PubMed source (useful if NCBI quota is exceeded)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Number of Monte Carlo synthesis passes (default: 2). "
            "More passes improve coverage at the cost of additional API calls. "
            "Use --passes 1 for a quick single-pass run during testing, "
            "--passes 3 for maximum coverage on high-stakes editions."
        ),
    )
    args = parser.parse_args()

    # Load .env for local development
    load_dotenv()

    logger.info("=" * 60)
    logger.info("Pharma Regulatory Pulse — Pipeline Starting")
    if args.dry_run:
        logger.info("MODE: DRY RUN (no emails will be sent)")
    elif args.draft_only:
        logger.info("MODE: DRAFT ONLY (no formatting or sending)")
    logger.info("=" * 60)

    try:
        # ── Stage 1: Load config ─────────────────────────────────────────
        config = load_config("config.yaml")
        newsletter_config = config.get("newsletter", {})
        email_config = config.get("email", {})

        if args.no_pubmed:
            config.setdefault("sources", {})["pubmed"] = False
            logger.info("PubMed source disabled via --no-pubmed flag")

        # ── Stage 2: Research ────────────────────────────────────────────
        logger.info("STAGE: Research — gathering articles from all sources")
        from research import research_topics
        articles = research_topics(config)
        logger.info("Research complete: %d articles found", len(articles))

        if not articles:
            logger.warning("No articles retrieved from any source — aborting")
            sys.exit(0)

        # ── Stage 3: Normalize (gpt-5.4-nano) ───────────────────────────
        # Per-article enrichment: structured scoring fields + cluster key.
        # Uses the worker model (nano) via the Responses API.
        # Gracefully skipped if OPENAI_API_KEY is absent or SDK is too old.
        logger.info(
            "STAGE: Normalization — enriching %d articles with gpt-5.4-nano",
            len(articles),
        )
        from normalize import normalize_articles
        articles = normalize_articles(articles, config)

        # ── Stage 3.5: Archive raw enriched data ────────────────────────
        logger.info("STAGE: Archival — saving full enriched corpus")
        run_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        archive_path = save_archive(articles, date_str=run_date)
        logger.info("Archived %d articles to %s", len(articles), archive_path)
        bundle_path = save_ingest_bundle(articles, archive_path, run_date)
        if bundle_path:
            logger.info("Saved KB ingest bundle to %s", bundle_path)

        # ── Stage 4: Deduplicate ─────────────────────────────────────────
        # Layer 0 uses nano's duplicate_cluster_key; Layers 1–2 are URL + TF-IDF.
        logger.info("STAGE: Deduplication")
        from dedup import deduplicate
        articles = deduplicate(articles)
        logger.info("After dedup: %d articles", len(articles))

        # ── Stage 5: Impact scoring + ranking ───────────────────────────
        scoring_config = config.get("scoring", {})
        if scoring_config.get("enabled", True):
            logger.info("STAGE: Article impact scoring")
            from score import score_and_rank
            articles = score_and_rank(
                articles,
                min_score=float(scoring_config.get("min_score", 2.0)),
                max_articles=int(scoring_config.get("max_articles", 40)),
            )
            logger.info("After scoring: %d articles", len(articles))

        # ── Stage 6: Minimum threshold check ────────────────────────────
        min_articles = int(newsletter_config.get("min_articles_to_publish", 3))
        if len(articles) < min_articles:
            logger.warning(
                "ABORT: Only %d articles (minimum: %d). Exiting cleanly.",
                len(articles), min_articles,
            )
            sys.exit(0)

        # ── Stage 7: Synthesize (gpt-5.4-mini) ──────────────────────────
        logger.info(
            "STAGE: Synthesis — Monte Carlo generation (%d passes)", args.passes
        )
        from synthesize import generate_newsletter
        date_str = datetime.now(tz=timezone.utc).strftime("%B %d, %Y")
        newsletter_md = generate_newsletter(
            articles, newsletter_config, date_str, num_passes=args.passes
        )
        logger.info("Newsletter generated: %d characters", len(newsletter_md))

        # ── Stage 8: Archive draft ───────────────────────────────────────
        draft_path = save_draft(newsletter_md, newsletter_config.get("title", "Newsletter"))

        if args.draft_only:
            logger.info("Draft-only mode — pipeline complete. Draft at: %s", draft_path)
            print(newsletter_md)
            return

        # ── Stage 9: Format as HTML email ───────────────────────────────
        logger.info("STAGE: Email formatting (MJML)")
        from format_email import format_newsletter_html
        html = format_newsletter_html(
            newsletter_md, newsletter_config, date_str,
            article_count=len(articles),
            articles=articles,
        )
        logger.info("HTML rendered: %d characters", len(html))

        if args.dry_run:
            logger.info("Dry-run mode — printing HTML to stdout (no emails sent)")
            print(html)
            logger.info("Dry run complete.")
            return

        # ── Stage 10: Deliver ────────────────────────────────────────────
        logger.info("STAGE: Delivery — sending to subscribers")
        from deliver import send_newsletter
        result = send_newsletter(html, newsletter_md, email_config)
        logger.info(
            "Delivery complete: %d/%d sent (%d failures)",
            result["success"], result["total"], result["failed"],
        )
        
        # ── Stage 11: Update History ─────────────────────────────────────
        # Only add to history if we're not in dry-run mode
        if not args.dry_run:
            logger.info("STAGE: Updating history for cross-week deduplication")
            update_history(articles)

        logger.info("=" * 60)
        logger.info("Pipeline finished successfully.")
        logger.info("=" * 60)

    except SystemExit:
        raise   # Allow clean exits to propagate
    except Exception:
        logger.error("PIPELINE FAILED — full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
