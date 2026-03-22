"""
normalize.py — Pharma Regulatory Pulse
Per-article normalization using gpt-5.4-nano via the OpenAI Responses API.

Stage 2.5 in the pipeline — runs after research, before deduplication.

Each raw article is enriched with structured fields that downstream stages
use for smarter deduplication (duplicate_cluster_key) and scoring
(audience_fit, novelty, trust_score):

    story_id              — stable URL-hash identifier, consistent across stages
    topic                 — regulatory subject category (e.g. "FDA Guidance")
    audience_fit          — 0–5: relevance to pharma regulatory professionals
    novelty               — 0–5: how new/non-obvious/action-relevant this week
    trust_score           — 0–5: source quality and evidentiary strength
    summary_2_sentences   — concise factual summary for display in synthesis
    why_it_matters        — one sentence on compliance/submission relevance
    duplicate_cluster_key — semantic event key for cluster-based dedup
                            (e.g. "FDA-NDA-lecanemab-2026-approval")

Model routing:
    worker  →  gpt-5.4-nano   (OPENAI_MODEL_WORKER env var)
    API     →  Responses API  (client.responses.create)
    Output  →  strict JSON schema, additionalProperties: false

Design rules (from handoff doc):
    - Never use the writer model for bulk per-article normalization
    - Use strict JSON schema outputs for all machine-to-machine stages
    - Keep story IDs stable across all stages
    - Graceful fallback: if normalization fails, article continues unfilled
    - Log model, prompt version, and timestamp for each run
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

logger = logging.getLogger(__name__)

# ── Prompt versioning ─────────────────────────────────────────────────────────
PROMPT_VERSION = "normalize-v1"

# ── Strict JSON schema for Responses API structured output ────────────────────
NORMALIZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "story_id": {
            "type": "string",
            "description": "Stable identifier — use the URL hash if unsure.",
        },
        "title": {"type": "string"},
        "source": {"type": "string"},
        "url": {"type": "string"},
        "published_at": {"type": ["string", "null"]},
        "topic": {
            "type": "string",
            "description": (
                "Regulatory subject category. Examples: "
                "'FDA Drug Approval', 'FDA Draft Guidance', 'ICH Harmonization', "
                "'EMA Marketing Authorization', 'Clinical Trial Designation', "
                "'GMP Enforcement', 'Drug Safety Communication', 'REMS Update'."
            ),
        },
        "audience_fit": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
            "description": "Relevance to pharma regulatory affairs professionals (0=irrelevant, 5=highly relevant).",
        },
        "novelty": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
            "description": "How new, non-obvious, or action-relevant this item is this week (0=stale/generic, 5=highly novel).",
        },
        "trust_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
            "description": (
                "Source quality: 5=official FDA/EMA/ICH primary source; "
                "4=established industry publisher; 3=reputable secondary; "
                "2=blog/commentary; 1=weak or unclear provenance."
            ),
        },
        "summary_2_sentences": {
            "type": "string",
            "description": "Factual, information-dense 2-sentence summary. No speculation.",
        },
        "why_it_matters": {
            "type": "string",
            "description": (
                "One sentence on compliance, submission, or pharmacovigilance relevance "
                "for the target audience. No marketing language."
            ),
        },
        "duplicate_cluster_key": {
            "type": "string",
            "description": (
                "Short semantic identifier grouping near-duplicate stories about the same "
                "regulatory event. Use kebab-case with agency, drug/topic, and event type. "
                "Example: 'fda-nda-lecanemab-approval', 'ich-q13-guideline-update', "
                "'ema-chmp-biosimilar-opinion-2026-03'. Stories on the same event MUST share "
                "the same key. Unrelated stories MUST have different keys."
            ),
        },
    },
    "required": [
        "story_id",
        "title",
        "source",
        "url",
        "published_at",
        "topic",
        "audience_fit",
        "novelty",
        "trust_score",
        "summary_2_sentences",
        "why_it_matters",
        "duplicate_cluster_key",
    ],
}

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a pharmaceutical regulatory newsletter research worker.

Task:
Read one source item and normalize it into a structured record for newsletter selection.

Critical rules (in priority order):
1. Be factual and concise. Do not invent facts not present in the source.
2. If the source is weak, unclear, or low-authority, lower trust_score accordingly.
3. If the item is old, repetitive, or derivative, lower novelty accordingly.
4. Use short, information-dense writing. No marketing language.
5. Output ONLY valid JSON matching the required schema — no preamble, no explanation.

Audience context:
Pharmaceutical regulatory affairs professionals, QA/QC managers, medical affairs teams,
pharmacovigilance specialists, and compliance officers at pharma/biotech companies.
They have deep domain knowledge — do not explain basic regulatory concepts.

Scoring guidance (0–5 integer scale):
- audience_fit : relevance to the target audience described above
- novelty      : how new, non-obvious, or immediately actionable this item is this week
- trust_score  : 5 = official FDA/EMA/ICH primary source
                 4 = established industry publisher (FDAnews, RAPS, Pharma Times)
                 3 = reputable secondary source
                 2 = blog, commentary, or analysis without primary sourcing
                 1 = weak provenance, speculative, or unclear origin

duplicate_cluster_key:
Short kebab-case semantic event key. Stories covering the EXACT SAME regulatory event
MUST share the same key. Unrelated stories MUST have different keys.
Examples: "fda-nda-lecanemab-approval", "ich-q13-guideline-update",
"fda-warning-letter-pfizer-mcpherson-2026-03"
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stable_story_id(url: str) -> str:
    """Deterministic story_id from URL — stable across pipeline stages."""
    return "story_" + hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def _build_user_content(article: dict[str, Any]) -> str:
    """Format a single article as the user message for nano."""
    content_preview = (article.get("content") or "")[:2000]
    return (
        f"Title: {article.get('title', '(no title)')}\n"
        f"Source name: {article.get('source_name', 'unknown')}\n"
        f"Source type: {article.get('source_type', 'unknown')}\n"
        f"URL: {article.get('url', '')}\n"
        f"Published: {article.get('published_date', 'unknown')}\n\n"
        f"Content:\n{content_preview}"
    )


def _fallback_article(article: dict[str, Any]) -> dict[str, Any]:
    """Return article with empty normalization fields when nano fails."""
    enriched = dict(article)
    enriched.update({
        "story_id":              _stable_story_id(article.get("url", "")),
        "topic":                 "",
        "audience_fit":          0,
        "novelty":               0,
        "trust_score":           0,
        "summary_2_sentences":   "",
        "why_it_matters":        "",
        "duplicate_cluster_key": "",
        "_normalized":           False,
    })
    return enriched


# ── Core normalization call ───────────────────────────────────────────────────

def _normalize_one(
    client: OpenAI,
    model: str,
    article: dict[str, Any],
) -> dict[str, Any]:
    """
    Call gpt-5.4-nano via the Responses API to normalize a single article.

    Uses strict JSON schema output (additionalProperties: false).
    Retries up to 3 times with exponential backoff (2s → 4s → 8s).
    Falls back gracefully on persistent failure.
    """
    user_content = _build_user_content(article)
    delays = [2, 4, 8]

    for attempt, delay in enumerate(delays, start=1):
        try:
            # Responses API — primary interface per handoff spec
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                text={
                    "format": {
                        "type":   "json_schema",
                        "name":   "normalized_article",
                        "schema": NORMALIZE_SCHEMA,
                        "strict": True,
                    }
                },
            )
            normalized: dict[str, Any] = json.loads(response.output_text)

            # Merge normalized fields back into the original article dict
            enriched = dict(article)
            enriched.update({
                "story_id":              normalized.get("story_id") or _stable_story_id(article.get("url", "")),
                "topic":                 normalized.get("topic", ""),
                "audience_fit":          int(normalized.get("audience_fit", 0)),
                "novelty":               int(normalized.get("novelty", 0)),
                "trust_score":           int(normalized.get("trust_score", 0)),
                "summary_2_sentences":   normalized.get("summary_2_sentences", ""),
                "why_it_matters":        normalized.get("why_it_matters", ""),
                "duplicate_cluster_key": normalized.get("duplicate_cluster_key", ""),
                "_normalized":           True,
            })
            return enriched

        except (RateLimitError, APIStatusError, APIConnectionError) as exc:
            logger.warning(
                "Normalize attempt %d/%d for '%s': %s",
                attempt, len(delays), article.get("title", "?")[:60], exc,
            )
            if attempt < len(delays):
                time.sleep(delay)

        except AttributeError:
            # Responses API not available in this SDK version — fall back
            logger.warning(
                "Responses API unavailable (openai SDK too old). "
                "Upgrade: pip install 'openai>=1.66.0'. Skipping normalization."
            )
            return _fallback_article(article)

        except Exception as exc:
            logger.warning(
                "Normalize failed for '%s': %s",
                article.get("title", "?")[:60], exc,
            )
            break

    return _fallback_article(article)


# ── Public entry point ────────────────────────────────────────────────────────

def normalize_articles(
    articles: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Normalize all articles using gpt-5.4-nano (OPENAI_MODEL_WORKER).

    Adds structured scoring fields and duplicate_cluster_key to each article
    dict. Articles that fail normalization are returned with empty enrichment
    fields so downstream stages continue uninterrupted.

    Rate-limited at ~0.15 s between calls (~6–7 req/s) — nano handles much
    higher throughput but this keeps costs predictable and avoids burst limits.

    Parameters
    ----------
    articles : list[dict]
        Raw articles from research.py.
    config : dict
        Loaded config.yaml content.

    Returns
    -------
    list[dict]
        Articles enriched with normalization fields.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping normalization stage.")
        return articles

    model = (
        os.getenv("OPENAI_MODEL_WORKER")
        or config.get("newsletter", {}).get("model_worker", "gpt-5.4-nano")
    )
    client = OpenAI(api_key=api_key)

    logger.info(
        "Normalization: %d articles → %s  [prompt_version=%s]",
        len(articles), model, PROMPT_VERSION,
    )

    enriched: list[dict[str, Any]] = []
    success_count = 0

    for i, article in enumerate(articles):
        result = _normalize_one(client, model, article)
        enriched.append(result)
        if result.get("_normalized"):
            success_count += 1

        # Brief pause between calls — adjust if hitting rate limits
        if i < len(articles) - 1:
            time.sleep(0.15)

    logger.info(
        "Normalization complete: %d/%d articles enriched by nano",
        success_count, len(articles),
    )
    return enriched
