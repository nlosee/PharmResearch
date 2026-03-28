"""
score.py — Pharma Regulatory Pulse
Article impact scoring layer, placed between deduplication and synthesis.

Two-path scoring depending on whether gpt-5.4-nano normalization ran:

  PATH A — Nano-enriched articles (preferred):
    Primary signal  : nano composite score from audience_fit (0–5),
                      trust_score (0–5), and novelty (0–5). These are
                      LLM-assessed against the specific regulatory audience.
    Secondary signal: recency bonus + content completeness.
    Weights         : audience_fit×2 + trust_score×1.5 + novelty×1 → (0–7)
                      + recency (0–2) + completeness (0–1) → raw max ≈ 10
    Normalised to 0–10 scale.

  PATH B — Non-normalized articles (fallback):
    Uses original keyword-based heuristic scoring:
    source_authority (0–5) + recency (0–2) + keywords (0–5) + completeness (0–2)
    → raw max ≈ 14, normalised to 0–10.

  GUIDELINE BOOST (+2.0):
    Articles identified as guideline/guidance updates receive a +2.0 bonus
    on their final score (both paths). This ensures guideline changes are
    prioritised above clinical trial and general news articles.

Output: articles sorted by score descending, with 'impact_score' field added.
        Articles below the configured minimum threshold are dropped.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source authority weights
# High-authority sources that indicate primary regulatory content
# ---------------------------------------------------------------------------

SOURCE_AUTHORITY: dict[str, float] = {
    "rss_fda":        5.0,   # FDA official RSS — highest authority
    "rss_ema":        5.0,   # EMA official RSS
    "openfda":        4.5,   # openFDA REST API — structured primary data
    "tavily":         4.0,   # Tavily structured search — high signal
    "rss_industry":   3.0,   # Drugs.com, industry news
    "pubmed":         3.0,   # Peer-reviewed literature
    "clinicaltrials": 3.0,   # ClinicalTrials.gov
    "google_news":    2.0,   # Broad news — lower authority
}

# ---------------------------------------------------------------------------
# High-value regulatory keywords (each match adds to score)
# ---------------------------------------------------------------------------

TIER_1_KEYWORDS = [
    # Approval actions
    r"\bPDUFA\b", r"\bNDA\b", r"\bBLA\b", r"\bANDA\b", r"\b505\(b\)\(2\)\b",
    r"\bCRL\b", r"complete response letter",
    # Guidance document signals
    r"\bguidance for industry\b", r"\bdraft guidance\b", r"\bfinal guidance\b",
    r"\bdocket no\b", r"\bFDA-\d{4}-[A-Z]-\d+\b",
    # Safety actions
    r"\bboxed warning\b", r"\bblack box\b", r"\bRISK evaluation\b", r"\bREMS\b",
    r"\bsafety communication\b", r"\bmarket withdrawal\b", r"\brecall\b",
    # International
    r"\bCHMP\b", r"\bEMA\b", r"\bmarketing authoriz", r"\bICH [A-Z]\d+\b",
    # Enforcement
    r"\bwarning letter\b", r"\b483\b", r"\bimport alert\b", r"\bcGMP\b",
]

TIER_2_KEYWORDS = [
    r"\bbiological\b", r"\bbiosimilar\b", r"\binterchangeabilit",
    r"\bbreakthrough therapy\b", r"\bfast track\b", r"\bpriority review\b",
    r"\bOrphan Drug\b", r"\bAccelerated Approval\b",
    r"\bpharmacovigilan", r"\badverse event\b",
    r"\binspection\b", r"\bclinical trial\b", r"\bphase [123] \b",
    r"\bregulatory submission\b", r"\bapproval\b",
]

_T1_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in TIER_1_KEYWORDS]
_T2_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in TIER_2_KEYWORDS]

# ---------------------------------------------------------------------------
# Guideline detection — articles matching these get a score boost
# ---------------------------------------------------------------------------

_GUIDELINE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bguideline\b", r"\bguidance\b", r"\bguidance for industry\b",
    r"\bdraft guidance\b", r"\bfinal guidance\b", r"\brecommendation update\b",
    r"\bpractice guideline\b", r"\bclinical practice\b",
    r"\bconsensus statement\b", r"\bposition statement\b",
    r"\bstandard of care\b", r"\btreatment protocol\b",
    r"\bICH [A-Z]\d+\b", r"\bdocket no\b",
]]

GUIDELINE_BOOST = 2.0   # added to final score for guideline articles


def _is_guideline_content(article: dict[str, Any]) -> bool:
    """Return True if the article appears to be a guideline or guidance update."""
    text = (
        f"{article.get('title', '')} {article.get('content', '')} "
        f"{article.get('source_name', '')}"
    )
    return any(pat.search(text) for pat in _GUIDELINE_PATTERNS)


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _recency_score(published_date: str) -> float:
    """
    Articles published in the last 48 h get +2.0.
    Articles published in the last 7 d get +1.0.
    Older or unparseable dates get 0.
    """
    if not published_date or published_date == "unknown":
        return 0.5   # slight neutral credit for undated articles

    # Try common date formats
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(published_date[:19], fmt)
            break
        except ValueError:
            continue
    else:
        return 0.5

    dt = dt.replace(tzinfo=timezone.utc)
    age = datetime.now(tz=timezone.utc) - dt
    if age <= timedelta(hours=48):
        return 2.0
    if age <= timedelta(days=7):
        return 1.0
    return 0.0


def _keyword_score(text: str) -> tuple[float, list[str]]:
    """
    Score the article text against regulatory keyword tiers.
    Returns (score, matched_keywords).
    """
    matched: list[str] = []
    score = 0.0

    for pat in _T1_PATTERNS:
        if pat.search(text):
            score += 1.5
            matched.append(pat.pattern)

    for pat in _T2_PATTERNS:
        if pat.search(text):
            score += 0.5
            matched.append(pat.pattern)

    return score, matched


def _content_completeness(content: str) -> float:
    """Score based on content richness — rewards substantial articles."""
    word_count = len(content.split())
    if word_count >= 300:
        return 2.0
    if word_count >= 100:
        return 1.0
    if word_count >= 30:
        return 0.5
    return 0.0


def _nano_composite(article: dict[str, Any]) -> float:
    """
    Build the primary signal from nano-assigned scores.

    Weights: audience_fit × 2.0 + trust_score × 1.5 + novelty × 1.0
    Max raw = 5×2 + 5×1.5 + 5×1 = 22.5 → mapped to 0–7 range.
    """
    af = float(article.get("audience_fit", 0))
    ts = float(article.get("trust_score",  0))
    nv = float(article.get("novelty",      0))
    raw = af * 2.0 + ts * 1.5 + nv * 1.0      # 0–22.5
    return min(raw / 22.5 * 7.0, 7.0)          # normalise to 0–7


def score_article(article: dict[str, Any]) -> dict[str, Any]:
    """
    Compute an impact score (0–10) for a single article.

    Routing:
      - If the article was enriched by gpt-5.4-nano (_normalized=True):
          PATH A: nano composite (0–7) + recency (0–2) + completeness (0–1)
      - Otherwise:
          PATH B: source_authority (0–5) + recency (0–2) + keywords (0–5)
                  + completeness (0–2), normalised to 0–10.

    Returns the article dict with 'impact_score' and 'matched_keywords' added.
    """
    article = dict(article)   # don't mutate the original
    combined_text = f"{article.get('title', '')} {article.get('content', '')}"

    recency     = _recency_score(article.get("published_date", ""))
    completeness = _content_completeness(article.get("content", ""))

    if article.get("_normalized"):
        # ── PATH A: nano-primary scoring ──────────────────────────────────
        nano   = _nano_composite(article)              # 0–7
        raw    = nano + recency + min(completeness, 1.0)   # max ≈ 10
        normalised = round(min(raw, 10.0), 2)
        matched_kws: list[str] = []   # keyword scan not needed for nano path
    else:
        # ── PATH B: heuristic fallback scoring ────────────────────────────
        source_type = article.get("source_type", "")
        authority   = SOURCE_AUTHORITY.get(source_type, 2.0)
        kw_score, matched_kws = _keyword_score(combined_text)
        kw_score   = min(kw_score, 5.0)
        raw        = authority + recency + kw_score + completeness  # max ≈ 14
        normalised = round(min(raw / 14.0 * 10.0, 10.0), 2)

    # ── Guideline boost ──────────────────────────────────────────────────
    is_guideline = _is_guideline_content(article)
    if is_guideline:
        normalised = round(min(normalised + GUIDELINE_BOOST, 10.0), 2)

    article["impact_score"]    = normalised
    article["matched_keywords"] = matched_kws[:5]
    article["_score_path"]     = "nano" if article.get("_normalized") else "heuristic"
    article["_guideline_boost"] = is_guideline
    return article


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_and_rank(
    articles: list[dict[str, Any]],
    min_score: float = 2.0,
    max_articles: int = 40,
) -> list[dict[str, Any]]:
    """
    Score all articles, drop low-signal content, and return the top N
    sorted by impact_score descending.

    Parameters
    ----------
    articles : list[dict]
        Deduplicated articles from dedup.py.
    min_score : float
        Articles scoring below this threshold are dropped (default 2.0).
    max_articles : int
        Maximum number of articles to pass to synthesis (default 40).
        Limits token budget and keeps the LLM focused.

    Returns
    -------
    list[dict]
        Scored, filtered, sorted articles with 'impact_score' field.
    """
    scored = [score_article(a) for a in articles]

    # Log score distribution
    if scored:
        avg = sum(a["impact_score"] for a in scored) / len(scored)
        logger.info(
            "Scoring complete — %d articles, avg score=%.2f, min_threshold=%.1f",
            len(scored), avg, min_score,
        )

    # Filter below threshold
    filtered = [a for a in scored if a["impact_score"] >= min_score]
    dropped = len(scored) - len(filtered)
    if dropped:
        logger.info(
            "Impact filter: dropped %d low-signal articles (score < %.1f)",
            dropped, min_score,
        )

    # Sort and cap
    ranked = sorted(filtered, key=lambda a: a["impact_score"], reverse=True)
    if len(ranked) > max_articles:
        logger.info(
            "Capping at %d articles (had %d after filtering)", max_articles, len(ranked)
        )
        ranked = ranked[:max_articles]

    # Log top-5 for transparency
    for i, a in enumerate(ranked[:5], start=1):
        logger.info(
            "  #%d [%.1f] %s (%s)",
            i, a["impact_score"], a.get("title", "")[:70], a.get("source_type", ""),
        )

    return ranked
