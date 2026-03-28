"""
dedup.py — Pharma Regulatory Pulse
Three-layer deduplication pipeline:

  Layer 0 — Cluster key dedup: groups near-duplicate stories by the
             duplicate_cluster_key assigned by gpt-5.4-nano in normalize.py.
             Within each cluster, only the highest-trust article is kept.
             No-op for articles without a cluster key (normalize skipped).

  Layer 1 — URL normalization + exact-match deduplication.

  Layer 2 — TF-IDF cosine similarity for semantic deduplication (threshold 0.70).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 0 — Cluster key dedup (nano-derived duplicate_cluster_key)
# ---------------------------------------------------------------------------

def _dedup_by_cluster_key(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group articles that share the same duplicate_cluster_key and keep only
    the representative for each cluster — the article with the highest
    trust_score (nano-assigned), breaking ties by source authority order.

    Articles with an empty or missing cluster key are passed through
    unchanged — this makes the layer a no-op when normalize.py was skipped.
    """
    # Separate articles that have a meaningful cluster key from those that don't
    clustered:   dict[str, list[dict[str, Any]]] = {}
    unclustered: list[dict[str, Any]] = []

    for article in articles:
        key = (article.get("duplicate_cluster_key") or "").strip()
        if key:
            clustered.setdefault(key, []).append(article)
        else:
            unclustered.append(article)

    survivors: list[dict[str, Any]] = []
    removed = 0

    for key, cluster in clustered.items():
        if len(cluster) == 1:
            survivors.append(cluster[0])
            continue

        # Keep the article with the highest trust_score; tie-break on
        # source_type authority (rss_fda/rss_ema > tavily > others)
        _authority_rank = {"rss_fda": 5, "rss_ema": 5, "openfda": 4,
                           "tavily": 3, "rss_industry": 2, "pubmed": 2,
                           "clinicaltrials": 2, "google_news": 1}
        best = max(
            cluster,
            key=lambda a: (
                int(a.get("trust_score", 0)),
                _authority_rank.get(a.get("source_type", ""), 0),
            ),
        )
        survivors.append(best)
        dropped = len(cluster) - 1
        removed += dropped
        logger.debug(
            "Cluster '%s': kept '%s', dropped %d duplicate(s)",
            key, best.get("title", "?")[:60], dropped,
        )

    if removed:
        logger.info(
            "Cluster key dedup: %d duplicate(s) removed across %d cluster(s)",
            removed, len(clustered),
        )

    # Preserve original ordering as much as possible: survivors first,
    # then unclustered, both in their original relative order.
    survivor_urls = {a.get("url", "") for a in survivors}
    result = []
    for article in articles:
        url = article.get("url", "")
        if url in survivor_urls or article in unclustered:
            result.append(article)
            survivor_urls.discard(url)   # avoid double-adding

    return result


# ---------------------------------------------------------------------------
# Layer 0.5 — Cross-week history dedup
# ---------------------------------------------------------------------------

def _dedup_by_history(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove articles whose normalised URLs are already in published_history.json.
    This prevents articles that were published in previous weeks from reappearing.
    """
    history_path = Path("published_history.json")
    if not history_path.exists():
        return articles

    try:
        with history_path.open("r", encoding="utf-8") as f:
            history = set(json.load(f))
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse published_history.json, skipping history dedup.")
        return articles

    if not history:
        return articles

    survivors: list[dict[str, Any]] = []
    
    # We use the exact same normalization as Layer 1 (URL dedup)
    for article in articles:
        url = article.get("url", "")
        # The history json contains normalized URLs directly
        if _normalize_url(url) not in history:
            survivors.append(article)
            
    return survivors


# ---------------------------------------------------------------------------
# Layer 1 — URL normalisation + exact dedup
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """
    Strip query params, fragments, trailing slashes, and www. prefix
    to produce a canonical URL key for deduplication.
    """
    try:
        parsed = urllib.parse.urlparse(url.strip().lower())
        # Remove www.
        netloc = re.sub(r"^www\.", "", parsed.netloc)
        # Strip query and fragment
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{netloc}{path}"
    except Exception:
        return url.strip().lower()


def _dedup_by_url(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first occurrence of each normalised URL."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for article in articles:
        key = _normalize_url(article.get("url", ""))
        if key not in seen:
            seen.add(key)
            unique.append(article)
    return unique


# ---------------------------------------------------------------------------
# Layer 2 — TF-IDF cosine similarity dedup
# ---------------------------------------------------------------------------

def _dedup_by_similarity(
    articles: list[dict[str, Any]],
    threshold: float = 0.70,
) -> list[dict[str, Any]]:
    """
    Vectorise title + content with TF-IDF and remove near-duplicate articles
    whose pairwise cosine similarity exceeds `threshold`.

    For any pair (i, j) where sim >= threshold, the later article (j) is dropped.
    """
    if len(articles) < 2:
        return articles

    # Build corpus: title + content for each article
    corpus = [
        f"{a.get('title', '')} {a.get('content', '')}" for a in articles
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError as exc:
        logger.warning("TF-IDF vectorization failed: %s — skipping semantic dedup", exc)
        return articles

    # Compute pairwise similarities (upper triangle)
    sim_matrix: np.ndarray = cosine_similarity(tfidf_matrix)

    # Mark articles to drop
    n = len(articles)
    drop: set[int] = set()
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            if sim_matrix[i, j] >= threshold:
                drop.add(j)
                logger.debug(
                    "Semantic dup removed [%d] ← [%d] (sim=%.2f): %s",
                    i, j, sim_matrix[i, j],
                    articles[j].get("title", "")[:60],
                )

    return [a for idx, a in enumerate(articles) if idx not in drop]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deduplicate(
    articles: list[dict[str, Any]],
    similarity_threshold: float = 0.70,
) -> list[dict[str, Any]]:
    """
    Run both deduplication layers and log a summary.

    Parameters
    ----------
    articles : list[dict]
        Raw articles from research.py.
    similarity_threshold : float
        Cosine similarity cutoff for semantic dedup (default 0.70).

    Returns
    -------
    list[dict]
        Deduplicated article list.
    """
    original_count = len(articles)

    # Layer 0 — Cluster key (nano-derived; no-op if normalization was skipped)
    after_cluster = _dedup_by_cluster_key(articles)
    cluster_removed = original_count - len(after_cluster)
    if cluster_removed:
        logger.info(
            "Cluster dedup: %d → %d (%d duplicates removed)",
            original_count, len(after_cluster), cluster_removed,
        )

    # Layer 0.5 — History (cross-week deduplication)
    after_history = _dedup_by_history(after_cluster)
    history_removed = len(after_cluster) - len(after_history)
    if history_removed:
        logger.info(
            "History dedup: %d → %d (%d previous-week duplicates removed)",
            len(after_cluster), len(after_history), history_removed,
        )

    # Layer 1 — URL
    after_url = _dedup_by_url(after_history)
    url_removed = len(after_history) - len(after_url)
    logger.info(
        "URL dedup: %d → %d (%d duplicates removed)",
        len(after_cluster), len(after_url), url_removed,
    )

    # Layer 2 — Semantic
    after_semantic = _dedup_by_similarity(after_url, threshold=similarity_threshold)
    semantic_removed = len(after_url) - len(after_semantic)
    logger.info(
        "Semantic dedup: %d → %d (%d duplicates removed)",
        len(after_url), len(after_semantic), semantic_removed,
    )

    total_removed = original_count - len(after_semantic)
    logger.info(
        "Deduplication: %d → %d articles (%d duplicates removed)",
        original_count, len(after_semantic), total_removed,
    )

    return after_semantic
