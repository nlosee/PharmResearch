"""
research.py — Pharma Regulatory Pulse
Gathers pharmaceutical regulatory news from multiple sources for the past 7 days.

Source hierarchy (per artifact best-practice analysis):
  PRIMARY   — Tavily API: structured, LLM-optimised results with summaries + citations
  SECONDARY — Official RSS: FDA, EMA, Drugs.com (authoritative primary sources)
  TERTIARY  — Google News RSS per topic (broad coverage supplement)
  ENRICHMENT— openFDA REST API (enforcement/recall structured data)
  ENRICHMENT— PubMed / NCBI E-Utilities (peer-reviewed regulatory literature)
  ENRICHMENT— ClinicalTrials.gov API v2 (designation milestones)

Tavily is the #1 recommended search tool for LLM pipelines (93.3 % accuracy
on SimpleQA; 1,000 free searches/month). It is used as the primary layer here,
with all other sources serving as supplementary signal.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Any

import feedparser
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _fetch_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
    max_attempts: int = 3,
    no_results_ok: bool = False,
) -> requests.Response | None:
    """GET with exponential backoff (2s → 4s → 8s).

    Args:
        no_results_ok: When True, HTTP 404 is treated as "zero results" and
            returns None immediately without retrying (e.g. openFDA returns
            404 when a date-range query finds no matching records).
    """
    delays = [2, 4, 8]
    for attempt, delay in enumerate(delays[:max_attempts], start=1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            # openFDA returns 404 JSON when a query matches zero records.
            # Treat that as "no results" rather than a retryable error.
            if no_results_ok and resp.status_code == 404:
                logger.debug("No results (404) for %s — query returned empty set", url)
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s", attempt, max_attempts, url, exc
            )
            if attempt < max_attempts:
                time.sleep(delay)
    logger.error("All %d attempts failed for %s", max_attempts, url)
    return None


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _cutoff_dt() -> datetime:
    """Return timezone-aware UTC datetime 7 days ago."""
    return datetime.now(tz=timezone.utc) - timedelta(days=7)


def _parse_entry_date(entry: Any) -> datetime | None:
    """Try to extract a timezone-aware published datetime from a feedparser entry."""
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return None


def _article(
    title: str,
    url: str,
    content: str,
    source_name: str,
    source_type: str,
    published_date: str,
) -> dict[str, str]:
    """Return a normalised article dict."""
    return {
        "title": title.strip(),
        "url": url.strip(),
        "content": content.strip()[:3000],   # cap content length
        "source_name": source_name,
        "source_type": source_type,
        "published_date": published_date,
    }


# ---------------------------------------------------------------------------
# PRIMARY: Tavily API (LLM-optimised structured search)
# ---------------------------------------------------------------------------

def _get_topics(config: dict) -> list[tuple[str, list[str], int]]:
    """
    Return a flat list of (topic, include_domains, days) tuples.

    - topic          : the Tavily search query string
    - include_domains: list of domains Tavily should restrict results to
                       (empty = search the whole web)
    - days           : how many days back to search (7 for CT topics, 30 for
                       guideline topics so infrequent society updates are caught)

    Within each topic_category, index 0 = clinical-trials topic (7-day, open
    web), index 1 = guidelines topic (30-day, restricted to guideline_hub_domains
    for that category so results come from authoritative society sources).

    Topics are interleaved round-robin across categories so every category
    gets equal Tavily quota.
    """
    newsletter = config.get("newsletter", {})
    categories: dict = newsletter.get("topic_categories", {})
    guideline_hub_domains: dict = config.get("sources", {}).get("guideline_hub_domains", {})

    if categories:
        # Build per-category lists of (topic, include_domains, days)
        cat_items: list[list[tuple[str, list[str], int]]] = []
        for cat_name, cat_topics in categories.items():
            domains = guideline_hub_domains.get(cat_name, [])
            items: list[tuple[str, list[str], int]] = []
            for idx, topic in enumerate(cat_topics):
                if idx == 1 and domains:
                    # Guideline topic: target society domains, 30-day window
                    items.append((topic, list(domains), 30))
                else:
                    # Clinical-trial topic: open web, 7-day window
                    items.append((topic, [], 7))
            cat_items.append(items)

        # Interleave round-robin
        result: list[tuple[str, list[str], int]] = []
        max_len = max((len(c) for c in cat_items), default=0)
        for i in range(max_len):
            for cat in cat_items:
                if i < len(cat):
                    result.append(cat[i])
        return result

    # Legacy flat format: topics (no domain restriction, 7-day window)
    return [(t, [], 7) for t in newsletter.get("topics", [])]


def ingest_tavily(config: dict) -> list[dict[str, str]]:
    """
    Query Tavily API for each configured topic.

    Tavily returns structured, LLM-optimised results with summaries and
    source citations — purpose-built for RAG pipelines. It is the highest-
    signal source in this stack (93.3 % SimpleQA accuracy per benchmark).

    Requires TAVILY_API_KEY environment variable.
    Falls back gracefully if the key is absent (logs a warning and skips).
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning(
            "TAVILY_API_KEY not set — skipping Tavily search. "
            "Add your key from https://tavily.com to improve research quality."
        )
        return []

    # Lazy import so the package is optional at install time
    try:
        from tavily import TavilyClient  # type: ignore[import]
    except ImportError:
        logger.warning(
            "'tavily-python' not installed. Run: pip install tavily-python>=0.5.0 "
            "— skipping Tavily source."
        )
        return []

    client = TavilyClient(api_key=api_key)
    topic_entries = _get_topics(config)
    max_per_topic: int = config.get("newsletter", {}).get("max_articles_per_topic", 5)
    all_articles: list[dict[str, str]] = []

    for topic, include_domains, days in topic_entries:
        domain_label = f" [{','.join(include_domains[:2])}]" if include_domains else ""
        # Exponential backoff retry inside the per-topic call
        for attempt, delay in enumerate([2, 4, 8], start=1):
            try:
                search_kwargs: dict[str, Any] = dict(
                    query=f"{topic} pharmaceutical regulatory",
                    search_depth="advanced",
                    max_results=max_per_topic,
                    include_answer=True,
                    days=days,
                )
                if include_domains:
                    search_kwargs["include_domains"] = include_domains
                result = client.search(**search_kwargs)
                break
            except Exception as exc:
                logger.warning("Tavily attempt %d for '%s' failed: %s", attempt, topic, exc)
                if attempt < 3:
                    time.sleep(delay)
                else:
                    result = None

        if not result:
            continue

        for item in result.get("results", []):
            title = item.get("title", "")
            url   = item.get("url", "")
            # Tavily provides a structured content snippet
            content = item.get("content", "") or item.get("raw_content", "")
            pub_date = item.get("published_date", "unknown") or "unknown"
            if title and url:
                source_label = f"Tavily{domain_label}: {topic[:35]}"
                all_articles.append(
                    _article(title, url, content, source_label, "tavily", pub_date)
                )

        logger.info(
            "Tavily [%s%s, %dd]: %d results",
            topic[:35], domain_label, days, len(result.get("results", [])),
        )

    logger.info("Tavily total: %d articles", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# RSS ingestion
# ---------------------------------------------------------------------------

def _ingest_rss_feed(
    feed_url: str, source_name: str, source_type: str
) -> list[dict[str, str]]:
    """Parse a single RSS/Atom feed and return articles from the last 7 days."""
    cutoff = _cutoff_dt()
    articles: list[dict[str, str]] = []

    try:
        parsed = feedparser.parse(feed_url)
    except Exception as exc:  # feedparser should never raise, but be safe
        logger.warning("feedparser error for %s: %s", feed_url, exc)
        return articles

    for entry in parsed.entries:
        pub_dt = _parse_entry_date(entry)
        if pub_dt and pub_dt < cutoff:
            continue   # older than 7 days

        title = getattr(entry, "title", "") or ""
        link = getattr(entry, "link", "") or ""
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""

        if not title or not link:
            continue

        pub_str = pub_dt.strftime("%Y-%m-%d") if pub_dt else "unknown"
        articles.append(_article(title, link, summary, source_name, source_type, pub_str))

    logger.info("RSS [%s]: %d articles", source_name, len(articles))
    return articles


def ingest_rss_feeds(config: dict) -> list[dict[str, str]]:
    """Ingest all configured RSS feeds."""
    all_articles: list[dict[str, str]] = []
    for feed in config.get("sources", {}).get("pharma_rss_feeds", []):
        articles = _ingest_rss_feed(
            feed["url"], feed["name"], feed.get("source_type", "rss")
        )
        all_articles.extend(articles)
    logger.info("RSS total: %d articles", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------------------------

def ingest_google_news(config: dict) -> list[dict[str, str]]:
    """Generate a Google News RSS URL per topic and ingest recent articles."""
    if not config.get("sources", {}).get("google_news_rss", False):
        return []

    all_articles: list[dict[str, str]] = []
    topic_entries = _get_topics(config)

    for topic, _domains, _days in topic_entries:
        encoded = urllib.parse.quote_plus(f"{topic} when:7d")
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
        articles = _ingest_rss_feed(url, f"Google News: {topic[:40]}", "google_news")
        all_articles.extend(articles)

    logger.info("Google News total: %d articles", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# openFDA REST API
# ---------------------------------------------------------------------------

def ingest_openfda(config: dict) -> list[dict[str, str]]:
    """Pull recent enforcement actions and label changes from openFDA.

    Key API notes:
    - Date range syntax: field:[YYYYMMDD TO YYYYMMDD] — spaces around TO, NOT +TO+.
      The requests library URL-encodes spaces as '+', which is what openFDA expects.
      Using literal '+' in the Python string causes requests to encode them as '%2B',
      which openFDA receives as literal plus signs → invalid Lucene range → 500 error.
    - openFDA returns HTTP 404 (not 200 with empty list) when a query has zero results.
      We pass no_results_ok=True so 404 is treated as "no data" without retrying.
    - sort= is omitted: report_date is not reliably sortable across all openFDA clusters.
      Results default to relevance order; we sort by pub_date downstream in score.py.
    - Enforcement uses a 30-day lookback: FDA batches enforcement reports weekly,
      so a 7-day window frequently returns zero results.
    """
    if not config.get("sources", {}).get("openfda", False):
        return []

    api_key = os.getenv("OPENFDA_API_KEY", "")
    base_url = os.getenv("OPENFDA_BASE_URL", "https://api.fda.gov")
    today_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    # Enforcement data is batched weekly by FDA — use 30-day window to avoid empty results
    cutoff_enforcement = (datetime.now(tz=timezone.utc) - timedelta(days=30)).strftime("%Y%m%d")
    # Label changes can be narrower
    cutoff_label = _cutoff_dt().strftime("%Y%m%d")
    all_articles: list[dict[str, str]] = []

    searches = config.get("sources", {}).get("openfda_searches", [
        {"endpoint": "drug/enforcement", "query": "", "description": "FDA Drug Enforcement"},
    ])

    for search in searches:
        endpoint = search.get("endpoint", "drug/enforcement")
        description = search.get("description", endpoint)
        url = f"{base_url}/{endpoint}.json"

        params: dict[str, Any] = {"limit": 10}
        if api_key:
            params["api_key"] = api_key

        # IMPORTANT: use spaces around TO, NOT literal '+'.
        # requests encodes spaces as '+' in the URL → openFDA receives valid Lucene syntax.
        # Literal '+' in Python string → requests encodes as '%2B' → server gets literal
        # '+TO+' → Lucene parse error → HTTP 500.
        if "enforcement" in endpoint:
            params["search"] = f"report_date:[{cutoff_enforcement} TO {today_str}]"
        elif "label" in endpoint:
            params["search"] = f"effective_time:[{cutoff_label} TO {today_str}]"

        resp = _fetch_with_retry(url, params=params, no_results_ok=True)
        if resp is None:
            continue

        try:
            data = resp.json()
        except ValueError:
            logger.warning("openFDA JSON parse error for %s", url)
            continue

        results = data.get("results", [])
        for item in results:
            # Enforcement record
            if "enforcement" in endpoint:
                title = item.get("product_description", "")[:120] or "FDA Enforcement Action"
                recall_num = item.get("recall_number", "")
                reason = item.get("reason_for_recall", "")
                company = item.get("recalling_firm", "")
                content = (
                    f"Recall #{recall_num} | Company: {company}\n"
                    f"Reason: {reason}\n"
                    f"Classification: {item.get('classification', '')}"
                )
                pub_str = item.get("report_date", "")[:8]
                if pub_str:
                    try:
                        pub_str = datetime.strptime(pub_str, "%Y%m%d").strftime("%Y-%m-%d")
                    except ValueError:
                        pass
                link = f"https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts"
                all_articles.append(
                    _article(f"FDA Recall: {title}", link, content, description, "openfda", pub_str)
                )
            else:
                title = str(item.get("openfda", {}).get("brand_name", ["Unknown"])[0])
                content = str(item)[:500]
                pub_str = str(item.get("effective_time", ""))[:8]
                link = "https://www.fda.gov/drugs/drug-approvals-and-databases"
                all_articles.append(
                    _article(f"FDA Label Update: {title}", link, content, description, "openfda", pub_str)
                )

    logger.info("openFDA total: %d articles", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# PubMed / NCBI E-Utilities
# ---------------------------------------------------------------------------

def ingest_pubmed(config: dict) -> list[dict[str, str]]:
    """Search PubMed for regulatory-relevant publications from the last 7 days."""
    if not config.get("sources", {}).get("pubmed", False):
        return []

    api_key = os.getenv("PUBMED_API_KEY", "")
    base = os.getenv("PUBMED_BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
    now = datetime.now(tz=timezone.utc)
    # Use 30-day window for PubMed — regulatory papers take weeks to be indexed
    # datetype=pdat (publication date) is more reliable than edat (entrez date)
    mindate = (now - timedelta(days=30)).strftime("%Y/%m/%d")
    maxdate = now.strftime("%Y/%m/%d")
    all_articles: list[dict[str, str]] = []

    queries: list[str] = config.get("sources", {}).get(
        "pubmed_queries", ["FDA guidance pharmaceutical regulatory"]
    )

    for query in queries:
        # Step 1 — esearch: get PMIDs
        search_params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 5,
            "datetype": "pdat",
            "mindate": mindate,
            "maxdate": maxdate,
            "usehistory": "y",
        }
        if api_key:
            search_params["api_key"] = api_key

        resp = _fetch_with_retry(f"{base}/esearch.fcgi", params=search_params)
        if resp is None:
            continue

        try:
            search_data = resp.json()
        except ValueError:
            continue

        pmids: list[str] = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            continue

        # Step 2 — esummary: get titles + abstracts
        summary_params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        if api_key:
            summary_params["api_key"] = api_key

        resp2 = _fetch_with_retry(f"{base}/esummary.fcgi", params=summary_params)
        if resp2 is None:
            continue

        try:
            summary_data = resp2.json()
        except ValueError:
            continue

        uids = summary_data.get("result", {}).get("uids", [])
        for uid in uids:
            item = summary_data["result"].get(uid, {})
            title = item.get("title", "PubMed Article")
            pub_date = item.get("pubdate", "")[:10]
            link = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
            authors = ", ".join(
                a.get("name", "") for a in item.get("authors", [])[:3]
            )
            source = item.get("source", "")
            content = (
                f"Journal: {source} | Authors: {authors}\n"
                f"Published: {pub_date}\n"
                f"Query context: {query}"
            )
            all_articles.append(
                _article(title, link, content, "PubMed", "pubmed", pub_date)
            )

    logger.info("PubMed total: %d articles", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# ClinicalTrials.gov API v2
# ---------------------------------------------------------------------------

def ingest_clinicaltrials(config: dict) -> list[dict[str, str]]:
    """Pull recently updated clinical trials with regulatory milestones."""
    if not config.get("sources", {}).get("clinicaltrials", False):
        return []

    base = os.getenv("CLINICALTRIALS_BASE_URL", "https://clinicaltrials.gov/api/v2")
    cutoff = _cutoff_dt()
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    all_articles: list[dict[str, str]] = []

    queries: list[str] = config.get("sources", {}).get(
        "clinicaltrials_queries", ["Breakthrough Therapy", "Fast Track"]
    )

    for query in queries:
        params: dict[str, Any] = {
            "query.term": query,
            "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{cutoff_str},MAX]",
            "pageSize": 5,
            "format": "json",
        }

        resp = _fetch_with_retry(f"{base}/studies", params=params)
        if resp is None:
            continue

        try:
            data = resp.json()
        except ValueError:
            continue

        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            id_module = proto.get("identificationModule", {})
            status_module = proto.get("statusModule", {})
            desc_module = proto.get("descriptionModule", {})

            nct_id = id_module.get("nctId", "")
            title = id_module.get("briefTitle", "Clinical Trial Update")
            status = status_module.get("overallStatus", "")
            last_update = status_module.get("lastUpdatePostDateStruct", {}).get("date", "")
            brief_summary = desc_module.get("briefSummary", "")[:500]
            link = f"https://clinicaltrials.gov/study/{nct_id}"

            content = (
                f"NCT ID: {nct_id} | Status: {status}\n"
                f"Designation query: {query}\n"
                f"{brief_summary}"
            )
            all_articles.append(
                _article(
                    f"[{query}] {title}",
                    link,
                    content,
                    "ClinicalTrials.gov",
                    "clinicaltrials",
                    last_update,
                )
            )

    logger.info("ClinicalTrials.gov total: %d articles", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def research_topics(config: dict) -> list[dict[str, str]]:
    """
    Orchestrate all research sources and return a unified article list.

    Parameters
    ----------
    config : dict
        Loaded config.yaml contents.

    Returns
    -------
    list[dict]
        Articles with keys: title, url, content, source_name, source_type, published_date
    """
    all_articles: list[dict[str, str]] = []

    # 1 — PRIMARY: Tavily (LLM-optimised structured search — highest signal)
    all_articles.extend(ingest_tavily(config))

    # 2 — SECONDARY: Official + industry RSS feeds (authoritative primary sources)
    all_articles.extend(ingest_rss_feeds(config))

    # 3 — TERTIARY: Google News RSS (broad coverage supplement)
    all_articles.extend(ingest_google_news(config))

    # 4 — ENRICHMENT: openFDA REST API (enforcement/recall structured data)
    all_articles.extend(ingest_openfda(config))

    # 5 — ENRICHMENT: PubMed (peer-reviewed regulatory literature)
    all_articles.extend(ingest_pubmed(config))

    # 6 — ENRICHMENT: ClinicalTrials.gov (designation milestones)
    all_articles.extend(ingest_clinicaltrials(config))

    logger.info("Research complete — total articles gathered: %d", len(all_articles))
    return all_articles
