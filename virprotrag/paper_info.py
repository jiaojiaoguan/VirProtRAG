"""
Fetch PubMed article metadata and OpenAlex bibliometric data.
Reference: step9_get_info.py, step11_get_paper_quality_info.py
"""

import json
import time
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from Bio import Entrez

logger = logging.getLogger(__name__)


def fetch_pubmed_abstracts(
    pmids: list[str],
    email: str,
    api_key: Optional[str] = None,
    batch_size: int = 200,
) -> dict[str, dict]:
    """Fetch title and abstract for a list of PMIDs via NCBI EFetch.

    Args:
        pmids: List of PubMed IDs.
        email: NCBI Entrez email.
        api_key: NCBI API key (optional).
        batch_size: Number of PMIDs per request (max ~200).

    Returns:
        Dict mapping PMID → {"title": str, "abstract": str}
    """
    if not pmids:
        return {}

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    articles = {}
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        ids_str = ",".join(batch)

        for attempt in range(3):
            try:
                handle = Entrez.efetch(
                    db="pubmed", id=ids_str, rettype="xml", retmode="text"
                )
                records = Entrez.read(handle)
                handle.close()

                for article in records.get("PubmedArticle", []):
                    medline = article.get("MedlineCitation", {})
                    art = medline.get("Article", {})
                    pmid = str(medline.get("PMID", ""))
                    title = art.get("ArticleTitle", "")
                    abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
                    abstract = " ".join(
                        str(a) if isinstance(a, str) else a.get("#text", "")
                        for a in abstract_parts
                    )
                    articles[pmid] = {"title": str(title), "abstract": str(abstract)}
                break
            except Exception as e:
                logger.warning(f"EFetch attempt {attempt + 1} failed: {e}")
                time.sleep(3)

        time.sleep(0.5)  # Rate-limit between batches

    return articles


def _safe_request(url: str, max_retries: int = 3, pause: int = 2) -> Optional[dict]:
    """GET request with automatic retry and rate-limit handling."""
    for i in range(max_retries):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("OpenAlex rate limit, waiting 5s...")
                time.sleep(5)
            elif resp.status_code == 404:
                return None
            else:
                logger.warning(f"Request failed {resp.status_code}: {url}")
        except Exception as e:
            logger.warning(f"Network error: {e} (attempt {i + 1}/{max_retries})")
        time.sleep(pause * (i + 1))
    return None


def fetch_openalex_metrics(
    pmids: list[str],
    email: Optional[str] = None,
    max_workers: int = 5,
) -> dict[str, dict]:
    """Fetch paper quality metrics from OpenAlex for a list of PMIDs.

    Retrieves: citation count, venue h-index, author h-indices.

    Args:
        pmids: List of PubMed IDs.
        email: OpenAlex polite pool email (optional, for higher rate limits).
        max_workers: Concurrent request threads.

    Returns:
        Dict mapping PMID → {
            "citation_count": int,
            "venue_h_index": float,
            "venue_name": str,
            "first_author_h_index": float,
            "last_author_h_index": float,
        }
    """
    if not pmids:
        return {}

    mailto = f"?mailto={email}" if email else ""

    def _get_one(pmid: str) -> tuple[str, Optional[dict]]:
        url = f"https://api.openalex.org/works/pmid:{pmid}{mailto}"
        data = _safe_request(url)
        if not data:
            return pmid, None

        result = {
            "citation_count": data.get("cited_by_count", 0) or 0,
            "venue_name": "",
            "venue_h_index": 0.0,
            "first_author_h_index": 0.0,
            "last_author_h_index": 0.0,
        }

        # Venue info
        venue = data.get("primary_location", {}) or {}
        source = venue.get("source", {}) or {}
        result["venue_name"] = source.get("display_name", "")

        # Authorship info
        authorship = data.get("authorships", [])
        if authorship:
            # First author
            first = authorship[0].get("author", {}) or {}
            first_id = first.get("id")
            if first_id:
                author_data = _safe_request(
                    first_id.replace("https://openalex.org/", "https://api.openalex.org/authors/")
                    + mailto
                )
                if author_data:
                    result["first_author_h_index"] = (
                        author_data.get("summary_stats", {}) or {}
                    ).get("h_index", 0.0) or 0.0

            # Last author
            if len(authorship) > 1:
                last = authorship[-1].get("author", {}) or {}
                last_id = last.get("id")
                if last_id:
                    author_data = _safe_request(
                        last_id.replace("https://openalex.org/", "https://api.openalex.org/authors/")
                        + mailto
                    )
                    if author_data:
                        result["last_author_h_index"] = (
                            author_data.get("summary_stats", {}) or {}
                        ).get("h_index", 0.0) or 0.0

        return pmid, result

    metrics: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get_one, pmid): pmid for pmid in pmids}
        for future in as_completed(futures):
            pmid, result = future.result()
            if result:
                metrics[pmid] = result

    return metrics
