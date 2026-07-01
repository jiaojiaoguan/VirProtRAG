"""
PubMed BM25 multi-round search via NCBI Entrez API.
Reference: step4_BM25_baseQ_synQ.py
"""

import re
import time
import logging
from typing import Optional

from Bio import Entrez

logger = logging.getLogger(__name__)


def _configure_entrez(email: str, api_key: Optional[str] = None):
    """Configure NCBI Entrez global settings."""
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    Entrez.max_tries = 5


def _deduplicate_keep_order(seq: list) -> list:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _search_pubmed(query: str, max_results: int) -> list[str]:
    """Search PubMed for a query, returning top PMIDs sorted by relevance."""
    for attempt in range(5):
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance",
            )
            record = Entrez.read(handle)
            handle.close()
            return _deduplicate_keep_order(record.get("IdList", []))
        except Exception as e:
            logger.warning(f"PubMed search attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    return []


def bm25_multi_round_search(
    query: str,
    synonym_query: str = "",
    top_k: int = 100,
    email: str = "",
    api_key: Optional[str] = None,
) -> list[str]:
    """Multi-round PubMed BM25 search with progressive query relaxation.

    The search strategy (Step 1–8) progressively broadens the query to
    maximize recall while prioritizing precise matches at the top.

    Reference: step4_BM25_baseQ_synQ.py multi_round_pubmed_search_new_query()

    Args:
        query: Base BM25 query expression.
        synonym_query: Optional synonym-expanded query from LLM.
        top_k: Maximum number of PMIDs to retrieve.
        email: NCBI Entrez email.
        api_key: NCBI API key (optional but recommended).

    Returns:
        List of PMIDs in ranked order.
    """
    _configure_entrez(email, api_key)

    all_pmids: list[str] = []
    fq = query.strip()

    if not fq:
        return []

    # ---- Step 1: Exact query with quoted terms ----
    retrieved = _search_pubmed(fq, top_k)
    all_pmids.extend(retrieved[:top_k])
    logger.debug(f"Step1: {len(retrieved)} PMIDs, total={len(all_pmids)}")

    # ---- Step 2: Synonym-expanded query (if available) ----
    sq = synonym_query.strip()
    if len(all_pmids) < top_k and sq:
        retrieved = _search_pubmed(sq, top_k)
        new_ids = [p for p in _deduplicate_keep_order(retrieved) if p not in all_pmids]
        all_pmids.extend(new_ids[: top_k - len(all_pmids)])
        logger.debug(f"Step2: {len(new_ids)} new, total={len(all_pmids)}")

    # ---- Step 3: Remove quotes (relax exact match) ----
    if len(all_pmids) < top_k:
        q3 = re.sub(r'"([^"]+)"', r"\1", fq)
        retrieved = _search_pubmed(q3, top_k)
        new_ids = [p for p in _deduplicate_keep_order(retrieved) if p not in all_pmids]
        all_pmids.extend(new_ids[: top_k - len(all_pmids)])
        logger.debug(f"Step3: {len(new_ids)} new, total={len(all_pmids)}")

    # ---- Step 4: Replace 'phage' with 'virus' ----
    if len(all_pmids) < top_k:
        q4 = re.sub(r"\b[Pp]hage\b", "virus", re.sub(r'"([^"]+)"', r"\1", fq))
        retrieved = _search_pubmed(q4, top_k)
        new_ids = [p for p in _deduplicate_keep_order(retrieved) if p not in all_pmids]
        all_pmids.extend(new_ids[: top_k - len(all_pmids)])
        logger.debug(f"Step4: {len(new_ids)} new, total={len(all_pmids)}")

    # ---- Pre-split query into protein and organism parts ----
    # Use split on ' AND ' to robustly separate the two halves,
    # avoiding fragile regex that breaks on nested parentheses or special chars.
    _and_parts = re.split(r"\s+AND\s+", fq, maxsplit=1)
    _protein_part = _and_parts[0].strip()
    _org_part = _and_parts[1].strip() if len(_and_parts) == 2 else ""
    # Strip outer parentheses if present
    _org_clean = re.sub(r'^\((.*)\)$', r'\1', _org_part).strip()

    # ---- Step 5: Remove organism constraint (protein-only) ----
    if len(all_pmids) < top_k and _protein_part:
        retrieved = _search_pubmed(_protein_part, top_k)
        new_ids = [p for p in _deduplicate_keep_order(retrieved) if p not in all_pmids]
        all_pmids.extend(new_ids[: top_k - len(all_pmids)])
        logger.debug(f"Step5: {len(new_ids)} new, total={len(all_pmids)}")

    # ---- Step 6: Organism only (quoted) ----
    if len(all_pmids) < top_k and _org_clean:
        q6 = f'"{_org_clean}"'
        retrieved = _search_pubmed(q6, top_k)
        new_ids = [p for p in _deduplicate_keep_order(retrieved) if p not in all_pmids]
        all_pmids.extend(new_ids[: top_k - len(all_pmids)])
        logger.debug(f"Step6: {len(new_ids)} new, total={len(all_pmids)}")

    # ---- Step 7: Organism only (unquoted) ----
    if len(all_pmids) < top_k and _org_clean:
        retrieved = _search_pubmed(_org_clean, top_k)
        new_ids = [p for p in _deduplicate_keep_order(retrieved) if p not in all_pmids]
        all_pmids.extend(new_ids[: top_k - len(all_pmids)])
        logger.debug(f"Step7: {len(new_ids)} new, total={len(all_pmids)}")

    # ---- Step 8: Organism only (phage→virus) ----
    if len(all_pmids) < top_k and _org_clean:
        q8 = re.sub(r"\b[Pp]hage\b", "virus", _org_clean)
        retrieved = _search_pubmed(q8, top_k)
        new_ids = [p for p in _deduplicate_keep_order(retrieved) if p not in all_pmids]
        all_pmids.extend(new_ids[: top_k - len(all_pmids)])
        logger.debug(f"Step8: {len(new_ids)} new, total={len(all_pmids)}")

    return all_pmids[:top_k]
