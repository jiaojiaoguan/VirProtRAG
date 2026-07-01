"""
LLM-based evidence type classification for retrieved publications.
Reference: step10_llm_for_exp.py

Supports persistent caching via CacheManager: evidence labels are keyed by
(pmid, protein_name) so the same paper re-encountered for the same protein
can skip the LLM call entirely.
"""

import json
import logging
from typing import Optional

from .llm_client import call_llm, extract_json_from_response
from .config import LLMConfig
from .cache import CacheManager

logger = logging.getLogger(__name__)

EVIDENCE_SYSTEM_PROMPT = """You are a molecular biology expert specializing in protein function annotation and familiar with UniProt evidence codes.

You will be given the name of a protein and a set of scientific publications (titles and abstracts).
For each publication, determine whether it provides *experimental evidence* supporting that protein's biological function, merely *infers* it without experiments, or does not discuss the protein's function at all.

Respond strictly in JSON format as a list of objects, one per paper, like this:

[
  {
    "pmid": "123456",
    "label": 0 or 1 or 2,
    "label_name": "NONE" or "INFERRED" or "EXPERIMENTAL",
    "confidence": float between 0 and 1
  },
  ...
]

Label definitions:

- **0 (NONE)**: No protein/gene function described.
- **1 (INFERRED)**: Similar ECO code ECO:0000303. Function is *predicted* or *implied* (no experimental validation).
- **2 (EXPERIMENTAL)**: Similar ECO code ECO:0000269. The study contains experiments directly testing or demonstrating protein function (e.g., enzymatic assays, mutagenesis affecting activity, knockout/overexpression altering phenotype, etc.).

Respond using **only valid JSON**, no explanatory text, no Markdown."""

EVIDENCE_BATCH_SIZE = 15


def _classify_one_batch(
    config: LLMConfig,
    protein_name: str,
    batch: list[dict],
    batch_idx: int,
    total_batches: int,
) -> dict[str, dict]:
    """Classify evidence for a single batch of publications (no cache logic)."""
    pub_texts = []
    for pub in batch:
        pub_texts.append(
            {
                "pmid": pub.get("pmid", ""),
                "title": pub.get("title", "")[:500],
                "abstract": pub.get("abstract", "")[:2000],
            }
        )

    user_prompt = (
        f"Protein names: {protein_name}\n\n"
        f"--- Associated publications (batch {batch_idx + 1}/{total_batches}, "
        f"{len(pub_texts)} papers) ---\n"
        + json.dumps(pub_texts, ensure_ascii=False, indent=2)
    )

    try:
        raw = call_llm(
            config,
            EVIDENCE_SYSTEM_PROMPT,
            user_prompt,
            task="evidence",
            temperature=0.0,
        )
        result = extract_json_from_response(raw)
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "401" in error_msg:
            hint = (
                f"Evidence classification batch {batch_idx + 1}/{total_batches} failed: "
                f"API authentication error. Check your {config.provider.upper()}_API_KEY. "
                f"Error: {error_msg[:200]}"
            )
        elif "Connection" in error_msg or "timeout" in error_msg:
            hint = (
                f"Evidence classification batch {batch_idx + 1}/{total_batches} failed: "
                f"Network error. Check your connection. Error: {error_msg[:200]}"
            )
        else:
            hint = (
                f"Evidence classification batch {batch_idx + 1}/{total_batches} failed: "
                f"{error_msg[:200]}"
            )
        logger.warning(hint)
        return {}

    evidence_map = {}
    if isinstance(result, list):
        for item in result:
            pmid = str(item.get("pmid", "")).strip()
            if pmid:
                evidence_map[pmid] = {
                    "label": item.get("label", 0),
                    "label_name": item.get("label_name", "NONE"),
                    "confidence": item.get("confidence", 0.0),
                }
    return evidence_map


def classify_evidence(
    config: LLMConfig,
    protein_name: str,
    publications: list[dict],
    cache: Optional[CacheManager] = None,
) -> dict[str, dict]:
    """Classify the evidence type of each publication using an LLM.

    Checks the persistent cache first: evidence is keyed by (pmid, protein_name)
    so previously classified papers are reused across pipeline runs.

    Args:
        config: LLM configuration.
        protein_name: Target protein name for context.
        publications: List of {"pmid": str, "title": str, "abstract": str}.
        cache: Optional CacheManager for persistent evidence caching.

    Returns:
        Dict mapping PMID → {"label": int, "label_name": str, "confidence": float}
    """
    if not publications:
        return {}

    # Build PMID → publication lookup for uncached items
    pub_by_pmid = {pub.get("pmid", ""): pub for pub in publications}

    # ---- Check cache first ----
    cached_hits: dict[str, dict] = {}
    uncached_pubs: list[dict] = []

    if cache is not None:
        for pub in publications:
            pmid = pub.get("pmid", "")
            hit = cache.get_evidence(pmid, protein_name)
            if hit is not None:
                cached_hits[pmid] = hit
            else:
                uncached_pubs.append(pub)
    else:
        uncached_pubs = publications

    cache_hit_count = len(cached_hits)
    uncache_count = len(uncached_pubs)

    if cache_hit_count > 0:
        logger.debug(
            f"Evidence cache: {cache_hit_count} hits, "
            f"{uncache_count} need LLM classification"
        )

    # ---- Classify only uncached publications ----
    fresh_map: dict[str, dict] = {}
    if uncached_pubs:
        batches = [
            uncached_pubs[i : i + EVIDENCE_BATCH_SIZE]
            for i in range(0, len(uncached_pubs), EVIDENCE_BATCH_SIZE)
        ]
        logger.debug(
            f"Classifying evidence for {len(uncached_pubs)} uncached papers "
            f"in {len(batches)} batch(es)"
        )

        for batch_idx, batch in enumerate(batches):
            batch_map = _classify_one_batch(
                config, protein_name, batch, batch_idx, len(batches)
            )
            fresh_map.update(batch_map)
            logger.debug(
                f"  Batch {batch_idx + 1}/{len(batches)}: {len(batch_map)} classified"
            )

        # Store fresh results in cache
        if cache is not None:
            for pmid, data in fresh_map.items():
                cache.set_evidence(pmid, protein_name, data)

    # Merge cached + fresh results
    result = dict(cached_hits)
    result.update(fresh_map)
    return result
