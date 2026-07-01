"""
MedCPT dense retrieval over PubMed embeddings via FAISS.
Reference: step6_medcpt_search.py

Requires: faiss-cpu, transformers, torch (pip install virprotrag[medcpt])
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def medcpt_search(
    queries: list[str],
    index_path: str,
    pmids_path: str,
    topk: int = 100,
    batch_size: int = 2048,
) -> list[dict]:
    """Search PubMed via MedCPT dense retrieval.

    Encodes each query with the MedCPT query encoder and searches
    the pre-built FAISS index of ~39M PubMed article embeddings.

    Args:
        queries: List of natural-language query strings.
        index_path: Path to MedCPT FAISS index file.
        pmids_path: Path to PMIDs JSON mapping file.
        topk: Number of top results per query.
        batch_size: Batch size for query encoding.

    Returns:
        List of dicts: [{"pmid": "...", "score": 0.95}, ...]
        Results are deduplicated across queries (best score kept).
    """
    try:
        import faiss
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "MedCPT requires faiss-cpu, transformers, and torch. "
            "Install with: pip install virprotrag[medcpt]"
        )

    if not queries:
        return []

    # Load FAISS index and PMID mapping
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    with open(pmids_path, "r") as f:
        pmids = json.load(f)
    logger.info(f"Index: {index.ntotal} vectors, PMIDs: {len(pmids)} entries")

    # Load MedCPT query encoder (local_files_only: compute nodes have no internet)
    logger.info("Loading MedCPT query encoder...")
    model = AutoModel.from_pretrained(
        "ncbi/MedCPT-Query-Encoder", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "ncbi/MedCPT-Query-Encoder", local_files_only=True
    )
    model.eval()

    # Aggregate results per-query (keep best score per PMID after dedup)
    best_score: dict[str, float] = {}

    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]

        encoded = tokenizer(
            batch, truncation=True, padding=True, return_tensors="pt", max_length=512
        )

        with torch.no_grad():
            query_embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()

        scores, indices = index.search(query_embeds, k=topk)

        for i in range(len(batch)):
            for j in range(topk):
                pmid = pmids[indices[i][j]]
                score = float(scores[i][j])
                if pmid not in best_score or score > best_score[pmid]:
                    best_score[pmid] = score

        if (start + batch_size) % (batch_size * 10) == 0:
            logger.info(f"Processed {start + len(batch)}/{len(queries)} queries")

    # Sort by score descending
    results = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
    return [{"pmid": pmid, "score": score} for pmid, score in results]


def medcpt_search_from_bm25_json(
    bm25_json_path: str,
    index_path: str,
    pmids_path: str,
    topk: int = 100,
    batch_size: int = 2048,
) -> dict:
    """Standalone MedCPT search reading queries from a BM25-phase JSON file.

    Designed for execution on HPC compute nodes: reads pre-built queries,
    runs pure FAISS dense retrieval, and writes results to a JSON file.
    Zero network dependency — only needs the FAISS index + PMIDs on disk.

    Args:
        bm25_json_path: Path to the JSON output from ``virprotrag bm25``.
        index_path: Path to MedCPT FAISS index file.
        pmids_path: Path to PMIDs JSON mapping file.
        topk: Number of top results per query.
        batch_size: Batch size for query encoding.

    Returns:
        dict with keys: source, num_queries, medcpt_results, retrieved_topk,
        runtime (elapsed_seconds, timestamp_utc).
    """
    import time
    from datetime import datetime, timezone

    t_start = time.time()

    # Load BM25 output
    with open(bm25_json_path, "r", encoding="utf-8") as f:
        bm25_data = json.load(f)

    queries = bm25_data.get("medcpt_queries", [])
    if not queries:
        logger.warning(f"No medcpt_queries found in {bm25_json_path}")
        return {
            "source": bm25_json_path,
            "num_queries": 0,
            "medcpt_results": [],
            "retrieved_topk": topk,
            "runtime": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.time() - t_start, 1),
            },
        }

    protein = bm25_data.get("query", {}).get("protein", "unknown")
    organism = bm25_data.get("query", {}).get("organism", "unknown")
    logger.info(f"MedCPT phase: {protein} ({organism}) — {len(queries)} queries")

    results = medcpt_search(
        queries=queries,
        index_path=index_path,
        pmids_path=pmids_path,
        topk=topk,
        batch_size=batch_size,
    )

    total_elapsed = time.time() - t_start

    return {
        "source": bm25_json_path,
        "num_queries": len(queries),
        "medcpt_results": results,
        "retrieved_topk": topk,
        "runtime": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(total_elapsed, 1),
        },
    }


def has_medcpt(index_path: str, pmids_path: str) -> bool:
    """Check whether MedCPT index and PMID mapping files exist."""
    import os
    return (
        index_path is not None
        and pmids_path is not None
        and os.path.isfile(index_path)
        and os.path.isfile(pmids_path)
    )
