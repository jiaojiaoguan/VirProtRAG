"""
Re-ranking with paper quality and evidence signals.
Reference: step12_combine_signals.py
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _normalize_log1p(values: np.ndarray) -> np.ndarray:
    """Log1p + min-max normalization to handle skewed distributions."""
    logged = np.log1p(np.nan_to_num(values, nan=0.0))
    return (logged - logged.min()) / (logged.max() - logged.min() + 1e-8)


def _normalize_linear(values: np.ndarray) -> np.ndarray:
    """Linear min-max normalization to [0, 1]."""
    v = np.nan_to_num(values, nan=0.0)
    return (v - v.min()) / (v.max() - v.min() + 1e-8)


def rerank(
    fused_results: list[dict],
    abstracts: dict[str, dict],
    evidence_map: dict[str, dict],
    openalex_metrics: Optional[dict[str, dict]] = None,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
) -> list[dict]:
    """Re-rank fused retrieval results by combining semantic, quality, and evidence scores.

    S(d) = α * S_semantic(d) + β * S_qual(d) + γ * S_evid(d)

    Paper quality (S_qual) combines: citation count, venue h-index, author h-indices.
    Evidence score (S_evid) weights: EXPERIMENTAL=1.0, INFERRED=0.5, NONE=0.0.

    Args:
        fused_results: Ranked list of {"pmid": str, "score": float, "score_norm": float}.
        abstracts: Dict PMID → {"title": str, "abstract": str}.
        evidence_map: Dict PMID → {"label": int, "label_name": str, ...}.
        openalex_metrics: Dict PMID → {"citation_count": int, ...} (optional).
        alpha: Weight for semantic/similarity score.
        beta: Weight for paper quality score.
        gamma: Weight for evidence score.

    Returns:
        Re-ranked list of {"pmid": str, "final_score": float, ...} with added fields.
    """
    if not fused_results:
        return []

    pmids = [r["pmid"] for r in fused_results]
    semantic_scores = np.array([r.get("score_norm", r.get("score", 0.0)) for r in fused_results])

    # --- Evidence scores ---
    evidence_weights = {"NONE": 0.0, "INFERRED": 0.5, "EXPERIMENTAL": 1.0}
    evid_scores = np.array([
        evidence_weights.get(evidence_map.get(p, {}).get("label_name", "NONE"), 0.0)
        for p in pmids
    ])

    # --- Paper quality scores ---
    if openalex_metrics:
        citations = np.array([openalex_metrics.get(p, {}).get("citation_count", 0) for p in pmids])
        venue_h = np.array([openalex_metrics.get(p, {}).get("venue_h_index", 0.0) for p in pmids])
        author_h = np.array([
            max(
                openalex_metrics.get(p, {}).get("first_author_h_index", 0.0),
                openalex_metrics.get(p, {}).get("last_author_h_index", 0.0),
            )
            for p in pmids
        ])

        qual_scores = (
            0.6 * _normalize_log1p(citations)
            + 0.2 * _normalize_linear(venue_h)
            + 0.2 * _normalize_linear(author_h)
        )
    else:
        # No quality data — assign neutral scores
        qual_scores = np.ones_like(semantic_scores) * 0.5

    # --- Combined score ---
    final = alpha * semantic_scores + beta * qual_scores + gamma * evid_scores

    # Build re-ranked output
    reranked = []
    for i, pmid in enumerate(pmids):
        item = {
            "pmid": pmid,
            "final_score": float(final[i]),
            "semantic_score": float(semantic_scores[i]),
            "evidence_score": float(evid_scores[i]),
            "quality_score": float(qual_scores[i]),
        }
        # Merge abstract info
        if pmid in abstracts:
            item["title"] = abstracts[pmid].get("title", "")
            item["abstract"] = abstracts[pmid].get("abstract", "")
        # Merge evidence label
        if pmid in evidence_map:
            item["evidence_label"] = evidence_map[pmid].get("label_name", "")
        reranked.append(item)

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked
