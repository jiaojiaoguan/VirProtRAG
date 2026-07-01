"""
Rank-aware Reciprocal Rank Fusion for BM25 + MedCPT result merging.
Reference: step8_RankRRF.py
"""

from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


def rrf_fusion_rank_aware(
    bm25_results: list[str],
    medcpt_results: list[dict],
    k: float = 60.0,
    w_bm25_max: float = 1.0,
    w_bm25_min: float = 1.0,
    w_medcpt_min: float = 1.0,
    w_medcpt_max: float = 5.0,
    topk_bm25: int = 1000,
    topk_medcpt: int = 1000,
) -> dict[str, float]:
    """Fuse BM25 and MedCPT results with rank-dependent weighting.

    BM25 gets higher weight at shallow ranks (where precision is strong),
    while MedCPT gets higher weight at deeper ranks (where semantic
    generalization improves recall).

    Formula (Eq. 4–6 from the paper):
        S(d) = w_B(r_B(d)) / (k + r_B(d)) + w_M(r_M(d)) / (k + r_M(d))
        w_B(r) = w_B_max - (w_B_max - w_B_min) * r / N_B
        w_M(r) = w_M_min + (w_M_max - w_M_min) * r / N_M

    Args:
        bm25_results: Ranked list of PMIDs from BM25 search.
        medcpt_results: Ranked list of {"pmid": ..., "score": ...} from MedCPT.
        k: RRF damping constant (default 60).
        w_bm25_max/min: BM25 weight range.
        w_medcpt_max/min: MedCPT weight range.
        topk_bm25: Total number of BM25 results considered.
        topk_medcpt: Total number of MedCPT results considered.

    Returns:
        Dict mapping PMID → fused RRF score (higher = better).
    """
    fused_scores: dict[str, float] = {}

    # BM25 scores
    for rank, pmid in enumerate(bm25_results):
        w_bm25 = w_bm25_max - (w_bm25_max - w_bm25_min) * (rank / max(1, topk_bm25))
        fused_scores[pmid] = fused_scores.get(pmid, 0.0) + w_bm25 / (k + rank + 1)

    # MedCPT scores
    for rank, item in enumerate(medcpt_results):
        pmid = item["pmid"]
        w_medcpt = w_medcpt_min + (w_medcpt_max - w_medcpt_min) * (rank / max(1, topk_medcpt))
        fused_scores[pmid] = fused_scores.get(pmid, 0.0) + w_medcpt / (k + rank + 1)

    return dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))


def normalize_scores(fused: dict[str, float]) -> list[dict]:
    """Min-max normalize fused scores to [0, 1].

    Returns a ranked list of {"pmid": str, "score": float, "score_norm": float}.
    """
    if not fused:
        return []

    scores = list(fused.values())
    s_min, s_max = min(scores), max(scores)
    denom = s_max - s_min if s_max > s_min else 1.0

    return [
        {"pmid": pmid, "score": score, "score_norm": (score - s_min) / denom}
        for pmid, score in fused.items()
    ]
