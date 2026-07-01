"""
Full annotation pipeline orchestrator for a single viral protein.
Orchestrates: query building → synonym expansion → BM25 search →
MedCPT search (optional) → RRF fusion → paper metadata fetch →
evidence classification → re-ranking → LLM generation.

Supports three-phase execution for HPC environments:
  1. run_bm25_phase()  — login node (synonyms + query + PubMed search)
  2. run_medcpt_phase() — compute node via medcpt_search.py (FAISS, no network)
  3. run_annotate_phase() — login node (RRF + abstracts + evidence + generation)
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from .config import LLMConfig, RuntimeConfig
from .cache import CacheManager
from .query_builder import build_bm25_query, build_medcpt_queries
from .synonym import expand_synonyms
from .bm25_search import bm25_multi_round_search
from .medcpt_search import medcpt_search, has_medcpt
from .rrf_fusion import rrf_fusion_rank_aware, normalize_scores
from .paper_info import fetch_pubmed_abstracts, fetch_openalex_metrics
from .evidence import classify_evidence
from .rerank import rerank
from .generator import generate_annotation

logger = logging.getLogger(__name__)

# Total number of main pipeline steps (for progress display)
_TOTAL_STEPS = 8


def _step_print(step_num: int, description: str, status: str = ""):
    """Print a clean one-line step indicator. detail goes to debug log."""
    prefix = f"  [{step_num}/{_TOTAL_STEPS}] {description}"
    if status:
        print(f"{prefix} ... {status}")
    else:
        print(f"{prefix} ...", end="", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: BM25 — synonym expansion + query construction + PubMed search
# ---------------------------------------------------------------------------

def run_bm25_phase(
    config: RuntimeConfig,
    protein_names: str,
    gene_names: str = "",
    organism: str = "",
    retrieval_topk: int = 100,
) -> dict:
    """Run the BM25 retrieval phase: synonyms → query building → PubMed BM25.

    Designed for execution on a login node (needs network for Entrez API and
    LLM for synonym expansion).  Saves all intermediate state so later phases
    can pick up without repeating work.

    Args:
        config: RuntimeConfig with LLM and Entrez settings.
        protein_names: Comma/semicolon-separated protein names.
        gene_names: Comma/semicolon-separated gene names (optional).
        organism: Comma/semicolon-separated organism names.
        retrieval_topk: Max PMIDs to retrieve via BM25 (default 100).

    Returns:
        dict with keys: query, synonyms, all_proteins_syn, bm25_query_used,
        bm25_query_syn, medcpt_queries, bm25_pmids, retrieval_topk, runtime.
    """
    t_start = time.time()

    short_protein = protein_names.split(",")[0].strip()
    print(f"\n{'='*60}")
    print(f"  VirProtRAG BM25 Phase: {short_protein} ({organism})")
    print(f"{'='*60}")

    # Validate config (synonym expansion needs LLM; BM25 needs Entrez)
    issues = config.validate()
    if issues:
        raise RuntimeError("Configuration error: " + "; ".join(issues))

    # ---- Step 1: Synonym expansion ----
    print("  [1/3] Synonym expansion ...", end="", flush=True)
    step_start = time.time()
    from .query_builder import _split_names
    protein_list = _split_names(protein_names)
    synonyms = expand_synonyms(config.llm, protein_list, organism)
    all_proteins_syn = protein_names
    if synonyms:
        all_proteins_syn = protein_names + ", " + ", ".join(synonyms)
    logger.debug(f"  Found {len(synonyms)} synonyms: {synonyms}")
    print(f" \u2713 {len(synonyms)} synonyms ({time.time() - step_start:.1f}s)")

    # ---- Step 2: Build queries ----
    print("  [2/3] Building BM25 query ...", end="", flush=True)
    step_start = time.time()
    bm25_base = build_bm25_query(protein_names, gene_names, organism)
    bm25_syn = build_bm25_query(all_proteins_syn, gene_names, organism)
    synonym_query = bm25_syn if bm25_syn != bm25_base else ""

    medcpt_queries = build_medcpt_queries(protein_names, gene_names, organism)
    if synonyms:
        medcpt_syn_queries = build_medcpt_queries(all_proteins_syn, gene_names, organism)
        # deduplicate
        seen = set(medcpt_queries)
        for q in medcpt_syn_queries:
            if q not in seen:
                medcpt_queries.append(q)
                seen.add(q)
    print(f" \u2713 {len(medcpt_queries)} MedCPT queries ({time.time() - step_start:.1f}s)")

    # ---- Step 3: BM25 multi-round search ----
    print("  [3/3] PubMed BM25 search ...", end="", flush=True)
    step_start = time.time()
    bm25_pmids = bm25_multi_round_search(
        query=bm25_base,
        synonym_query=synonym_query,
        top_k=retrieval_topk,
        email=config.entrez_email or "",
        api_key=config.entrez_api_key,
    )
    print(f" \u2713 {len(bm25_pmids)} PMIDs ({time.time() - step_start:.1f}s)")

    total_elapsed = time.time() - t_start
    print(f"\n  BM25 phase complete: {total_elapsed:.1f}s\n")

    return {
        "query": {
            "protein": protein_names,
            "gene": gene_names,
            "organism": organism,
        },
        "synonyms": synonyms,
        "all_proteins_syn": all_proteins_syn,
        "bm25_query_used": bm25_base,
        "bm25_query_syn": synonym_query,
        "medcpt_queries": medcpt_queries,
        "bm25_pmids": bm25_pmids,
        "retrieval_topk": retrieval_topk,
        "runtime": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(total_elapsed, 1),
        },
    }


# ---------------------------------------------------------------------------
# Phase 3: Annotate — RRF fusion → abstracts → evidence → generation
# ---------------------------------------------------------------------------

def run_annotate_phase(
    config: RuntimeConfig,
    protein_names: str,
    gene_names: str = "",
    organism: str = "",
    bm25_pmids: Optional[list[str]] = None,
    medcpt_results: Optional[list[dict]] = None,
    topk: int = 10,
    retrieval_topk: int = 100,
    skip_dense: bool = False,
    skip_quality: bool = False,
) -> dict:
    """Run the annotation phase: RRF fusion → abstracts → evidence → generation.

    Designed for execution on a login node (needs network for PubMed/OpenAlex
    APIs and LLM for evidence classification / generation).  Accepts
    pre-computed BM25 PMIDs and/or MedCPT results from earlier phases.

    Auto-detection logic:
      - Both bm25_pmids + medcpt_results → RRF rank-aware fusion.
      - Only bm25_pmids → BM25-only (flat score based on rank).
      - Neither → raises ValueError (run full pipeline or bm25 phase first).

    Args:
        config: RuntimeConfig with LLM and Entrez settings.
        protein_names: Comma/semicolon-separated protein names.
        gene_names: Comma/semicolon-separated gene names (optional).
        organism: Comma/semicolon-separated organism names.
        bm25_pmids: Pre-computed PMID list from BM25 phase.
        medcpt_results: Pre-computed MedCPT results (list of {"pmid":...,"score":...}).
        topk: Number of top papers for final generation.
        retrieval_topk: Max PMIDs for RRF fusion.
        skip_dense: If True, skip MedCPT (alias for setting medcpt_results=None).
        skip_quality: If True, skip OpenAlex paper quality scoring.

    Returns:
        dict with full annotation result (same schema as annotate_single_protein).
    """
    t_start = time.time()

    short_protein = protein_names.split(",")[0].strip()
    print(f"\n{'='*60}")
    print(f"  VirProtRAG Annotate Phase: {short_protein} ({organism})")
    print(f"{'='*60}")

    issues = config.validate()
    if issues:
        raise RuntimeError("Configuration error: " + "; ".join(issues))

    if skip_dense:
        medcpt_results = None

    if bm25_pmids is None and medcpt_results is None:
        raise ValueError(
            "At least one of bm25_pmids or medcpt_results must be provided. "
            "Run 'virprotrag bm25' first, or use the full 'virprotrag annotate' "
            "command without --bm25/--medcpt flags to run from scratch."
        )
    if bm25_pmids is None:
        bm25_pmids = []

    # Initialize cache
    cache = CacheManager(config.cache_path)
    cache.load()

    result = {
        "entry": "USER_PROVIDED",
        "query": {
            "protein": protein_names,
            "gene": gene_names,
            "organism": organism,
        },
        "generated_annotation": "",
        "supporting_evidence": [],
        "synonyms_found": [],
        "bm25_query_used": "",
        "pipeline_yield": {},
        "evidence_distribution": {},
        "runtime": {},
    }

    # ---- RRF fusion ----
    _step_print(1, "RRF fusion")
    step_start = time.time()
    if medcpt_results:
        fused = rrf_fusion_rank_aware(
            bm25_results=bm25_pmids,
            medcpt_results=medcpt_results,
            topk_bm25=len(bm25_pmids),
            topk_medcpt=len(medcpt_results),
        )
        print(f" \u2713 BM25+MedCPT RRF fusion: {len(fused)} unique PMIDs "
              f"({time.time() - step_start:.1f}s)")
    else:
        n = min(len(bm25_pmids), retrieval_topk)
        fused = {pmid: 1.0 - i / max(n, 1) for i, pmid in enumerate(bm25_pmids[:n])}
        print(f" \u2713 BM25-only: {len(fused)} PMIDs ({time.time() - step_start:.1f}s)")

    fused_normalized = normalize_scores(fused)
    logger.debug(f"  Fused {len(fused_normalized)} unique PMIDs")

    # ---- Fetch paper metadata ----
    _step_print(2, "Fetching paper abstracts")
    step_start = time.time()
    top_fused_pmids = [d["pmid"] for d in fused_normalized[:100]]
    abstracts = fetch_pubmed_abstracts(
        top_fused_pmids,
        email=config.entrez_email or "",
        api_key=config.entrez_api_key,
    )
    print(f" \u2713 {len(abstracts)} abstracts ({time.time() - step_start:.1f}s)")

    # ---- Paper quality (OpenAlex, optional) ----
    openalex_metrics = {}
    quality_count = 0
    if not skip_quality:
        _step_print(3, "Paper quality (OpenAlex)")
        step_start = time.time()

        oa_cached, oa_uncached = cache.get_openalex_batch(top_fused_pmids)
        openalex_metrics = dict(oa_cached)
        quality_count = len(oa_cached)

        if oa_uncached:
            try:
                fresh_metrics = fetch_openalex_metrics(
                    oa_uncached, email=config.openalex_email
                )
                for pmid, metrics in fresh_metrics.items():
                    cache.set_openalex(pmid, metrics)
                openalex_metrics.update(fresh_metrics)
                quality_count = len(openalex_metrics)
            except Exception as e:
                logger.warning(f"OpenAlex fetch failed: {e}")
                print(f"\n    OpenAlex scoring failed, continuing with {quality_count} cached scores.")

        cache_hint = f" ({len(oa_cached)} cached)" if oa_cached else ""
        print(f" \u2713 {quality_count} scored{cache_hint} ({time.time() - step_start:.1f}s)")
    else:
        logger.debug("Paper quality scoring skipped")

    # ---- Evidence classification ----
    step_idx = 4 if not skip_quality else 3
    _step_print(step_idx, "Evidence classification (LLM)")
    full_pubs = []
    for d in fused_normalized[:100]:
        pmid = d["pmid"]
        if pmid in abstracts:
            full_pubs.append({
                "pmid": pmid,
                "title": abstracts[pmid].get("title", ""),
                "abstract": abstracts[pmid].get("abstract", ""),
            })

    step_start = time.time()
    from .query_builder import _split_names
    protein_list = _split_names(protein_names)
    evidence_map = classify_evidence(
        config.llm,
        protein_list[0] if protein_list else protein_names,
        full_pubs,
        cache=cache,
    )
    evid_dist = {"EXPERIMENTAL": 0, "INFERRED": 0, "NONE": 0}
    for _, v in evidence_map.items():
        label = v.get("label_name", "NONE")
        if label in evid_dist:
            evid_dist[label] += 1
    result["evidence_distribution"] = evid_dist
    print(f" \u2713 {len(evidence_map)} papers classified "
          f"(EXP={evid_dist['EXPERIMENTAL']}, INF={evid_dist['INFERRED']}, "
          f"NONE={evid_dist['NONE']}) ({time.time() - step_start:.1f}s)")

    # ---- Re-ranking ----
    step_idx += 1
    _step_print(step_idx, "Re-ranking")
    step_start = time.time()
    reranked = rerank(
        fused_results=fused_normalized[:100],
        abstracts=abstracts,
        evidence_map=evidence_map,
        openalex_metrics=openalex_metrics if openalex_metrics else None,
    )
    print(f" \u2713 {len(reranked)} candidates ranked ({time.time() - step_start:.1f}s)")

    # ---- Generate annotation ----
    step_idx += 1
    _step_print(step_idx, "Generating annotation (LLM)")
    step_start = time.time()
    top_pubs = reranked[:topk]
    gen_result = generate_annotation(
        config.llm,
        protein_names=protein_names,
        gene_names=gene_names,
        organism=organism,
        publications=top_pubs,
    )

    result["generated_annotation"] = gen_result.get("Summary", str(gen_result))
    result["supporting_evidence"] = [
        {
            "pmid": p["pmid"],
            "title": p.get("title", ""),
            "evidence_label": p.get("evidence_label", ""),
            "final_score": round(p.get("final_score", 0), 4),
            "semantic_score": round(p.get("semantic_score", 0), 4),
            "evidence_score": round(p.get("evidence_score", 0), 4),
            "quality_score": round(p.get("quality_score", 0), 4),
        }
        for p in top_pubs
    ]
    print(f" \u2713 done ({time.time() - step_start:.1f}s)")

    # ---- Pipeline yield ----
    result["pipeline_yield"] = {
        "bm25_retrieved": len(bm25_pmids),
        "medcpt_retrieved": len(medcpt_results) if medcpt_results else 0,
        "rrf_fused": len(fused_normalized),
        "abstracts_fetched": len(abstracts),
        "evidence_classified": len(evidence_map),
        "reranked": len(reranked),
        "used_for_generation": topk,
    }

    # ---- Runtime ----
    total_elapsed = time.time() - t_start
    result["runtime"] = {
        "model_synonym": "n/a (pre-computed)",
        "model_evidence": config.llm.get_model("evidence"),
        "model_generation": config.llm.get_model("generation"),
        "provider": config.llm.provider,
        "skip_dense": medcpt_results is None,
        "skip_quality": skip_quality,
        "topk": topk,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(total_elapsed, 1),
    }

    cache.save()
    cache.print_stats()

    print(f"\n  Annotate phase complete: {total_elapsed:.1f}s\n")
    return result


# ---------------------------------------------------------------------------
# Legacy: full pipeline (backward compatible)
# ---------------------------------------------------------------------------


def annotate_single_protein(
    config: RuntimeConfig,
    protein_names: str,
    gene_names: str = "",
    organism: str = "",
    topk: int = 10,
    retrieval_topk: int = 100,
    skip_dense: bool = False,
    skip_quality: bool = False,
) -> dict:
    """Run the full VirProtRAG pipeline for a single viral protein.

    Backward-compatible wrapper that runs BM25 phase + optional MedCPT +
    annotation phase in one shot.  If you are on an HPC where MedCPT needs
    to be split to a compute node, use:

        virprotrag bm25 ...        # login node
        virprotrag medcpt ...      # compute node (SGE/qsub)
        virprotrag annotate ...    # login node

    Args:
        config: Runtime configuration (LLM, Entrez, MedCPT paths, etc.).
        protein_names: Comma/semicolon-separated protein names.
        gene_names: Comma/semicolon-separated gene names (optional).
        organism: Comma/semicolon-separated organism names.
        topk: Number of top papers to use for final generation.
        retrieval_topk: Maximum PMIDs to retrieve per search engine (default 100).
        skip_dense: If True, skip MedCPT and use BM25 only.
        skip_quality: If True, skip OpenAlex paper quality scoring.

    Returns:
        dict with full annotation result.
    """
    t_start = time.time()

    short_protein = protein_names.split(",")[0].strip()
    print(f"\n{'='*60}")
    print(f"  VirProtRAG: {short_protein} ({organism})")
    print(f"{'='*60}")

    # ---- Phase 1: BM25 ----
    bm25_output = run_bm25_phase(
        config=config,
        protein_names=protein_names,
        gene_names=gene_names,
        organism=organism,
        retrieval_topk=retrieval_topk,
    )

    # ---- Phase 2: MedCPT (optional) ----
    medcpt_results = None
    if not skip_dense and config.has_medcpt:
        print("  [MedCPT] Dense retrieval ...", end="", flush=True)
        step_start = time.time()
        try:
            medcpt_results = medcpt_search(
                queries=bm25_output["medcpt_queries"],
                index_path=config.medcpt_index_path,
                pmids_path=config.medcpt_pmids_path,
                topk=retrieval_topk,
            )
            print(f" \u2713 {len(medcpt_results)} PMIDs ({time.time() - step_start:.1f}s)")
        except Exception as e:
            logger.warning(f"MedCPT search failed: {e}. Falling back to BM25 only.")
            print(f"\n    MedCPT search failed, continuing with BM25 only.")

    # ---- Phase 3: Annotate ----
    result = run_annotate_phase(
        config=config,
        protein_names=protein_names,
        gene_names=gene_names,
        organism=organism,
        bm25_pmids=bm25_output["bm25_pmids"],
        medcpt_results=medcpt_results,
        topk=topk,
        retrieval_topk=retrieval_topk,
        skip_quality=skip_quality,
    )

    # Enrich with synonym info from BM25 phase
    result["synonyms_found"] = bm25_output.get("synonyms", [])
    result["bm25_query_used"] = bm25_output.get("bm25_query_used", "")
    result["runtime"]["model_synonym"] = config.llm.get_model("synonym")

    total_elapsed = time.time() - t_start
    result["runtime"]["elapsed_seconds"] = round(total_elapsed, 1)

    print(f"  Total time: {total_elapsed:.1f}s\n")
    return result
