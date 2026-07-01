"""
Command-line interface for VirProtRAG.
Provides five entry points: bm25, medcpt (phases); annotate, batch, search-fasta (subcommands).
"""

import argparse
import json
import logging
import os
import sys
import csv

from .config import RuntimeConfig
from .search_fasta import search_and_annotate

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _write_report(result: dict, report_path: str):
    """Write a clean human-readable Markdown report alongside the JSON."""
    q = result.get("query", {})
    ev = result.get("supporting_evidence", [])
    run = result.get("runtime", {})

    lines = []
    lines.append(f"# VirProtRAG Annotation Report\n")
    lines.append(f"**Protein:** {q.get('protein', 'N/A')}  ")
    lines.append(f"**Gene:** {q.get('gene', 'N/A')}  ")
    lines.append(f"**Organism:** {q.get('organism', 'N/A')}  ")
    lines.append(f"**Date:** {run.get('timestamp_utc', 'N/A')[:19]}  ")
    lines.append(f"**Pipeline:** {run.get('provider', '?')}/{run.get('model_generation', '?')} | "
                 f"topk={run.get('topk', '?')} | elapsed={run.get('elapsed_seconds', '?')}s\n")

    # --- Annotation ---
    lines.append("---\n")
    lines.append("## Generated Annotation\n")
    lines.append(result.get("generated_annotation", "*No annotation generated.*"))
    lines.append("")

    # --- Evidence table ---
    lines.append("---\n")
    lines.append("## Supporting Evidence\n")
    if ev:
        lines.append("| # | PMID | Title | Evidence | Score |")
        lines.append("|---|------|-------|----------|-------|")
        for i, e in enumerate(ev, 1):
            pmid = e.get("pmid", "")
            title = e.get("title", "")[:120]
            label = e.get("evidence_label", "NONE")
            # Visual indicator for evidence strength
            if label == "EXPERIMENTAL":
                label_display = "🧪 EXPERIMENTAL"
            elif label == "INFERRED":
                label_display = "📖 INFERRED"
            else:
                label_display = "— NONE"
            score = e.get("final_score", 0)
            lines.append(f"| {i} | [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}) | {title} | {label_display} | {score:.4f} |")
    else:
        lines.append("*No supporting evidence found.*")
    lines.append("")


    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄 Report saved to {report_path}")


def cmd_bm25(args):
    """Entry: BM25 retrieval phase (synonyms + PubMed search).

    Runs on login node. Saves queries, synonyms, and BM25 results to a JSON
    file that can be consumed by 'virprotrag medcpt' and 'virprotrag annotate'.
    """
    from .pipeline import run_bm25_phase  # lazy import (needs biopython)

    _setup_logging(args.verbose)

    config = RuntimeConfig.from_env(model=args.model)
    issues = config.validate()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease set the required environment variables. See .env.example for guidance.")
        sys.exit(1)

    result = run_bm25_phase(
        config=config,
        protein_names=args.protein,
        gene_names=args.gene or "",
        organism=args.organism,
        retrieval_topk=args.retrieval_topk,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ BM25 results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_medcpt(args):
    """Entry: MedCPT dense retrieval from BM25 output.

    Pure compute, zero network dependency. Reads queries from the BM25-phase
    JSON file and runs FAISS dense retrieval.  Designed for SGE/qsub compute
    nodes where internet is unavailable but FAISS index is on shared storage.
    """
    from .medcpt_search import medcpt_search_from_bm25_json  # lazy import (needs torch/faiss)

    _setup_logging(args.verbose)

    # MedCPT only needs index paths, not LLM or Entrez
    index_path = args.index_path or os.getenv("MEDCPT_FAISS_INDEX_PATH")
    pmids_path = args.pmids_path or os.getenv("MEDCPT_PMIDS_PATH")

    if not index_path or not os.path.isfile(index_path):
        print(f"❌ MedCPT FAISS index not found.")
        print(f"   Expected at: {index_path or 'MEDCPT_FAISS_INDEX_PATH (not set)'}")
        print(f"   Set MEDCPT_FAISS_INDEX_PATH in .env or use --index-path.")
        sys.exit(1)
    if not pmids_path or not os.path.isfile(pmids_path):
        print(f"❌ MedCPT PMIDs mapping not found.")
        print(f"   Expected at: {pmids_path or 'MEDCPT_PMIDS_PATH (not set)'}")
        print(f"   Set MEDCPT_PMIDS_PATH in .env or use --pmids-path.")
        sys.exit(1)

    result = medcpt_search_from_bm25_json(
        bm25_json_path=args.input,
        index_path=index_path,
        pmids_path=pmids_path,
        topk=args.retrieval_topk,
        batch_size=args.batch_size,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ MedCPT results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_annotate(args):
    """Entry: Annotation phase — generates functional annotation.

    Supports three modes:
      1. Full pipeline (no --bm25/--medcpt flags): backward-compatible, runs
         BM25 + optional MedCPT + annotation from scratch.
      2. With --bm25 (and optional --medcpt): loads pre-computed retrieval
         results and skips directly to RRF fusion → annotation.
    """
    from .pipeline import annotate_single_protein, run_annotate_phase  # lazy import (needs biopython)
    _setup_logging(args.verbose)

    config = RuntimeConfig.from_env(model=args.model)

    # --- Mode 2: Load pre-computed results ---
    if args.bm25:
        if not os.path.isfile(args.bm25):
            print(f"❌ BM25 input file not found: {args.bm25}")
            sys.exit(1)

        with open(args.bm25, "r", encoding="utf-8") as f:
            bm25_data = json.load(f)

        bm25_pmids = bm25_data.get("bm25_pmids", [])
        protein = bm25_data["query"]["protein"]
        gene = bm25_data["query"].get("gene", "")
        organism = bm25_data["query"]["organism"]

        # Load optional MedCPT results
        medcpt_results = None
        if args.medcpt:
            if not os.path.isfile(args.medcpt):
                print(f"❌ MedCPT input file not found: {args.medcpt}")
                sys.exit(1)
            with open(args.medcpt, "r", encoding="utf-8") as f:
                medcpt_data = json.load(f)
            medcpt_results = medcpt_data.get("medcpt_results", [])
            if medcpt_results:
                print(f"📎 Loaded {len(medcpt_results)} MedCPT results from {args.medcpt}")
            else:
                print(f"⚠️  MedCPT file has no results; falling back to BM25-only.")

        if not bm25_pmids:
            print("❌ BM25 input file contains no PMIDs.")
            sys.exit(1)

        result = run_annotate_phase(
            config=config,
            protein_names=protein,
            gene_names=gene,
            organism=organism,
            bm25_pmids=bm25_pmids,
            medcpt_results=medcpt_results,
            topk=args.topk,
            retrieval_topk=bm25_data.get("retrieval_topk", 100),
            skip_quality=args.skip_quality,
        )

    # --- Mode 1: Full pipeline (backward compatible) ---
    else:
        issues = config.validate()
        if issues:
            print("Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nPlease set the required environment variables. See .env.example for guidance.")
            sys.exit(1)

        if args.check:
            print("Running preflight checks...")
            pf = config.preflight_check()
            if pf["status"] == "error":
                print("\nPreflight check FAILED:")
                for issue in pf["issues"]:
                    print(f"  \u2717 {issue}")
                sys.exit(1)
            if pf["warnings"]:
                for w in pf["warnings"]:
                    print(f"  \u26a0 {w}")
            if pf["status"] == "ok":
                print("  \u2713 All connectivity checks passed.\n")

        result = annotate_single_protein(
            config=config,
            protein_names=args.protein,
            gene_names=args.gene or "",
            organism=args.organism,
            topk=args.topk,
            skip_dense=args.skip_dense,
            skip_quality=args.skip_quality,
        )

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON saved to {args.output}")

        # Auto-generate companion Markdown report
        report_path = args.output.replace(".json", "_report.md")
        if report_path == args.output:
            report_path = args.output + "_report.md"
        _write_report(result, report_path)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Batch mode helpers — shared CSV reading
# ---------------------------------------------------------------------------

def _batch_read_input(input_path: str) -> list[dict]:
    """Read and validate TSV input, return normalized rows.

    Uses tab-separated values (TSV). Within a single field, use commas
    to separate multiple values (e.g., protein_name values).

    Raises:
        ValueError: on missing file, empty content, or missing required columns.
    """
    if not os.path.exists(input_path):
        raise ValueError(f"Input file not found: {input_path}")

    sep = "	"
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=sep)
        if not reader.fieldnames:
            raise ValueError("Input file has no header row.")
        rows = list(reader)

    if not rows:
        raise ValueError("Input file contains no data rows.")

    required = {"protein_name", "organism"}
    file_cols = set(c.lower() for c in reader.fieldnames)
    missing = required - file_cols
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: protein_name, organism. Optional: gene_name, entry")

    norm_rows = []
    for row in rows:
        norm = {}
        for k, v in row.items():
            kl = k.lower().replace(" ", "_")
            if kl in ("protein_name", "protein_names"):
                norm["protein_name"] = v
            elif kl in ("gene_name", "gene_names"):
                norm["gene_name"] = v
            elif kl in ("organism", "organism_name"):
                norm["organism"] = v
            elif kl in ("entry", "entry_id", "id"):
                norm["entry"] = v
        norm_rows.append(norm)
    return norm_rows


# ---------------------------------------------------------------------------
# Batch BM25 phase
# ---------------------------------------------------------------------------

def _batch_bm25(config, input_path, output_dir, retrieval_topk, resume):
    """Run BM25 phase for every row in a TSV file.

    Saves ``{entry}_bm25.json`` for each protein in *output_dir*.
    """
    from .pipeline import run_bm25_phase  # lazy import (needs biopython)
    rows = _batch_read_input(input_path)
    os.makedirs(output_dir, exist_ok=True)

    import glob as _glob
    done = set()
    if resume:
        for f in _glob.glob(os.path.join(output_dir, "*_bm25.json")):
            done.add(os.path.basename(f).replace("_bm25.json", ""))
        if done:
            print(f"🔁 Resuming: {len(done)} entries already have BM25 output.")

    total = len(rows)
    for i, row in enumerate(rows):
        entry_id = (row.get("entry") or row.get("protein_name") or f"row_{i}").strip()
        if entry_id in done:
            continue

        protein = row.get("protein_name", "").strip()
        gene = row.get("gene_name", "").strip()
        organism = row.get("organism", "").strip()

        if not protein or not organism:
            print(f"⚠️  [{i+1}/{total}] {entry_id}: skipping (missing protein/organism)")
            continue

        print(f"[{i+1}/{total}] BM25: {entry_id} — {protein} ({organism})")

        result = run_bm25_phase(
            config=config,
            protein_names=protein,
            gene_names=gene,
            organism=organism,
            retrieval_topk=retrieval_topk,
        )

        out_path = os.path.join(output_dir, f"{entry_id}_bm25.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  → {out_path}")

    remaining = total - len(done)
    print(f"✅ Batch BM25 complete. {remaining} files in {output_dir}/")


# ---------------------------------------------------------------------------
# Batch MedCPT phase  (loads FAISS + encoder once → processes all files)
# ---------------------------------------------------------------------------

def _batch_medcpt(input_dir, output_dir, retrieval_topk, batch_size,
                  index_path, pmids_path, resume):
    """Run MedCPT dense retrieval for all *_bm25.json files in *input_dir*.

    Loads the FAISS index and MedCPT encoder **once** — suitable for a single
    SGE compute-node job processing many proteins.
    """
    import glob as _glob
    import time
    from datetime import datetime, timezone

    index_path = index_path or os.getenv("MEDCPT_FAISS_INDEX_PATH")
    pmids_path = pmids_path or os.getenv("MEDCPT_PMIDS_PATH")

    if not index_path or not os.path.isfile(index_path):
        print(f"❌ MedCPT FAISS index not found: {index_path}")
        sys.exit(1)
    if not pmids_path or not os.path.isfile(pmids_path):
        print(f"❌ MedCPT PMIDs mapping not found: {pmids_path}")
        sys.exit(1)

    bm25_files = sorted(_glob.glob(os.path.join(input_dir, "*_bm25.json")))
    if not bm25_files:
        print(f"❌ No *_bm25.json files found in {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    done = set()
    if resume:
        for f in _glob.glob(os.path.join(output_dir, "*_medcpt.json")):
            done.add(os.path.basename(f).replace("_medcpt.json", ""))
        if done:
            print(f"🔁 Resuming: {len(done)} entries already have MedCPT output.")

    # --- Load FAISS index (once) ---
    import faiss
    import numpy as np  # noqa: F811
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    with open(pmids_path, "r") as f:
        pmids = json.load(f)
    logger.info(f"Index: {index.ntotal} vectors, PMIDs: {len(pmids)} entries")

    # --- Load encoder (once) ---
    logger.info("Loading MedCPT query encoder (local_files_only)...")
    model = AutoModel.from_pretrained(
        "ncbi/MedCPT-Query-Encoder", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "ncbi/MedCPT-Query-Encoder", local_files_only=True
    )
    model.eval()

    # --- Process each BM25 file ---
    for fpath in bm25_files:
        entry_id = os.path.basename(fpath).replace("_bm25.json", "")
        if entry_id in done:
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            bm25_data = json.load(f)

        queries = bm25_data.get("medcpt_queries", [])
        if not queries:
            print(f"⚠️  {entry_id}: no medcpt_queries, skipping")
            continue

        protein = bm25_data.get("query", {}).get("protein", "unknown")
        print(f"MedCPT: {entry_id} — {protein} ({len(queries)} queries)")

        t_start = time.time()
        best_score: dict[str, float] = {}

        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            encoded = tokenizer(
                batch, truncation=True, padding=True, return_tensors="pt",
                max_length=512,
            )
            with torch.no_grad():
                query_embeds = (
                    model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
                )
            scores, indices = index.search(query_embeds, k=retrieval_topk)
            for qi in range(len(batch)):
                for j in range(retrieval_topk):
                    pmid = pmids[indices[qi][j]]
                    score = float(scores[qi][j])
                    if pmid not in best_score or score > best_score[pmid]:
                        best_score[pmid] = score

        results = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
        medcpt_results = [{"pmid": p, "score": s} for p, s in results]

        result = {
            "source": fpath,
            "num_queries": len(queries),
            "medcpt_results": medcpt_results,
            "retrieved_topk": retrieval_topk,
            "runtime": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.time() - t_start, 1),
            },
        }

        out_path = os.path.join(output_dir, f"{entry_id}_medcpt.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  → {len(medcpt_results)} PMIDs → {out_path}")

    remaining = len(bm25_files) - len(done)
    print(f"✅ Batch MedCPT complete. {remaining} files in {output_dir}/")


# ---------------------------------------------------------------------------
# Batch annotate phase
# ---------------------------------------------------------------------------

def _batch_annotate(config, input_dir, output_file, topk, skip_quality,
                    medcpt_dir, resume):
    """Run annotation phase for all *_bm25.json files in *input_dir*.

    Pairs each BM25 JSON with an optional MedCPT JSON from *medcpt_dir*.
    """
    from .pipeline import annotate_single_protein, run_annotate_phase  # lazy import (needs biopython)
    import glob as _glob

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    bm25_files = sorted(_glob.glob(os.path.join(input_dir, "*_bm25.json")))
    if not bm25_files:
        print(f"❌ No *_bm25.json files found in {input_dir}")
        sys.exit(1)

    # Resume support
    done_entries = set()
    if resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                e = row.get("entry", "").strip()
                if e:
                    done_entries.add(e)
        print(f"🔁 Resuming: {len(done_entries)} entries already completed.")

    for fpath in bm25_files:
        entry_id = os.path.basename(fpath).replace("_bm25.json", "")
        if entry_id in done_entries:
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            bm25_data = json.load(f)

        query_info = bm25_data.get("query", {})
        protein = query_info.get("protein", "")
        gene = query_info.get("gene", "")
        organism = query_info.get("organism", "")
        bm25_pmids = bm25_data.get("bm25_pmids", [])

        if not bm25_pmids:
            print(f"⚠️  {entry_id}: no BM25 PMIDs, skipping")
            continue

        # Check for optional MedCPT
        medcpt_results = None
        if medcpt_dir:
            mp = os.path.join(medcpt_dir, f"{entry_id}_medcpt.json")
            if os.path.isfile(mp):
                with open(mp, "r", encoding="utf-8") as f:
                    md = json.load(f)
                medcpt_results = md.get("medcpt_results", [])

        print(f"Annotate: {entry_id} — {protein} ({organism})"
              + (" [BM25+MedCPT]" if medcpt_results else " [BM25-only]"))

        result = run_annotate_phase(
            config=config,
            protein_names=protein,
            gene_names=gene,
            organism=organism,
            bm25_pmids=bm25_pmids,
            medcpt_results=medcpt_results,
            topk=topk,
            retrieval_topk=bm25_data.get("retrieval_topk", 100),
            skip_quality=skip_quality,
        )

        new_row = {
            "entry": entry_id,
            "protein_name": protein,
            "gene_name": gene,
            "organism": organism,
            "generated_annotation": result.get("generated_annotation", ""),
            "supporting_pmids": ";".join(
                ev["pmid"] for ev in result.get("supporting_evidence", [])
            ),
        }
        _write_tsv(output_file, [new_row], done_entries)
        done_entries.add(entry_id)

    print(f"✅ Batch annotate complete. Results in {output_file}")


# ---------------------------------------------------------------------------
# cmd_batch — dispatcher
# ---------------------------------------------------------------------------

def cmd_batch(args):
    """Batch processing entry point.

    Without --phase: all-in-one pipeline (BM25 + optional MedCPT + annotate)
    per row, producing a single annotations.tsv.

    With --phase bm25|medcpt|annotate: only runs the specified pipeline phase
    across all rows, producing per-protein intermediate JSON files.
    """
    from .pipeline import annotate_single_protein, run_bm25_phase, run_annotate_phase  # lazy import
    from .medcpt_search import medcpt_search_from_bm25_json  # lazy import (needs torch/faiss)
    _setup_logging(args.verbose)

    # ---- Phased batch ----
    if args.phase == "bm25":
        config = RuntimeConfig.from_env(model=args.model)
        issues = config.validate()
        if issues:
            print("❌ Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        _batch_bm25(
            config=config,
            input_path=args.input,
            output_dir=args.output or "./batch_bm25/",
            retrieval_topk=getattr(args, "retrieval_topk", 100),
            resume=args.resume,
        )
        return

    if args.phase == "medcpt":
        _batch_medcpt(
            input_dir=args.input,
            output_dir=args.output or "./batch_medcpt/",
            retrieval_topk=getattr(args, "retrieval_topk", 100),
            batch_size=getattr(args, "batch_size", 2048),
            index_path=getattr(args, "index_path", None),
            pmids_path=getattr(args, "pmids_path", None),
            resume=args.resume,
        )
        return

    if args.phase == "annotate":
        config = RuntimeConfig.from_env(model=args.model)
        issues = config.validate()
        if issues:
            print("❌ Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        if args.output and args.output.endswith(".tsv"):
            output_file = args.output
        else:
            output_dir = args.output or "./batch_results/"
            output_file = os.path.join(output_dir, "annotations.tsv")
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        _batch_annotate(
            config=config,
            input_dir=args.input,
            output_file=output_file,
            topk=getattr(args, "topk", 10),
            skip_quality=getattr(args, "skip_quality", False),
            medcpt_dir=getattr(args, "medcpt_dir", None),
            resume=args.resume,
        )
        return

    # ---- All-in-one batch (existing behavior, no --phase) ----
    config = RuntimeConfig.from_env(model=args.model)
    issues = config.validate()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease set the required environment variables. See .env.example for guidance.")
        sys.exit(1)

    # Optional preflight connectivity check
    if args.check:
        print("Running preflight checks...")
        pf = config.preflight_check()
        if pf["status"] == "error":
            print("\nPreflight check FAILED:")
            for issue in pf["issues"]:
                print(f"  \u2717 {issue}")
            sys.exit(1)
        if pf["warnings"]:
            for w in pf["warnings"]:
                print(f"  \u26a0 {w}")
        if pf["status"] == "ok":
            print("  \u2713 All connectivity checks passed.\n")

    try:
        norm_rows = _batch_read_input(args.input)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Setup output
    if args.output and args.output.endswith(".tsv"):
        output_file = args.output
    else:
        output_dir = args.output or "./virprotrag_output/"
        output_file = os.path.join(output_dir, "annotations.tsv")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Check for resume
    done_entries = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                entry = row.get("entry", "").strip()
                if entry:
                    done_entries.add(entry)
        print(f"🔁 Resuming: {len(done_entries)} entries already completed.")

    pending = [r for r in norm_rows if (r.get("entry") or r.get("protein_name") or "").strip() not in done_entries]
    print(f"🧬 Processing {len(pending)}/{len(norm_rows)} entries...")

    for i, row in enumerate(pending):
        entry_id = (row.get("entry") or row.get("protein_name") or f"row_{i}").strip()
        protein = row.get("protein_name", "").strip()
        gene = row.get("gene_name", "").strip()
        organism = row.get("organism", "").strip()

        if not protein or not organism:
            logger.warning(f"Skipping row {i}: missing protein_name or organism")
            continue

        print(f"[{i+1}/{len(pending)}] {entry_id}: {protein} ({organism})")
        result = annotate_single_protein(
            config=config,
            protein_names=protein,
            gene_names=gene,
            organism=organism,
            topk=args.topk,
            skip_dense=args.skip_dense,
            skip_quality=args.skip_quality,
        )

        new_row = {
            "entry": entry_id,
            "protein_name": protein,
            "gene_name": gene,
            "organism": organism,
            "generated_annotation": result.get("generated_annotation", ""),
            "supporting_pmids": ";".join(
                ev["pmid"] for ev in result.get("supporting_evidence", [])
            ),
        }

        # Flush to file after every protein
        _write_tsv(output_file, [new_row], done_entries)

    print(f"✅ Batch complete. Results saved to {output_file}")


def _write_tsv(output_file: str, results: list[dict], done_entries: set):
    """Append results to TSV output file."""
    fieldnames = ["entry", "protein_name", "gene_name", "organism", "generated_annotation", "supporting_pmids"]
    write_header = not os.path.exists(output_file)
    with open(output_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# search-fasta command
# ---------------------------------------------------------------------------

def cmd_search_fasta(args):
    """Entry 4: Align protein FASTA to viral DB and cross-reference annotations."""
    _setup_logging(args.verbose)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    db_prefix = args.db or os.path.join(data_dir, "viral_proteins")
    uniprot_tsv = args.uniprot_tsv
    if not uniprot_tsv:
        # Auto-detect the UniProt TSV in data dir
        import glob as _glob
        tsv_files = _glob.glob(os.path.join(data_dir, "uniprotkb_*.tsv"))
        uniprot_tsv = tsv_files[0] if tsv_files else ""
    virprotrag_db = args.virprotrag_db or os.path.join(data_dir, "ViProtRAG_database.jsonl")
    seqbench_db = args.seqbench_db or os.path.join(data_dir, "SeqBench.jsonl")
    output_tsv = args.output or "search_results.tsv"

    # Validate files exist
    for label, path in [
        ("Query FASTA", args.query),
        ("DIAMOND DB", db_prefix + ".dmnd"),
        ("UniProt TSV", uniprot_tsv),
        ("VirProtRAG DB", virprotrag_db),
        ("SeqBench DB", seqbench_db),
    ]:
        if not os.path.exists(path):
            print(f"⚠️  {label} not found: {path}")
            if label == "DIAMOND DB":
                print("   Run 'diamond makedb --in viral_proteins.fasta -d viral_proteins' first.")
                sys.exit(1)

    config = RuntimeConfig.from_env()
    diamond_bin = config.diamond_path

    search_and_annotate(
        query_fasta=args.query,
        db_prefix=db_prefix,
        uniprot_tsv=uniprot_tsv,
        virprotrag_jsonl=virprotrag_db,
        seqbench_jsonl=seqbench_db,
        output_tsv=output_tsv,
        diamond_bin=diamond_bin,
        threads=args.threads,
        min_identity=args.min_identity,
        max_evalue=args.evalue,
        min_qcov=args.min_qcov,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="virprotrag",
        description="VirProtRAG: Literature-grounded viral protein function annotation",
    )
    # --- Top-level --phase for single-protein phased mode ---
    # When --phase is set without a subcommand, the single-protein phased pipeline runs.
    # With the 'batch' subcommand, --phase triggers batch-phased mode.
    parser.add_argument("--phase", choices=["bm25", "medcpt", "annotate"],
                        help="Run a single pipeline phase (single-protein mode; "
                             "or combine with 'batch' subcommand for batch-phased)")

    # Single-protein phased-mode arguments (also shared by annotate subcommand)
    parser.add_argument("--protein", help="Protein name(s), comma-separated")
    parser.add_argument("--gene", default="", help="Gene name(s), comma-separated")
    parser.add_argument("--organism", help="Organism name(s), comma-separated")
    parser.add_argument("--input", help="Input JSON (medcpt phase) or pre-computed BM25 JSON")
    parser.add_argument("--bm25", help="Path to BM25-phase JSON (annotate phase, pre-computed input)")
    parser.add_argument("--medcpt", help="Path to MedCPT-phase JSON (annotate phase, enables RRF fusion)")
    parser.add_argument("--output", help="Output file path (JSON or TSV depending on phase)")
    parser.add_argument("--topk", type=int, default=10, help="Top papers for generation (default: 10)")
    parser.add_argument("--retrieval-topk", type=int, default=100, help="Max PMIDs per retrieval (default: 100)")
    parser.add_argument("--batch-size", type=int, default=2048, help="Encoding batch size for MedCPT (default: 2048)")
    parser.add_argument("--index-path", help="MedCPT FAISS index path (only for --phase medcpt, overrides MEDCPT_FAISS_INDEX_PATH env)")
    parser.add_argument("--pmids-path", help="MedCPT PMIDs mapping path (only for --phase medcpt, overrides MEDCPT_PMIDS_PATH env)")
    parser.add_argument("--skip-dense", action="store_true", help="Skip MedCPT (full pipeline mode)")
    parser.add_argument("--skip-quality", action="store_true", help="Skip OpenAlex quality scoring")
    parser.add_argument("--model", help="LLM model name (overrides LLM_MODEL env var and provider defaults)")
    parser.add_argument("--check", action="store_true", help="Preflight connectivity checks")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--resume", action="store_true", help="Resume from interrupted run (batch mode)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- Entry: annotate (all-in-one only) ----
    p1 = subparsers.add_parser("annotate", help="Annotation phase, or full pipeline if no --bm25 flag")
    # Full pipeline mode (used when --bm25 is NOT provided)
    p1.add_argument("--protein", help="Protein name(s), comma-separated (required without --bm25)")
    p1.add_argument("--gene", default="", help="Gene name(s), comma-separated")
    p1.add_argument("--organism", help="Organism name(s), comma-separated (required without --bm25)")
    p1.add_argument("--skip-dense", action="store_true", help="Skip MedCPT (full pipeline mode only)")
    p1.add_argument("--check", action="store_true", help="Run preflight checks (full pipeline mode only)")
    # Pre-computed input mode
    p1.add_argument("--bm25", help="Path to BM25-phase JSON output (skips BM25 step)")
    p1.add_argument("--medcpt", help="Path to MedCPT-phase JSON output (enables RRF fusion)")
    # Common options
    p1.add_argument("--topk", type=int, default=10, help="Number of top papers for generation (default: 10)")
    p1.add_argument("--skip-quality", action="store_true", help="Skip OpenAlex paper quality scoring")
    p1.add_argument("--model", help="LLM model name (overrides LLM_MODEL env var and provider defaults)")
    p1.add_argument("--output", help="Output JSON file path (default: stdout)")
    p1.add_argument("--verbose", action="store_true", help="Verbose logging")

    # ---- Entry 2: batch ----
    p3 = subparsers.add_parser("batch", help="Batch processing from TSV (all-in-one or phased)")
    p3.add_argument("--phase", choices=["bm25", "medcpt", "annotate"],
                    help="Run only this pipeline phase (omit for all-in-one)")
    # Input / output
    p3.add_argument("--input", required=True, help="Path to input TSV file (or directory of JSONs for medcpt/annotate phases)")
    p3.add_argument("--output", help="Output directory or TSV path (default: ./virprotrag_output/)")
    # All-in-one options (ignored in phased mode)
    p3.add_argument("--topk", type=int, default=10, help="Number of top papers for generation (default: 10)")
    p3.add_argument("--skip-dense", action="store_true", help="Skip MedCPT (all-in-one mode only)")
    p3.add_argument("--skip-quality", action="store_true", help="Skip OpenAlex paper quality scoring")
    p3.add_argument("--model", help="LLM model name (overrides LLM_MODEL env var and provider defaults)")
    p3.add_argument("--resume", action="store_true", help="Resume from previous interrupted run")
    p3.add_argument("--verbose", action="store_true", help="Verbose logging")
    p3.add_argument("--check", action="store_true", help="Run preflight API connectivity checks (all-in-one mode only)")
    # Phase-specific options
    p3.add_argument("--retrieval-topk", type=int, default=100, help="Max PMIDs per retrieval (bm25/medcpt phases)")
    p3.add_argument("--batch-size", type=int, default=2048, help="Encoding batch size (medcpt phase)")
    p3.add_argument("--index-path", help="MedCPT FAISS index path (only for --phase medcpt, overrides MEDCPT_FAISS_INDEX_PATH env)")
    p3.add_argument("--pmids-path", help="MedCPT PMIDs mapping path (only for --phase medcpt, overrides MEDCPT_PMIDS_PATH env)")
    p3.add_argument("--medcpt-dir", help="Directory of *_medcpt.json files (annotate phase, enables RRF fusion)")

    # ---- Entry 3: search-fasta ----
    p4 = subparsers.add_parser("search-fasta", help="Align protein FASTA to viral DB and cross-reference annotations")
    p4.add_argument("--query", required=True, help="Path to query protein FASTA file")
    p4.add_argument("--db", help="DIAMOND database prefix (default: data/viral_proteins)")
    p4.add_argument("--uniprot-tsv", help="UniProt annotation TSV (default: data/uniprotkb_*.tsv)")
    p4.add_argument("--virprotrag-db", help="VirProtRAG database JSONL (default: data/ViProtRAG_database.jsonl)")
    p4.add_argument("--seqbench-db", help="SeqBench database JSONL (default: data/SeqBench.jsonl)")
    p4.add_argument("--output", help="Output TSV file path (default: search_results.tsv)")
    p4.add_argument("--threads", type=int, default=8, help="DIAMOND threads (default: 8)")
    p4.add_argument("--min-identity", type=float, default=40.0, help="Min identity %% (default: 40)")
    p4.add_argument("--evalue", type=float, default=1e-5, help="Max e-value (default: 1e-5)")
    p4.add_argument("--min-qcov", type=float, default=40.0, help="Min query coverage %% (default: 40)")
    p4.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # ---- Single-protein phased mode (--phase without subcommand) ----
    if args.phase and not args.command:
        if args.phase == "bm25":
            cmd_bm25(args)
        elif args.phase == "medcpt":
            cmd_medcpt(args)
        elif args.phase == "annotate":
            cmd_annotate(args)
        return

    # ---- Legacy subcommand dispatch (annotate / batch) ----
    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "search-fasta":
        cmd_search_fasta(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
