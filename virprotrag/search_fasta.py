"""
search_fasta.py — Case 3: Protein FASTA alignment + cross-reference annotation lookup.

User provides a protein FASTA file. We run DIAMOND blastp against the viral
reference protein database, filter hits (identity >= 40%, e-value <= 1e-5,
coverage >= 40%), then cross-reference each hit's UniProt Entry against:

1. UniProt TSV — all downloaded annotation fields (protein name, gene, organism, etc.)
2. ViProtRAG_database.jsonl — VirProtRAG generated functional summaries
3. SeqBench.jsonl — sequence-based benchmark functional annotations

Output is a single TSV with all columns merged.
"""

import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

DIAMOND_OUTFMT = "qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovhsp"

# Mapping Greek letters → ASCII to prevent GBK encoding issues
# (e.g., κ UTF-8 bytes \xCE\xBA get decoded as 魏 in GBK/CP936)
_GREEK_TO_ASCII = {
    "\u0391": "Alpha",   "\u03B1": "alpha",
    "\u0392": "Beta",    "\u03B2": "beta",
    "\u0393": "Gamma",   "\u03B3": "gamma",
    "\u0394": "Delta",   "\u03B4": "delta",
    "\u0395": "Epsilon", "\u03B5": "epsilon",
    "\u0396": "Zeta",    "\u03B6": "zeta",
    "\u0397": "Eta",     "\u03B7": "eta",
    "\u0398": "Theta",   "\u03B8": "theta",
    "\u0399": "Iota",    "\u03B9": "iota",
    "\u039A": "Kappa",   "\u03BA": "kappa",
    "\u039B": "Lambda",  "\u03BB": "lambda",
    "\u039C": "Mu",      "\u03BC": "mu",
    "\u039D": "Nu",      "\u03BD": "nu",
    "\u039E": "Xi",      "\u03BE": "xi",
    "\u039F": "Omicron", "\u03BF": "omicron",
    "\u03A0": "Pi",      "\u03C0": "pi",
    "\u03A1": "Rho",     "\u03C1": "rho",
    "\u03A3": "Sigma",   "\u03C3": "sigma",  "\u03C2": "sigma",
    "\u03A4": "Tau",     "\u03C4": "tau",
    "\u03A5": "Upsilon", "\u03C5": "upsilon",
    "\u03A6": "Phi",     "\u03C6": "phi",
    "\u03A7": "Chi",     "\u03C7": "chi",
    "\u03A8": "Psi",     "\u03C8": "psi",
    "\u03A9": "Omega",   "\u03C9": "omega",
    # Coptic / Latin extensions that cause GBK issues
    "\u0190": "E",       # Latin capital open E (COPƐ → COPE)
}


def normalize_greek(text: str) -> str:
    """Replace Greek/Coptic characters with ASCII equivalents.

    Prevents garbled characters when TSV is opened with GBK/CP936 encoding
    (e.g., κ → kappa, not 魏).
    """
    if not text:
        return text
    return "".join(_GREEK_TO_ASCII.get(ch, ch) for ch in text)


def run_diamond_blastp(
    query_fasta: str,
    db_path: str,
    diamond_bin: str = "diamond",
    threads: int = 8,
    evalue: float = 1e-5,
    tmpdir: Optional[str] = None,
) -> list[dict]:
    """Run DIAMOND blastp and return parsed hits as list of dicts.

    DIAMOND outfmt 6 columns:
        qseqid sseqid pident length mismatch gapopen
        qstart qend sstart send evalue bitscore qcovhsp
    """
    if tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        outfile = os.path.join(tmpdir, "diamond_results.txt")
    else:
        outfile = tempfile.mktemp(suffix="_diamond.txt")

    cmd = [
        diamond_bin, "blastp",
        "--db", db_path,
        "--query", query_fasta,
        "--outfmt", "6",
    ] + DIAMOND_OUTFMT.split() + [
        "--evalue", str(evalue),
        "--max-target-seqs", "0",
        "--threads", str(threads),
        "--out", outfile,
        "--quiet",
    ]

    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error("DIAMOND failed: %s", e.stderr)
        raise RuntimeError(f"DIAMOND blastp failed: {e.stderr}") from e

    if not os.path.exists(outfile) or os.path.getsize(outfile) == 0:
        logger.warning("DIAMOND produced no hits.")
        return []

    hits = []
    with open(outfile) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 13:
                continue
            hit = {
                "qseqid": parts[0],
                "sseqid": parts[1],
                "pident": float(parts[2]),
                "length": int(parts[3]),
                "mismatch": int(parts[4]),
                "gapopen": int(parts[5]),
                "qstart": int(parts[6]),
                "qend": int(parts[7]),
                "sstart": int(parts[8]),
                "send": int(parts[9]),
                "evalue": float(parts[10]),
                "bitscore": float(parts[11]),
                "qcovhsp": float(parts[12]),
            }
            hits.append(hit)

    logger.info("DIAMOND returned %d raw hits", len(hits))
    return hits


def filter_hits(
    hits: list[dict],
    min_identity: float = 40.0,
    max_evalue: float = 1e-5,
    min_qcov: float = 40.0,
) -> list[dict]:
    """Filter DIAMOND hits by identity, e-value, and query coverage."""
    filtered = []
    for h in hits:
        if h["pident"] >= min_identity and h["evalue"] <= max_evalue and h["qcovhsp"] >= min_qcov:
            filtered.append(h)
    logger.info("Filtered: %d / %d hits pass thresholds (id>=%.0f%%, eval<=%s, qcov>=%.0f%%)",
                len(filtered), len(hits), min_identity, max_evalue, min_qcov)
    return filtered


def load_uniprot_tsv(tsv_path: str) -> dict[str, dict]:
    """Load UniProt TSV into a dict keyed by Entry.

    Returns dict: Entry -> {column_name: value, ...}
    """
    if not os.path.exists(tsv_path):
        logger.warning("UniProt TSV not found: %s", tsv_path)
        return {}

    db = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entry = row.get("Entry", "").strip()
            if entry:
                db[entry] = dict(row)
    logger.info("Loaded %d entries from UniProt TSV", len(db))
    return db


def load_jsonl_db(jsonl_path: str, key_field: str = "Entry") -> dict[str, dict]:
    """Load a JSONL database into a dict keyed by *key_field*."""
    if not os.path.exists(jsonl_path):
        logger.warning("JSONL not found: %s", jsonl_path)
        return {}

    db = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get(key_field, "").strip()
            if key:
                db[key] = rec
    logger.info("Loaded %d entries from %s", len(db), os.path.basename(jsonl_path))
    return db


def search_and_annotate(
    query_fasta: str,
    db_prefix: str,
    uniprot_tsv: str,
    virprotrag_jsonl: str,
    seqbench_jsonl: str,
    output_tsv: str,
    diamond_bin: str = "diamond",
    threads: int = 8,
    min_identity: float = 40.0,
    max_evalue: float = 1e-5,
    min_qcov: float = 40.0,
) -> str:
    """Full pipeline: DIAMOND search → filter → cross-reference → TSV output.

    Returns the path to the output TSV file.
    """
    # Step 1: DIAMOND blastp
    print(f"🔍 Running DIAMOND blastp against {db_prefix}...")
    hits = run_diamond_blastp(
        query_fasta=query_fasta,
        db_path=db_prefix,
        diamond_bin=diamond_bin,
        threads=threads,
        evalue=max_evalue,
    )

    if not hits:
        print("⚠️  No DIAMOND hits found.")
        # Write empty TSV with header (BOM for Excel compatibility)
        with open(output_tsv, "w", newline="", encoding="utf-8-sig") as f:
            f.write("Query_ID\tHit_Entry\tIdentity\tE_value\tBit_Score\tCoverage\t"
                    "Gene_Names\tOrganism\tProtein_Names\tLength\tFunction_CC\t"
                    "Taxonomic_Lineage\tVirus_Hosts\tEntry_Name\t"
                    "VirProtRAG\tSeqBench\n")
        return output_tsv

    # Step 2: Filter
    hits = filter_hits(hits, min_identity=min_identity, max_evalue=max_evalue, min_qcov=min_qcov)

    if not hits:
        print("⚠️  No hits pass filtering thresholds.")
        with open(output_tsv, "w", newline="", encoding="utf-8-sig") as f:
            f.write("Query_ID\tHit_Entry\tIdentity\tE_value\tBit_Score\tCoverage\t"
                    "Gene_Names\tOrganism\tProtein_Names\tLength\tFunction_CC\t"
                    "Taxonomic_Lineage\tVirus_Hosts\tEntry_Name\t"
                    "VirProtRAG\tSeqBench\n")
        return output_tsv

    # Step 3: Load lookup databases
    print("📚 Loading annotation databases...")
    uniprot_db = load_uniprot_tsv(uniprot_tsv)
    virprotrag_db = load_jsonl_db(virprotrag_jsonl, key_field="Entry")
    seqbench_db = load_jsonl_db(seqbench_jsonl, key_field="Entry")

    # Step 4: Cross-reference and build output rows
    print(f"🔗 Cross-referencing {len(hits)} hits...")
    rows = []
    seen = set()  # deduplicate (query, hit) pairs

    # Define output columns
    out_columns = [
        "Query_ID", "Hit_Entry", "Identity", "E_value", "Bit_Score", "Coverage",
        "Gene_Names", "Organism", "Protein_Names", "Length", "Function_CC",
        "Taxonomic_Lineage", "Virus_Hosts", "Entry_Name",
        "VirProtRAG", "SeqBench",
    ]

    for hit in hits:
        query_id = hit["qseqid"]
        hit_entry = hit["sseqid"]
        pair_key = (query_id, hit_entry)
        if pair_key in seen:
            continue
        seen.add(pair_key)

        # UniProt annotation
        uniprot = uniprot_db.get(hit_entry, {})

        # VirProtRAG
        vp = virprotrag_db.get(hit_entry, {})
        vp_summary = vp.get("Summary", "")

        # SeqBench
        sb = seqbench_db.get(hit_entry, {})
        sb_text = sb.get("SeqBench", "")

        row = {
            "Query_ID": query_id,
            "Hit_Entry": hit_entry,
            "Identity": f"{hit['pident']:.1f}",
            "E_value": f"{hit['evalue']:.2e}",
            "Bit_Score": f"{hit['bitscore']:.1f}",
            "Coverage": f"{hit['qcovhsp']:.1f}",
            "Gene_Names": normalize_greek(uniprot.get("Gene Names", "")),
            "Organism": normalize_greek(uniprot.get("Organism", "")),
            "Protein_Names": normalize_greek(uniprot.get("Protein names", "")),
            "Length": uniprot.get("Length", ""),
            "Function_CC": normalize_greek(uniprot.get("Function [CC]", "")),
            "Taxonomic_Lineage": normalize_greek(uniprot.get("Taxonomic lineage", "")),
            "Virus_Hosts": normalize_greek(uniprot.get("Virus hosts", "")),
            "Entry_Name": normalize_greek(uniprot.get("Entry Name", "")),
            "VirProtRAG": normalize_greek(vp_summary),
            "SeqBench": normalize_greek(sb_text),
        }
        rows.append(row)

    # Step 5: Write TSV (with BOM for Excel compatibility)
    os.makedirs(os.path.dirname(output_tsv) or ".", exist_ok=True)
    with open(output_tsv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Output: {len(rows)} rows → {output_tsv}")

    # Summary stats
    with_vp = sum(1 for r in rows if r["VirProtRAG"])
    with_sb = sum(1 for r in rows if r["SeqBench"])
    unique_hits = len(set(r["Hit_Entry"] for r in rows))
    unique_queries = len(set(r["Query_ID"] for r in rows))
    print(f"   {unique_queries} query proteins matched {unique_hits} unique reference entries")
    print(f"   {with_vp} with VirProtRAG annotation, {with_sb} with SeqBench annotation")

    return output_tsv
