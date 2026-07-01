# VirProtRAG: Literature-Grounded Viral Protein Function Annotation

VirProtRAG is a retrieval-augmented generation (RAG) framework for viral protein function annotation. It retrieves relevant literature from PubMed, re-ranks papers by evidence quality and type, and synthesizes evidence-grounded functional descriptions with explicit PMID citations.

---

## Three Use Cases at a Glance

| #   | Use Case                 | Command                   | Input                   | Output                            | Best For                               |
| --- | ------------------------ | ------------------------- | ----------------------- | --------------------------------- | -------------------------------------- |
| 1   | **Single Protein RAG**   | `virprotrag annotate`     | Protein name + organism | JSON + Markdown report            | One-off function annotation            |
| 2   | **Batch Protein RAG**    | `virprotrag batch`        | TSV file                | TSV annotations                   | Proteome-scale annotation              |
| 3   | **Protein FASTA Search** | `virprotrag search-fasta` | FASTA file              | TSV with alignments + annotations | Homology-based lookup against viral DB |

## Use Case 1: Single Protein RAG

Annotate one  viral proteins by name. The pipeline searches PubMed, classifies evidence strength, and generates a literature-grounded functional description.

### 1.1 All-in-One (simplest)

Runs BM25 + MedCPT (if available) + annotation in a single command. **MedCPT is automatically skipped** when the environment variables (`MEDCPT_FAISS_INDEX_PATH`, `MEDCPT_PMIDS_PATH`) are not set or the index files are missing — no manual `--skip-dense` flag needed.

```bash
# Full pipeline (BM25 + MedCPT + annotation)
virprotrag annotate \
    --protein "Spike glycoprotein" \
    --gene "S" \
    --organism "SARS-CoV-2" \
    --output spike_annotation.json

# Multi-protein (comma-separated)
virprotrag annotate \
    --protein "terminase large subunit, portal protein, capsid protein" \
    --gene "terL, 20, 23" \
    --organism "Escherichia phage T4" \
    --output t4_proteins.json
```

**Key arguments:**

| Argument         | Required | Default | Description                                    |
| ---------------- | -------- | ------- | ---------------------------------------------- |
| `--protein`      | Yes      | —       | Protein name(s), comma-separated               |
| `--gene`         | No       | —       | Gene name(s), comma-separated                  |
| `--organism`     | Yes      | —       | Host/virus name(s)                             |
| `--topk`         | No       | 10      | Papers used for final generation               |
| `--skip-dense`   | No       | False   | Skip MedCPT semantic search                    |
| `--skip-quality` | No       | False   | Skip OpenAlex paper quality scoring            |
| `--output`       | No       | stdout  | JSON output path (auto-generates `_report.md`) |
| `--check`        | No       | —       | Run connectivity preflight check               |
| `--verbose`      | No       | False   | Debug logging                                  |

**Output:** JSON file + companion Markdown report (e.g., `spike_annotation.json` + `spike_annotation_report.md`).

### 1.2 Phased Execution (HPC clusters)

For HPC environments where the login node cannot load the MedCPT FAISS index (~103 GB), split into three phases:

**Phase 1 — BM25 (login node, needs internet):**

```bash
virprotrag --phase bm25 \
    --protein "terminase large subunit" \
    --gene "terL" \
    --organism "Escherichia phage T4" \
    --output bm25_output.json \
```

This saves synonyms + queries + PubMed PMIDs to `bm25_output.json`.

**Phase 2 — MedCPT (compute node, zero network, submit via SGE/qsub):**

```bash
virprotrag --phase medcpt --input bm25_output.json --output medcpt_output.json --verbose
```

**Phase 3 — Annotation (login node):**

```bash
# With BM25 + MedCPT → RRF fusion
virprotrag annotate \
    --bm25 bm25_output.json \
    --medcpt medcpt_output.json \
    --output annotation.json
```

**Phased mode arguments (`--bm25`/`--medcpt`):**

| Argument         | Required | Default | Description                                    |
| ---------------- | -------- | ------- | ---------------------------------------------- |
| `--bm25`         | Yes      | —       | Path to BM25-phase JSON                        |
| `--medcpt`       | No       | —       | Path to MedCPT-phase JSON (enables RRF fusion) |
| `--topk`         | No       | 10      | Papers for generation                          |
| `--skip-quality` | No       | False   | Skip OpenAlex scoring                          |
| `--output`       | No       | stdout  | Output JSON path                               |
| `--verbose`      | No       | False   | Debug logging                                  |

---

## Use Case 2: Batch Protein RAG

Process dozens or hundreds of proteins from a TSV file. Supports two modes: all-in-one (single command per protein) and phased (split into BM25/MedCPT/Annotate stages for HPC).

### 2.1 All-in-One Batch

```bash
virprotrag batch \
    --input examples/sample_proteins.tsv \
    --output batch_results/sample_proteins.tsv
```

**Input format (TSV):**

| column         | required | description                                             |
| -------------- | -------- | ------------------------------------------------------- |
| `protein_name` | Yes      | Protein name                                            |
| `organism`     | Yes      | Host/virus name                                         |
| `gene_name`    | No       | Gene name                                               |
| `entry`        | No       | Unique ID (auto-generated from protein_name if missing) |

Example `sample_proteins.tsv`:

```tsv
entry    protein_name    gene_name    organism
HCV_core    Core protein    C,Hepatitis    C virus
SARS2_S    Spike glycoprotein    S    SARS-CoV-2
EBOV_NP    Nucleoprotein    NEbola virus
```

A pre-made example is at `examples/sample_proteins.tsv`.

**Batch all-in-one arguments:**

| Argument         | Required | Default                             | Description                    |
| ---------------- | -------- | ----------------------------------- | ------------------------------ |
| `--input`        | Yes      | —                                   | TSV file path                  |
| `--output`       | No       | `virprotrag_output/annotations.tsv` | Output TSV path                |
| `--topk`         | No       | 10                                  | Papers for generation          |
| `--skip-dense`   | No       | False                               | Skip MedCPT                    |
| `--skip-quality` | No       | False                               | Skip OpenAlex scoring          |
| `--resume`       | No       | —                                   | Skip already-completed entries |
| `--verbose`      | No       | False                               | Debug logging                  |
| `--check`        | No       | —                                   | Preflight connectivity check   |

### 2.2 Phased Batch (HPC)

Split the batch pipeline into three phases — ideal for processing hundreds of proteins on a cluster:

**Phase 1 — Batch BM25 (login node):**

```bash
virprotrag batch --phase bm25 \
    --input proteins.tsv \
    --output batch_bm25/ \
    --retrieval-topk 100 \
    --resume
```

Creates `batch_bm25/{entry}_bm25.json` for each protein.

**Phase 2 — Batch MedCPT (compute node):**

```bash
virprotrag batch --phase medcpt \
    --input batch_bm25/ \
    --output batch_medcpt/ \
    --retrieval-topk 100 \
    --resume
```

Creates `batch_medcpt/{entry}_medcpt.json`. Loads FAISS index and MedCPT encoder once — efficient for processing many proteins in one job.

**Phase 3 — Batch Annotate:**

```bash
# BM25 + MedCPT (RRF fusion)
virprotrag batch --phase annotate \
    --input batch_bm25/ \
    --medcpt-dir batch_medcpt/ \
    --output batch_results/annotations_fused.tsv \
    --resume
```

**Phased batch arguments:**

| Argument           | Phase        | Description                                                     |
| ------------------ | ------------ | --------------------------------------------------------------- |
| `--input`          | All          | TSV (bm25 phase) or directory of JSONs (medcpt/annotate phases) |
| `--output`         | All          | Output directory or TSV path                                    |
| `--retrieval-topk` | bm25, medcpt | Max PMIDs per search (default: 100)                             |
| `--batch-size`     | medcpt       | Encoding batch size (default: 2048)                             |
| `--index-path`     | medcpt       | FAISS index path (overrides env)                                |
| `--pmids-path`     | medcpt       | PMIDs mapping path (overrides env)                              |
| `--medcpt-dir`     | annotate     | Directory of `*_medcpt.json` files for RRF fusion               |
| `--topk`           | annotate     | Papers for generation (default: 10)                             |
| `--skip-quality`   | annotate     | Skip OpenAlex scoring                                           |
| `--resume`         | All          | Skip already-processed entries                                  |

---

## Use Case 3: Protein FASTA Search

Given a FASTA file of viral protein sequences, run DIAMOND BLASTP against the viral reference database, filter hits, and cross-reference each match against three annotation sources:

1. **UniProt** — curated annotation fields (gene, organism, function, taxonomy, hosts)
2. **ViProtRAG** — LLM-generated functional summaries
3. **SeqBench** — sequence-based benchmark annotations

### 3.1 Prerequisites

**Data files** (put in `data/` directory):

| File                       | Size    | Description                                       |
| -------------------------- | ------- | ------------------------------------------------- |
| `viral_proteins.dmnd`      | ~8 MB   | DIAMOND BLASTP index (built via `diamond makedb`) |
| `uniprotkb_*.tsv`          | ~19 MB  | UniProt annotation fields                         |
| `ViProtRAG_database.jsonl` | ~7.7 MB | VirProtRAG functional summaries                   |
| `SeqBench.jsonl`           | ~9.6 MB | Sequence benchmark annotations                    |

The pre-built `viral_proteins.dmnd` is included in the repository for convenience.
However, the DIAMOND binary index is **version-dependent** — if your local DIAMOND
version differs, `search-fasta` will fail with a version mismatch error.
In that case, rebuild the index from the source FASTA:

```bash
# Rebuild DIAMOND database from the viral protein FASTA
diamond makedb --in data/viral_proteins.fasta --db data/viral_proteins --threads 8
```

This overwrites `data/viral_proteins.dmnd` with a version-compatible index.
The FASTA file contains 17,517 reviewed UniProt viral protein sequences with
simplified headers (`>ENTRY_ID`).

### 3.2 Run Search

```bash
virprotrag search-fasta \
    --query examples/test_query.fasta \
    --output search_results.tsv \
    --threads 16
```

The tool auto-detects database files from the `data/` directory. Override with explicit paths:

```bash
virprotrag search-fasta \
    --query my_viral_proteins.fasta \
    --db /path/to/viral_proteins \
    --uniprot-tsv /path/to/uniprotkb.tsv \
    --virprotrag-db /path/to/ViProtRAG_database.jsonl \
    --seqbench-db /path/to/SeqBench.jsonl \
    --output results.tsv \
    --threads 16 \
    --min-identity 50 \
    --evalue 1e-10
```

### 3.3 Search Arguments

| Argument          | Required | Default                         | Description                |
| ----------------- | -------- | ------------------------------- | -------------------------- |
| `--query`         | Yes      | —                               | Query FASTA file path      |
| `--db`            | No       | `data/viral_proteins`           | DIAMOND database prefix    |
| `--uniprot-tsv`   | No       | `data/uniprotkb_*.tsv`          | UniProt annotation TSV     |
| `--virprotrag-db` | No       | `data/ViProtRAG_database.jsonl` | VirProtRAG summaries JSONL |
| `--seqbench-db`   | No       | `data/SeqBench.jsonl`           | SeqBench annotations JSONL |
| `--output`        | No       | `search_results.tsv`            | Output TSV path            |
| `--threads`       | No       | 8                               | DIAMOND threads            |
| `--min-identity`  | No       | 40.0                            | Minimum identity %         |
| `--evalue`        | No       | 1e-5                            | Maximum e-value            |
| `--min-qcov`      | No       | 40.0                            | Minimum query coverage %   |
| `--verbose`       | No       | False                           | Debug logging              |

### 3.4 Output Format (TSV)

The output TSV uses UTF-8 with BOM for Excel compatibility. Greek characters (e.g., NF-κB) are normalized to ASCII (e.g., NF-kappaB) to prevent encoding issues.

| Column              | Source       | Description                      |
| ------------------- | ------------ | -------------------------------- |
| `Query_ID`          | FASTA header | Query sequence ID                |
| `Hit_Entry`         | DIAMOND      | Matched UniProt entry            |
| `Identity`          | DIAMOND      | Sequence identity (%)            |
| `E_value`           | DIAMOND      | E-value                          |
| `Bit_Score`         | DIAMOND      | Bit score                        |
| `Coverage`          | DIAMOND      | Query coverage (%)               |
| `Gene_Names`        | UniProt      | Gene names                       |
| `Organism`          | UniProt      | Source organism                  |
| `Protein_Names`     | UniProt      | Protein description              |
| `Length`            | UniProt      | Sequence length                  |
| `Function_CC`       | UniProt      | Curated function (from comments) |
| `Taxonomic_Lineage` | UniProt      | Full taxonomy                    |
| `Virus_Hosts`       | UniProt      | Known virus hosts                |
| `Entry_Name`        | UniProt      | Entry name                       |
| `VirProtRAG`        | ViProtRAG DB | RAG-generated functional summary |
| `SeqBench`          | SeqBench DB  | Sequence-benchmark annotation    |

---

## Installation

### From source (recommended)

```bash
git clone https://github.com/jiaojiaoguan/VirProtRAG.git
cd VirProtRAG
conda create -n virprotrag python=3.10 -y
conda activate virprotrag
pip install -e .
```

### Verify

```bash
virprotrag --help
# Expected subcommands: annotate, batch, search-fasta
```

---

## Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
vim .env
```

### Minimal `.env` (using DeepSeek for example)

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-key-here
ENTREZ_EMAIL=yourname@gmail.com
```

Want to use a different model? Just add one line:

```bash
LLM_MODEL=gpt-4.1    # overrides all tasks with this model
```

### LLM Providers

| Provider        | Default Models                                         | API Key Env         | Base URL      |
| --------------- | ------------------------------------------------------ | ------------------- | ------------- |
| **DeepSeek**    | `deepseek-chat` (all tasks)                            | `DEEPSEEK_API_KEY`  | Auto-detected |
| **OpenAI**      | `gpt-4o-mini` / `gpt-4o`                               | `OPENAI_API_KEY`    | Auto-detected |
| **Anthropic**   | `claude-haiku-4-20250514` / `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` | Auto-detected |
| **Zhipu (GLM)** | `glm-4-flash` / `glm-4`                                | `ZHIPU_API_KEY`     | Auto-detected |

> **Important:** Do NOT manually set `LLM_BASE_URL` unless you use a proxy. The correct endpoints are auto-detected.

#### Customizing the Model

You can override the default model in three ways (highest priority first):

| Method              | Scope         | Example                       |
| ------------------- | ------------- | ----------------------------- |
| `--model` CLI flag  | Single run    | `--model gpt-4.1`             |
| `LLM_MODEL` env var | All runs      | `LLM_MODEL=gpt-4.1`           |
| Per-task env vars   | Specific task | `LLM_GENERATION_MODEL=gpt-4o` |

Example — use a different model for a single run:

```bash
virprotrag annotate --protein "capsid protein" --organism "Escherichia phage T4" \
    --model gpt-4.1 --output capsid.json
```

Example — set a default model in `.env`:

```bash
LLM_MODEL=gpt-4.1    # all VirProtRAG tasks will use this model
```

### Optional environment variables

| Variable                  | Purpose                                                          |
| ------------------------- | ---------------------------------------------------------------- |
| `ENTREZ_API_KEY`          | Boosts PubMed rate 3→10 req/s                                    |
| `MEDCPT_FAISS_INDEX_PATH` | Path to MedCPT FAISS index                                       |
| `MEDCPT_PMIDS_PATH`       | Path to MedCPT PMIDs mapping                                     |
| `OPENALEX_EMAIL`          | Paper quality scoring (citation count, journal prestige)         |
| `DIAMOND_PATH`            | Path to DIAMOND binary (default: `diamond` on PATH)              |
| `VIRPROTRAG_CACHE`        | Cross-protein cache path (reuses paper metadata across proteins) |
| `LLM_MODEL`               | Override model for ALL tasks (single entry, e.g., `gpt-4.1`)     |
| `LLM_SYNONYM_MODEL`       | Override model for synonym expansion                             |
| `LLM_EVIDENCE_MODEL`      | Override model for evidence classification                       |
| `LLM_GENERATION_MODEL`    | Override model for function generation                           |

## Output Format (Single Protein / Batch)

```json
{
  "entry": "USER_PROVIDED",
  "query": {
    "protein": "capsid protein",
    "gene": "",
    "organism": "Escherichia phage"
  },
  "generated_annotation": "FUNCTION: Forms the icosahedral capsid... [PMID:12345678, PMID:23456789]",
  "supporting_evidence": [
    {
      "pmid": "12345678",
      "title": "Structure of the Escherichia phage capsid...",
      "evidence_label": "EXPERIMENTAL",
      "final_score": 0.8921,
      "semantic_score": 0.92,
      "evidence_score": 1.0,
      "quality_score": 0.75
    }
  ],
  "evidence_distribution": {
    "EXPERIMENTAL": 23,
    "INFERRED": 45,
    "NONE": 32
  },
  "pipeline_yield": {
    "bm25_retrieved": 100,
    "medcpt_retrieved": 100,
    "rrf_fused": 145,
    "abstracts_fetched": 100,
    "evidence_classified": 100,
    "reranked": 100,
    "used_for_generation": 10
  },
  "runtime": {
    "provider": "deepseek",
    "model_generation": "deepseek-chat",
    "topk": 10,
    "elapsed_seconds": 45.3
  }
}
```

**Key fields:**

- `generated_annotation` — Evidence-grounded functional description with `[PMID:...]` citations
- `pipeline_yield` — Papers surviving each stage (diagnose the funnel)
- `evidence_distribution` — EXPERIMENTAL / INFERRED / NONE counts (assesses evidence quality)
- `supporting_evidence[].*_score` — Three dimensions: semantic relevance, evidence strength, paper quality

**Batch mode** outputs a TSV with columns: `entry`, `protein_name`, `gene_name`, `organism`, `generated_annotation`, `supporting_pmids`.

---

## Cross-Protein Caching

When annotating multiple related proteins, the cache reuses previously fetched data:

| Cached content                  | Shared across proteins?                                          |
| ------------------------------- | ---------------------------------------------------------------- |
| Paper metadata (title/abstract) | Yes (same paper)                                                 |
| OpenAlex quality scores         | Yes (citation counts are protein-independent)                    |
| Evidence classification         | No (paper may be "experimental" for protein A, "inferred" for B) |

Enable: `VIRPROTRAG_CACHE=/path/to/cache.json` in `.env`. At exit, prints: `[cache] 42 hits / 10 misses`.

---

## Citation

```
Guan, J., Shang, J., Peng, C., & Sun, Y. (2025).
VirProtRAG: Literature-grounded viral protein function annotation
with retrieval-augmented generation.
```

## License

MIT License.
