"""
VirProtRAG: Literature-grounded viral protein function annotation
with retrieval-augmented generation.
"""

__version__ = "1.0.0"

# Safe imports — search_fasta has no external deps
from .config import LLMConfig, RuntimeConfig
from .search_fasta import search_and_annotate

# Pipeline and MedCPT require biopython + transformers (optional)
try:
    from .pipeline import annotate_single_protein, run_bm25_phase, run_annotate_phase
    from .medcpt_search import medcpt_search, medcpt_search_from_bm25_json
except ImportError:
    pass
