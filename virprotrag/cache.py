"""
Persistent cache for evidence classification and OpenAlex quality scores.

Evidence is protein-dependent: the same PMID may have different labels
for different proteins.  Cache key = (pmid, protein).
OpenAlex quality is protein-independent: cache key = pmid only.

Usage:
    cache = CacheManager(path="/home/user/.virprotrag_cache.json")
    cache.load()

    # Evidence
    cached = cache.get_evidence("12345678", "capsid protein")
    if cached is None:
        label = classify_via_llm(...)
        cache.set_evidence("12345678", "capsid protein", label)

    # OpenAlex
    cached = cache.get_openalex("12345678")
    if cached is None:
        metrics = fetch_openalex(...)
        cache.set_openalex("12345678", metrics)

    cache.save()   # persist to disk
"""

import json
import os
import threading
import time
from typing import Optional


class CacheManager:
    """Thread-safe persistent cache for VirProtRAG pipeline.

    Evidence: dict[protein_name][pmid] -> {label, label_name, confidence}
    OpenAlex: dict[pmid] -> {cited_by_count, ...}
    """

    def __init__(self, path: Optional[str] = None):
        # Default path: project root or home directory
        if path is None:
            home = os.path.expanduser("~")
            path = os.path.join(home, ".virprotrag_cache.json")
        self._path = path
        self._lock = threading.Lock()
        self._dirty = False
        self._evidence: dict[str, dict[str, dict]] = {}  # protein -> pmid -> data
        self._openalex: dict[str, dict] = {}  # pmid -> data
        self._hits = 0
        self._misses = 0

    # ---- load / save ----

    def load(self):
        """Load cache from disk. No-op if file doesn't exist."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._evidence = data.get("evidence", {})
                self._openalex = data.get("openalex", {})
        except (json.JSONDecodeError, OSError) as e:
            # Corrupt cache — start fresh
            print(f"  [cache] Warning: could not load cache ({e}), starting fresh.")
            self._evidence = {}
            self._openalex = {}

    def save(self):
        """Persist cache to disk (only if modified)."""
        if not self._dirty:
            return
        with self._lock:
            data = {
                "version": "1",
                "evidence": self._evidence,
                "openalex": self._openalex,
            }
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        # Atomic write via temp file
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._path)
            self._dirty = False
        except OSError as e:
            print(f"  [cache] Warning: could not save cache: {e}")

    # ---- evidence (protein-dependent) ----

    def get_evidence(self, pmid: str, protein: str) -> Optional[dict]:
        """Return cached evidence label for a (pmid, protein) pair, or None."""
        with self._lock:
            result = self._evidence.get(protein, {}).get(pmid)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def set_evidence(self, pmid: str, protein: str, data: dict):
        """Store evidence label for (pmid, protein)."""
        with self._lock:
            if protein not in self._evidence:
                self._evidence[protein] = {}
            self._evidence[protein][pmid] = dict(data)
            self._evidence[protein][pmid]["cached_at"] = time.time()
            self._dirty = True

    def get_evidence_batch(
        self, pmids: list[str], protein: str
    ) -> tuple[dict[str, dict], list[str]]:
        """Return (cached_map, uncached_pmids) for a batch of PMIDs.

        cached_map: pmid -> cached_data (for pmids with cache hits)
        uncached_pmids: list of pmids that need LLM classification
        """
        cached = {}
        uncached = []
        for pmid in pmids:
            hit = self.get_evidence(pmid, protein)
            if hit is not None:
                cached[pmid] = hit
            else:
                uncached.append(pmid)
        return cached, uncached

    # ---- OpenAlex (protein-independent) ----

    def get_openalex(self, pmid: str) -> Optional[dict]:
        """Return cached OpenAlex metrics for a PMID, or None."""
        with self._lock:
            result = self._openalex.get(pmid)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def set_openalex(self, pmid: str, data: dict):
        """Store OpenAlex metrics for a PMID."""
        with self._lock:
            self._openalex[pmid] = dict(data)
            self._openalex[pmid]["cached_at"] = time.time()
            self._dirty = True

    def get_openalex_batch(
        self, pmids: list[str]
    ) -> tuple[dict[str, dict], list[str]]:
        """Return (cached_map, uncached_pmids) for a batch of PMIDs."""
        cached = {}
        uncached = []
        for pmid in pmids:
            hit = self.get_openalex(pmid)
            if hit is not None:
                cached[pmid] = hit
            else:
                uncached.append(pmid)
        return cached, uncached

    # ---- stats ----

    @property
    def stats(self) -> dict:
        with self._lock:
            total_evidence = sum(len(v) for v in self._evidence.values())
        return {
            "evidence_entries": total_evidence,
            "evidence_proteins": len(self._evidence),
            "openalex_entries": len(self._openalex),
            "hits": self._hits,
            "misses": self._misses,
            "path": self._path,
        }

    def print_stats(self):
        s = self.stats
        print(
            f"  [cache] {s['hits']} hits / {s['misses']} misses "
            f"(evidence: {s['evidence_entries']} entries for {s['evidence_proteins']} proteins, "
            f"openalex: {s['openalex_entries']} entries)"
        )
