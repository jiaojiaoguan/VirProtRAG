"""
LLM configuration and environment variable management.

Environment variables (loaded from shell or .env file):
    LLM_PROVIDER          — openai | anthropic | deepseek | zhipu
    OPENAI_API_KEY        — OpenAI API key
    ANTHROPIC_API_KEY     — Anthropic API key
    DEEPSEEK_API_KEY      — DeepSeek API key
    ZHIPU_API_KEY         — Zhipu API key
    LLM_BASE_URL          — optional custom endpoint for proxies
    LLM_MODEL             — override model for ALL tasks (single entry)
    LLM_SYNONYM_MODEL     — override model for synonym expansion
    LLM_EVIDENCE_MODEL    — override model for evidence classification
    LLM_GENERATION_MODEL  — override model for function generation

    ENTREZ_EMAIL          — NCBI required
    ENTREZ_API_KEY        — NCBI API key (recommended)

    MEDCPT_FAISS_INDEX_PATH — path to MedCPT FAISS index
    MEDCPT_PMIDS_PATH       — path to MedCPT PMIDs mapping

    OPENALEX_EMAIL        — OpenAlex API email (optional)
    DIAMOND_PATH          — path to DIAMOND binary (optional)
    VIRPROTRAG_CACHE      — path to cache file (default: ~/.virprotrag_cache.json)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Literal

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

Provider = Literal["openai", "anthropic", "deepseek", "zhipu"]

DEFAULT_MODELS = {
    "openai": {
        "synonym": "gpt-4o-mini",
        "evidence": "gpt-4o-mini",
        "generation": "gpt-4o",
    },
    "anthropic": {
        "synonym": "claude-haiku-4-20250514",
        "evidence": "claude-haiku-4-20250514",
        "generation": "claude-sonnet-4-20250514",
    },
    "deepseek": {
        "synonym": "deepseek-chat",
        "evidence": "deepseek-chat",
        "generation": "deepseek-chat",
    },
    "zhipu": {
        "synonym": "glm-4-flash",
        "evidence": "glm-4-flash",
        "generation": "glm-4",
    },
}


@dataclass
class LLMConfig:
    """Central configuration for LLM access across all VirProtRAG modules."""

    provider: Provider = "openai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Model selection (priority: per-task > --model CLI > LLM_MODEL env > default)
    model: Optional[str] = None          # single model for all tasks
    synonym_model: Optional[str] = None
    evidence_model: Optional[str] = None
    generation_model: Optional[str] = None

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> "LLMConfig":
        """Load configuration from environment variables.

        Args:
            model: Model name override (from --model CLI flag). Takes
                   precedence over LLM_MODEL env var.
        """
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider not in ("openai", "anthropic", "deepseek", "zhipu"):
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. "
                f"Must be one of: openai, anthropic, deepseek, zhipu."
            )

        # Resolve API key by provider
        key_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
        }
        api_key = os.getenv(key_env_map[provider])
        if not api_key:
            print(
                f"⚠️  Warning: {key_env_map[provider]} is not set. "
                f"LLM calls will fail unless you provide an API key at runtime."
            )

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL"),
            model=model or os.getenv("LLM_MODEL"),
            synonym_model=os.getenv("LLM_SYNONYM_MODEL"),
            evidence_model=os.getenv("LLM_EVIDENCE_MODEL"),
            generation_model=os.getenv("LLM_GENERATION_MODEL"),
        )

    def get_model(self, task: Literal["synonym", "evidence", "generation"]) -> str:
        """Return the model name for a given task.

        Priority: per-task override > LLM_MODEL > provider default.
        """
        override = getattr(self, f"{task}_model", None)
        if override:
            return override
        if self.model:
            return self.model
        return DEFAULT_MODELS[self.provider][task]


@dataclass
class RuntimeConfig:
    """Global runtime configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig.from_env)

    # NCBI / PubMed
    entrez_email: Optional[str] = None
    entrez_api_key: Optional[str] = None

    # MedCPT
    medcpt_index_path: Optional[str] = None
    medcpt_pmids_path: Optional[str] = None

    @property
    def has_medcpt(self) -> bool:
        return (
            self.medcpt_index_path is not None
            and self.medcpt_pmids_path is not None
            and os.path.isfile(self.medcpt_index_path)
            and os.path.isfile(self.medcpt_pmids_path)
        )

    # OpenAlex
    openalex_email: Optional[str] = None

    # DIAMOND
    diamond_path: str = "diamond"

    # Cache
    cache_path: Optional[str] = None

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> "RuntimeConfig":
        """Load full runtime configuration from environment.

        Args:
            model: Model name override (from --model CLI flag).
        """
        return cls(
            llm=LLMConfig.from_env(model=model),
            entrez_email=os.getenv("ENTREZ_EMAIL"),
            entrez_api_key=os.getenv("ENTREZ_API_KEY"),
            medcpt_index_path=os.getenv("MEDCPT_FAISS_INDEX_PATH"),
            medcpt_pmids_path=os.getenv("MEDCPT_PMIDS_PATH"),
            openalex_email=os.getenv("OPENALEX_EMAIL"),
            diamond_path=os.getenv("DIAMOND_PATH", "diamond"),
            cache_path=os.getenv("VIRPROTRAG_CACHE"),
        )

    def validate(self) -> list[str]:
        """Check required config and return a list of issues (empty = all good).

        This is a fast, offline check (no network calls).
        Use preflight_check() for API connectivity verification.
        """
        issues = []
        if not self.entrez_email:
            issues.append(
                "ENTREZ_EMAIL is required for PubMed API access. "
                "Set it in your .env file or environment."
            )
        if not self.llm.api_key:
            key_env = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "zhipu": "ZHIPU_API_KEY",
            }.get(self.llm.provider, "API key")
            issues.append(
                f"No API key found for provider '{self.llm.provider}'. "
                f"Please set {key_env} in your .env file or environment."
            )
        return issues

    def preflight_check(self) -> dict:
        """Run connectivity checks and return {status, issues, warnings}.

        Sends minimal API calls to verify:
        - LLM API key is valid and reachable
        - PubMed E-Utilities accessible
        - MedCPT index files exist (if configured)

        Returns dict with:
            status: "ok" | "warning" | "error"
            issues: list of error strings (blocking)
            warnings: list of warning strings (non-blocking)
        """
        issues = []
        warnings = []

        # 1. LLM API connectivity
        provider = self.llm.provider
        if self.llm.api_key:
            try:
                from openai import OpenAI

                # Determine base_url
                base_url = self.llm.base_url
                if not base_url:
                    _default_urls = {
                        "deepseek": "https://api.deepseek.com",
                        "zhipu": "https://open.bigmodel.cn/api/paas/v4",
                    }
                    base_url = _default_urls.get(provider)

                client_kwargs = {"api_key": self.llm.api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url

                client = OpenAI(**client_kwargs, timeout=10)
                # Lightweight check: list models (does not incur cost)
                client.models.list()
            except Exception as e:
                error_msg = str(e)
                if "403" in error_msg or "401" in error_msg:
                    issues.append(
                        f"LLM authentication failed for provider '{provider}'. "
                        f"Check your API key. Error: {error_msg[:200]}"
                    )
                elif "Connection" in error_msg or "timeout" in error_msg:
                    issues.append(
                        f"Cannot reach LLM API for provider '{provider}'. "
                        f"Check your network / VPN. Error: {error_msg[:200]}"
                    )
                else:
                    warnings.append(
                        f"LLM preflight check warning for '{provider}': {error_msg[:200]}"
                    )
        else:
            issues.append(
                f"No API key configured for '{provider}'. LLM calls will fail."
            )

        # 2. PubMed / Entrez connectivity (lightweight)
        if self.entrez_email:
            try:
                from Bio import Entrez
                Entrez.email = self.entrez_email
                if self.entrez_api_key:
                    Entrez.api_key = self.entrez_api_key
                # Quick search that returns minimal results
                handle = Entrez.esearch(
                    db="pubmed", term="virus", retmax=1, retmode="xml"
                )
                handle.close()
            except Exception as e:
                warnings.append(
                    f"PubMed API check warning: {str(e)[:200]}. "
                    f"Search may still work with retries."
                )

        # 3. MedCPT index files
        if self.medcpt_index_path and not os.path.isfile(self.medcpt_index_path):
            warnings.append(
                f"MedCPT FAISS index not found at: {self.medcpt_index_path}"
            )
        if self.medcpt_pmids_path and not os.path.isfile(self.medcpt_pmids_path):
            warnings.append(
                f"MedCPT PMIDs mapping not found at: {self.medcpt_pmids_path}"
            )

        # Determine overall status
        if issues:
            status = "error"
        elif warnings:
            status = "warning"
        else:
            status = "ok"

        return {"status": status, "issues": issues, "warnings": warnings}
