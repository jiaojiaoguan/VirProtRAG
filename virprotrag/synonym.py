"""
LLM-based protein name synonym expansion.
Reference: step2_syn_query_extension.py
"""

import json
from .llm_client import call_llm, extract_json_from_response
from .config import LLMConfig

SYNONYM_SYSTEM_PROMPT = """You are a virology and molecular biology expert who is deeply familiar with viral protein nomenclature and synonym usage across databases (UniProt, RefSeq, GenBank) and experimental literature (PubMed).

Your task:
Given a JSON input describing an organism and its protein names (which may include ORF labels, polyprotein cleavage products, or locus tags), identify reliable synonyms used in the literature for each protein within that organism.

### Synonym inclusion rules
- Include both full and short forms that clearly co‑refer to the same protein entity.
- Include descriptive names only if they explicitly refer to the same protein in UniProt or PubMed.
- Do **not** invent speculative names, predicted functions, or inferred homologs.

### Output policy
- Only include synonyms you are **highly confident** are correct for that organism.
- Do not repeat any protein name that already appears in the input.
- If no reliable synonym exists, return an empty list.
- Respond with **valid JSON only**, no Markdown or free text.
- Do not output raw reasoning or commentary.

### Output format
{
  "synonyms": [
    {"<protein1>": ["<synonym_a>", "<synonym_b>", ...]},
    {"<protein2>": ["<synonym_c>", ...]},
    ...
  ]
}"""


def expand_synonyms(
    config: LLMConfig,
    protein_names: list[str],
    organism: str,
) -> list[str]:
    """Use LLM to find literature synonyms for given protein names.

    Args:
        config: LLM configuration.
        protein_names: List of protein name strings.
        organism: Organism name (e.g. 'SARS-CoV-2').

    Returns:
        List of newly discovered synonym strings (excluding originals).
    """
    if not protein_names:
        return []

    input_json = {
        "organism_name": organism,
        "proteins": protein_names,
    }

    context = (
        f"The following JSON describes an organism ('{organism}') and its proteins. "
        f"Identify highly reliable synonyms used in research literature for each protein, "
        f"focusing on canonical short forms or official UniProt designations.\n\n"
    )
    user_prompt = context + json.dumps(input_json, ensure_ascii=False, indent=2)

    try:
        raw = call_llm(config, SYNONYM_SYSTEM_PROMPT, user_prompt, task="synonym", temperature=0.2)
        result = extract_json_from_response(raw)
    except Exception as e:
        error_msg = str(e)
        # Give actionable hints based on common errors
        hints = []
        if "403" in error_msg or "401" in error_msg or "Authentication" in error_msg:
            hints.append(
                f"API authentication failed. Verify your "
                f"{config.provider.upper()}_API_KEY is correct."
            )
            if config.provider in ("deepseek", "zhipu"):
                hints.append(
                    f"Make sure LLM_BASE_URL points to the correct "
                    f"{config.provider} endpoint. (Auto-detected by default)"
                )
        elif "Connection" in error_msg or "timeout" in error_msg or "Name or service not known" in error_msg:
            hints.append(
                "Cannot reach the API server. Check your network connection or VPN."
            )
        elif "model" in error_msg.lower():
            model = config.get_model("synonym")
            hints.append(
                f"Model '{model}' may not exist or be accessible. "
                f"Check LLM_SYNONYM_MODEL or your provider's available models."
            )
        hint_text = "\n  ".join(hints) if hints else ""
        if hint_text:
            print(f"\n  Synonym expansion failed: {e}\n  Hints:\n  {hint_text}")
        else:
            print(f"\n  Synonym expansion failed: {e}")
        return []

    # Collect new synonyms not already in original list
    new_synonyms = set()
    syn_list = result.get("synonyms", [])
    if not isinstance(syn_list, list):
        return []

    for item in syn_list:
        if isinstance(item, dict):
            for _, syns in item.items():
                if isinstance(syns, list):
                    for s in syns:
                        s_clean = s.strip()
                        if s_clean and s_clean not in protein_names:
                            new_synonyms.add(s_clean)

    return sorted(new_synonyms)
