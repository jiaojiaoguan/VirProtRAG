"""
Query construction from viral protein metadata.
Reference: step1_build_base_query.py, step5_build_base_query_MedCPT.py
"""

import re


def _clean_text(text: str) -> str:
    """Remove redundant spaces and special characters."""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\[\]{}]", "", text)
    return text


def _split_names(value: str) -> list[str]:
    """Split comma/semicolon-separated names into a cleaned list."""
    if not value or not value.strip():
        return []
    parts = re.split(r"[,;]+", value.strip())
    return [_clean_text(p) for p in parts if p.strip()]


def _extract_protein_aliases(text: str) -> list[str]:
    """Extract main name and parenthetical aliases from a protein name string."""
    if not text or not text.strip():
        return []
    text = re.sub(r"\(EC [^)]+\)", "", text)  # remove EC numbers
    names = []
    main = text.split("(")[0].strip()
    if main:
        names.append(_clean_text(main))
    for alias in re.findall(r"\(([^)]+)\)", text):
        alias = alias.strip()
        if alias and not alias.startswith("EC "):
            names.append(_clean_text(alias))
    return [n for n in names if len(n) > 1]


def _parse_cleaved_sections(text: str) -> list[str]:
    """Parse [Cleaved into:] and [Includes:] blocks for sub-protein names."""
    if "[" not in text:
        return []
    names = []
    for block in re.findall(r"\[(Cleaved into|Includes):(.+?)\]", text):
        for part in re.split(r";|,", block[1]):
            part = part.strip()
            if part:
                names.extend(_extract_protein_aliases(part))
    return names


def build_protein_name_list(protein_names: str) -> list[str]:
    """Given raw protein names (comma-separated), return deduplicated name list.

    Handles UniProt-style polyprotein names with [Cleaved into:] sections.
    """
    raw_list = _split_names(protein_names)
    all_names = set()
    for name in raw_list:
        # Extract before brackets
        before = name.split("[")[0]
        all_names.update(_extract_protein_aliases(before))
        # Parse cleavage sections
        all_names.update(_parse_cleaved_sections(name))
    return sorted([n for n in all_names if len(n) > 1])


def build_organism_name_list(organism: str) -> list[str]:
    """Given raw organism string, return main name + parenthetical aliases."""
    raw = str(organism).strip()
    if not raw:
        return []
    names = [_clean_text(raw.split("(")[0])]
    for segment in re.findall(r"\(([^)]+)\)", raw):
        cleaned = _clean_text(segment)
        if cleaned and cleaned not in names:
            names.append(cleaned)
    return names


def build_bm25_query(
    protein_names: str,
    gene_names: str,
    organism: str,
) -> str:
    """Build a BM25 PubMed query expression from protein metadata.

    Format: (protein1 OR protein2 OR gene1 ...) AND (organism1 OR organism2 ...)

    Reference: step1_build_base_query.py final_query_expr construction.
    """
    protein_list = _split_names(protein_names)
    gene_list = _split_names(gene_names)
    org_list = build_organism_name_list(organism)

    # Build left part (protein + gene terms)
    left_terms = []
    for p in protein_list:
        left_terms.append(f'"{p}"')
    for g in gene_list:
        left_terms.append(f'"{g}"')

    left_str = " OR ".join(left_terms) if left_terms else ""

    # Build right part (organism terms)
    right_terms = [f'"{o}"' for o in org_list]
    right_str = " OR ".join(right_terms) if right_terms else ""

    if left_str and right_str:
        return f"({left_str}) AND ({right_str})"
    elif left_str:
        return f"({left_str})"
    elif right_str:
        return f"({right_str})"
    else:
        return ""


def build_medcpt_queries(
    protein_names: str,
    gene_names: str,
    organism: str,
) -> list[str]:
    """Build multi-view MedCPT queries covering complementary functional aspects.

    Generates queries for different functional dimensions:
    - catalytic activity
    - biological process
    - molecular function
    - interaction partners
    - subcellular localization
    - structural features
    - post-translational modifications

    Reference: step5_build_base_query_MedCPT.py and medcpt JSONL query templates.
    """
    protein_list = _split_names(protein_names)
    gene_list = _split_names(gene_names)
    org_list = build_organism_name_list(organism)

    # Use the first protein name and organism as primary identifiers
    primary_protein = protein_list[0] if protein_list else "target protein"
    primary_gene = gene_list[0] if gene_list else ""
    primary_org = org_list[0] if org_list else ""

    # Build synonym list for use in queries
    all_identifiers = protein_list + gene_list
    name_variants = " OR ".join(
        f'"{n}"' for n in all_identifiers[:5]
    )  # limit to 5 names

    aspects = [
        f"Catalytic mechanism and active residues of {primary_protein} in {primary_org}.",
        f"Enzymatic reaction and substrate specificity of {primary_protein} from {primary_org}.",
        f"Biological process involving {primary_protein} in {primary_org}.",
        f"Functional pathway of {primary_protein} in {primary_org}.",
        f"Molecular function and biochemical role of {primary_protein} in {primary_org}.",
        f"Interaction partners of {primary_protein} from {primary_org}.",
        f"Host proteins interacting with {primary_protein} in {primary_org}.",
        f"Complex formation and subunit associations of {primary_protein} from {primary_org}.",
        f"Subcellular localization of {primary_protein} in {primary_org}.",
        f"Post‑translational modifications of {primary_protein} in {primary_org}.",
        f"3D structure and conserved domains of {primary_protein} in {primary_org}.",
        f"Functional domains or motifs determining {primary_protein} activity in {primary_org}.",
    ]

    if primary_gene:
        aspects.append(
            f"Gene {primary_gene} encoding {primary_protein} function in {primary_org}."
        )

    return aspects
