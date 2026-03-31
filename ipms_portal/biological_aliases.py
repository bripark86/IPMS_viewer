from __future__ import annotations

import re
from typing import Optional

from ipms_portal.constants import BAF_SUBUNITS

BAF_SUBUNIT_SET = set(BAF_SUBUNITS)

# Expanded SWI/SNF (BAF) aliases -> canonical gene symbol.
BAF_ALIAS_TO_GENE: dict[str, str] = {
    "SMARCC1": "SMARCC1",
    "BAF155": "SMARCC1",
    "SMARCC2": "SMARCC2",
    "BAF170": "SMARCC2",
    "SMARCB1": "SMARCB1",
    "BAF47": "SMARCB1",
    "SMARCE1": "SMARCE1",
    "BAF57": "SMARCE1",
    "SMARCD1": "SMARCD1",
    "BAF60A": "SMARCD1",
    "SMARCD2": "SMARCD2",
    "BAF60B": "SMARCD2",
    "SMARCD3": "SMARCD3",
    "BAF60C": "SMARCD3",
    "DPF1": "DPF1",
    "DPF2": "DPF2",
    "BAF45B": "DPF2",
    "SMARCA2": "SMARCA2",
    "BRM": "SMARCA2",
    "SMARCA4": "SMARCA4",
    "BRG1": "SMARCA4",
    "ACTL6A": "ACTL6A",
    "ACTL6B": "ACTL6B",
    "ACTB": "ACTB",
    "ARID1A": "ARID1A",
    "ARID1B": "ARID1B",
    "ARID2": "ARID2",
    "PBRM1": "PBRM1",
    "BRD7": "BRD7",
    "PHF10": "PHF10",
    "BRD9": "BRD9",
    "BICRA": "BICRA",
    "GLTSCR1": "BICRA",
    "BICRAL": "BICRAL",
    "GLTSCR1L": "BICRAL",
    "BCL7A": "BCL7A",
    "BCL7B": "BCL7B",
    "BCL7C": "BCL7C",
}


def expand_biological_target_string(bio: str) -> set[str]:
    """
    Expand lab shorthand like 'SMARCD1/2/3' into individual gene symbols for search/indexing.
    """
    if not bio or not str(bio).strip() or str(bio).upper() in ("N/A", "NA"):
        return set()
    s = str(bio).strip().upper().replace(" ", "")
    if s == "SMARCD1/2/3" or s == "SMARCD1-2-3":
        return {"SMARCD1", "SMARCD2", "SMARCD3"}
    parts = re.split(r"[/,;|]+", s)
    out: set[str] = set()
    for p in parts:
        p = p.strip()
        if p and p not in ("OR", "AND"):
            out.add(p)
    return out


def resolve_biological_fields(
    label: str | None,
    stem: str | None,
    *,
    bait_gene_guess: str | None,
) -> tuple[str, str, str]:
    """
    Returns: (biological_target, domain_details, display_bait_label)
    biological_target: resolved protein/gene target string for tables (N/A if unknown).
    domain_details: isolation / domain note.
    display_bait_label: bait-like label for UI batching (gene symbol if inferred, else N/A).
    """
    label_u = (label or "").upper()
    stem_u = (stem or "").upper()
    blob = f"{label_u} {stem_u}"

    bio = "N/A"
    domain = "N/A"
    display = bait_gene_guess or "N/A"

    if re.search(r"\bSWIFT\b", blob):
        bio = "SMARCD1/2/3"
        domain = "Domain in Isolation"
        display = display if display != "N/A" else "SMARCD1/2/3 (SWIFT)"
    elif re.search(r"\bSANT\b", blob):
        bio = "N/A"
        domain = "SANT Domain Isolation"
    elif re.search(r"\bBRD\b", blob):
        bio = "N/A"
        domain = "Bromodomain Isolation"

    # If no domain shorthand matched, promote inferred bait gene to biological target (e.g. SMARCA2).
    if bio == "N/A" and bait_gene_guess:
        guess = str(bait_gene_guess).strip().upper()
        if guess and guess not in ("N/A", "NA", "NONE"):
            bio = guess

    return bio, domain, display


def infer_bait_gene_from_label(label: str | None, stem: str | None = None) -> Optional[str]:
    """
    Heuristic: find a known BAF subunit token in the sample label and/or filename stem
    (e.g. SMARCA4 in the file name when the parsed label omits it).
    """
    tokens: list[str] = []
    if label:
        tokens.extend(str(label).replace("-", "_").split("_"))
    if stem:
        tokens.extend(str(stem).replace("-", "_").split("_"))
    for tok in tokens:
        t = tok.strip().upper()
        if not t:
            continue
        if t in BAF_ALIAS_TO_GENE:
            return BAF_ALIAS_TO_GENE[t]
        if t in BAF_SUBUNIT_SET:
            return t
    return None
