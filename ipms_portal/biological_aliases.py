from __future__ import annotations

import re
from typing import Optional

from ipms_portal.constants import BAF_SUBUNITS

BAF_SUBUNIT_SET = set(BAF_SUBUNITS)


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

    return bio, domain, display


def infer_bait_gene_from_label(label: str | None) -> Optional[str]:
    """
    Heuristic: find a known BAF subunit token inside the sample label.
    """
    if not label:
        return None
    for tok in str(label).replace("-", "_").split("_"):
        t = tok.strip().upper()
        if t in BAF_SUBUNIT_SET:
            return t
    return None
