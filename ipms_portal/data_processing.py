from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from ipms_portal.biological_aliases import infer_bait_gene_from_label, resolve_biological_fields

BAF_SUBUNITS = {
    "SMARCA4", "SMARCA2", "SMARCB1", "SMARCC1", "SMARCC2", "ARID1A", "ARID1B",
    "ARID2", "PBRM1", "BRD7", "BRD9", "ACTL6A", "ACTL6B", "BCL7A", "BCL7B",
    "BCL7C", "DPF1", "DPF2", "DPF3", "PHF10", "SS18", "SS18L1", "SMARCD1",
    "SMARCD2", "SMARCD3", "GLTSCR1",
}
COMMON_CELL_LINES = {"K562", "MOLM13", "G401", "A549", "OCILY1"}


@dataclass(frozen=True)
class ExperimentMeta:
    """One row per CSV under Data/ (recursive)."""

    file_key: str
    rel_path: str
    filename: str
    path: str
    mtime: float
    investigator: Optional[str]
    session_id: Optional[str]
    initials: Optional[str]
    label: Optional[str]


def _normalize_col(s: str) -> str:
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _read_text_lines_with_fallback(path: str, max_lines: int = 300) -> list[str]:
    for enc in ("utf-8", "latin1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                lines: list[str] = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)
                return lines
        except Exception:
            continue
    return []


def _guess_header_row(path: str, max_lines: int = 200) -> int:
    lines = _read_text_lines_with_fallback(path, max_lines=max_lines)
    if not lines:
        return 0
    score_terms = (
        "gene symbol",
        "gene",
        "symbol",
        "accession",
        "peptide",
        "spectral count",
        "total peptides",
        "count",
        "num peptides",
        "lda probability",
        "confidence",
        "prob",
        "score",
    )
    best_i, best_score = 0, -1
    for i, raw in enumerate(lines):
        ll = raw.lower()
        score = sum(1 for t in score_terms if t in ll)
        if score > best_score:
            best_score = score
            best_i = i
    return best_i if best_score >= 2 else 0


def _read_csv_flexible(path: str, header_idx: int) -> pd.DataFrame:
    # Try common encodings first; tolerate malformed bytes/rows.
    tries = [
        {"encoding": "utf-8", "encoding_errors": "ignore"},
        {"encoding": "latin1"},
    ]
    last_exc: Exception | None = None
    for opts in tries:
        try:
            return pd.read_csv(
                path,
                header=header_idx,
                engine="python",
                sep=None,
                dtype=str,
                on_bad_lines="skip",
                **opts,
            )
        except Exception as e:
            last_exc = e
            continue
    if last_exc is not None:
        raise last_exc
    raise ValueError(f"Unable to read CSV: {path}")


def _resolve_any_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    cols = list(df.columns)
    ncols = {_normalize_col(c): c for c in cols}

    def canon(x: str) -> str:
        return re.sub(r"[\s_]+", "", _normalize_col(x)).lower()

    for a in aliases:
        an = _normalize_col(a)
        if an in ncols:
            return ncols[an]
    alias_canon = [canon(a) for a in aliases]
    for c in cols:
        cc = canon(c)
        if cc in alias_canon:
            return c
    for c in cols:
        cc = canon(c)
        if any(ac in cc for ac in alias_canon):
            return c
    return None


def _confidence_numeric(s: pd.Series) -> pd.Series:
    raw = pd.to_numeric(s, errors="coerce")
    pct = raw.notna() & (raw > 1.0) & (raw <= 100.0)
    raw = raw.where(~pct, raw / 100.0)
    raw = raw.where(raw.notna() & (raw >= 0.0) & (raw <= 1.0), np.nan)
    return raw.fillna(0.5).astype(float)


def load_and_aggregate_csv(path: str) -> pd.DataFrame:
    header_idx = _guess_header_row(path, max_lines=250)
    df_raw = _read_csv_flexible(path, header_idx)
    df_raw.columns = [_normalize_col(c) for c in df_raw.columns]

    col_gene = _resolve_any_column(df_raw, ("Gene Symbol", "Gene", "Symbol", "Accession"))
    col_pep = _resolve_any_column(df_raw, ("Peptide", "Sequence"))
    col_spec = _resolve_any_column(df_raw, ("Spectral Count", "Total Peptides", "Count", "Num Peptides"))
    col_conf = _resolve_any_column(df_raw, ("LDA Probability", "Confidence", "Prob", "Score"))

    n = len(df_raw)
    if n == 0:
        out = pd.DataFrame(columns=["Gene Symbol", "Spectral Count", "Unique Peptides", "Confidence Score", "Log_Prob"])
        out.attrs["ipms_lda_probability_column_in_csv"] = False
        return out

    gene = (
        df_raw[col_gene].astype(str).str.strip()
        if col_gene is not None
        else pd.Series([f"UNKNOWN_{i+1}" for i in range(n)], index=df_raw.index)
    )
    gene = gene.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    gene = gene.infer_objects(copy=False).fillna(
        pd.Series([f"UNKNOWN_{i+1}" for i in range(n)], index=df_raw.index)
    )

    if col_spec is not None:
        spec_row = pd.to_numeric(df_raw[col_spec], errors="coerce").fillna(1.0)
    elif col_pep is not None:
        spec_row = pd.Series(1.0, index=df_raw.index)
    else:
        spec_row = pd.Series(1.0, index=df_raw.index)

    pep_token = (
        df_raw[col_pep].astype(str).str.strip()
        if col_pep is not None
        else pd.Series([f"ROW_{i+1}" for i in range(n)], index=df_raw.index)
    )
    pep_token = pep_token.replace({"": np.nan, "nan": np.nan}).fillna(
        pd.Series([f"ROW_{i+1}" for i in range(n)], index=df_raw.index)
    )

    conf_row = _confidence_numeric(df_raw[col_conf]) if col_conf is not None else pd.Series(0.5, index=df_raw.index)

    tmp = pd.DataFrame(
        {
            "Gene Symbol": gene.astype(str),
            "_spec": spec_row.astype(float),
            "_pep": pep_token.astype(str),
            "_conf": conf_row.astype(float),
        }
    )

    grp = tmp.groupby("Gene Symbol", dropna=False, sort=False)
    out = pd.DataFrame(
        {
            "Spectral Count": grp["_spec"].sum(),
            "Unique Peptides": grp["_pep"].nunique(dropna=True),
            "Confidence Score": grp["_conf"].mean(),
        }
    ).reset_index()

    cs = pd.to_numeric(out["Confidence Score"], errors="coerce").fillna(0.5).clip(lower=1e-12, upper=1.0)
    out["Confidence Score"] = cs
    out["Log_Prob"] = -np.log10(cs)
    out.attrs["ipms_lda_probability_column_in_csv"] = bool(col_conf is not None)
    return out


def parse_filename_fuzzy(stem: str) -> tuple[Optional[str], Optional[str], str]:
    """
    Label-agnostic parsing:
    - Session_ID: first two numeric blocks found left-to-right.
    - Initials: third underscore block when present.
    - Sample Label: everything after initials.
    """
    parts = [p for p in str(stem).split("_") if p != ""]
    if not parts:
        return None, None, ""

    nums: list[str] = []
    for p in parts:
        if p.isdigit():
            nums.append(p)
        if len(nums) == 2:
            break
    session_id = f"{nums[0]}_{nums[1]}" if len(nums) == 2 else "Unknown"

    initials = parts[2] if len(parts) >= 3 and parts[2] else "Unknown"
    label = "_".join(parts[3:]) if len(parts) >= 4 else stem
    return session_id, initials, (label if str(label).strip() else stem)


def _parse_label_heuristics(label: str) -> tuple[str, str | None, str | None]:
    toks = [t for t in str(label).split("_") if t]
    if not toks:
        return "Unknown", None, None
    bait = None
    cell = None
    kept: list[str] = []
    for tok in toks:
        up = tok.strip().upper()
        if bait is None and up in BAF_SUBUNITS:
            bait = up
            continue
        if cell is None and up in COMMON_CELL_LINES:
            cell = up
            continue
        kept.append(tok)
    sample_label = "_".join(kept) if kept else "_".join(toks)
    return sample_label, bait, cell


def extract_metadata_from_filename(filename: str) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Back-compat shim: approximate (cell_line, bait, replicate) from fuzzy label.
    May return Nones when not inferrable.
    """
    base = os.path.basename(filename)
    stem = base[: -len(".csv")] if base.lower().endswith(".csv") else base
    _sid, _ini, label = parse_filename_fuzzy(stem)
    bait = infer_bait_gene_from_label(label, stem)
    cell = None
    if label:
        toks = label.split("_")
        if toks:
            cell = toks[0] if toks[0] else None
    return cell, bait, None


def _investigator_from_rel(rel_path: str) -> Optional[str]:
    norm = os.path.normpath(rel_path)
    parent = os.path.dirname(norm)
    if not parent or parent in (".", ""):
        return None
    return parent.split(os.sep)[0]


def scan_csv_files(data_dir: str) -> list[ExperimentMeta]:
    """
    Recursively scan Data/[Investigator]/... for CSVs.
    Sorted newest -> oldest by mtime.
    """
    data_root = os.path.abspath(os.path.normpath(data_dir))
    print(f"[IPMS Debug] scan_csv_files: os.walk starting at {data_root!r}")
    print(f"[IPMS Debug] scan_csv_files: os.getcwd() = {os.getcwd()!r}")

    metas: list[ExperimentMeta] = []
    if not os.path.isdir(data_root):
        print(f"[IPMS Debug] scan_csv_files: not a directory or missing: {data_root!r}")
        return metas

    for root, _dirs, files in os.walk(data_root):
        for name in files:
            nlow = name.lower()
            if name.startswith('.'):
                continue
            if "_processed" in nlow:
                continue
            if not nlow.endswith(".csv"):
                continue
            path = os.path.join(root, name)
            try:
                rel = os.path.relpath(path, data_root)
            except ValueError:
                continue
            rel_posix = rel.replace(os.sep, "/")
            stem = name[: -len(".csv")] if name.lower().endswith(".csv") else name
            session_id, initials, label = parse_filename_fuzzy(stem)
            inv = _investigator_from_rel(rel)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            metas.append(
                ExperimentMeta(
                    file_key=rel_posix,
                    rel_path=rel_posix,
                    filename=name,
                    path=path,
                    mtime=mtime,
                    investigator=inv,
                    session_id=session_id,
                    initials=initials,
                    label=label if label else None,
                )
            )

    metas.sort(key=lambda m: m.mtime, reverse=True)
    print(f"[IPMS Debug] scan_csv_files: discovered {len(metas)} CSV(s) (incl. nested e.g. Data/Janet_Liu/*.csv)")
    return metas


def enrich_meta_dict(meta: ExperimentMeta) -> dict[str, object]:
    """UI / portal metadata dict including biological alias resolution."""
    inv_s = meta.investigator if meta.investigator else None
    label = meta.label or ""
    stem = meta.filename[: -len(".csv")] if meta.filename.lower().endswith(".csv") else meta.filename
    sample_label, bait_from_name, cell_line = _parse_label_heuristics(label)
    bait_guess = bait_from_name or infer_bait_gene_from_label(sample_label, stem)
    bio, domain, disp_bait = resolve_biological_fields(meta.label, stem, bait_gene_guess=bait_guess)

    return {
        "file_key": meta.file_key,
        "rel_path": meta.rel_path,
        "filename": meta.filename,
        "path": meta.path,
        "mtime": meta.mtime,
        "investigator": inv_s,
        "session_id": meta.session_id or "Unknown",
        "initials": meta.initials or "Unknown",
        "label": sample_label or "Unknown",
        "cell_line": cell_line,
        "bait": disp_bait if disp_bait != "N/A" else (bait_guess or "Unknown"),
        "replicate": None,
        "biological_target": bio,
        "domain_details": domain,
    }


def add_baf_core_indicator(df_gene_level: pd.DataFrame, baf_subunits: list[str]) -> pd.DataFrame:
    df = df_gene_level.copy()
    baf_set = {g.upper() for g in baf_subunits}
    df["is_baf_core"] = df["Gene Symbol"].astype(str).str.strip().str.upper().isin(baf_set)
    return df


def add_experiment_biological_columns(df_gene: pd.DataFrame, *, biological_target: str, domain_details: str) -> pd.DataFrame:
    out = df_gene.copy()
    out["Biological Target"] = biological_target
    out["Domain/Details"] = domain_details
    return out
