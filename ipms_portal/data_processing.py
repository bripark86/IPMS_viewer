from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from ipms_portal.biological_aliases import infer_bait_gene_from_label, resolve_biological_fields


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


def _find_header_row(path: str, required_markers: tuple[str, str], max_lines: int = 250) -> int:
    gene_marker, peptide_marker = required_markers
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            if idx >= max_lines:
                break
            if gene_marker in line and peptide_marker in line:
                return idx
    raise ValueError(
        f"Could not find header row containing both markers {required_markers} within first {max_lines} lines."
    )


def _select_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {_normalize_col(c): c for c in df.columns}

    def resolve(target: str) -> Optional[str]:
        target_n = _normalize_col(target)
        if target_n in normalized:
            return normalized[target_n]
        target_f = re.sub(r"[\s_]+", "", target_n).lower()
        for col in df.columns:
            col_f = re.sub(r"[\s_]+", "", _normalize_col(col)).lower()
            if col_f == target_f:
                return col
        return None

    def resolve_lda_probability_column() -> Optional[str]:
        """
        Use the numeric LDA Probability column only — never LDA Score or unrelated LDA* columns.
        """
        # Exact / near-exact headers first
        for want in ("LDA Probability", "LDA probability", "Prob LDA"):
            wn = _normalize_col(want)
            if wn in normalized:
                return normalized[wn]
            wf = re.sub(r"[\s_]+", "", wn).lower()
            for col in df.columns:
                cf = re.sub(r"[\s_]+", "", _normalize_col(col)).lower()
                if cf == wf:
                    return col
        # Fuzzy: contains lda + prob, excludes score (never use LDA Score)
        candidates: list[str] = []
        for col in df.columns:
            cf = re.sub(r"[\s_]+", "", _normalize_col(col)).lower()
            if "lda" not in cf:
                continue
            if "score" in cf:
                continue
            if "prob" in cf or cf.endswith("prob") or "probability" in cf:
                candidates.append(col)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        def _prob_col_rank(col: str) -> tuple[int, str]:
            cf = re.sub(r"[\s_]+", "", _normalize_col(col)).lower()
            pri = 0
            if "probability" in cf:
                pri -= 3
            if "prob" in cf:
                pri -= 1
            return (pri, col)

        return min(candidates, key=_prob_col_rank)

    col_gene = resolve("Gene Symbol")
    col_peptide = resolve("Peptide")
    col_lda = resolve_lda_probability_column()
    col_xcorr = resolve("XCorr")

    if col_gene is None or col_peptide is None:
        raise ValueError("Missing required columns for aggregation (Gene Symbol and/or Peptide).")

    keep = {"Gene Symbol": col_gene, "Peptide": col_peptide}
    if col_lda is not None:
        keep["LDA Probability"] = col_lda
    if col_xcorr is not None:
        keep["XCorr"] = col_xcorr

    df2 = df.copy()
    df2 = df2[list(keep.values())].rename(columns={v: k for k, v in keep.items()})
    return df2


def _peptide_lda_prob_numeric(df: pd.DataFrame) -> pd.Series:
    """
    Per-peptide LDA probability: coerce with pd.to_numeric(..., errors='coerce').
    Missing / non-numeric / out-of-range values stay NaN (no floor, no fill)—those rows
    do not contribute valid probability mass at gene aggregation time.
    Values in (1, 100] are treated as percents and scaled to (0, 1].
    """
    if "LDA Probability" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    raw = df["LDA Probability"]
    if raw.dtype == object or str(raw.dtype) == "string":
        s = raw.astype(str).str.strip()
        u = s.str.upper()
        bad = u.isin(["", "N/A", "NA", "NAN", "NONE", "NULL", "-", "—"])
        s = s.where(~bad, np.nan)
        lda = pd.to_numeric(s, errors="coerce")
    else:
        lda = pd.to_numeric(raw, errors="coerce")
    lda = lda.astype(float)
    pct = lda.notna() & (lda > 1.0) & (lda <= 100.0)
    lda = lda.where(~pct, lda / 100.0)
    lda = lda.where(lda.notna() & (lda > 0.0) & (lda <= 1.0), np.nan)
    return lda


def load_and_aggregate_csv(path: str) -> pd.DataFrame:
    header_idx = _find_header_row(
        path, required_markers=("Gene Symbol", "Peptide"), max_lines=300
    )
    df_raw = pd.read_csv(
        path,
        header=header_idx,
        engine="python",
        sep=None,
        dtype=str,
        on_bad_lines="skip",
    )

    df_raw.columns = [_normalize_col(c) for c in df_raw.columns]
    df = _select_and_rename_columns(df_raw)

    df["Gene Symbol"] = df["Gene Symbol"].astype(str).str.strip()
    df["Peptide"] = df["Peptide"].astype(str).str.strip()

    df["LDA_Prob"] = _peptide_lda_prob_numeric(df)

    grouped = df.groupby("Gene Symbol", dropna=False, sort=False)
    spectral_count = grouped.size().rename("Spectral Count")
    unique_peptides = grouped["Peptide"].nunique(dropna=True).rename("Unique Peptides")
    avg_lda = grouped["LDA_Prob"].mean().rename("Confidence Score")

    out = pd.concat([spectral_count, unique_peptides, avg_lda], axis=1).reset_index()
    cs = pd.to_numeric(out["Confidence Score"], errors="coerce")
    out["Confidence Score"] = cs
    out["Log_Prob"] = np.nan
    valid_p = cs.notna() & (cs > 0.0) & (cs <= 1.0)
    out.loc[valid_p, "Log_Prob"] = -np.log10(cs.loc[valid_p])

    return out


def parse_filename_fuzzy(stem: str) -> tuple[Optional[str], Optional[str], str]:
    """
    Session ID: first two numeric blocks (left-to-right).
    Initials: token immediately following the second numeric block.
    Label: remainder of stem (do not special-case trailing digits as replicates).
    """
    parts = [p for p in str(stem).split("_") if p != ""]
    if not parts:
        return None, None, ""

    numeric_positions: list[int] = []
    for i, p in enumerate(parts):
        if p.isdigit():
            numeric_positions.append(i)
        if len(numeric_positions) == 2:
            break

    if len(numeric_positions) < 2:
        return None, None, "_".join(parts)

    i0, i1 = numeric_positions[0], numeric_positions[1]
    session_id = f"{parts[i0]}_{parts[i1]}"
    j = i1 + 1
    if j >= len(parts):
        return session_id, None, ""

    initials = parts[j]
    label = "_".join(parts[j + 1 :]) if j + 1 < len(parts) else ""
    return session_id, initials, label


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
            if not name.lower().endswith(".csv"):
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
    bait_guess = infer_bait_gene_from_label(meta.label, stem)
    bio, domain, disp_bait = resolve_biological_fields(meta.label, stem, bait_gene_guess=bait_guess)

    cell_line = None
    if label:
        toks = label.split("_")
        if toks and toks[0]:
            cell_line = toks[0]

    return {
        "file_key": meta.file_key,
        "rel_path": meta.rel_path,
        "filename": meta.filename,
        "path": meta.path,
        "mtime": meta.mtime,
        "investigator": inv_s,
        "session_id": meta.session_id,
        "initials": meta.initials,
        "label": meta.label,
        "cell_line": cell_line,
        "bait": disp_bait if disp_bait != "N/A" else (bait_guess or "N/A"),
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
