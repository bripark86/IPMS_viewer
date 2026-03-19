from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentMeta:
    filename: str
    path: str
    mtime: float
    cell_line: Optional[str]
    bait: Optional[str]
    replicate: Optional[int]


def _normalize_col(s: str) -> str:
    # Normalize to make messy headers robust.
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _find_header_row(path: str, required_markers: tuple[str, str], max_lines: int = 250) -> int:
    """
    Scan the file line-by-line to find the first row containing both required markers.
    Returns a 0-based row index usable as pandas `header=<idx>`.
    """
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
    """
    Make sure we have canonical columns:
    - Gene Symbol
    - Peptide
    - LDA Probability
    - XCorr (optional)
    """
    normalized = {_normalize_col(c): c for c in df.columns}
    # Try direct matches first
    def resolve(target: str) -> Optional[str]:
        target_n = _normalize_col(target)
        if target_n in normalized:
            return normalized[target_n]
        # Fallback: collapse spaces and lowercase
        target_f = re.sub(r"[\s_]+", "", target_n).lower()
        for col in df.columns:
            col_f = re.sub(r"[\s_]+", "", _normalize_col(col)).lower()
            if col_f == target_f:
                return col
        return None

    col_gene = resolve("Gene Symbol")
    col_peptide = resolve("Peptide")
    col_lda = resolve("LDA Probability")
    col_xcorr = resolve("XCorr")

    if col_gene is None or col_peptide is None:
        raise ValueError("Missing required columns for aggregation (Gene Symbol and/or Peptide).")
    if col_lda is None:
        # We'll allow LDA Probability missing; confidence/log values become NaN.
        col_lda = None

    keep = {"Gene Symbol": col_gene, "Peptide": col_peptide}
    if col_lda is not None:
        keep["LDA Probability"] = col_lda
    if col_xcorr is not None:
        keep["XCorr"] = col_xcorr

    df2 = df.copy()
    df2 = df2[list(keep.values())].rename(columns={v: k for k, v in keep.items()})
    return df2


def load_and_aggregate_csv(path: str, required_eps: float = 1e-12) -> pd.DataFrame:
    """
    Load one peptide-level CSV (robust header detection), then aggregate to gene-level.
    Output schema includes:
    - Gene Symbol
    - Spectral Count
    - Unique Peptides
    - Confidence Score (mean of LDA Probability)
    - Log_Prob = -log10(avg probability + eps)
    """
    header_idx = _find_header_row(
        path, required_markers=("Gene Symbol", "Peptide"), max_lines=300
    )
    df_raw = pd.read_csv(
        path,
        header=header_idx,
        engine="python",
        sep=None,  # infer delimiter
        dtype=str,
        on_bad_lines="skip",
    )

    # Normalize column names; then select/rename to canonical set.
    df_raw.columns = [_normalize_col(c) for c in df_raw.columns]
    df = _select_and_rename_columns(df_raw)

    # Ensure correct dtypes for aggregation.
    df["Gene Symbol"] = df["Gene Symbol"].astype(str).str.strip()
    df["Peptide"] = df["Peptide"].astype(str).str.strip()

    if "LDA Probability" in df.columns:
        df["LDA Probability"] = pd.to_numeric(df["LDA Probability"], errors="coerce")
    else:
        df["LDA Probability"] = np.nan

    grouped = df.groupby("Gene Symbol", dropna=False, sort=False)
    spectral_count = grouped.size().rename("Spectral Count")
    unique_peptides = grouped["Peptide"].nunique(dropna=True).rename("Unique Peptides")
    # pandas groupby reductions already skip NaN by default.
    avg_lda = grouped["LDA Probability"].mean().rename("Confidence Score")

    out = pd.concat([spectral_count, unique_peptides, avg_lda], axis=1).reset_index()
    out["Log_Prob"] = -np.log10(out["Confidence Score"].fillna(0.0).clip(lower=0.0) + required_eps)
    out.loc[out["Confidence Score"].isna(), "Log_Prob"] = np.nan

    return out


_FILENAME_REPLICATE = re.compile(r"(?P<replicate>\d+)(?:\D|$)")


def extract_metadata_from_filename(filename: str) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Expected filename pattern resembles:
    89849_118070_jl_A549_SMARCA2_1.csv
    We parse the last 3 underscore-separated segments as:
    cell_line, bait, replicate.
    """
    stem = os.path.basename(filename)
    if stem.lower().endswith(".csv"):
        stem = stem[: -len(".csv")]

    parts = stem.split("_")
    if len(parts) < 4:
        return None, None, None

    cell_line = parts[-3] or None
    bait = parts[-2] or None
    rep_part = parts[-1]

    m = _FILENAME_REPLICATE.search(rep_part)
    replicate = int(m.group("replicate")) if m else None
    return cell_line, bait, replicate


def scan_csv_files(data_dir: str) -> list[ExperimentMeta]:
    """
    Scan for CSVs and produce ExperimentMeta objects.
    Sorted newest -> oldest by file modification time.
    """
    metas: list[ExperimentMeta] = []
    for entry in os.scandir(data_dir):
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith(".csv"):
            continue
        cell_line, bait, replicate = extract_metadata_from_filename(entry.name)
        metas.append(
            ExperimentMeta(
                filename=entry.name,
                path=entry.path,
                mtime=entry.stat().st_mtime,
                cell_line=cell_line,
                bait=bait,
                replicate=replicate,
            )
        )

    metas.sort(key=lambda m: m.mtime, reverse=True)
    return metas


def add_baf_core_indicator(df_gene_level: pd.DataFrame, baf_subunits: list[str]) -> pd.DataFrame:
    df = df_gene_level.copy()
    baf_set = set(baf_subunits)
    df["is_baf_core"] = df["Gene Symbol"].isin(baf_set)
    return df

