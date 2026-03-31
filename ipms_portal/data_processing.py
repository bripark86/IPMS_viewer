from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from ipms_portal.biological_aliases import infer_bait_gene_from_label, resolve_biological_fields
from ipms_portal.constants import (
    BAF_SUBUNITS as BAF_SUBUNIT_LIST,
    COMMON_CELL_LINES,
    METADATA_SPLIT_REGEX,
)

pd.set_option("future.no_silent_downcasting", True)

BAF_SUBUNITS = set(BAF_SUBUNIT_LIST)
COMMON_CELL_LINES_SET = {str(x).upper() for x in COMMON_CELL_LINES}


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
    gene = gene.replace({"": np.nan, "nan": np.nan, "None": np.nan}).infer_objects(copy=False)
    gene = gene.fillna(pd.Series([f"UNKNOWN_{i+1}" for i in range(n)], index=df_raw.index)).astype(str)

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


def get_experiment_data(
    file_path: str,
    *,
    biological_target: str = "Unknown",
    domain_details: str = "Unknown",
) -> pd.DataFrame:
    """
    Heavy path: read CSV, aggregate peptide→gene, add BAF flags and experiment columns.
    Call only when a specific experiment is activated (lazy loading).
    """
    try:
        df_gene = load_and_aggregate_csv(file_path)
        lda_csv_ok = bool(df_gene.attrs.get("ipms_lda_probability_column_in_csv", True))
    except Exception:
        base = os.path.basename(file_path)
        stem = base[: -len(".csv")] if base.lower().endswith(".csv") else base
        df_gene = pd.DataFrame(
            {
                "Gene Symbol": [stem.upper() or "UNKNOWN"],
                "Spectral Count": [1.0],
                "Unique Peptides": [1],
                "Confidence Score": [0.5],
                "Log_Prob": [-np.log10(0.5)],
            }
        )
        lda_csv_ok = False
        df_gene.attrs["ipms_lda_probability_column_in_csv"] = False

    df_gene = add_baf_core_indicator(df_gene, list(BAF_SUBUNIT_LIST))
    df_gene = add_experiment_biological_columns(
        df_gene,
        biological_target=str(biological_target),
        domain_details=str(domain_details),
    )
    df_gene.attrs["ipms_lda_probability_column_in_csv"] = lda_csv_ok
    return df_gene


def load_and_process_file(
    file_path: str,
    *,
    biological_target: str = "Unknown",
    domain_details: str = "Unknown",
) -> pd.DataFrame:
    """Alias used by app lazy-loading path."""
    return get_experiment_data(
        file_path,
        biological_target=biological_target,
        domain_details=domain_details,
    )


def crawl_metadata(data_dir: str, file_count: int) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Lightweight startup crawl: os.walk + filename metadata only (no CSV read).
    """
    _ = file_count
    data_root = os.path.abspath(os.path.normpath(data_dir))
    metas = scan_csv_files(data_root)
    experiments_rows: list[dict[str, Any]] = []
    meta_by_file: dict[str, dict[str, Any]] = {}

    for meta in metas:
        enriched = enrich_meta_dict(meta)
        d = dict(enriched)
        meta_by_file[meta.file_key] = d
        experiments_rows.append(
            {
                "file_key": meta.file_key,
                "filename": meta.filename,
                "investigator": d.get("investigator") or "Unknown",
                "session_id": d.get("session_id") or "Unknown",
                "initials": d.get("initials") or "Unknown",
                "label": d.get("label") or "Unknown",
                "cell_line": d.get("cell_line") or "Unknown",
                "bait": d.get("bait") or "Unknown",
                "biological_target": d.get("biological_target") or "Unknown",
                "domain_details": d.get("domain_details") or "Unknown",
                "mtime": meta.mtime,
                "path": d.get("path") or meta.path,
                "n_proteins": pd.NA,
                "n_baf_core": pd.NA,
            }
        )

    experiments_df = pd.DataFrame(experiments_rows)
    if not experiments_df.empty and "mtime" in experiments_df.columns:
        experiments_df = experiments_df.sort_values("mtime", ascending=False).reset_index(drop=True)
    return experiments_df, meta_by_file


def crawl_filenames(data_dir: str, file_count: int) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Alias used by app startup path."""
    return crawl_metadata(data_dir, file_count)


def parse_filename_fuzzy(stem: str) -> tuple[Optional[str], Optional[str], str]:
    """
    Label-agnostic parsing:
    - Session_ID: first two numeric blocks found left-to-right.
    - Initials: third underscore block when present.
    - Sample Label: everything after initials.
    """
    parts = [p for p in re.split(METADATA_SPLIT_REGEX, str(stem)) if p != ""]
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
    toks = [t for t in re.split(METADATA_SPLIT_REGEX, str(label)) if t]
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
        if cell is None and up in COMMON_CELL_LINES_SET:
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
        "resolved_target": (
            bio
            if str(bio).strip().upper() not in ("", "N/A", "NA")
            else (disp_bait if str(disp_bait).strip().upper() not in ("", "N/A", "NA") else (sample_label or "—"))
        ),
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
