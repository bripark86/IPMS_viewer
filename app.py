from __future__ import annotations

import os
import pathlib
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ipms_portal.biological_aliases import (
    BAF_SUBUNIT_SET,
    expand_biological_target_string,
    infer_bait_gene_from_label,
)
from ipms_portal.constants import BAF_CORE_COLOR, BAF_SUBUNITS
from ipms_portal.data_processing import (
    add_baf_core_indicator,
    add_experiment_biological_columns,
    enrich_meta_dict,
    load_and_aggregate_csv,
    scan_csv_files,
)

DATA_SOURCE_DIR_DEFAULT = "/Users/sp1665/Downloads/IPMS/Janet_Liu"
PROJECT_ROOT = os.path.dirname(__file__)
DATA_EMPTY_MESSAGE = "No datasets found in /Data. Please add IP-MS CSVs to begin analysis."
DATA_ROOT = pathlib.Path(__file__).parent.resolve() / "Data"
# Avoid StreamlitAPIException: never assign to portal_dataset_select after its widget is created;
# set this before the main selectbox runs, then pop and apply.
_PENDING_PORTAL_DATASET_KEY = "_pending_portal_dataset"
ACTIVE_DATASET_STATE_KEY = "active_dataset_id"
CURRENT_PROJECT_BAITS = {"BCL7A", "BCL7B", "BCL7C", "SMARCE1"}


def _apply_global_css() -> None:
    styles_path = os.path.join(PROJECT_ROOT, "styles.css")
    if os.path.exists(styles_path):
        with open(styles_path, "r", encoding="utf-8", errors="replace") as f:
            css = f.read()
        # Always inject CSS inside a <style> tag so it renders correctly.
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        return

    # Fallback CSS if styles.css isn't present.
    st.markdown(
        """
        <style>
          body { background-color: #0E1117; color: #E6EDF3; }
          .sidebar .sidebar-content { background-color: #0B0E14; }
          .stMetric { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 12px; }
          div[data-testid="stMetricLabel"] { color: rgba(230,237,243,0.8); font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_local_data_dir(data_dir: pathlib.Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)


def _count_csv_files(data_dir: str) -> int:
    try:
        count = 0
        for _root, _dirs, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    count += 1
        return count
    except FileNotFoundError:
        return 0


def save_uploaded_csvs_to_local_data(
    uploaded_files: list[Any],
    dest_dir: str,
    *,
    subfolder: str = "Imported",
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Save uploaded CSVs under Data/<subfolder>/ (creates nested layout).
    Returns counts: {'saved': X, 'skipped': Y}.
    """
    target_root = os.path.join(dest_dir, subfolder.strip().replace("..", "").strip(os.sep) or "Imported")
    os.makedirs(target_root, exist_ok=True)
    saved = 0
    skipped = 0

    for uploaded in uploaded_files:
        if uploaded is None:
            continue
        name = getattr(uploaded, "name", "")
        if not str(name).lower().endswith(".csv"):
            continue
        safe_name = os.path.basename(name)
        dst_path = os.path.join(target_root, safe_name)
        if (not overwrite) and os.path.exists(dst_path):
            skipped += 1
            continue
        with open(dst_path, "wb") as f:
            f.write(uploaded.getbuffer())
        saved += 1

    return {"saved": saved, "skipped": skipped}


def _pretty_dataset_label(file_key: str, meta: dict[str, Any]) -> str:
    inv = meta.get("investigator") or "N/A"
    sid = meta.get("session_id") or "N/A"
    bait = meta.get("bait") or "N/A"
    lbl = meta.get("label") or "N/A"
    return f"[{inv}] {sid} | Bait:{bait} | {lbl} | {file_key}"


def _meta_matches_bait(meta: dict[str, Any], bait_q: str) -> bool:
    q = bait_q.strip().upper()
    if not q:
        return False
    if str(meta.get("bait", "")).upper() == q:
        return True
    if q in expand_biological_target_string(str(meta.get("biological_target", ""))):
        return True
    lab = str(meta.get("label", "")).upper()
    if q in lab.split("_"):
        return True
    return False


def _lab_consensus_table(
    run_keys: list[str],
    aggregated_by_file: dict[str, pd.DataFrame],
    *,
    min_fraction: float = 0.5,
) -> pd.DataFrame:
    if not run_keys:
        return pd.DataFrame()
    gene_sets: list[set[str]] = []
    spec_lists: list[pd.Series] = []
    for rk in run_keys:
        df = aggregated_by_file.get(rk)
        if df is None or df.empty:
            gene_sets.append(set())
            spec_lists.append(pd.Series(dtype=float))
            continue
        gs = set(df.loc[df["Spectral Count"].fillna(0) > 0, "Gene Symbol"].astype(str))
        gene_sets.append(gs)
        spec_lists.append(df.set_index("Gene Symbol")["Spectral Count"])
    if not gene_sets:
        return pd.DataFrame()

    universe: set[str] = set()
    for gs in gene_sets:
        universe |= gs

    rows: list[dict[str, Any]] = []
    eff_n = len(gene_sets)
    for gene in sorted(universe):
        cnt = sum(1 for gs in gene_sets if gene in gs)
        if eff_n <= 0:
            continue
        if (cnt / eff_n) <= min_fraction:
            continue
        vals: list[float] = []
        for s in spec_lists:
            if gene in s.index:
                try:
                    vals.append(float(s.loc[gene]))
                except Exception:
                    continue
        if not vals:
            continue
        rows.append(
            {
                "Gene Symbol": gene,
                "Runs With Hit": cnt,
                "Fraction of Runs": round(cnt / eff_n, 3),
                "Mean Spectral Count": float(np.mean(vals)),
                "is_baf_core": gene in BAF_SUBUNIT_SET,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(by=["Fraction of Runs", "Mean Spectral Count"], ascending=[False, False]).reset_index(
        drop=True
    )
    return out


def _pick_top_interactor(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    # Stable tie-breaking: Spectral Count desc, then Confidence Score desc, then Gene Symbol asc.
    df2 = df.copy()
    df2["Confidence Score"] = pd.to_numeric(df2["Confidence Score"], errors="coerce")
    df2 = df2.sort_values(
        by=["Spectral Count", "Confidence Score", "Gene Symbol"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return str(df2.iloc[0]["Gene Symbol"])


def _styled_gene_table(df: pd.DataFrame) -> Any:
    # Conditional formatting: highlight BAF core genes in the Gene Symbol column.
    def highlight_gene_symbol(row: pd.Series) -> list[str]:
        if bool(row.get("is_baf_core", False)):
            return [f"background-color: {BAF_CORE_COLOR}; color: #FFFFFF; font-weight: 700;"]
        return [""]

    return df.style.apply(highlight_gene_symbol, axis=1, subset=["Gene Symbol"])


@st.cache_data(show_spinner=True)
def load_portal_data(
    data_dir: str,
    refresh_token: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, Any]], list[str]]:
    """
    Returns:
    - experiments_df: one row per file (metadata + file_key)
    - aggregated_by_file: mapping file_key -> gene-level aggregated df (with is_baf_core, bio columns)
    - meta_by_file: mapping file_key -> dict meta fields (for UI labeling)
    """
    _ = refresh_token  # included to force cache invalidation on refresh
    data_root = os.path.abspath(os.path.normpath(data_dir))
    print(f"[IPMS Debug] load_portal_data: scanning repository Data at {data_root!r} (refresh_token={refresh_token})")
    metas = scan_csv_files(data_root)
    experiments_rows: list[dict[str, Any]] = []
    aggregated_by_file: dict[str, pd.DataFrame] = {}
    meta_by_file: dict[str, dict[str, Any]] = {}
    skipped_files: list[str] = []

    for meta in metas:
        try:
            df_gene = load_and_aggregate_csv(meta.path)
            df_gene = add_baf_core_indicator(df_gene, BAF_SUBUNITS)
            enriched = enrich_meta_dict(meta)
            df_gene = add_experiment_biological_columns(
                df_gene,
                biological_target=str(enriched.get("biological_target", "N/A")),
                domain_details=str(enriched.get("domain_details", "N/A")),
            )
            aggregated_by_file[meta.file_key] = df_gene

            experiments_rows.append(
                {
                    "file_key": meta.file_key,
                    "filename": meta.filename,
                    "investigator": enriched.get("investigator"),
                    "session_id": enriched.get("session_id"),
                    "initials": enriched.get("initials"),
                    "label": enriched.get("label"),
                    "cell_line": enriched.get("cell_line"),
                    "bait": enriched.get("bait"),
                    "biological_target": enriched.get("biological_target"),
                    "domain_details": enriched.get("domain_details"),
                    "mtime": meta.mtime,
                    "path": enriched.get("path"),
                    "n_proteins": int(df_gene["Gene Symbol"].nunique()),
                    "n_baf_core": int(df_gene["is_baf_core"].sum()),
                }
            )
            meta_by_file[meta.file_key] = dict(enriched)
        except Exception:
            skipped_files.append(meta.file_key)
            continue

    experiments_df = pd.DataFrame(experiments_rows)
    if not experiments_df.empty and "mtime" in experiments_df.columns:
        experiments_df = experiments_df.sort_values("mtime", ascending=False).reset_index(drop=True)

    return experiments_df, aggregated_by_file, meta_by_file, skipped_files


def _infer_bait_gene_for_volcano(meta: dict[str, Any]) -> str | None:
    """Gene symbol for bait marker (diamond). Returns None if not inferable (e.g. EV only)."""
    bait_raw = str(meta.get("bait") or "").strip()
    if bait_raw and bait_raw.upper() not in ("N/A", "NA", "—", ""):
        m = re.match(r"^([A-Za-z0-9]+)", bait_raw)
        if m:
            return m.group(1).upper()
    ig = infer_bait_gene_from_label(meta.get("label"))
    return ig.upper() if ig else None


def _short_dataset_label(file_key: str, meta: dict[str, Any], *, max_len: int = 32) -> str:
    fn = str(meta.get("filename") or file_key.split("/")[-1])
    return (fn[: max_len - 1] + "…") if len(fn) > max_len else fn


def _log_prob_y_with_jitter(log_prob: pd.Series, gene_symbols: pd.Series) -> np.ndarray:
    """Slight Y jitter when many genes share the same -log10(p) so markers don't stack."""
    y = log_prob.astype(float).to_numpy()
    if len(y) == 0:
        return y
    genes = tuple(gene_symbols.astype(str).tolist())
    y_key = tuple(np.round(y, 9).tolist())
    seed = abs(hash((genes, y_key))) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    y_r = np.round(y, 12)
    _, inv, counts = np.unique(y_r, return_inverse=True, return_counts=True)
    jitter = np.zeros(len(y), dtype=float)
    dup = counts[inv] > 1
    if dup.any():
        jitter[dup] = rng.normal(0.0, 0.028, size=int(dup.sum()))
    return y + jitter


def _normalize_total_spectral_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Spectral Count scaled to PPM (×1e6) of the run's total spectral count."""
    out = df.copy()
    sc = pd.to_numeric(out["Spectral Count"], errors="coerce").fillna(0.0)
    tot = float(sc.sum())
    if tot > 0:
        out["Spectral Count"] = sc / tot * 1_000_000.0
    else:
        out["Spectral Count"] = sc
    return out


def _make_volcano_figure(
    active_dataset: pd.DataFrame,
    *,
    bait_gene: str | None,
    force_log_x: bool | None = None,
    xaxis_range: tuple[float, float] | None = None,
    yaxis_range: tuple[float, float] | None = None,
    emphasize_baf_only: bool = False,
    xaxis_title: str | None = None,
) -> go.Figure:
    """
    Volcano: X = Spectral Count (abundance proxy), Y = -log10(LDA probability).
    BAF subunits: red circles. Bait: diamond. If emphasize_baf_only, dim non-BAF preys.
    """
    base_color = "#00A6ED"
    core_color = BAF_CORE_COLOR
    bait_non_baf_color = "#FFD700"

    df = active_dataset.copy()
    df["Spectral Count"] = pd.to_numeric(df["Spectral Count"], errors="coerce")
    df["Log_Prob"] = pd.to_numeric(df["Log_Prob"], errors="coerce")
    df = df.dropna(subset=["Log_Prob", "Spectral Count"], how="any")
    df = df[df["Spectral Count"] > 0]

    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=xaxis_title or "Spectral Count",
            yaxis_title="-log10(LDA Probability)",
        )
        return fig

    df = df.copy()
    df["Log_Prob_plot"] = _log_prob_y_with_jitter(df["Log_Prob"], df["Gene Symbol"])

    genes_u = df["Gene Symbol"].astype(str).str.strip().str.upper()
    bait_u = bait_gene.strip().upper() if bait_gene else None
    is_bait = genes_u == bait_u if bait_u else pd.Series(False, index=df.index)
    is_baf = df["is_baf_core"] == True

    df_bait = df.loc[is_bait]
    df_baf_circle = df.loc[is_baf & ~is_bait]
    df_prey = df.loc[~is_baf & ~is_bait]

    pos = df["Spectral Count"].astype(float)
    pos_only = pos[pos > 0]
    use_log_x = False
    if force_log_x is not None:
        use_log_x = force_log_x
    elif len(pos_only) >= 3:
        ratio = float(pos_only.max()) / float(max(pos_only.min(), 1.0))
        use_log_x = pos_only.max() > 80 or ratio >= 25.0

    fig = go.Figure()

    def _scatter_trace(
        d: pd.DataFrame,
        *,
        name: str,
        color: str,
        size: int,
        symbol: str | None,
        prey_dimmed: bool = False,
    ) -> None:
        if d.empty:
            return
        opacity = 0.14 if prey_dimmed else 0.9
        mk: dict[str, Any] = dict(
            size=size,
            color=color,
            opacity=opacity,
            line=dict(width=0.5, color="#FFFFFF"),
        )
        if symbol:
            mk["symbol"] = symbol
        fig.add_trace(
            go.Scatter(
                x=d["Spectral Count"],
                y=d["Log_Prob_plot"],
                mode="markers",
                name=name,
                marker=mk,
                customdata=np.stack(
                    [
                        d["Gene Symbol"].astype(str).values,
                        d["Unique Peptides"].values,
                        d["Spectral Count"].values,
                        d["Log_Prob"].values,
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Spectral Count: %{customdata[2]}<br>"
                    "Unique Peptides: %{customdata[1]}<br>"
                    "-log10(LDA prob): %{customdata[3]:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    prey_dim = bool(emphasize_baf_only)
    _scatter_trace(df_prey, name="Prey / other", color=base_color, size=7, symbol=None, prey_dimmed=prey_dim)
    _scatter_trace(df_baf_circle, name="BAF core (26)", color=core_color, size=12, symbol=None)

    if bait_u and not df_bait.empty:
        bcol = core_color if bait_u in BAF_SUBUNIT_SET else bait_non_baf_color
        _scatter_trace(df_bait, name=f"Bait ({bait_u})", color=bcol, size=18, symbol="diamond")

    default_xtitle = "Spectral Count (log scale)" if use_log_x else "Spectral Count"
    xaxis_cfg: dict[str, Any] = dict(
        title=(xaxis_title or default_xtitle),
        gridcolor="rgba(255,255,255,0.08)",
        type="log" if use_log_x else "linear",
    )
    if xaxis_range is not None:
        lo, hi = float(xaxis_range[0]), float(xaxis_range[1])
        if lo <= 0 or hi <= 0:
            pass
        elif use_log_x:
            xaxis_cfg["range"] = [np.log10(max(lo, 1e-12)), np.log10(max(hi, 1e-12))]
        else:
            xaxis_cfg["range"] = [lo, hi]

    yaxis_cfg: dict[str, Any] = dict(gridcolor="rgba(255,255,255,0.08)", title="-log10(LDA Probability)")
    if yaxis_range is not None:
        yaxis_cfg["range"] = [float(yaxis_range[0]), float(yaxis_range[1])]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=10, t=30, b=30),
        xaxis=xaxis_cfg,
        yaxis=yaxis_cfg,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _volcano_xy_limits_for_compare(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    normalize_total: bool,
) -> tuple[bool, tuple[float, float], tuple[float, float]]:
    """Shared log/linear X choice and padded numeric ranges for paired volcanoes."""
    frames: list[pd.DataFrame] = []
    for raw in (df_a, df_b):
        d = raw.copy()
        if normalize_total:
            d = _normalize_total_spectral_column(d)
        d["Spectral Count"] = pd.to_numeric(d["Spectral Count"], errors="coerce")
        d["Log_Prob"] = pd.to_numeric(d["Log_Prob"], errors="coerce")
        d = d.dropna(subset=["Log_Prob", "Spectral Count"], how="any")
        d = d[d["Spectral Count"] > 0]
        frames.append(d)
    combo = pd.concat(frames, axis=0, ignore_index=True)
    if combo.empty:
        return False, (0.0, 1.0), (0.0, 1.0)

    pos = combo["Spectral Count"].astype(float)
    yv = combo["Log_Prob"].astype(float)
    use_log_x = False
    if len(pos) >= 3:
        ratio = float(pos.max()) / float(max(float(pos.min()), 1.0))
        use_log_x = float(pos.max()) > 80 or ratio >= 25.0

    x_lo, x_hi = float(pos.min()), float(pos.max())
    y_lo, y_hi = float(yv.min()), float(yv.max())
    x_pad = (x_hi - x_lo) * 0.06 + 1e-6
    y_pad = (y_hi - y_lo) * 0.06 + 1e-6
    if use_log_x:
        x_lo = max(x_lo * 0.85, 1e-12)
        x_hi = x_hi * 1.15
    else:
        x_lo = max(0.0, x_lo - x_pad)
        x_hi = x_hi + x_pad
    y_lo = y_lo - y_pad
    y_hi = y_hi + y_pad
    return use_log_x, (x_lo, x_hi), (y_lo, y_hi)


def _correlation_heatmap_spectral(
    selected_keys: list[str],
    aggregated_by_file: dict[str, pd.DataFrame],
    meta_by_file: dict[str, dict[str, Any]],
    *,
    normalize_total: bool,
    baf_genes_only: bool = False,
) -> go.Figure:
    """Pearson correlation of log1p(spectral counts) across experiments."""
    short_labels: dict[str, str] = {
        k: _short_dataset_label(k, meta_by_file.get(k, {})) for k in selected_keys
    }
    col_names = [short_labels[k] for k in selected_keys]

    genes: set[str] = set()
    for k in selected_keys:
        df = aggregated_by_file.get(k)
        if df is None or df.empty:
            continue
        m = pd.to_numeric(df["Spectral Count"], errors="coerce").fillna(0.0) > 0
        genes.update(df.loc[m, "Gene Symbol"].astype(str).str.strip().str.upper().tolist())

    if baf_genes_only:
        genes = genes & BAF_SUBUNIT_SET

    if not genes:
        fig = go.Figure()
        t = (
            "No BAF subunits with spectral signal across these runs"
            if baf_genes_only
            else "No genes with spectral signal to correlate"
        )
        fig.update_layout(template="plotly_dark", title=t)
        return fig

    mat = pd.DataFrame(0.0, index=sorted(genes), columns=col_names, dtype=float)
    for k, c in zip(selected_keys, col_names):
        df = aggregated_by_file.get(k)
        if df is None or df.empty:
            continue
        d = df.copy()
        gs = d["Gene Symbol"].astype(str).str.strip().str.upper()
        sc = pd.to_numeric(d["Spectral Count"], errors="coerce").fillna(0.0)
        if normalize_total and float(sc.sum()) > 0:
            sc = sc / float(sc.sum()) * 1_000_000.0
        sub = pd.Series(sc.values, index=gs)
        sub = sub[~sub.index.duplicated(keep="first")]
        for g in mat.index:
            if g in sub.index:
                mat.at[g, c] = float(sub.loc[g])

    logm = np.log1p(mat)
    corr = logm.corr()
    corr = corr.fillna(0.0)
    z_heat = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(z_heat, 1.0)

    labels = list(corr.columns)
    fig = go.Figure(
        data=go.Heatmap(
            z=z_heat,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="r"),
            hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=(
            "Spectral-count correlation (Pearson on log₁ₚ signal"
            + ("; PPM-normalized" if normalize_total else "")
            + ("; BAF subunits only" if baf_genes_only else "")
            + ")"
        ),
        font=dict(color="#E6EDF3"),
        margin=dict(l=100, r=40, t=60, b=100),
        xaxis=dict(side="bottom", tickangle=-35),
        yaxis=dict(autorange="reversed"),
        height=max(420, 80 + 48 * len(labels)),
    )
    return fig


@st.cache_data(show_spinner=False)
def draw_volcano_plot(
    active_dataset_id: str, active_dataset: pd.DataFrame, *, bait_gene: str | None
) -> go.Figure:
    return _make_volcano_figure(active_dataset, bait_gene=bait_gene)


@st.cache_data(show_spinner=False)
def draw_complex_coverage(active_dataset_id: str, active_dataset: pd.DataFrame) -> go.Figure:
    """All BAF_SUBUNITS on Y-axis; missing subunits show 0 (gaps visible). Order = canonical list."""
    gs_u = active_dataset["Gene Symbol"].astype(str).str.strip().str.upper()
    sc = pd.to_numeric(active_dataset["Spectral Count"], errors="coerce").fillna(0.0)
    counts = pd.DataFrame({"_g": gs_u, "_s": sc}).groupby("_g", sort=False)["_s"].sum()

    rows: list[dict[str, Any]] = []
    for gene in BAF_SUBUNITS:
        v = float(counts.get(gene, 0.0) or 0.0)
        rows.append({"Gene Symbol": gene, "Spectral Count": v})
    merged = pd.DataFrame(rows)

    merged["present"] = merged["Spectral Count"] > 0
    merged["color"] = np.where(merged["present"], BAF_CORE_COLOR, "rgba(120,120,120,0.35)")

    fig = go.Figure(
        go.Bar(
            x=merged["Spectral Count"],
            y=merged["Gene Symbol"].astype(str),
            orientation="h",
            marker=dict(color=merged["color"].tolist()),
            hovertemplate="BAF subunit: %{y}<br>Spectral Count: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=180, r=10, t=30, b=30),
        xaxis_title="Spectral Count",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=11), categoryorder="array", categoryarray=list(BAF_SUBUNITS)),
        title=dict(text="BAF complex coverage (0 = not detected)", font=dict(size=14)),
    )
    return fig


def _comparison_table(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    normalize_total: bool = False,
) -> pd.DataFrame:
    """Outer join on Gene Symbol. Presence (Venn) uses raw spectral > 0; display counts optionally PPM-normalized."""
    a = df_a.copy()
    b = df_b.copy()
    a_raw = pd.to_numeric(a["Spectral Count"], errors="coerce")
    b_raw = pd.to_numeric(b["Spectral Count"], errors="coerce")
    tot_a, tot_b = float(a_raw.sum()), float(b_raw.sum())

    if normalize_total and tot_a > 0:
        a_disp = a_raw / tot_a * 1_000_000.0
    else:
        a_disp = a_raw
    if normalize_total and tot_b > 0:
        b_disp = b_raw / tot_b * 1_000_000.0
    else:
        b_disp = b_raw

    a = a.rename(
        columns={
            "Unique Peptides": "Unique Peptides_A",
            "Confidence Score": "Confidence Score_A",
            "Log_Prob": "Log_Prob_A",
            "is_baf_core": "is_baf_core",
        }
    )
    b = b.rename(
        columns={
            "Unique Peptides": "Unique Peptides_B",
            "Confidence Score": "Confidence Score_B",
            "Log_Prob": "Log_Prob_B",
            "is_baf_core": "is_baf_core",
        }
    )
    a["Spectral Count_A_raw"] = a_raw
    a["Spectral Count_A"] = a_disp
    b["Spectral Count_B_raw"] = b_raw
    b["Spectral Count_B"] = b_disp
    a = a.drop(columns=[c for c in ("Spectral Count",) if c in a.columns], errors="ignore")
    b = b.drop(columns=[c for c in ("Spectral Count",) if c in b.columns], errors="ignore")

    merged = pd.merge(a, b, on="Gene Symbol", how="outer", suffixes=("", "_dup"))

    has_a = merged["Spectral Count_A_raw"].fillna(0) > 0
    has_b = merged["Spectral Count_B_raw"].fillna(0) > 0
    merged["category"] = np.where(has_a & has_b, "Both", np.where(has_a, "Only A", "Only B"))

    if "is_baf_core_dup" in merged.columns:
        merged["is_baf_core"] = merged["is_baf_core"].fillna(merged["is_baf_core_dup"])
        merged = merged.drop(columns=["is_baf_core_dup"])

    merged["max_spectral"] = merged[["Spectral Count_A", "Spectral Count_B"]].max(axis=1, skipna=True)
    merged = merged.sort_values(
        by=["category", "max_spectral", "Gene Symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )

    return merged


def _render_comparative_analytics_tab(
    *,
    selected_keys: list[str],
    dataset_keys_all: list[str],
    aggregated_by_file: dict[str, pd.DataFrame],
    meta_by_file: dict[str, dict[str, Any]],
) -> None:
    if len(dataset_keys_all) < 2:
        st.info("Upload at least two IP-MS CSVs under `/Data` to enable comparative analytics.")
        return

    sel = [str(k) for k in selected_keys if k in aggregated_by_file]
    if len(sel) < 2:
        st.info("In the sidebar, pick **two or more experiments** under **Comparative Analytics**.")
        return

    st.markdown(
        "Compare enrichment across runs. **Two** selections → side-by-side volcanoes and an overlap table; "
        "**three or more** → correlation heatmap of spectral profiles."
    )
    r1, r2 = st.columns(2)
    with r1:
        normalize = st.toggle(
            "Total spectral-count normalization (PPM)",
            value=True,
            key="cmp_normalize_total",
            help="Scale each run so total spectral counts sum to 1e6, comparable MS depth across experiments.",
        )
    with r2:
        baf_only = st.toggle(
            "Show only BAF subunits",
            value=False,
            key="cmp_baf_only",
            help="Restrict the overlap table to BAF members; dim non-BAF points in volcanoes; heatmap uses BAF genes only.",
        )

    st.markdown("#### Selected experiments")
    per_row = 4
    for start in range(0, len(sel), per_row):
        chunk = sel[start : start + per_row]
        cols = st.columns(len(chunk))
        for ci, fk in enumerate(chunk):
            m = meta_by_file.get(fk, {})
            with cols[ci]:
                st.markdown(f"**{_short_dataset_label(fk, m, max_len=44)}**")
                st.caption(
                    f"**Investigator:** {m.get('investigator') or '—'}  \n"
                    f"**Session:** {m.get('session_id') or '—'}  \n"
                    f"**Bait:** {m.get('bait') or '—'}  \n"
                    f"**Label:** {m.get('label') or '—'}"
                )

    if len(sel) == 2:
        k_a, k_b = sel[0], sel[1]
        df_a = aggregated_by_file[k_a]
        df_b = aggregated_by_file[k_b]
        lab_a = _short_dataset_label(k_a, meta_by_file.get(k_a, {}))
        lab_b = _short_dataset_label(k_b, meta_by_file.get(k_b, {}))
        bait_a = _infer_bait_gene_for_volcano(meta_by_file.get(k_a, {}))
        bait_b = _infer_bait_gene_for_volcano(meta_by_file.get(k_b, {}))

        use_log_x, xr, yr = _volcano_xy_limits_for_compare(df_a, df_b, normalize_total=normalize)

        xt = (
            "Spectral signal (PPM, log scale)"
            if normalize and use_log_x
            else "Spectral signal (PPM)"
            if normalize
            else "Spectral Count (log scale)"
            if use_log_x
            else "Spectral Count"
        )

        va = _normalize_total_spectral_column(df_a) if normalize else df_a.copy()
        vb = _normalize_total_spectral_column(df_b) if normalize else df_b.copy()

        st.markdown("#### Side-by-side volcano plots (matched axes)")
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"{lab_a}")
            fig_a = _make_volcano_figure(
                va,
                bait_gene=bait_a,
                force_log_x=use_log_x,
                xaxis_range=xr,
                yaxis_range=yr,
                emphasize_baf_only=baf_only,
                xaxis_title=xt,
            )
            st.plotly_chart(
                fig_a,
                use_container_width=True,
                key=f"cmpv_a_{k_a}_{k_b}_{int(normalize)}_{int(baf_only)}",
            )
        with c2:
            st.caption(f"{lab_b}")
            fig_b = _make_volcano_figure(
                vb,
                bait_gene=bait_b,
                force_log_x=use_log_x,
                xaxis_range=xr,
                yaxis_range=yr,
                emphasize_baf_only=baf_only,
                xaxis_title=xt,
            )
            st.plotly_chart(
                fig_b,
                use_container_width=True,
                key=f"cmpv_b_{k_a}_{k_b}_{int(normalize)}_{int(baf_only)}",
            )

        st.markdown("#### Overlap / unique interactors")
        if normalize:
            st.caption(
                "Spectral columns show **PPM** (normalized); **Common vs unique** is still defined by raw detection (spectral count > 0 in that run)."
            )

        try:
            merged = _comparison_table(df_a, df_b, normalize_total=normalize)
        except Exception as e:
            st.error(f"Could not build comparison table: {e}")
            return

        if merged.empty or "category" not in merged.columns:
            st.warning("No overlap rows were produced for this pair.")
            return

        overlap_n = int((merged["category"] == "Both").sum())
        only_a_n = int((merged["category"] == "Only A").sum())
        only_b_n = int((merged["category"] == "Only B").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Common interactors", overlap_n)
        m2.metric(f"Unique to {lab_a}", only_a_n)
        m3.metric(f"Unique to {lab_b}", only_b_n)

        sp_lbl = "Spectral (PPM)" if normalize else "Spectral Count"
        show_cols = [
            "Gene Symbol",
            "category",
            "Spectral Count_A",
            "Spectral Count_B",
            "Unique Peptides_A",
            "Unique Peptides_B",
            "Confidence Score_A",
            "Confidence Score_B",
            "is_baf_core",
        ]
        use_show = [c for c in show_cols if c in merged.columns]
        df_cmp = merged[use_show].copy()
        df_cmp["is_baf_core"] = df_cmp["is_baf_core"].fillna(False)
        df_cmp["_cat_ord"] = df_cmp["category"].map({"Both": 0, "Only A": 1, "Only B": 2}).fillna(3)
        df_cmp = df_cmp.sort_values(
            by=["is_baf_core", "_cat_ord", "Spectral Count_A", "Spectral Count_B"],
            ascending=[False, True, False, False],
            kind="mergesort",
        )
        df_cmp["Overlap group"] = df_cmp["category"].map(
            {
                "Both": "Common interactors",
                "Only A": f"Unique to {lab_a}",
                "Only B": f"Unique to {lab_b}",
            }
        )
        df_cmp = df_cmp.drop(columns=["category", "_cat_ord"])
        col_a = f"{lab_a} — {sp_lbl}"
        col_b = f"{lab_b} — {sp_lbl}"
        df_cmp = df_cmp.rename(columns={"Spectral Count_A": col_a, "Spectral Count_B": col_b})
        front = ["Gene Symbol", "Overlap group", col_a, col_b]
        rest = [c for c in df_cmp.columns if c not in front]
        df_cmp = df_cmp[front + rest]

        if baf_only:
            df_cmp = df_cmp[df_cmp["is_baf_core"] == True]
            if df_cmp.empty:
                st.info("No BAF subunits in overlap table for this pair (try turning the filter off).")

        if not df_cmp.empty:
            st.dataframe(
                _styled_gene_table(df_cmp),
                use_container_width=True,
                height=min(560, 120 + 22 * min(len(df_cmp), 80)),
            )

    else:
        st.markdown("#### Spectral-count correlation heatmap")
        st.caption(
            "Pearson **r** on **log(1 + spectral counts)** per gene across runs (high r → similar spectral profiles)."
        )
        fig_h = _correlation_heatmap_spectral(
            sel,
            aggregated_by_file,
            meta_by_file,
            normalize_total=normalize,
            baf_genes_only=baf_only,
        )
        st.plotly_chart(
            fig_h,
            use_container_width=True,
            key=f"cmp_corr_{abs(hash(tuple(sel)))}_{int(normalize)}_{int(baf_only)}",
        )


def _dataset_appearance_for_subunit(df_gene: pd.DataFrame, subunit: str) -> bool:
    u = subunit.strip().upper()
    sy = df_gene["Gene Symbol"].astype(str).str.strip().str.upper()
    return bool(((sy == u) & (df_gene["Spectral Count"].fillna(0) > 0)).any())


def search_gene_across_datasets(
    gene_query: str,
    aggregated_by_file: dict[str, pd.DataFrame],
    meta_by_file: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Cross-dataset index over cached gene-level frames. SWIFT/SMARCD1–3 mapping: runs whose
    resolved Biological Target contains the query appear in the index (with observed quantities).
    """
    q = str(gene_query).strip()
    if not q:
        return pd.DataFrame()

    q_upper = q.upper()
    rows: list[dict[str, Any]] = []

    for file_key, df_gene in aggregated_by_file.items():
        if df_gene is None or df_gene.empty or "Gene Symbol" not in df_gene.columns:
            continue
        meta = meta_by_file.get(file_key, {})
        bio = str(meta.get("biological_target", "N/A"))
        bio_expanded = expand_biological_target_string(bio)
        index_match = q_upper in bio_expanded

        mask = df_gene["Gene Symbol"].astype(str).str.strip().str.upper() == q_upper
        hit = df_gene.loc[mask]

        if hit.empty:
            if not index_match:
                continue
            rows.append(
                {
                    "Investigator": meta.get("investigator") or "N/A",
                    "Session ID": meta.get("session_id") or "N/A",
                    "Label": meta.get("label") or "N/A",
                    "Bio-Target": bio,
                    "Spectral Count": 0,
                    "Unique Peptides": 0,
                    "Avg LDA": np.nan,
                    "Dataset ID": meta.get("filename") or file_key.split("/")[-1],
                    "_file_key": file_key,
                }
            )
            continue

        row = hit.iloc[0]
        rows.append(
            {
                "Investigator": meta.get("investigator") or "N/A",
                "Session ID": meta.get("session_id") or "N/A",
                "Label": meta.get("label") or "N/A",
                "Bio-Target": bio,
                "Spectral Count": row.get("Spectral Count"),
                "Unique Peptides": row.get("Unique Peptides"),
                "Avg LDA": row.get("Confidence Score"),
                "Dataset ID": meta.get("filename") or file_key.split("/")[-1],
                "_file_key": file_key,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["Spectral Count"] = pd.to_numeric(out["Spectral Count"], errors="coerce")
    out = out.sort_values("Spectral Count", ascending=False, kind="mergesort").reset_index(drop=True)
    return out


def _style_global_search_results(df_display: pd.DataFrame, *, target_is_baf_core: bool) -> Any:
    """Highlight all rows when the searched protein is a BAF core subunit."""
    if not target_is_baf_core or df_display.empty:
        return df_display

    def highlight_row(_row: pd.Series) -> list[str]:
        style = f"background-color: {BAF_CORE_COLOR}; color: #FFFFFF; font-weight: 600;"
        return [style] * len(_row)

    return df_display.style.apply(highlight_row, axis=1)


def _render_global_search_tab(
    *,
    gene_input: str,
    aggregated_by_file: dict[str, pd.DataFrame],
    meta_by_file: dict[str, dict[str, Any]],
) -> None:
    st.markdown("### Global Protein Search")
    st.caption(
        "Search across all experiments using cached gene-level results (no extra file reads)."
    )

    q = gene_input.strip()
    if not q:
        st.info("Enter a Gene Symbol in the sidebar under **Global Protein Search** to search the library.")
        return

    results = search_gene_across_datasets(q, aggregated_by_file, meta_by_file)
    target_baf = q.strip().upper() in BAF_SUBUNIT_SET

    if target_baf:
        st.markdown(
            "<span style='background:#FF4B4B;color:#fff;padding:4px 10px;border-radius:8px;font-weight:700;'>BAF Core</span>",
            unsafe_allow_html=True,
        )

    if results.empty:
        st.info(f"No detections found for **{q}** in current library.")
        return

    r_obs = results[results["Spectral Count"].fillna(0) > 0]
    max_sc = int(r_obs["Spectral Count"].max()) if not r_obs.empty and r_obs["Spectral Count"].notna().any() else 0
    r_rank = results.copy()
    r_rank["Avg LDA"] = pd.to_numeric(r_rank["Avg LDA"], errors="coerce")
    r_rank = r_rank.sort_values(
        by=["Spectral Count", "Avg LDA"],
        ascending=[False, False],
        kind="mergesort",
    )
    top = r_rank.iloc[0]
    meta0 = meta_by_file.get(str(top.get("_file_key", "")), {})
    primary_bait = meta0.get("bait") or meta0.get("biological_target") or "N/A"

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Hits", f"{len(results)}")
    m2.metric("Highest Enrichment", f"{max_sc}")
    m3.metric("Primary Interaction", str(primary_bait))

    df_show = results.drop(columns=["_file_key"], errors="ignore").copy()

    st.markdown("#### Results")
    styler = _style_global_search_results(df_show, target_is_baf_core=target_baf)
    st.dataframe(styler, use_container_width=True, height=min(520, 42 + 36 * len(df_show)))

    st.markdown("#### Open in Dataset Browser")
    for i, r in results.iterrows():
        fk = r.get("_file_key")
        if not fk:
            continue
        ds_id = r.get("Dataset ID", fk)
        label = f"Open **{ds_id}**"
        if st.button(label, key=f"global_open_{fk}_{i}"):
            st.session_state[_PENDING_PORTAL_DATASET_KEY] = str(fk)
            st.rerun()


def _compute_storage_mb_under_dir(data_dir: str) -> float:
    total_bytes = 0
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                fp = os.path.join(root, f)
                try:
                    total_bytes += os.path.getsize(fp)
                except OSError:
                    continue
    return total_bytes / (1024 * 1024)


def _aggregated_download_name(meta: dict[str, Any], fallback_filename: str) -> str:
    bait = (meta.get("bait") or "UnknownBait").replace(" ", "")
    cell = (meta.get("cell_line") or "UnknownCell").replace(" ", "")
    return f"{bait}_{cell}_Aggregated.csv"


def _vault_resolve_target_from_meta(meta: dict[str, Any]) -> str:
    """Prefer biological target, then inferred bait; fall back to sample label."""
    bt = str(meta.get("biological_target") or "").strip()
    if bt and bt.upper() not in ("N/A", "NA", ""):
        return bt
    bait = str(meta.get("bait") or "").strip()
    if bait and bait.upper() not in ("N/A", "NA", ""):
        return bait
    lbl = str(meta.get("label") or "").strip()
    return lbl if lbl else "—"


def _vault_target_display_is_baf(target_display: str) -> bool:
    toks = re.split(r"[^A-Z0-9]+", str(target_display).upper())
    return any(t in BAF_SUBUNIT_SET for t in toks if t)


def _style_vault_experiment_table(df: pd.DataFrame) -> Any:
    """Highlight BAF-related targets in the Target column."""

    def highlight_row(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        if "Target" not in row.index:
            return styles
        pos = row.index.get_loc("Target")
        if isinstance(pos, (np.ndarray, slice)):
            return styles
        if _vault_target_display_is_baf(str(row["Target"])):
            styles[int(pos)] = (
                f"background-color: {BAF_CORE_COLOR}; color: #FFFFFF; font-weight: 600;"
            )
        return styles

    return df.style.apply(highlight_row, axis=1)


def _vault_action_label(fk: str, meta_by_file: dict[str, dict[str, Any]]) -> str:
    m = meta_by_file.get(fk, {})
    inv = m.get("investigator") or "—"
    sid = m.get("session_id") or "—"
    tgt = _vault_resolve_target_from_meta(m)
    fn = str(m.get("filename") or fk.split("/")[-1])
    return f"{inv} | {sid} | {tgt} | {fn}"


def _render_data_vault_tab(
    *,
    experiments_df: pd.DataFrame,
    aggregated_by_file: dict[str, pd.DataFrame],
    meta_by_file: dict[str, dict[str, Any]],
    data_dir: str,
) -> None:
    st.markdown("### BAF-Vault: Experiment Directory")
    if experiments_df.empty:
        st.info(DATA_EMPTY_MESSAGE)
        return

    file_keys = experiments_df["file_key"].astype(str).tolist()
    unique_baits = int(experiments_df["bait"].dropna().nunique()) if "bait" in experiments_df.columns else 0
    storage_mb = _compute_storage_mb_under_dir(data_dir)

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Experiments Stored", f"{len(file_keys)}")
    s2.metric("Unique Baits", f"{unique_baits}")
    s3.metric("Storage Used (MB)", f"{storage_mb:.2f}")

    st.markdown("#### Filter & layout")
    bait_query = st.text_input("Search (bait / target / label / filename)", placeholder="e.g. SMARCA4, SWIFT, BCL7")
    bait_q = bait_query.strip().upper()

    work = experiments_df.copy()
    if bait_q:
        def row_matches(r: pd.Series) -> bool:
            parts = [
                str(r.get("bait", "")),
                str(r.get("biological_target", "")),
                str(r.get("label", "")),
                str(r.get("filename", "")),
                str(r.get("file_key", "")),
                str(r.get("investigator", "")),
            ]
            blob = " ".join(parts).upper()
            return bait_q in blob

        work = work[work.apply(row_matches, axis=1)]

    fks = work["file_key"].astype(str)
    work["Target"] = work["file_key"].astype(str).map(lambda fk: _vault_resolve_target_from_meta(meta_by_file.get(fk, {})))
    work["Date"] = pd.to_datetime(work["mtime"], unit="s", errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    work["File"] = work["file_key"].astype(str).map(
        lambda fk: str(meta_by_file.get(fk, {}).get("filename") or fk.split("/")[-1])
    )
    work["Domain"] = work["file_key"].astype(str).map(lambda fk: str(meta_by_file.get(fk, {}).get("domain_details") or "—"))

    group_mode = st.radio(
        "Group rows by",
        ["None (flat table)", "Investigator", "Session ID"],
        horizontal=True,
        key="vault_group_mode",
    )

    display_base = work[
        [
            "investigator",
            "session_id",
            "Target",
            "label",
            "File",
            "Domain",
            "Date",
        ]
    ].rename(
        columns={
            "investigator": "Investigator",
            "session_id": "Session ID",
            "label": "Label",
        }
    )

    if display_base.empty:
        st.caption("No experiments match the current filter.")
        return

    def sort_key_inv(s: str) -> tuple[Any, ...]:
        return (str(s).upper() == "N/A", str(s).upper() == "NONE", str(s).lower())

    def sort_key_session(s: str) -> tuple[Any, ...]:
        return (str(s).upper() in ("N/A", "NONE", ""), str(s))

    st.markdown("#### Experiments")
    row_h = min(520, 80 + 28 * len(display_base))

    if group_mode == "None (flat table)":
        view = display_base.sort_values(
            by=["Investigator", "Session ID", "Target", "File"],
            key=lambda c: c.map(str) if c.name in ("Investigator", "Session ID", "Target", "File") else c,
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
        st.dataframe(_style_vault_experiment_table(view), use_container_width=True, height=row_h)
    elif group_mode == "Investigator":
        invs = sorted(display_base["Investigator"].fillna("—").unique(), key=sort_key_inv)
        for inv in invs:
            sub = display_base[display_base["Investigator"].fillna("—") == inv].sort_values(
                by=["Session ID", "Target", "File"], kind="mergesort"
            )
            with st.expander(f"{inv} — {len(sub)} experiment(s)", expanded=len(invs) <= 12):
                st.dataframe(
                    _style_vault_experiment_table(sub.reset_index(drop=True)),
                    use_container_width=True,
                    height=min(360, 60 + 26 * len(sub)),
                )
    else:
        sids = sorted(display_base["Session ID"].fillna("—").unique(), key=sort_key_session)
        for sid in sids:
            sub = display_base[display_base["Session ID"].fillna("—") == sid].sort_values(
                by=["Investigator", "Target", "File"], kind="mergesort"
            )
            with st.expander(f"{sid} — {len(sub)} experiment(s)", expanded=len(sids) <= 16):
                st.dataframe(
                    _style_vault_experiment_table(sub.reset_index(drop=True)),
                    use_container_width=True,
                    height=min(360, 60 + 26 * len(sub)),
                )

    st.markdown("#### Actions")
    st.caption("Pick one experiment from the filtered list above, then download or delete.")
    action_keys = work["file_key"].astype(str).tolist()
    sel = st.selectbox(
        "Experiment",
        options=action_keys,
        format_func=lambda fk: _vault_action_label(fk, meta_by_file),
        key="vault_action_select",
    )
    meta_sel = meta_by_file.get(sel, {})
    fpath = str(meta_sel.get("path") or os.path.join(data_dir, sel))
    fn = str(meta_sel.get("filename") or sel.split("/")[-1])
    df_agg = aggregated_by_file.get(sel)

    ac1, ac2, ac3 = st.columns([1, 1, 2])
    with ac1:
        if df_agg is not None and not df_agg.empty:
            df_dl = df_agg.sort_values(
                by=["is_baf_core", "Spectral Count", "Confidence Score"],
                ascending=[False, False, False],
                kind="mergesort",
            )
            st.download_button(
                "Download aggregated CSV",
                data=df_dl.to_csv(index=False).encode("utf-8"),
                file_name=_aggregated_download_name(meta_sel, fn),
                mime="text/csv",
                key="vault_download_selected",
            )
        else:
            st.caption("No aggregated data.")
    with ac2:
        del_conf = st.checkbox("Confirm delete", key="vault_delete_confirm")
    with ac3:
        if st.button("Delete selected file", key="vault_delete_btn", type="primary"):
            if not del_conf:
                st.warning("Enable **Confirm delete** before removing a run.")
            else:
                try:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                    load_portal_data.clear()
                    st.success(f"Deleted `{sel}`.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")


def main() -> None:
    st.set_page_config(page_title="BAF-Complex IP-MS Analytics Portal", layout="wide")
    # Inject CSS at app startup so it applies globally.
    _apply_global_css()

    data_dir = DATA_ROOT
    print(f"[IPMS Debug] App using Data root: {str(data_dir)!r}")
    print(f"[IPMS Debug] os.getcwd(): {os.getcwd()!r}")

    st.title("BAF-Vault: IP-MS Encyclopedia")
    st.caption(
        "Nested `Data/[Investigator]/` repository with peptide→gene aggregation, SWIFT alias mapping, and lab-scale search."
    )

    matching: list[str] = []
    with st.sidebar:
        _ensure_local_data_dir(data_dir)

        if "data_refresh_token" not in st.session_state:
            st.session_state["data_refresh_token"] = 0
        st.markdown("### Data Directory & File Management")
        st.caption(f"Local data: `{str(data_dir)}` (use `Data/[Investigator]/file.csv`)")

        upload_subfolder = st.text_input(
            "Upload subfolder under Data/",
            value="Imported",
            help="Creates `Data/<subfolder>/` and places uploaded CSVs there.",
        )
        overwrite = st.checkbox("Overwrite existing files on upload", value=False)
        uploaded_files = st.file_uploader(
            "Upload IP-MS CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload one or more CSVs. Files are saved under Data/<subfolder>/ for analysis.",
        )
        if uploaded_files:
            try:
                counts = save_uploaded_csvs_to_local_data(
                    uploaded_files,
                    str(data_dir),
                    overwrite=overwrite,
                    subfolder=str(upload_subfolder or "Imported"),
                )
                st.success(
                    f"Upload complete: saved {counts['saved']} file(s), skipped {counts['skipped']} existing file(s)."
                )
            except Exception as e:
                st.error(f"Upload failed: {e}")

        if st.button("Refresh Data"):
            st.session_state["data_refresh_token"] += 1
            load_portal_data.clear()

        local_csv_count = _count_csv_files(str(data_dir))
        if local_csv_count == 0:
            st.info(DATA_EMPTY_MESSAGE)
            st.stop()

        st.markdown("---")
        st.markdown("### Dataset Browser")
        try:
            experiments_df, aggregated_by_file, meta_by_file, skipped_files = load_portal_data(
                str(data_dir),
                st.session_state["data_refresh_token"],
            )
        except Exception as e:
            st.error(
                "Could not load experiment library. One or more files may be corrupted or unreadable. "
                f"Details: {e}"
            )
            st.stop()

        if experiments_df.empty:
            st.warning("No CSV datasets were parsed successfully. Please check CSV format/headers.")
            st.stop()

        if skipped_files:
            st.caption(f"Skipped {len(skipped_files)} file(s) due to parsing issues.")

        dataset_keys_all = experiments_df["file_key"].astype(str).tolist()
        meta_lookup = meta_by_file
        labels = {k: _pretty_dataset_label(k, meta_lookup.get(k, {})) for k in dataset_keys_all}

        # Apply investigator / global-search "set active" before portal_dataset_select widget is created.
        if _PENDING_PORTAL_DATASET_KEY in st.session_state:
            pending_fk = st.session_state.pop(_PENDING_PORTAL_DATASET_KEY)
            if pending_fk and str(pending_fk) in dataset_keys_all:
                st.session_state["portal_project_view"] = "All Experiments"
                st.session_state["portal_dataset_select"] = str(pending_fk)
            elif pending_fk:
                st.warning(f"Could not switch active dataset (not in library): {pending_fk!r}")

        # Bait-based project grouping for active experiments.
        current_project_keys = [
            k
            for k in dataset_keys_all
            if str(meta_lookup.get(k, {}).get("bait", "")).upper() in CURRENT_PROJECT_BAITS
        ]
        if current_project_keys:
            if "portal_project_view" not in st.session_state:
                st.session_state["portal_project_view"] = "Current Projects"
            st.radio(
                "Project View",
                ["Current Projects", "All Experiments"],
                horizontal=True,
                key="portal_project_view",
            )
            project_view = str(st.session_state.get("portal_project_view", "All Experiments"))
        else:
            st.session_state["portal_project_view"] = "All Experiments"
            project_view = "All Experiments"

        dataset_keys_for_picker = (
            current_project_keys if project_view == "Current Projects" else dataset_keys_all
        )

        if not dataset_keys_for_picker:
            dataset_keys_for_picker = dataset_keys_all

        ds_cur = st.session_state.get("portal_dataset_select")
        if ds_cur not in dataset_keys_for_picker and dataset_keys_for_picker:
            st.session_state["portal_dataset_select"] = dataset_keys_for_picker[0]

        st.selectbox(
            "Select dataset",
            options=dataset_keys_for_picker,
            format_func=lambda k: labels.get(k, k),
            key="portal_dataset_select",
        )

        st.markdown("#### Browse by Investigator")
        by_inv: dict[str, list[str]] = defaultdict(list)
        for fk in dataset_keys_all:
            inv = str(meta_lookup.get(fk, {}).get("investigator") or "N/A")
            by_inv[inv].append(fk)
        for inv in sorted(by_inv.keys(), key=lambda s: (s == "N/A", s.lower())):
            safe = str(inv).replace("/", "_").replace(" ", "_")[:48]
            with st.expander(f"{inv} ({len(by_inv[inv])})"):
                pick = st.selectbox(
                    "Experiments",
                    options=sorted(by_inv[inv]),
                    key=f"inv_pick_{safe}",
                    format_func=lambda kk: labels.get(kk, kk),
                )
                if st.button("Set as active dataset", key=f"inv_set_{safe}", type="primary"):
                    st.session_state[_PENDING_PORTAL_DATASET_KEY] = str(pick)
                    st.rerun()

        st.markdown("---")
        st.markdown("### Comparative Analytics")
        st.multiselect(
            "Experiments to compare (select ≥2)",
            options=dataset_keys_all,
            format_func=lambda kk: labels.get(kk, kk),
            key="compare_analytics_pick",
            help="Two runs → **Comparative Analytics** tab shows side-by-side volcanoes + overlap table. Three or more → correlation heatmap.",
        )

        st.markdown("---")
        st.markdown("### Global Protein Search")
        st.text_input(
            "Gene Symbol (all experiments)",
            placeholder="e.g. ACTL6A, MYC",
            key="global_gene_query",
            help="Open the **Global Search** tab for the full cross-experiment table and metrics.",
        )

        st.markdown("---")
        st.markdown("### Subunit Quick-Search")
        selected_subunit = st.selectbox("BAF subunit", options=BAF_SUBUNITS, index=0)

        matching.clear()
        for key in dataset_keys_all:
            df_gene = aggregated_by_file.get(key)
            if df_gene is None:
                continue
            if _dataset_appearance_for_subunit(df_gene, selected_subunit):
                matching.append(key)

        st.write(f"Detected in {len(matching)} experiment(s).")

        if matching:
            if st.session_state.get("subunit_jump_dataset") not in matching:
                st.session_state["subunit_jump_dataset"] = matching[0]
            st.selectbox(
                "Jump to matching dataset",
                options=matching,
                format_func=lambda k: labels.get(k, k),
                key="subunit_jump_dataset",
            )

    # Subunit jump overrides the main dataset picker when matches exist.
    if matching:
        selected_dataset = str(st.session_state.get("subunit_jump_dataset", dataset_keys_all[0]))
    else:
        selected_dataset = str(st.session_state.get("portal_dataset_select", dataset_keys_all[0]))

    try:
        df_gene = aggregated_by_file[selected_dataset]
    except KeyError:
        st.error(
            f"Selected dataset is not available: {selected_dataset!r}. Choose another run in the sidebar."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data for the selected experiment. Details: {e}")
        st.stop()

    prev_active = st.session_state.get(ACTIVE_DATASET_STATE_KEY)
    if prev_active is not None and prev_active != selected_dataset:
        draw_volcano_plot.clear()
        draw_complex_coverage.clear()
    st.session_state[ACTIVE_DATASET_STATE_KEY] = selected_dataset

    bait_gene_for_volcano = _infer_bait_gene_for_volcano(meta_by_file.get(selected_dataset, {}))

    tab_main, tab_compare, tab_vault, tab_global, tab_batch = st.tabs(
        [
            "Main Dashboard",
            "Comparative Analytics",
            "Data Vault",
            "Global Search",
            "Batch Subunit (Consensus)",
        ]
    )

    with tab_main:
        # KPI metrics
        total_proteins = int(df_gene["Gene Symbol"].nunique())
        baf_detected = int(df_gene["is_baf_core"].sum())
        top_interactor = _pick_top_interactor(df_gene)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Proteins", f"{total_proteins}")
        c2.metric("BAF Subunits Found", f"{baf_detected}")
        c3.metric("Top Interactor", top_interactor if top_interactor else "NA")

        st.markdown("### Volcano Plot")
        st.plotly_chart(
            draw_volcano_plot(
                selected_dataset,
                df_gene,
                bait_gene=bait_gene_for_volcano,
            ),
            use_container_width=True,
            key=f"main_volcano_{selected_dataset}",
        )

        st.markdown("### Complex Coverage (BAF Subunits Pulled Down)")
        st.plotly_chart(
            draw_complex_coverage(selected_dataset, df_gene),
            use_container_width=True,
            key=f"main_coverage_{selected_dataset}",
        )

        st.markdown("### Gene-Level Aggregated Table")
        df_view = df_gene.copy()
        df_view["Confidence Score"] = pd.to_numeric(df_view["Confidence Score"], errors="coerce")
        df_view["Spectral Count"] = pd.to_numeric(df_view["Spectral Count"], errors="coerce")
        # Default table order: strongest spectral signal first (stable ties).
        df_view = df_view.sort_values(
            by=["Spectral Count", "Confidence Score", "Gene Symbol"],
            ascending=[False, False, True],
            kind="mergesort",
        )

        base_cols = [
            "Gene Symbol",
            "Biological Target",
            "Domain/Details",
            "Spectral Count",
            "Unique Peptides",
            "Confidence Score",
            "Log_Prob",
            "is_baf_core",
        ]
        use_cols = [c for c in base_cols if c in df_view.columns]
        df_table = df_view[use_cols].copy()
        df_table = df_table.rename(columns={"Log_Prob": "-log10(avg Prob)"})

        styler = _styled_gene_table(df_table)
        st.dataframe(styler, use_container_width=True, height=520)

        meta_cur = meta_by_file.get(selected_dataset, {})
        dl = df_view.sort_values(
            by=["Spectral Count", "Confidence Score", "Gene Symbol"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        st.download_button(
            "Download Aggregated Data",
            data=dl.to_csv(index=False).encode("utf-8"),
            file_name=_aggregated_download_name(meta_cur, str(meta_cur.get("filename") or selected_dataset)),
            mime="text/csv",
            key="download_main_active",
        )

    with tab_compare:
        _render_comparative_analytics_tab(
            selected_keys=list(st.session_state.get("compare_analytics_pick", [])),
            dataset_keys_all=dataset_keys_all,
            aggregated_by_file=aggregated_by_file,
            meta_by_file=meta_by_file,
        )

    with tab_vault:
        _render_data_vault_tab(
            experiments_df=experiments_df,
            aggregated_by_file=aggregated_by_file,
            meta_by_file=meta_by_file,
            data_dir=str(data_dir),
        )

    with tab_global:
        _render_global_search_tab(
            gene_input=str(st.session_state.get("global_gene_query", "")),
            aggregated_by_file=aggregated_by_file,
            meta_by_file=meta_by_file,
        )

    with tab_batch:
        st.markdown("### Batch Subunit Analysis (Lab Consensus)")
        st.caption("Aggregate every experiment where the inferred bait matches your selection (>50% prevalence).")

        baits = sorted(
            {
                str(x).strip()
                for x in experiments_df["bait"].dropna().tolist()
                if str(x).strip() not in ("", "N/A", "nan")
            }
        )
        manual = st.text_input("Type a bait if missing from list", value="", key="batch_bait_manual")
        options = sorted(set(baits + ([manual.strip()] if manual.strip() else [])))

        if not options:
            st.info("No bait options yet. Run imports or enter a bait name above.")
        else:
            pick = st.selectbox("Bait / target", options=options, key="batch_bait_select")
            runs = [fk for fk in dataset_keys_all if _meta_matches_bait(meta_by_file.get(fk, {}), pick)]
            st.metric("Matching experiments", len(runs))
            if not runs:
                st.warning("No experiments matched this bait (check filenames/labels).")
            else:
                with st.expander("Show Experiment Details", expanded=False):
                    st.markdown("#### Matching metadata")
                    run_rows: list[dict[str, Any]] = []
                    for fk in sorted(runs):
                        m = meta_by_file.get(fk, {})
                        run_rows.append(
                            {
                                "Investigator": m.get("investigator") or "—",
                                "Session ID": m.get("session_id") or "—",
                                "Label": m.get("label") or "—",
                                "Cell Line": m.get("cell_line") or "—",
                                "File": m.get("filename") or str(fk).split("/")[-1],
                            }
                        )
                    meta_runs = pd.DataFrame(run_rows)
                    meta_runs = meta_runs.sort_values(
                        by=["Investigator", "Session ID", "File"],
                        kind="mergesort",
                    ).reset_index(drop=True)

                    researchers = sorted(
                        {str(x) for x in meta_runs["Investigator"].tolist() if str(x) not in ("—", "N/A", "nan", "")}
                    )
                    if researchers:
                        st.markdown(
                            "**Contributing researchers:** "
                            + ", ".join(f"**{r}**" for r in researchers)
                        )
                    else:
                        st.caption("Investigator not parsed for some runs; check `Data/[Investigator]/` layout.")

                    inv_counts = meta_runs["Investigator"].value_counts()
                    cell_counts = meta_runs["Cell Line"].value_counts()

                    bc1, bc2 = st.columns(2)
                    h_inv = max(180, 48 + 28 * len(inv_counts))
                    h_cell = max(180, 48 + 28 * len(cell_counts))
                    with bc1:
                        st.caption("Runs by investigator")
                        fig_i = go.Figure(
                            go.Bar(
                                x=inv_counts.values,
                                y=inv_counts.index.astype(str),
                                orientation="h",
                                marker=dict(color="#00A6ED"),
                            )
                        )
                        fig_i.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=120, r=10, t=10, b=30),
                            height=h_inv,
                            xaxis_title="Number of runs",
                            yaxis_title="",
                        )
                        st.plotly_chart(fig_i, use_container_width=True)
                    with bc2:
                        st.caption("Runs by cell line")
                        fig_c = go.Figure(
                            go.Bar(
                                x=cell_counts.values,
                                y=cell_counts.index.astype(str),
                                orientation="h",
                                marker=dict(color="#7C4DFF"),
                            )
                        )
                        fig_c.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=120, r=10, t=10, b=30),
                            height=h_cell,
                            xaxis_title="Number of runs",
                            yaxis_title="",
                        )
                        st.plotly_chart(fig_c, use_container_width=True)

                    st.dataframe(meta_runs, use_container_width=True, height=min(360, 72 + 28 * len(meta_runs)))

                cons = _lab_consensus_table(runs, aggregated_by_file, min_fraction=0.5)
                if cons.empty:
                    st.info("No proteins exceeded 50% prevalence across these runs.")
                else:
                    st.dataframe(cons, use_container_width=True, height=520)
                    st.download_button(
                        "Download Consensus Table",
                        data=cons.to_csv(index=False).encode("utf-8"),
                        file_name=f"Consensus_{str(pick).replace('/', '_')}_50pct.csv",
                        mime="text/csv",
                        key="download_consensus_batch",
                    )


if __name__ == "__main__":
    main()

