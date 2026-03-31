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
    crawl_metadata,
    get_experiment_data,
    load_and_aggregate_csv,
)

DATA_SOURCE_DIR_DEFAULT = "/Users/sp1665/Downloads/IPMS/Janet_Liu"
PROJECT_ROOT = os.path.dirname(__file__)
DATA_EMPTY_MESSAGE = "No datasets found in /Data. Please add IP-MS CSVs to begin analysis."
DATA_ROOT = pathlib.Path(__file__).parent.resolve() / "Data"
# Avoid StreamlitAPIException: never assign to portal_dataset_select after its widget is created;
# set this before the main selectbox runs, then pop and apply.
_PENDING_PORTAL_DATASET_KEY = "_pending_portal_dataset"
ACTIVE_DATASET_STATE_KEY = "active_dataset_id"
ACTIVE_DF_STATE_KEY = "active_df"
ACTIVE_DATASET_PICK_KEY = "global_active_dataset"
_DATASET_SWITCH_RERUN_KEY = "_dataset_switch_rerun"
# Session-only cache of gene-level frames (background index for Global Search).
INDEXED_AGGREGATES_KEY = "indexed_aggregated_by_file"
CURRENT_PROJECT_BAITS = {"BCL7A", "BCL7B", "BCL7C", "SMARCE1"}
# Horizontal reference line on volcano: y = -log10(p); change here or wire to UI later.
VOLCANO_CONFIDENCE_P_CUTOFF = 0.05
# Subtle prey points (non-BAF); BAF uses BAF_CORE_COLOR (#FF4B4B).
VOLCANO_PREY_COLOR = "#5C6E82"
# Publication volcano (FC vs −log10 p)
VOLCANO_PUB_GREY = "#9E9E9E"
VOLCANO_PUB_RED = "#D62728"
VOLCANO_PUB_BLUE = "#1F78B4"
VOLCANO_PUB_BAIT = "#FDB462"
# When using mean-spectral baseline (no Mock/EV), tiny IP lists make the mean unstable — use this constant instead.
VOLCANO_MIN_GENES_FOR_MEAN_BASELINE = 8
VOLCANO_MEAN_BASELINE_FALLBACK = 1.0


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
    return str(meta.get("bait", "")).upper() == q


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
    out = out.sort_values(by=["Mean Spectral Count", "Fraction of Runs"], ascending=[False, False]).reset_index(
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


@st.cache_data(show_spinner="Accessing BAF-Vault...")
def process_csv(csv_path: str, csv_mtime: float) -> pd.DataFrame:
    """Legacy shim: aggregate directly from source CSV (no _PROCESSED writes)."""
    _ = csv_mtime
    try:
        d = load_and_aggregate_csv(str(csv_path))
    except Exception:
        # permissive fallback keeps file searchable instead of skipping it entirely
        p = pathlib.Path(csv_path)
        d = pd.DataFrame(
            {
                "Gene Symbol": [p.stem.upper() or "UNKNOWN"],
                "Spectral Count": [1.0],
                "Unique Peptides": [1],
                "Confidence Score": [0.5],
                "Log_Prob": [-np.log10(0.5)],
            }
        )
        d.attrs["ipms_lda_probability_column_in_csv"] = False
    return d


@st.cache_data(show_spinner=False)
def _cached_crawl_metadata(data_dir: str, file_count: int) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Filesystem crawl + filename metadata only (fast startup)."""
    print(f"[IPMS Debug] crawl_metadata: {data_dir!r} (file_count={file_count})")
    return crawl_metadata(data_dir, file_count)


@st.cache_data(show_spinner=False)
def _cached_aggregate_experiment(path: str, mtime: float, bio: str, domain: str) -> pd.DataFrame:
    """One-file heavy aggregation (cached by path + mtime + meta strings)."""
    return get_experiment_data(path, biological_target=bio, domain_details=domain)


def _session_indexed_aggregates() -> dict[str, pd.DataFrame]:
    if INDEXED_AGGREGATES_KEY not in st.session_state:
        st.session_state[INDEXED_AGGREGATES_KEY] = {}
    return st.session_state[INDEXED_AGGREGATES_KEY]


def _clear_portal_caches() -> None:
    _cached_crawl_metadata.clear()
    _cached_aggregate_experiment.clear()
    st.session_state.pop(INDEXED_AGGREGATES_KEY, None)


def _ensure_dataset_in_session_index(file_key: str, meta_by_file: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Load aggregate into session index (background index for Global Search). Uses st.cache_data under the hood."""
    idx = _session_indexed_aggregates()
    if file_key in idx:
        return idx[file_key]
    meta = meta_by_file[file_key]
    path = str(meta.get("path") or "")
    mtime = float(meta.get("mtime") or 0.0)
    bio = str(meta.get("biological_target") or "Unknown")
    dom = str(meta.get("domain_details") or "Unknown")
    df = _cached_aggregate_experiment(path, mtime, bio, dom)
    idx[file_key] = df
    return df


def _stem_from_meta_filename(filename: str | None) -> str | None:
    fn = str(filename or "").strip()
    if not fn:
        return None
    return fn[: -len(".csv")] if fn.lower().endswith(".csv") else fn


def _infer_bait_gene_for_volcano(meta: dict[str, Any]) -> str | None:
    """Gene symbol for bait marker (diamond). Returns None if not inferable (e.g. EV only)."""
    bait_raw = str(meta.get("bait") or "").strip()
    if bait_raw and bait_raw.upper() not in ("N/A", "NA", "—", ""):
        m = re.match(r"^([A-Za-z0-9]+)", bait_raw)
        if m:
            return m.group(1).upper()
    ig = infer_bait_gene_from_label(meta.get("label"), _stem_from_meta_filename(meta.get("filename")))
    return ig.upper() if ig else None


def _short_dataset_label(file_key: str, meta: dict[str, Any], *, max_len: int = 32) -> str:
    fn = str(meta.get("filename") or file_key.split("/")[-1])
    return (fn[: max_len - 1] + "…") if len(fn) > max_len else fn


def _friendly_experiment_label(file_key: str, meta: dict[str, Any]) -> str:
    sid = str(meta.get("session_id") or "NoSession")
    bait = str(meta.get("bait") or "N/A")
    label = str(meta.get("label") or "N/A")
    label = label.replace(".csv", "")
    return f"[{sid}] - {bait} - {label}"


def _investigator_folder_from_file_key(file_key: str) -> str | None:
    norm = str(file_key).replace("\\", "/").strip("/")
    if "/" not in norm:
        return None
    return norm.split("/")[0]


def _meta_suggests_mock_or_ev(meta: dict[str, Any]) -> bool:
    """True if filename/label/bait indicates Mock IP or empty-vector (EV) control."""
    blob = " ".join(
        [
            str(meta.get("label") or ""),
            str(meta.get("filename") or ""),
            str(meta.get("bait") or ""),
            str(meta.get("biological_target") or ""),
        ]
    ).upper()
    blob = blob.replace("-", "_")
    if re.search(r"\bMOCK\b", blob):
        return True
    if re.search(r"\bEV\b", blob):
        return True
    if "EMPTY" in blob and "VECTOR" in blob:
        return True
    if "EMPTY_VECTOR" in blob or "NEGATIVE" in blob:
        return True
    return False


def _auto_control_dataset_key(
    active_key: str,
    all_keys: list[str],
    meta_by_file: dict[str, dict[str, Any]],
) -> str | None:
    """Pick a Mock/EV CSV in the same investigator subfolder as the active run (if any)."""
    inv_active = _investigator_folder_from_file_key(active_key)
    if not inv_active:
        return None
    candidates: list[str] = []
    for k in all_keys:
        if k == active_key:
            continue
        if _investigator_folder_from_file_key(k) != inv_active:
            continue
        if _meta_suggests_mock_or_ev(meta_by_file.get(k, {})):
            candidates.append(k)
    return sorted(candidates)[0] if candidates else None


def _prepare_volcano_foldchange_frame(
    df_bait: pd.DataFrame,
    df_control: pd.DataFrame | None,
    *,
    lda_column_present: bool = True,
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Gene-level log2 fold change with +1 pseudocount:
    log2((bait_sc+1)/(control_sc+1)), or vs dataset mean spectral count if no control.
    """
    diags: list[str] = []
    d = df_bait.copy()
    d["Gene Symbol"] = d["Gene Symbol"].astype(str).str.strip().str.upper()
    sc_b = pd.to_numeric(d["Spectral Count"], errors="coerce").fillna(0.0)
    n_prot = int(len(sc_b))
    if df_control is not None and not df_control.empty:
        c = df_control.copy()
        c["Gene Symbol"] = c["Gene Symbol"].astype(str).str.strip().str.upper()
        idx = pd.to_numeric(c.groupby("Gene Symbol", sort=False)["Spectral Count"].sum(), errors="coerce").fillna(0.0)
        sc_m = d["Gene Symbol"].map(idx).fillna(0.0).astype(float)
        mode = "mock_ev"
    else:
        if n_prot < VOLCANO_MIN_GENES_FOR_MEAN_BASELINE:
            mu = float(VOLCANO_MEAN_BASELINE_FALLBACK)
            diags.append(
                f"**Baseline:** only **{n_prot}** proteins in this IP — using fixed baseline spectral count "
                f"**{VOLCANO_MEAN_BASELINE_FALLBACK:g}** instead of the mean (fold-change stability)."
            )
        else:
            mu = float(sc_b.mean()) if n_prot else 0.0
        sc_m = pd.Series(np.full(len(d), mu, dtype=float), index=d.index)
        mode = "mean_baseline"
    d["Log2FC"] = np.log2((sc_b.astype(float) + 1.0) / (sc_m + 1.0))
    d["Log_Prob"] = pd.to_numeric(d["Log_Prob"], errors="coerce")
    n_before = len(d)
    d = d.dropna(subset=["Log_Prob"])
    if d.empty and n_before > 0 and lda_column_present:
        diags.append(
            "**LDA signal:** every gene has missing or invalid **Log_Prob** after aggregation "
            "(needs numeric LDA probability in (0, 1] at peptide level)."
        )
    return d, mode, diags


def _make_discovery_volcano_figure(active_dataset: pd.DataFrame, *, bait_gene: str | None) -> go.Figure:
    """Fallback: spectral count vs unique peptides when LDA / FC volcano has no valid y."""
    d = active_dataset.copy()
    d["Spectral Count"] = pd.to_numeric(d["Spectral Count"], errors="coerce")
    if "Unique Peptides" not in d.columns:
        d["Unique Peptides"] = 0.0
    d["Unique Peptides"] = pd.to_numeric(d["Unique Peptides"], errors="coerce").fillna(0.0)
    d = d.dropna(subset=["Spectral Count"])
    d = d[d["Spectral Count"] > 0]

    if d.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Discovery view: no proteins with spectral count > 0",
            margin=dict(l=56, r=36, t=48, b=48),
        )
        return fig

    genes_u = d["Gene Symbol"].astype(str).str.strip().str.upper()
    bait_u = bait_gene.strip().upper() if bait_gene else None
    is_bait = genes_u == bait_u if bait_u else pd.Series(False, index=d.index)
    has_baf = "is_baf_core" in d.columns

    d_bait_only = d.loc[is_bait].copy()
    d_rest = d.loc[~is_bait].copy()
    if has_baf:
        ib = d_rest["is_baf_core"] == True
        d_baf_ring = d_rest.loc[ib]
        d_prey = d_rest.loc[~ib]
    else:
        d_baf_ring = d_rest.iloc[0:0]
        d_prey = d_rest

    pos = d["Spectral Count"].astype(float)
    use_log_x = len(pos) >= 3 and (float(pos.max()) > 80 or float(pos.max()) / max(float(pos.min()), 1.0) >= 25.0)

    fig = go.Figure()

    def _tr(sub: pd.DataFrame, *, name: str, color: str, size: int) -> None:
        if sub.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=sub["Spectral Count"],
                y=sub["Unique Peptides"],
                mode="markers",
                name=name,
                marker=dict(color=color, size=size, opacity=0.85, line=dict(width=0.35, color="rgba(0,0,0,0.2)")),
                customdata=sub["Gene Symbol"].astype(str).values,
                hovertemplate="<b>%{customdata}</b><br>Spectral count: %{x}<br>Unique peptides: %{y}<extra></extra>",
            )
        )

    _tr(d_prey, name="Prey", color=VOLCANO_PUB_GREY, size=6)
    _tr(d_baf_ring, name="BAF subunit", color=VOLCANO_PUB_RED, size=9)

    if bait_u and not d_bait_only.empty:
        fig.add_trace(
            go.Scatter(
                x=d_bait_only["Spectral Count"],
                y=d_bait_only["Unique Peptides"],
                mode="markers",
                name=f"Bait ({bait_u})",
                marker=dict(
                    size=15,
                    color=VOLCANO_PUB_BAIT,
                    symbol="diamond",
                    line=dict(width=1, color="#333333"),
                ),
                customdata=d_bait_only["Gene Symbol"].astype(str).values,
                hovertemplate="<b>%{customdata}</b> (bait)<br>Spectral count: %{x}<br>Unique peptides: %{y}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        title=dict(text="Discovery view: spectral count vs unique peptides", font=dict(size=14)),
        font=dict(color="#222222"),
        margin=dict(l=56, r=36, t=56, b=48),
        xaxis=dict(
            title="Spectral count",
            type="log" if use_log_x else "linear",
            gridcolor="#E0E0E0",
        ),
        yaxis=dict(title="Unique peptides", gridcolor="#E0E0E0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig


def _make_publication_volcano_figure(
    df: pd.DataFrame,
    *,
    bait_gene: str | None,
    log2_fc_threshold: float,
    neg_log10_p_threshold: float,
    n_top_labels: int,
) -> go.Figure:
    """Fold-change volcano: white background, quadrant colors, cutoffs, top-hit labels."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No genes with valid LDA probability")
        return fig

    d = df.copy()
    d["Log_Prob_plot"] = _log_prob_y_with_jitter(d["Log_Prob"], d["Gene Symbol"])

    fc_t = float(log2_fc_threshold)
    y_t = float(neg_log10_p_threshold)

    d["quad"] = "ns"
    d.loc[(d["Log2FC"] >= fc_t) & (d["Log_Prob"] >= y_t), "quad"] = "up"
    d.loc[(d["Log2FC"] <= -fc_t) & (d["Log_Prob"] >= y_t), "quad"] = "down"

    genes_u = d["Gene Symbol"].astype(str).str.strip().str.upper()
    bait_u = bait_gene.strip().upper() if bait_gene else None
    is_bait = genes_u == bait_u if bait_u else pd.Series(False, index=d.index)

    if "Unique Peptides" not in d.columns:
        d["Unique Peptides"] = 0

    d_bg = d.loc[(d["quad"] == "ns") & ~is_bait].copy()
    d_up = d.loc[(d["quad"] == "up") & ~is_bait].copy()
    d_dn = d.loc[(d["quad"] == "down") & ~is_bait].copy()
    d_bait_only = d.loc[is_bait].copy()

    fig = go.Figure()

    def _trace(sub: pd.DataFrame, *, name: str, color: str, size: int) -> None:
        if sub.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=sub["Log2FC"],
                y=sub["Log_Prob_plot"],
                mode="markers",
                name=name,
                marker=dict(size=size, color=color, opacity=0.85, line=dict(width=0.35, color="rgba(0,0,0,0.25)")),
                customdata=np.stack(
                    [
                        sub["Gene Symbol"].astype(str).values,
                        sub["Spectral Count"].values,
                        sub["Unique Peptides"].values,
                        sub["Log2FC"].values,
                        sub["Log_Prob"].values,
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Log2 FC: %{customdata[3]:.3f}<br>"
                    "Spectral count: %{customdata[1]}<br>"
                    "Unique peptides: %{customdata[2]}<br>"
                    "−log10 LDA p: %{customdata[4]:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    _trace(d_bg, name="Non-significant", color=VOLCANO_PUB_GREY, size=5)
    _trace(d_dn, name="Depleted", color=VOLCANO_PUB_BLUE, size=7)
    _trace(d_up, name="Enriched", color=VOLCANO_PUB_RED, size=7)

    if bait_u and not d_bait_only.empty:
        fig.add_trace(
            go.Scatter(
                x=d_bait_only["Log2FC"],
                y=d_bait_only["Log_Prob_plot"],
                mode="markers",
                name=f"Bait ({bait_u})",
                marker=dict(
                    size=16,
                    color=VOLCANO_PUB_BAIT,
                    symbol="diamond",
                    line=dict(width=1, color="#333333"),
                ),
                customdata=np.stack(
                    [
                        d_bait_only["Gene Symbol"].astype(str).values,
                        d_bait_only["Spectral Count"].values,
                        d_bait_only["Unique Peptides"].values,
                        d_bait_only["Log2FC"].values,
                        d_bait_only["Log_Prob"].values,
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (bait)<br>"
                    "Log2 FC: %{customdata[3]:.3f}<br>"
                    "Spectral count: %{customdata[1]}<br>"
                    "Unique peptides: %{customdata[2]}<br>"
                    "−log10 LDA p: %{customdata[4]:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    abs_fc = np.abs(d["Log2FC"].astype(float).to_numpy())
    xmax_raw = float(np.nanmax(abs_fc)) if np.isfinite(abs_fc).any() else 1.0
    xmax = max(3.0, xmax_raw * 1.15 + 0.5)
    y_max = max(0.5, float(d["Log_Prob_plot"].max()) * 1.12)

    fig.add_vline(x=fc_t, line_dash="dash", line_color="#757575", line_width=1)
    fig.add_vline(x=-fc_t, line_dash="dash", line_color="#757575", line_width=1)
    fig.add_hline(y=y_t, line_dash="dash", line_color="#757575", line_width=1)

    ann: list[dict[str, Any]] = []
    top_enr = (
        d.loc[(d["quad"] == "up")]
        .sort_values(["Log_Prob", "Log2FC"], ascending=[False, False])
        .head(max(5, min(int(n_top_labels), 15)))
    )
    for _, row in top_enr.iterrows():
        ann.append(
            dict(
                x=float(row["Log2FC"]),
                y=float(row["Log_Prob_plot"]),
                text=str(row["Gene Symbol"]),
                showarrow=True,
                arrowhead=2,
                arrowsize=0.55,
                arrowwidth=0.8,
                arrowcolor="#404040",
                ax=0,
                ay=-36,
                font=dict(size=10, color="#222222"),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.12)",
                borderwidth=1,
                borderpad=3,
            )
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(color="#222222", size=11),
        margin=dict(l=56, r=36, t=40, b=48),
        xaxis=dict(
            title=dict(text="Log<sub>2</sub> fold change (bait / baseline)", font=dict(size=13)),
            range=[-xmax, xmax],
            gridcolor="#E0E0E0",
            zeroline=True,
            zerolinecolor="#BDBDBD",
            zerolinewidth=1,
        ),
        yaxis=dict(
            title=dict(text="Significance (−log<sub>10</sub> LDA probability)", font=dict(size=13)),
            range=[0, y_max],
            gridcolor="#E0E0E0",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=ann,
        height=520,
    )
    return fig


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


def _spectral_x_with_jitter(spectral: pd.Series, *, dataset_id: str) -> np.ndarray:
    """Small X jitter (0.1-0.2) so low integer counts do not stack in vertical bands."""
    x = spectral.astype(float).to_numpy()
    if len(x) == 0:
        return x
    seed = abs(hash((dataset_id, tuple(np.round(x, 6).tolist())))) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    amp = rng.uniform(0.1, 0.2, size=len(x))
    sign = rng.choice(np.array([-1.0, 1.0]), size=len(x))
    xj = x + sign * amp
    xj = np.where(xj <= 0.01, x + amp, xj)
    return xj


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
    dataset_id: str,
    force_log_x: bool | None = None,
    xaxis_range: tuple[float, float] | None = None,
    yaxis_range: tuple[float, float] | None = None,
    emphasize_baf_only: bool = False,
    xaxis_title: str | None = None,
    confidence_p_threshold: float | None = VOLCANO_CONFIDENCE_P_CUTOFF,
) -> go.Figure:
    """
    Volcano: X = Spectral Count (abundance), Y = -log10(LDA probability).
    BAF subunits (26): #FF4B4B. Other prey: muted blue-grey. Optional dashed p-cutoff line.
    """
    prey_color = VOLCANO_PREY_COLOR
    core_color = BAF_CORE_COLOR
    bait_non_baf_color = "#FFD700"

    df = active_dataset.copy()
    df["Spectral Count"] = pd.to_numeric(df["Spectral Count"], errors="coerce").fillna(0.0)
    df["Log_Prob"] = pd.to_numeric(df["Log_Prob"], errors="coerce")
    df = df.dropna(subset=["Log_Prob"], how="any")
    df = df[df["Spectral Count"] >= 0]

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
    df["Spectral Count_plus1"] = df["Spectral Count"].astype(float) + 1.0
    df["Spectral Count_plot"] = _spectral_x_with_jitter(df["Spectral Count_plus1"], dataset_id=dataset_id)

    genes_u = df["Gene Symbol"].astype(str).str.strip().str.upper()
    bait_u = bait_gene.strip().upper() if bait_gene else None
    is_bait = genes_u == bait_u if bait_u else pd.Series(False, index=df.index)
    is_baf = df["is_baf_core"] == True

    df_bait = df.loc[is_bait]
    df_baf_circle = df.loc[is_baf & ~is_bait]
    df_prey = df.loc[~is_baf & ~is_bait]

    pos = df["Spectral Count_plus1"].astype(float)
    pos_only = pos[pos > 0]
    use_log_x = False
    if force_log_x is not None:
        use_log_x = force_log_x
    else:
        use_log_x = float(pos_only.max()) >= 30.0 if len(pos_only) else False

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
        opacity = 0.14 if prey_dimmed else 0.82
        line_col = "rgba(255,255,255,0.35)" if prey_dimmed else "rgba(57, 71, 91, 0.65)"
        mk: dict[str, Any] = dict(
            size=size,
            color=color,
            opacity=opacity,
            line=dict(width=0.5, color=line_col),
        )
        if symbol:
            mk["symbol"] = symbol
        fig.add_trace(
            go.Scatter(
                x=d["Spectral Count_plot"],
                y=d["Log_Prob_plot"],
                mode="markers",
                name=name,
                marker=mk,
                customdata=np.stack(
                    [
                        d["Gene Symbol"].astype(str).values,
                        d["Unique Peptides"].values,
                        d["Spectral Count"].values,
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Spectral count: %{customdata[2]}<br>"
                    "Unique peptides: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    prey_dim = bool(emphasize_baf_only)
    _scatter_trace(df_prey, name="Prey / other", color=prey_color, size=7, symbol=None, prey_dimmed=prey_dim)
    _scatter_trace(df_baf_circle, name="BAF core (26)", color=core_color, size=12, symbol=None)

    if bait_u and not df_bait.empty:
        bcol = core_color if bait_u in BAF_SUBUNIT_SET else bait_non_baf_color
        _scatter_trace(df_bait, name=f"Bait ({bait_u})", color=bcol, size=18, symbol="diamond")

    default_xtitle = "Spectral Count (+1, log scale)" if use_log_x else "Spectral Count (+1)"
    xaxis_cfg: dict[str, Any] = dict(
        title=(xaxis_title or default_xtitle),
        gridcolor="rgba(255,255,255,0.08)",
        type="log" if use_log_x else "linear",
    )
    if use_log_x:
        x_upper = np.log10(max(float(pos_only.max()), 1.0)) if len(pos_only) else 1.0
        x_upper = min(max(3.0, x_upper * 1.05), 3.0)
        xaxis_cfg["range"] = [0.0, x_upper]
    else:
        xaxis_cfg["range"] = [0.0, max(float(pos.max()) * 1.1, 3.0)]
    if xaxis_range is not None and not use_log_x:
        lo, hi = float(xaxis_range[0]), float(xaxis_range[1])
        if np.isfinite(lo) and np.isfinite(hi):
            xaxis_cfg["range"] = [max(0.0, lo), max(lo + 1e-6, hi)]

    yaxis_cfg: dict[str, Any] = dict(gridcolor="rgba(255,255,255,0.08)", title="-log10(LDA Probability)")
    if yaxis_range is not None:
        yaxis_cfg["range"] = [float(yaxis_range[0]), float(yaxis_range[1])]
    else:
        y_max = float(df["Log_Prob_plot"].max())
        yaxis_cfg["range"] = [0.0, max(y_max * 1.1, 0.05)]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=10, t=30, b=30),
        xaxis=xaxis_cfg,
        yaxis=yaxis_cfg,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Label top 5 by spectral count and BAF genes above significance threshold.
    top5 = df.sort_values(["Spectral Count", "Log_Prob"], ascending=[False, False]).head(5)
    baf_sig = df[(df["is_baf_core"] == True) & (df["Log_Prob"] > 2.0)]
    ann_df = pd.concat([top5, baf_sig], axis=0).drop_duplicates(subset=["Gene Symbol"], keep="first")
    for _, r in ann_df.iterrows():
        fig.add_annotation(
            x=float(r["Spectral Count_plot"]),
            y=float(r["Log_Prob_plot"]),
            text=str(r["Gene Symbol"]),
            showarrow=True,
            arrowhead=1,
            arrowsize=0.6,
            arrowwidth=0.8,
            arrowcolor="rgba(220,220,220,0.8)",
            ax=14,
            ay=-20,
            font=dict(size=10, color="#E6EDF3"),
            bgcolor="rgba(14,17,23,0.72)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            borderpad=2,
        )

    if (
        confidence_p_threshold is not None
        and 0.0 < float(confidence_p_threshold) < 1.0
        and not df.empty
    ):
        y_cut = float(-np.log10(float(confidence_p_threshold)))
        fig.add_hline(
            y=y_cut,
            line_dash="dash",
            line_color="rgba(230, 237, 243, 0.5)",
            line_width=1,
            annotation_text=f"p = {float(confidence_p_threshold):g}",
            annotation_position="right",
            annotation_font_size=11,
            annotation_font_color="rgba(230, 237, 243, 0.9)",
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


def draw_volcano_plot(
    active_dataset: pd.DataFrame,
    dataset_id: str,
    *,
    bait_gene: str | None,
    confidence_p_threshold: float | None = VOLCANO_CONFIDENCE_P_CUTOFF,
) -> go.Figure:
    """Reactive discovery volcano (Spectral Count vs -log10(LDA Probability))."""
    return _make_volcano_figure(
        active_dataset,
        bait_gene=bait_gene,
        dataset_id=dataset_id,
        confidence_p_threshold=confidence_p_threshold,
    )


def draw_complex_coverage(active_dataset_id: str, active_dataset: pd.DataFrame) -> go.Figure:
    """
    Recomputes BAF subunit coverage from `active_dataset` on every call (no cache).
    All BAF_SUBUNITS on Y-axis; missing subunits show 0. Order = canonical list.
    """
    _ = active_dataset_id
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
    merged = merged.sort_values("Spectral Count", ascending=False, kind="mergesort")

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
        yaxis=dict(tickfont=dict(size=11), autorange="reversed", categoryorder="array", categoryarray=merged["Gene Symbol"].astype(str).tolist()),
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
                dataset_id=f"cmp_{k_a}",
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
                dataset_id=f"cmp_{k_b}",
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
    q = str(gene_query).strip()
    if not q:
        return pd.DataFrame()

    q_upper = q.upper()
    rows: list[dict[str, Any]] = []

    for file_key, df_gene in aggregated_by_file.items():
        if df_gene is None or df_gene.empty or "Gene Symbol" not in df_gene.columns:
            continue
        meta = meta_by_file.get(file_key, {})

        mask = df_gene["Gene Symbol"].astype(str).str.strip().str.upper() == q_upper
        hit = df_gene.loc[mask]
        if hit.empty:
            continue

        row = hit.iloc[0]
        rows.append(
            {
                "Investigator": meta.get("investigator") or "Unknown",
                "Session_ID": meta.get("session_id") or "Unknown",
                "Bait": meta.get("bait") or "Unknown",
                "Sample Label": meta.get("label") or "Unknown",
                "Spectral Count": row.get("Spectral Count"),
                "Unique Peptides": row.get("Unique Peptides"),
                "Avg Prob": row.get("Confidence Score"),
                "_file_key": file_key,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["Spectral Count"] = pd.to_numeric(out["Spectral Count"], errors="coerce")
    out["Avg Prob"] = pd.to_numeric(out["Avg Prob"], errors="coerce")
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
    st.markdown("### Global Search")
    n_idx = len(aggregated_by_file)
    st.caption(
        f"**Session index:** {n_idx} experiment(s) with loaded gene-level data. "
        "Open runs from the Dataset Browser or other tabs first — only indexed files are searched (fast startup)."
    )

    q = st.text_input(
        "Gene Symbol",
        placeholder="e.g. ACTL6A, SMARCA4",
        key="global_gene_query_tab",
    ).strip()
    if not q:
        st.info("Enter a gene symbol to search across all experiments.")
        return

    results = search_gene_across_datasets(q, aggregated_by_file, meta_by_file)
    if results.empty:
        st.info(
            f"No detections for **{q}** in the session index ({n_idx} file(s)). "
            "Open experiments from the sidebar to add them to the index, then search again."
        )
        return

    r_obs = results[results["Spectral Count"].fillna(0) > 0]
    max_sc = int(r_obs["Spectral Count"].max()) if not r_obs.empty and r_obs["Spectral Count"].notna().any() else 0
    top = results.iloc[0]
    top_bait = top.get("Bait", "N/A")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Hits", f"{len(results)}")
    m2.metric("Highest Enrichment", f"{max_sc}")
    m3.metric("Top Bait", str(top_bait))

    df_show = results[
        ["Investigator", "Bait", "Sample Label", "Spectral Count", "Avg Prob", "_file_key"]
    ].copy()

    st.markdown("#### Results")
    st.dataframe(
        df_show.drop(columns=["_file_key"], errors="ignore"),
        use_container_width=True,
        height=min(520, 42 + 36 * len(df_show)),
    )

    st.markdown("#### Open in Dataset Browser")
    for i, r in results.iterrows():
        fk = r.get("_file_key")
        if not fk:
            continue
        ds_id = f"{r.get('Bait', 'Unknown')} | {r.get('Sample Label', 'Unknown')}"
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
    rt = str(meta.get("resolved_target") or "").strip()
    if rt and rt.upper() not in ("N/A", "NA", ""):
        return rt
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
    bait_query = st.text_input(
        "Global filter (bait / target / label / filename / investigator)",
        placeholder="e.g. SMARCA4, SWIFT, BCL7, Jordan_Otto",
    )
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
        ["None (Flat)", "Investigator", "Session ID"],
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

    if group_mode == "None (Flat)":
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
    if sel not in aggregated_by_file:
        with st.spinner("Processing IP-MS Data..."):
            _ensure_dataset_in_session_index(sel, meta_by_file)
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
                    _clear_portal_caches()
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


    matching: list[str] = []
    with st.sidebar:
        _ensure_local_data_dir(data_dir)

        st.markdown("### Data Directory & File Management")

        if st.button("Refresh Data"):
            _clear_portal_caches()
            st.rerun()

        local_csv_count = _count_csv_files(str(data_dir))
        if local_csv_count == 0:
            st.info(DATA_EMPTY_MESSAGE)
            st.stop()

        st.markdown("---")
        st.markdown("### Dataset Browser")
        try:
            experiments_df, meta_by_file = _cached_crawl_metadata(
                str(data_dir),
                local_csv_count,
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

        dataset_keys_all = experiments_df["file_key"].astype(str).tolist()
        meta_lookup = meta_by_file
        labels = {k: _pretty_dataset_label(k, meta_lookup.get(k, {})) for k in dataset_keys_all}
        aggregated_by_file = _session_indexed_aggregates()

        # Apply investigator / global-search "set active" before portal_dataset_select widget is created.
        if _PENDING_PORTAL_DATASET_KEY in st.session_state:
            pending_fk = st.session_state.pop(_PENDING_PORTAL_DATASET_KEY)
            if pending_fk and str(pending_fk) in dataset_keys_all:
                st.session_state["portal_project_view"] = "All Experiments"
                st.session_state[ACTIVE_DATASET_PICK_KEY] = str(pending_fk)
                st.session_state[_DATASET_SWITCH_RERUN_KEY] = True
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

        st.markdown("#### Browse by Investigator")
        by_inv: dict[str, list[str]] = defaultdict(list)
        for fk in dataset_keys_for_picker:
            inv = str(meta_lookup.get(fk, {}).get("investigator") or "Unknown")
            by_inv[inv].append(fk)
        inv_options = sorted(by_inv.keys(), key=lambda s: s.lower())
        if "browser_investigator" not in st.session_state or st.session_state["browser_investigator"] not in inv_options:
            st.session_state["browser_investigator"] = inv_options[0] if inv_options else "Unknown"
        st.selectbox("Investigator", options=inv_options, key="browser_investigator")
        inv_selected = str(st.session_state.get("browser_investigator", inv_options[0] if inv_options else "Unknown"))
        inv_keys = by_inv.get(inv_selected, [])
        if inv_keys:
            if st.session_state.get("browser_dataset_pick") not in inv_keys:
                cur = str(st.session_state.get(ACTIVE_DATASET_PICK_KEY, inv_keys[0]))
                st.session_state["browser_dataset_pick"] = cur if cur in inv_keys else inv_keys[0]
            st.selectbox(
                "Experiment",
                options=inv_keys,
                format_func=lambda fk: _vault_action_label(fk, meta_lookup),
                key="browser_dataset_pick",
            )
            picked_fk = str(st.session_state.get("browser_dataset_pick", inv_keys[0]))
            if picked_fk != st.session_state.get(ACTIVE_DATASET_PICK_KEY):
                st.session_state[ACTIVE_DATASET_PICK_KEY] = picked_fk
                st.rerun()

            if picked_fk not in aggregated_by_file:
                with st.spinner("Processing IP-MS Data..."):
                    _ensure_dataset_in_session_index(picked_fk, meta_lookup)

            st.caption("Actions")
            meta_sel = meta_lookup.get(picked_fk, {})
            fpath = str(meta_sel.get("path") or os.path.join(str(data_dir), picked_fk))
            fn = str(meta_sel.get("filename") or picked_fk.split("/")[-1])
            df_agg = aggregated_by_file.get(picked_fk)
            b1, b2 = st.columns([1, 1])
            with b1:
                if df_agg is not None and not df_agg.empty:
                    df_dl = df_agg.sort_values(
                        by=["is_baf_core", "Spectral Count", "Confidence Score"],
                        ascending=[False, False, False],
                        kind="mergesort",
                    )
                    st.download_button(
                        "Download Aggregated CSV",
                        data=df_dl.to_csv(index=False).encode("utf-8"),
                        file_name=_aggregated_download_name(meta_sel, fn),
                        mime="text/csv",
                        key="sidebar_download_selected",
                    )
                else:
                    st.caption("No aggregated data.")
            with b2:
                del_conf_sidebar = st.checkbox("Confirm Delete", key="sidebar_delete_confirm")
                if st.button("Delete Selected File", key="sidebar_delete_btn"):
                    if not del_conf_sidebar:
                        st.warning("Enable **Confirm Delete** first.")
                    else:
                        try:
                            if os.path.exists(fpath):
                                os.remove(fpath)
                            _clear_portal_caches()
                            st.success(f"Deleted `{picked_fk}`.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

        st.markdown("---")
        st.markdown("### Upload IP-MS CSVs")
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            try:
                counts = save_uploaded_csvs_to_local_data(uploaded_files, str(data_dir), subfolder="Imported", overwrite=False)
                st.success(f"Upload complete: saved {counts['saved']} file(s), skipped {counts['skipped']} existing file(s).")
                _cached_crawl_metadata.clear()
            except Exception as e:
                st.error(f"Upload failed: {e}")

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
        st.markdown("### Subunit Quick-Search")
        selected_subunit = st.selectbox("BAF subunit", options=BAF_SUBUNITS, index=0)

        matches: list[str] = []
        for key in dataset_keys_all:
            dfg = aggregated_by_file.get(key)
            if dfg is None or dfg.empty:
                continue
            if _dataset_appearance_for_subunit(dfg, selected_subunit):
                matches.append(key)
        matching = matches
        st.caption("Only experiments already loaded in this session (indexed) are scanned.")
        st.write(f"Detected in {len(matching)} experiment(s).")

        if matching:
            if st.session_state.get("subunit_jump_dataset") not in matching:
                st.session_state["subunit_jump_dataset"] = matching[0]
            st.selectbox(
                "Jump to matching dataset",
                options=matching,
                format_func=lambda k: _friendly_experiment_label(k, meta_lookup.get(k, {})),
                key="subunit_jump_dataset",
            )
            jump_fk = str(st.session_state.get("subunit_jump_dataset", matching[0]))
            if jump_fk != st.session_state.get(ACTIVE_DATASET_PICK_KEY):
                st.session_state[ACTIVE_DATASET_PICK_KEY] = jump_fk
                st.rerun()

    if st.session_state.pop(_DATASET_SWITCH_RERUN_KEY, False):
        st.rerun()

    selected_dataset = str(st.session_state.get(ACTIVE_DATASET_PICK_KEY, dataset_keys_all[0]))
    aggregated_by_file = _session_indexed_aggregates()

    if selected_dataset not in meta_by_file:
        st.error(
            f"Selected dataset is not available: {selected_dataset!r}. Choose another run in the sidebar."
        )
        st.stop()

    if selected_dataset not in aggregated_by_file:
        with st.spinner("Processing IP-MS Data..."):
            _ensure_dataset_in_session_index(selected_dataset, meta_by_file)

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

    prev_dataset_id = st.session_state.get(ACTIVE_DATASET_STATE_KEY)
    if prev_dataset_id is not None and prev_dataset_id != selected_dataset:
        st.toast(
            f"Loading dataset: {labels.get(selected_dataset, selected_dataset)}…",
            icon="📊",
        )
    st.session_state[ACTIVE_DATASET_STATE_KEY] = selected_dataset
    st.session_state[ACTIVE_DATASET_PICK_KEY] = selected_dataset
    st.session_state["active_filename"] = str(meta_by_file.get(selected_dataset, {}).get("filename") or selected_dataset.split("/")[-1])
    st.session_state[ACTIVE_DF_STATE_KEY] = df_gene.copy()
    active_df: pd.DataFrame = st.session_state[ACTIVE_DF_STATE_KEY]

    bait_gene_for_volcano = _infer_bait_gene_for_volcano(meta_by_file.get(selected_dataset, {}))
    with st.sidebar:
        bait_opts = sorted(set(list(BAF_SUBUNITS) + ([bait_gene_for_volcano] if bait_gene_for_volcano else [])))
        bait_opts = [b for b in bait_opts if b]
        bait_opts = ["Auto (inferred)"] + bait_opts + ["None"]
        idx0 = 0
        if bait_gene_for_volcano and bait_gene_for_volcano in bait_opts:
            idx0 = bait_opts.index(bait_gene_for_volcano)
        bait_pick = st.selectbox(
            "Confirm/Select Bait",
            options=bait_opts,
            index=idx0,
            key=f"bait_confirm_{selected_dataset}",
            help="Override inferred bait when filename metadata is messy.",
        )
    if bait_pick == "Auto (inferred)":
        bait_gene_for_volcano = bait_gene_for_volcano
    elif bait_pick == "None":
        bait_gene_for_volcano = None
    else:
        bait_gene_for_volcano = str(bait_pick).strip().upper()

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

        st.subheader(f"Volcano Plot: {st.session_state['active_filename']}")

        fig_volc = draw_volcano_plot(
            active_df,
            selected_dataset,
            bait_gene=bait_gene_for_volcano,
            confidence_p_threshold=VOLCANO_CONFIDENCE_P_CUTOFF,
        )
        st.plotly_chart(
            fig_volc,
            use_container_width=True,
            key=f"main_volcano_{selected_dataset}_{bait_gene_for_volcano}",
        )

        st.markdown("### Complex Coverage (BAF Subunits Pulled Down)")
        st.plotly_chart(
            draw_complex_coverage(selected_dataset, active_df),
            use_container_width=True,
            key=f"main_coverage_{selected_dataset}",
        )

        st.markdown("### Gene-Level Aggregated Table")
        df_view = active_df.copy()
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
        cmp_pick = [str(k) for k in st.session_state.get("compare_analytics_pick", []) if k in meta_by_file]
        idx_cmp = _session_indexed_aggregates()
        miss_cmp = [k for k in cmp_pick if k not in idx_cmp]
        if miss_cmp:
            with st.spinner("Processing IP-MS Data..."):
                for k in miss_cmp:
                    _ensure_dataset_in_session_index(k, meta_by_file)
        _render_comparative_analytics_tab(
            selected_keys=list(st.session_state.get("compare_analytics_pick", [])),
            dataset_keys_all=dataset_keys_all,
            aggregated_by_file=_session_indexed_aggregates(),
            meta_by_file=meta_by_file,
        )

    with tab_vault:
        _render_data_vault_tab(
            experiments_df=experiments_df,
            aggregated_by_file=_session_indexed_aggregates(),
            meta_by_file=meta_by_file,
            data_dir=str(data_dir),
        )

    with tab_global:
        _render_global_search_tab(
            gene_input=str(st.session_state.get("global_gene_query", "")),
            aggregated_by_file=_session_indexed_aggregates(),
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

                idx_batch = _session_indexed_aggregates()
                miss_batch = [k for k in runs if k not in idx_batch]
                if miss_batch:
                    with st.spinner("Processing IP-MS Data..."):
                        for k in miss_batch:
                            _ensure_dataset_in_session_index(k, meta_by_file)
                cons = _lab_consensus_table(runs, _session_indexed_aggregates(), min_fraction=0.5)
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

