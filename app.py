from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ipms_portal.biological_aliases import BAF_SUBUNIT_SET, expand_biological_target_string
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
LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
DATA_EMPTY_MESSAGE = "No datasets found in /Data. Please add IP-MS CSVs to begin analysis."
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


def _ensure_local_data_dir() -> None:
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)


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


def _scatter_matrix_spectral(
    selected_keys: list[str],
    aggregated_by_file: dict[str, pd.DataFrame],
    meta_by_file: dict[str, dict[str, Any]],
) -> go.Figure:
    genes: set[str] = set()
    for k in selected_keys:
        df = aggregated_by_file.get(k)
        if df is not None and not df.empty:
            genes.update(df["Gene Symbol"].astype(str).tolist())

    short_labels: dict[str, str] = {}
    for k in selected_keys:
        m = meta_by_file.get(k, {})
        fn = m.get("filename") or k.split("/")[-1]
        short_labels[k] = (fn[:28] + "…") if len(fn) > 28 else fn

    rows_data: list[dict[str, Any]] = []
    for g in genes:
        row: dict[str, Any] = {"Gene": g, "is_baf": g in BAF_SUBUNIT_SET}
        for k in selected_keys:
            df = aggregated_by_file.get(k)
            if df is None or df.empty:
                row[short_labels[k]] = 0.0
                continue
            hit = df[df["Gene Symbol"].astype(str) == g]
            row[short_labels[k]] = float(hit["Spectral Count"].iloc[0]) if not hit.empty else 0.0
        rows_data.append(row)

    dfm = pd.DataFrame(rows_data)
    dim_cols = [short_labels[k] for k in selected_keys]
    colors = [BAF_CORE_COLOR if bool(b) else "#00A6ED" for b in dfm["is_baf"]]
    sizes = [11 if bool(b) else 6 for b in dfm["is_baf"]]

    dims = [dict(label=c, values=dfm[c]) for c in dim_cols]
    fig = go.Figure(
        data=go.Splom(
            dimensions=dims,
            marker=dict(
                color=colors,
                size=sizes,
                line=dict(width=0.4, color="rgba(255,255,255,0.65)"),
                opacity=0.9,
            ),
            text=dfm["Gene"],
            hovertemplate="<b>%{text}</b><br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<extra></extra>",
            showupperhalf=True,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Spectral Count — multi-experiment scatter matrix (BAF genes emphasized)",
        dragmode="select",
        font=dict(color="#E6EDF3"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=640,
    )
    return fig


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
    metas = scan_csv_files(data_dir)
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


def _volcano_plot(df_gene: pd.DataFrame) -> go.Figure:
    base_color = "#00A6ED"
    core_color = BAF_CORE_COLOR

    df_gene = df_gene.copy()
    df_gene = df_gene.dropna(subset=["Log_Prob", "Spectral Count"], how="any")

    if df_gene.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Spectral Count",
            yaxis_title="-log10(Avg LDA Probability)",
        )
        return fig

    df_core = df_gene[df_gene["is_baf_core"] == True]
    df_non = df_gene[df_gene["is_baf_core"] == False]

    fig = go.Figure()
    if not df_non.empty:
        fig.add_trace(
            go.Scatter(
                x=df_non["Spectral Count"],
                y=df_non["Log_Prob"],
                mode="markers",
                name="Non-BAF",
                marker=dict(size=8, color=base_color, opacity=0.85),
                customdata=np.stack(
                    [
                        df_non["Gene Symbol"].astype(str).values,
                        df_non["Unique Peptides"].values,
                        df_non["Spectral Count"].values,
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Unique Peptides: %{customdata[1]}<br>"
                    "Spectral Count: %{customdata[2]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    if not df_core.empty:
        fig.add_trace(
            go.Scatter(
                x=df_core["Spectral Count"],
                y=df_core["Log_Prob"],
                mode="markers",
                name="BAF Core",
                marker=dict(size=14, color=core_color, line=dict(width=1, color="#FFFFFF"), opacity=0.95),
                customdata=np.stack(
                    [
                        df_core["Gene Symbol"].astype(str).values,
                        df_core["Unique Peptides"].values,
                        df_core["Spectral Count"].values,
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Unique Peptides: %{customdata[1]}<br>"
                    "Spectral Count: %{customdata[2]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=10, t=30, b=30),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Spectral Count",
        yaxis_title="-log10(Avg LDA Probability)",
    )
    return fig


def _coverage_plot(df_gene: pd.DataFrame) -> go.Figure:
    # Horizontal coverage: spectral count per BAF subunit (0 if absent).
    df_gene2 = df_gene.copy()
    sub_df = pd.DataFrame({"Gene Symbol": BAF_SUBUNITS})
    merged = sub_df.merge(df_gene2[["Gene Symbol", "Spectral Count"]], on="Gene Symbol", how="left")
    merged["Spectral Count"] = merged["Spectral Count"].fillna(0).astype(float)

    merged["present"] = merged["Spectral Count"] > 0
    merged["color"] = np.where(merged["present"], BAF_CORE_COLOR, "rgba(255,255,255,0.20)")

    merged = merged.sort_values("Spectral Count", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=merged["Spectral Count"],
            y=merged["Gene Symbol"],
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
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig


def _comparison_table(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    a = df_a.rename(
        columns={
            "Spectral Count": "Spectral Count_A",
            "Unique Peptides": "Unique Peptides_A",
            "Confidence Score": "Confidence Score_A",
            "Log_Prob": "Log_Prob_A",
            "is_baf_core": "is_baf_core",
        }
    )
    b = df_b.rename(
        columns={
            "Spectral Count": "Spectral Count_B",
            "Unique Peptides": "Unique Peptides_B",
            "Confidence Score": "Confidence Score_B",
            "Log_Prob": "Log_Prob_B",
            "is_baf_core": "is_baf_core",
        }
    )
    merged = pd.merge(a, b, on="Gene Symbol", how="outer", suffixes=("", "_dup"))

    # Determine category
    has_a = merged["Spectral Count_A"].notna()
    has_b = merged["Spectral Count_B"].notna()
    merged["category"] = np.where(has_a & has_b, "Both", np.where(has_a, "Only A", "Only B"))

    # is_baf_core may appear in both; coalesce.
    if "is_baf_core_dup" in merged.columns:
        merged["is_baf_core"] = merged["is_baf_core"].fillna(merged["is_baf_core_dup"])
        merged = merged.drop(columns=["is_baf_core_dup"])

    # Sort: Both first, then descending max spectral count.
    merged["max_spectral"] = merged[["Spectral Count_A", "Spectral Count_B"]].max(axis=1, skipna=True)
    merged = merged.sort_values(
        by=["category", "max_spectral", "Gene Symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )

    return merged


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
            st.session_state["portal_project_view"] = "All Experiments"
            st.session_state["portal_dataset_select"] = fk
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

    st.markdown("#### Search & Metadata")
    bait_query = st.text_input("Search by Bait", placeholder="Type bait (e.g., BCL7, SMARCE1)")
    bait_query_u = bait_query.strip().upper()

    filtered_df = experiments_df.copy()
    if bait_query_u:
        filtered_df = filtered_df[
            filtered_df["bait"].fillna("").astype(str).str.upper().str.contains(bait_query_u, na=False)
        ]

    meta_table = filtered_df.copy()
    meta_table["Processed Date"] = pd.to_datetime(meta_table["mtime"], unit="s", errors="coerce")
    meta_table["Processed Date"] = meta_table["Processed Date"].dt.strftime("%Y-%m-%d %H:%M")
    show = meta_table[
        [
            "file_key",
            "investigator",
            "session_id",
            "label",
            "bait",
            "biological_target",
            "domain_details",
            "Processed Date",
        ]
    ].rename(
        columns={
            "file_key": "Path",
            "investigator": "Investigator",
            "session_id": "Session ID",
            "label": "Label",
            "bait": "Bait",
            "biological_target": "Bio-Target",
            "domain_details": "Domain/Details",
        }
    )
    st.dataframe(show, use_container_width=True, height=280)

    st.markdown("#### Experiment Cards")
    keys_for_cards = filtered_df["file_key"].astype(str).tolist()
    if not keys_for_cards:
        st.caption("No experiments match the current bait filter.")
        return

    cols_per_row = 3
    for i in range(0, len(keys_for_cards), cols_per_row):
        row_keys = keys_for_cards[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, fk in enumerate(row_keys):
            meta = meta_by_file.get(fk, {})
            bait = meta.get("bait") or "NA"
            inv = meta.get("investigator") or "NA"
            sid = meta.get("session_id") or "NA"
            lbl = meta.get("label") or "NA"
            fn = meta.get("filename") or fk.split("/")[-1]
            is_baf_bait = str(bait).upper() in BAF_SUBUNIT_SET
            bait_tag = " <span style='color:#FF4B4B;'>● BAF</span>" if is_baf_bait else ""
            df_agg = aggregated_by_file.get(fk)
            fpath = str(meta.get("path") or os.path.join(data_dir, fk))

            with cols[j]:
                st.markdown(
                    (
                        "<div class='card'>"
                        f"<h4 style='margin:0 0 6px 0;'><b>{bait}</b>{bait_tag}</h4>"
                        f"<div style='opacity:0.9; margin-bottom:4px;'>Investigator: {inv}</div>"
                        f"<div style='opacity:0.9; margin-bottom:4px;'>Session: {sid}</div>"
                        f"<div style='opacity:0.9; margin-bottom:8px;'>Label: {lbl}</div>"
                        f"<div style='font-size:0.9rem; opacity:0.8; word-break:break-word;'>{fn}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

                if df_agg is not None and not df_agg.empty:
                    df_download = df_agg.copy().sort_values(
                        by=["is_baf_core", "Spectral Count", "Confidence Score"],
                        ascending=[False, False, False],
                        kind="mergesort",
                    )
                    csv_bytes = df_download.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Aggregated Data",
                        data=csv_bytes,
                        file_name=_aggregated_download_name(meta, fn),
                        mime="text/csv",
                        key=f"download_{fk}",
                    )
                else:
                    st.caption("Aggregated data unavailable for download.")

                confirm_key = f"confirm_delete_{fk}"
                delete_key = f"delete_{fk}"
                st.checkbox("Confirm delete", key=confirm_key)
                if st.button("Delete", key=delete_key):
                    if st.session_state.get(confirm_key, False):
                        try:
                            if os.path.exists(fpath):
                                os.remove(fpath)
                            load_portal_data.clear()
                            st.success(f"Deleted `{fk}`.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed for `{fk}`: {e}")
                    else:
                        st.warning("Tick 'Confirm delete' before deleting this file.")


def main() -> None:
    st.set_page_config(page_title="BAF-Complex IP-MS Analytics Portal", layout="wide")
    # Inject CSS at app startup so it applies globally.
    _apply_global_css()

    st.title("BAF-Vault: IP-MS Encyclopedia")
    st.caption(
        "Nested `Data/[Investigator]/` repository with peptide→gene aggregation, SWIFT alias mapping, and lab-scale search."
    )

    matching: list[str] = []
    with st.sidebar:
        _ensure_local_data_dir()

        if "data_refresh_token" not in st.session_state:
            st.session_state["data_refresh_token"] = 0
        st.markdown("### Data Directory & File Management")
        st.caption(f"Local data: `{LOCAL_DATA_DIR}` (use `Data/[Investigator]/file.csv`)")

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
                    LOCAL_DATA_DIR,
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

        local_csv_count = _count_csv_files(LOCAL_DATA_DIR)
        if local_csv_count == 0:
            st.info(DATA_EMPTY_MESSAGE)
            st.stop()

        st.markdown("---")
        st.markdown("### Dataset Browser")
        experiments_df, aggregated_by_file, meta_by_file, skipped_files = load_portal_data(
            LOCAL_DATA_DIR,
            st.session_state["data_refresh_token"],
        )

        if experiments_df.empty:
            st.warning("No CSV datasets were parsed successfully. Please check CSV format/headers.")
            st.stop()

        if skipped_files:
            st.caption(f"Skipped {len(skipped_files)} file(s) due to parsing issues.")

        dataset_keys_all = experiments_df["file_key"].astype(str).tolist()
        meta_lookup = meta_by_file
        labels = {k: _pretty_dataset_label(k, meta_lookup.get(k, {})) for k in dataset_keys_all}

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
                    st.session_state["portal_dataset_select"] = pick
                    st.session_state["portal_project_view"] = "All Experiments"
                    st.rerun()

        st.markdown("---")
        st.markdown("### Multi-Compare (2–4)")
        st.multiselect(
            "Experiments for scatter matrix",
            options=dataset_keys_all,
            format_func=lambda kk: labels.get(kk, kk),
            key="multi_compare_pick",
            max_selections=4,
            help="Select 2–4 runs, then open the **Multi-Experiment Compare** tab.",
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

    df_gene = aggregated_by_file[selected_dataset]

    tab_main, tab_cmp, tab_vault, tab_global, tab_batch, tab_multi = st.tabs(
        [
            "Main Dashboard",
            "Comparison Mode",
            "Data Vault",
            "Global Search",
            "Batch Subunit (Consensus)",
            "Multi-Experiment Compare",
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
        st.plotly_chart(_volcano_plot(df_gene), use_container_width=True)

        st.markdown("### Complex Coverage (BAF Subunits Pulled Down)")
        st.plotly_chart(_coverage_plot(df_gene), use_container_width=True)

        st.markdown("### Gene-Level Aggregated Table")
        df_view = df_gene.copy()
        df_view["Confidence Score"] = pd.to_numeric(df_view["Confidence Score"], errors="coerce")
        df_view["Spectral Count"] = pd.to_numeric(df_view["Spectral Count"], errors="coerce")
        # Put BAF core genes first while preserving highest-signal rows near top.
        df_view = df_view.sort_values(
            by=["is_baf_core", "Spectral Count", "Confidence Score"],
            ascending=[False, False, False],
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
            by=["is_baf_core", "Spectral Count", "Confidence Score"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        st.download_button(
            "Download Aggregated Data",
            data=dl.to_csv(index=False).encode("utf-8"),
            file_name=_aggregated_download_name(meta_cur, str(meta_cur.get("filename") or selected_dataset)),
            mime="text/csv",
            key="download_main_active",
        )

    with tab_cmp:
        st.markdown("### Dataset Comparison (Overlap vs Unique Interactors)")
        ds_options = dataset_keys_all
        if len(ds_options) < 2:
            st.info("Comparison needs at least two datasets in `/Data`.")
            return

        idx_a = ds_options.index(selected_dataset) if selected_dataset in ds_options else 0
        idx_b = 1 if len(ds_options) > 1 else 0

        colA, colB = st.columns(2)
        with colA:
            dataset_a = st.selectbox("Dataset A", options=ds_options, index=idx_a)
        with colB:
            dataset_b = st.selectbox("Dataset B", options=ds_options, index=idx_b if idx_b != idx_a else 0)

        try:
            if dataset_a not in aggregated_by_file or dataset_b not in aggregated_by_file:
                st.error("Comparison failed: one or both selected datasets could not be loaded.")
                return

            df_a = aggregated_by_file[dataset_a]
            df_b = aggregated_by_file[dataset_b]
            merged = _comparison_table(df_a, df_b)
            if merged.empty or "category" not in merged.columns:
                st.error("Comparison failed: overlap categories were not generated.")
                return

            overlap = int((merged["category"] == "Both").sum())
            only_a = int((merged["category"] == "Only A").sum())
            only_b = int((merged["category"] == "Only B").sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Overlapping Interactors", f"{overlap}")
            m2.metric("Unique to A", f"{only_a}")
            m3.metric("Unique to B", f"{only_b}")

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
            df_cmp = merged[show_cols].copy()
            df_cmp = df_cmp.rename(columns={"category": "Overlap Category"})
            df_cmp["is_baf_core"] = df_cmp["is_baf_core"].fillna(False)
            df_cmp = df_cmp.sort_values(
                by=["is_baf_core", "Overlap Category", "Spectral Count_A", "Spectral Count_B"],
                ascending=[False, True, False, False],
                kind="mergesort",
            )

            styler_cmp = _styled_gene_table(df_cmp)
            st.dataframe(styler_cmp, use_container_width=True, height=560)

            # Optional: show a small summary for BAF-core overlap categories.
            st.markdown("### BAF Core Emphasis")
            core_only = df_cmp[df_cmp["is_baf_core"] == True]
            if not core_only.empty and "Overlap Category" in core_only.columns:
                core_counts = core_only["Overlap Category"].value_counts().to_dict()
                st.write(core_counts)
            elif not core_only.empty:
                st.caption("BAF-core genes found, but overlap category values are unavailable.")
            else:
                st.caption("No BAF-core genes found in either dataset.")
        except Exception as e:
            st.error(f"Comparison failed: unable to compute overlap for the selected datasets. Details: {e}")

    with tab_vault:
        _render_data_vault_tab(
            experiments_df=experiments_df,
            aggregated_by_file=aggregated_by_file,
            meta_by_file=meta_by_file,
            data_dir=LOCAL_DATA_DIR,
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

    with tab_multi:
        st.markdown("### Multi-Experiment Comparison (Scatter Matrix)")
        sel = [str(x) for x in st.session_state.get("multi_compare_pick", [])]
        if len(sel) < 2:
            st.info("In the sidebar, select **2–4 experiments** under **Multi-Compare**.")
        elif len(sel) > 4:
            st.warning("Please choose at most 4 experiments in the sidebar.")
        else:
            st.plotly_chart(
                _scatter_matrix_spectral(sel, aggregated_by_file, meta_by_file),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

