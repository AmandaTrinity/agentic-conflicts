"""Shared analysis utilities and data loading for the replication package."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterator
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

STRATEGY_ORDER = ["V1", "V2", "CC", "CB", "NC", "NN", "Imprecise"]

_RAW_TO_CANONICAL = {
    "V1": "V1",
    "V2": "V2",
    "ConcatV1V2": "CC",
    "ConcatV2V1": "CC",
    "Combination": "CB",
    "New code": "NC",
    "None": "NN",
    "CC": "CC",
    "CB": "CB",
    "NC": "NC",
    "NN": "NN",
    "Imprecise": "Imprecise",
    # Postponed chunks are localized unresolved-marker commits. They are
    # counted separately through strategy_raw, but folded into Imprecise for
    # the analytical buckets reported in the paper.
    "Postponed": "Imprecise",
}

STRATEGY_PALETTE = {
    "V1": "#0072B2",
    "V2": "#D55E00",
    "CC": "#009E73",
    "CB": "#CC79A7",
    "NC": "#E69F00",
    "NN": "#56B4E9",
    "Imprecise": "#BFBFBF",
}

RESOLVER_ORDER = ["agent", "human"]
RESOLVER_PALETTE = {"agent": "#0072B2", "human": "#D55E00"}

TOP_N_LANGUAGES = 8

# Natural keys for deduplication
_MERGE_KEY = ["repo_full_name", "merge_sha"]
_CHUNK_KEY = ["repo_full_name", "merge_sha", "file_path", "chunk_index"]

# Stratification axes
STRATA = {
    "agent": "agent",
    "language": "language_top",
    "pr_task_type": "pr_task_type",
}

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "results"


@dataclass
class AnalysisTables:
    """Raw Parquet tables produced by the pipeline."""

    universe: pd.DataFrame
    internal_merges: pd.DataFrame
    classified_chunks: pd.DataFrame
    resolved_chunks: pd.DataFrame = None
    extraction_errors: pd.DataFrame = None
    resolver_labels: pd.DataFrame = None

    def __post_init__(self):
        # Fill in None values with empty DataFrames
        if self.resolved_chunks is None:
            self.resolved_chunks = pd.DataFrame()
        if self.extraction_errors is None:
            self.extraction_errors = pd.DataFrame()
        if self.resolver_labels is None:
            self.resolver_labels = pd.DataFrame()


def _read_parquet_optional(path: Path) -> pd.DataFrame:
    """Return an empty DataFrame if the file doesn't exist."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _dedup_on(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Sort by (subset + pr_id) and keep the first row per subset."""
    if df.empty or not all(c in df.columns for c in subset):
        return df
    sort_cols = list(subset)
    if "pr_id" in df.columns:
        sort_cols.append("pr_id")
    return df.sort_values(sort_cols).drop_duplicates(subset=subset, keep="first")


def _pr_context(universe: pd.DataFrame) -> pd.DataFrame:
    """One row per pr_id carrying cross-cutting strata."""
    if universe.empty:
        return universe
    cols = [c for c in (
        "pr_id", "full_name", "agent", "language",
        "pr_task_type", "state", "merged_at",
    ) if c in universe.columns]
    ctx = universe[cols].drop_duplicates(subset=["pr_id"])
    return ctx


def _merge_context(internal_merges: pd.DataFrame) -> pd.DataFrame:
    """One row per physical merge carrying resolver attribution."""
    if internal_merges.empty:
        return internal_merges
    cols = [c for c in (
        "pr_id", "merge_sha", "repo_full_name", "author",
        "committer", "resolver_type",
    ) if c in internal_merges.columns]
    return _dedup_on(internal_merges[cols], _MERGE_KEY)


def _apply_language_topn(df: pd.DataFrame, n: int = TOP_N_LANGUAGES) -> pd.DataFrame:
    """Collapse the language long tail into 'Other'."""
    if "language" not in df.columns:
        return df
    df = df.copy()
    lang = df["language"].fillna("Unknown").replace("", "Unknown")
    top = lang.value_counts().head(n).index.tolist()
    df["language_top"] = lang.where(lang.isin(top), other="Other")
    return df


def _canonicalize_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Fold raw strategy labels into the paper's seven reported buckets."""
    if "strategy" not in df.columns:
        return df
    df = df.copy()
    df["strategy_raw"] = df["strategy"]
    df["strategy"] = df["strategy"].map(_RAW_TO_CANONICAL).fillna("Imprecise")
    df["strategy"] = pd.Categorical(df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    return df


def load_tables(
    data_dir: str = None, deduplicate: bool = True, include_resolved_text: bool = False
) -> AnalysisTables:
    """Load analysis tables from data directory.

    Args:
        include_resolved_text: keep resolved_chunks' raw v1/base/v2/resolution
            conflict text. Defaults to False -- no current analysis reads
            resolved_chunks at all, so this table sat in memory for the
            entire Stage 3-7 run carrying full conflict text for nothing;
            at full-dataset scale that's a real chunk of the 62GB+RAM/8GB
            swap exhaustion seen during analysis. Pass True for future
            analyses that need the actual resolved text (e.g. Goal 2).
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    universe = _read_parquet_optional(data_dir / "universe.parquet")
    internal_merges = _read_parquet_optional(data_dir / "internal_merges.parquet")
    classified_chunks = _read_parquet_optional(data_dir / "classified_chunks.parquet")
    resolved_chunks = _read_parquet_optional(data_dir / "resolved_chunks.parquet")
    if not include_resolved_text:
        resolved_chunks = resolved_chunks.drop(
            columns=["v1", "base", "v2", "resolution"], errors="ignore"
        )
    extraction_errors = _read_parquet_optional(data_dir / "extraction_errors.parquet")

    if deduplicate:
        internal_merges = _dedup_on(internal_merges, _MERGE_KEY)
        classified_chunks = _dedup_on(classified_chunks, _CHUNK_KEY)

    tables = AnalysisTables(
        universe=universe,
        internal_merges=internal_merges,
        classified_chunks=classified_chunks,
        resolved_chunks=resolved_chunks,
        extraction_errors=extraction_errors,
    )

    # Add resolver_labels attribute
    resolver_labels = _read_parquet_optional(data_dir / "resolver_labels.parquet")
    tables.resolver_labels = resolver_labels

    return tables

# IC
def build_merge_artifact_frame(tables: AnalysisTables) -> pd.DataFrame:
    """Merge-level analysis frame with artifact-type composition (Q1).

    Unlike build_merge_frame (which counts conflicting chunks — a Q2
    structural measure), this frame aggregates over *distinct conflicting
    files* per merge, categorized by artifact type, to avoid conflating
    artifact type (Q1) with conflict volume (Q2).
    """
    from .file_category import categorize_filepath

    merges = tables.internal_merges
    if merges.empty:
        merges = merges.copy()
        merges["n_files_conflicting"] = pd.Series(dtype=int)
        return merges

    merges = _dedup_on(merges, _MERGE_KEY).copy()

    if tables.classified_chunks.empty:
        merges["n_files_conflicting"] = 0
        return merges

    # Collapse to one row per distinct conflicting FILE per merge (not per
    # chunk): a file with 10 chunks and a file with 1 chunk must each count
    # once here, or M1's composition would silently encode chunk volume.
    files = tables.classified_chunks[
        ["repo_full_name", "merge_sha", "file_path"]
    ].drop_duplicates()

    files = files.copy()
    files["file_category"] = files["file_path"].apply(categorize_filepath)

    # n_files_conflicting: total distinct conflicting files per merge (used
    # as the denominator for M1 proportions and to detect n=1 edge cases,
    # e.g. for M2's "misto" threshold rule).
    n_files = (
        files.groupby(["repo_full_name", "merge_sha"])
        .size()
        .rename("n_files_conflicting")
        .reset_index()
    )

    # Wide composition table: one column per artifact type, values = count
    # of distinct files of that type in the merge. This is the raw input
    composition = (
        files.groupby(["repo_full_name", "merge_sha", "file_category"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("n_files_")
        .reset_index()
    )

    merges = merges.merge(n_files, on=["repo_full_name", "merge_sha"], how="left")
    merges = merges.merge(composition, on=["repo_full_name", "merge_sha"], how="left")

    merges["n_files_conflicting"] = merges["n_files_conflicting"].fillna(0).astype(int)
    category_cols = [c for c in merges.columns if c.startswith("n_files_") and c != "n_files_conflicting"]
    merges[category_cols] = merges[category_cols].fillna(0).astype(int)

    # Join with PR context
    pctx = _pr_context(tables.universe)
    if not pctx.empty:
        merges = merges.merge(pctx, on="pr_id", how="left")
        merges = _apply_language_topn(merges)

    return merges


def build_chunk_frame(tables: AnalysisTables, include_text: bool = False) -> pd.DataFrame:
    """Chunk-level analysis frame with strategy and context.

    Args:
        include_text: keep the raw v1/base/v2/resolution conflict text
            columns from classified_chunks. Defaults to False because no
            current RQ1/RQ2 analysis reads this text (only the precomputed
            *_loc counts and strategy labels) -- at full-dataset scale
            (1.5M+ chunks), carrying it through the two merges below
            duplicates that text on every merge, which was observed
            exhausting 62GB RAM plus the entire 8GB swap during RQ2
            analysis. Pass True for analyses that need the actual text
            (e.g. comparing resolution content, as planned for Goal 2).
    """
    from .file_category import categorize_filepath

    chunks = tables.classified_chunks
    if chunks.empty:
        chunks = chunks.copy()
        chunks["strategy"] = pd.Series(dtype=object)
        chunks["strategy_raw"] = pd.Series(dtype=object)
        return chunks

    if not include_text:
        chunks = chunks.drop(
            columns=["v1", "base", "v2", "resolution"], errors="ignore"
        )

    chunks = _canonicalize_strategy(chunks)

    # Join resolver attribution on the physical merge. This is required for
    # RQ2, where per-agent strategy distributions include only agent-resolved
    # chunks and humans are the reference distribution.
    mctx = _merge_context(tables.internal_merges)
    if not mctx.empty:
        join_cols = [c for c in ("repo_full_name", "merge_sha") if c in chunks.columns and c in mctx.columns]
        if join_cols and "resolver_type" in mctx.columns:
            chunks = chunks.merge(
                mctx[join_cols + ["resolver_type"]],
                on=join_cols,
                how="left",
            )

    # Add file category
    if "file_path" in chunks.columns:
        chunks["file_category"] = chunks["file_path"].apply(categorize_filepath)

    # Join with PR context
    pctx = _pr_context(tables.universe)
    if not pctx.empty:
        chunks = chunks.merge(pctx, on="pr_id", how="left")
        chunks = _apply_language_topn(chunks)

    return chunks


def build_merge_frame(tables: AnalysisTables) -> pd.DataFrame:
    """Merge-level analysis frame with chunk counts and context."""
    merges = tables.internal_merges
    if merges.empty:
        merges = merges.copy()
        merges["n_chunks"] = pd.Series(dtype=int)
        merges["has_conflict"] = pd.Series(dtype=bool)
        return merges

    merges = _dedup_on(merges, _MERGE_KEY).copy()

    # Count chunks per merge
    if not tables.classified_chunks.empty:
        chunk_src = _dedup_on(tables.classified_chunks, _CHUNK_KEY)
        per_merge = (
            chunk_src
            .groupby(["repo_full_name", "merge_sha"])
            .size()
            .rename("n_chunks")
            .reset_index()
        )
        merges = merges.merge(per_merge, on=["repo_full_name", "merge_sha"], how="left")
        merges["n_chunks"] = merges["n_chunks"].fillna(0).astype(int)
    else:
        merges["n_chunks"] = 0

    merges["has_conflict"] = merges["n_chunks"] > 0

    # Join with PR context
    pctx = _pr_context(tables.universe)
    if not pctx.empty:
        merges = merges.merge(pctx, on="pr_id", how="left")
        merges = _apply_language_topn(merges)

    return merges


def stratum_order(df: pd.DataFrame, axis: str) -> list[str]:
    """Get ordering of strata by frequency."""
    col = STRATA.get(axis, axis)
    if col not in df.columns:
        return []
    values = df[col].fillna("Unknown").astype(str)
    return values.value_counts().index.tolist()


def stratify(df: pd.DataFrame, axis: str) -> Iterator[tuple[str, pd.DataFrame]]:
    """Iterate over (stratum_value, sub_df) pairs."""
    col = STRATA.get(axis, axis)
    if col not in df.columns:
        return iter([])
    values = df[col].fillna("Unknown").astype(str)
    order = values.value_counts().index.tolist()
    for value in order:
        yield value, df[values == value]


def setup_style() -> None:
    """Configure matplotlib/seaborn for publication figures."""
    if not sns or not plt:
        return
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
    })


def save_fig(fig: plt.Figure, name: str, directory: Path | str = None) -> tuple[Path, Path]:
    """Save figure as both PDF and PNG."""
    if directory is None:
        directory = FIGURES_DIR
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    pdf_path = directory / f"{name}.pdf"
    png_path = directory / f"{name}.png"

    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path, png_path


__all__ = [
    'AnalysisTables',
    'load_tables',
    'build_chunk_frame',
    'build_merge_frame',
    'build_merge_artifact_frame',
    'stratum_order',
    'stratify',
    'setup_style',
    'save_fig',
    'STRATEGY_ORDER',
    'STRATEGY_PALETTE',
    'RESOLVER_ORDER',
    'RESOLVER_PALETTE',
    'TOP_N_LANGUAGES',
    'STRATA',
    'DATA_DIR',
    'FIGURES_DIR',
]
