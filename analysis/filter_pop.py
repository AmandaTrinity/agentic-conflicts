"""Filter AnalysisTables down to AIDev-pop (repos with >= N stars).

Mirrors the AIDev-pop criterion used by Haque et al. (2026) -- restricting
to popular/starred repositories both (a) tests whether findings generalize
beyond that paper's scope, and (b) as a side effect, filters out the kind
of zero-star "agent demo" repos (e.g. DevinDemo, juice-shop-demo) found to
skew per-agent repository-concentration stats, without needing a bespoke
synthetic-repo heuristic.
"""

import logging
from pathlib import Path
from typing import Set

import pandas as pd

from .common import AnalysisTables


def _qualifying_repos(aidev_dir: Path, min_stars: int) -> Set[str]:
    """Repo full_names with >= min_stars, per all_repository.parquet."""
    repo_df = pd.read_parquet(
        Path(aidev_dir) / "all_repository.parquet", columns=["full_name", "stars"]
    )
    qualifying = repo_df.loc[repo_df["stars"] >= min_stars, "full_name"]
    return set(qualifying.dropna())


def filter_to_pop(
    tables: AnalysisTables, aidev_dir: Path, min_stars: int = 100
) -> AnalysisTables:
    """Return a new AnalysisTables restricted to repos with >= min_stars."""
    repos = _qualifying_repos(aidev_dir, min_stars)
    logging.info(
        f"AIDev-pop filter (>= {min_stars} stars): {len(repos):,} qualifying repos"
    )

    def _filter(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df.empty or col not in df.columns:
            return df
        return df[df[col].isin(repos)].copy()

    return AnalysisTables(
        universe=_filter(tables.universe, "full_name"),
        internal_merges=_filter(tables.internal_merges, "repo_full_name"),
        classified_chunks=_filter(tables.classified_chunks, "repo_full_name"),
        resolved_chunks=_filter(tables.resolved_chunks, "repo_full_name"),
        extraction_errors=_filter(tables.extraction_errors, "repo_full_name"),
        resolver_labels=_filter(tables.resolver_labels, "repo_full_name"),
    )
