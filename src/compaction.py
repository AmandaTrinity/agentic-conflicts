"""Batched JSONL -> Parquet compaction, to keep mining-time disk usage bounded.

Without this, the append-only JSONL intermediates (`pr_commits.jsonl`,
`extraction_errors.jsonl`, ...) grow without bound for the entire run and
are only ever converted to Parquet once, at the very end. At AIDev v5's
scale (~957k repositories) this was measured (from a partial run) to reach
roughly 1.2TB before that final conversion would ever happen -- far more
than fits on disk, and more than would fit in memory to convert even if it
did.

``flush_batch`` is called periodically (every N repositories) during
mining: it snapshots each JSONL's current content into a small,
already-compressed Parquet *batch* file and clears the JSONL, so the JSONL
working set never grows past what accumulated since the last flush (Parquet
also compresses the highly repetitive text in these tables -- e.g. the
same fixed error message repeated for every excluded integration merge --
far better than raw JSONL does).

``merge_batches`` is called once, after mining finishes (and can safely be
re-run standalone to finalize a crashed/interrupted run): it concatenates
every batch -- plus any existing canonical Parquet, so calling it more than
once is safe -- into the single canonical Parquet file the rest of the
pipeline expects, deduplicating exactly like the original one-shot
aggregation did, then deletes the batches.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# jsonl filename -> (canonical parquet filename, dedup subset or None for no dedup)
_TARGETS: Dict[str, Tuple[str, Optional[List[str]]]] = {
    "internal_merges.jsonl": ("internal_merges.parquet", ["repo_full_name", "pr_id", "merge_sha"]),
    "conflict_chunks.jsonl": ("conflict_chunks.parquet", ["repo_full_name", "merge_sha", "file_path", "chunk_index"]),
    "resolved_chunks.jsonl": ("resolved_chunks.parquet", ["repo_full_name", "merge_sha", "file_path", "chunk_index"]),
    "classified_chunks.jsonl": ("classified_chunks.parquet", ["repo_full_name", "merge_sha", "file_path", "chunk_index"]),
    "extraction_errors.jsonl": ("extraction_errors.parquet", None),
    "pr_commits.jsonl": ("pr_commits.parquet", ["pr_id", "sha"]),
    "pr_chronology_errors.jsonl": ("pr_chronology_errors.parquet", None),
}


def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed line in {path.name}")
    return records


def _batch_dir(data_dir: Path) -> Path:
    d = data_dir / "_compaction_batches"
    d.mkdir(parents=True, exist_ok=True)
    return d


def flush_batch(data_dir: Path) -> None:
    """Snapshot each pending JSONL into a Parquet batch file, then clear it."""
    data_dir = Path(data_dir)
    batch_dir = _batch_dir(data_dir)
    batch_id = uuid.uuid4().hex[:10]

    flushed_any = False
    for jsonl_name, (parquet_name, _dedup_keys) in _TARGETS.items():
        jsonl_path = data_dir / jsonl_name
        if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
            continue

        records = _read_jsonl(jsonl_path)
        jsonl_path.unlink(missing_ok=True)
        if not records:
            continue

        stem = parquet_name[: -len(".parquet")]
        batch_path = batch_dir / f"{stem}.{batch_id}.parquet"
        pd.DataFrame(records).to_parquet(batch_path)
        flushed_any = True

    if flushed_any:
        logging.info(f"[COMPACT] Flushed JSONL to batch parquet (batch {batch_id})")


def merge_batches(data_dir: Path) -> None:
    """Merge all batch Parquet files (plus any existing canonical Parquet)
    into the canonical Parquet files the rest of the pipeline expects.

    Safe to call more than once (e.g. after resuming a crashed run) or when
    there is nothing pending (no-op).
    """
    data_dir = Path(data_dir)
    batch_dir = _batch_dir(data_dir)

    for jsonl_name, (parquet_name, dedup_keys) in _TARGETS.items():
        stem = parquet_name[: -len(".parquet")]
        batch_paths = sorted(batch_dir.glob(f"{stem}.*.parquet"))
        canonical_path = data_dir / parquet_name

        frames = []
        if canonical_path.exists():
            frames.append(pd.read_parquet(canonical_path))
        frames.extend(pd.read_parquet(p) for p in batch_paths)

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        if dedup_keys and all(k in combined.columns for k in dedup_keys):
            before = len(combined)
            combined = combined.drop_duplicates(subset=dedup_keys, keep="first")
            after = len(combined)
            if before != after:
                logging.info(f"  Merge dedup {parquet_name}: {before:,} -> {after:,} rows")

        combined.to_parquet(canonical_path)
        for p in batch_paths:
            p.unlink(missing_ok=True)
        if batch_paths:
            logging.info(
                f"[COMPACT] Merged {len(batch_paths)} batch(es) -> {parquet_name} ({len(combined):,} rows)"
            )

    try:
        next(batch_dir.iterdir())
    except StopIteration:
        batch_dir.rmdir()
    except FileNotFoundError:
        pass


__all__ = ["flush_batch", "merge_batches"]
