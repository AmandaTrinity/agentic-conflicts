#!/usr/bin/env python3
"""Download only the AIDev subsets the mining pipeline actually reads.

The full hao-li/AIDev-7.6M (v5) dataset ships 14 subsets (~957k repositories,
7.69M PR rows across all_pull_request alone). `build_universe()` in
launch_pipeline.py only ever reads three of them: `all_pull_request`,
`all_repository`, and `pr_task_type`. Downloading the rest (issue,
pr_comments, pr_commit_details, pr_reviews, pr_timeline, related_issue,
all_user, repository, pull_request, pr_review_comments, pr_commits) would
multiply disk usage for data nothing in this repo reads.

This pulls the pre-converted Parquet shards for just those three subsets
from the dataset's `refs/convert/parquet` revision (the same files backing
the HF dataset viewer) instead of cloning the whole repository, and lays
them out as `<subset>.parquet` under the destination directory so they're
a drop-in `--aidev-dir` for launch_pipeline.py.

Usage:
    python download_aidev.py --dest ./data/aidev-v5
    python download_aidev.py --dest ./data/aidev-v5 --repo-id hao-li/AIDev-7.6M
"""

import argparse
import logging
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "hao-li/AIDev-7.6M"
SUBSETS = ["all_pull_request", "all_repository", "pr_task_type"]


def download_aidev_subsets(dest: Path, subsets=SUBSETS, repo_id: str = REPO_ID) -> None:
    """Download `subsets` from `repo_id`'s auto-converted Parquet revision into `dest`.

    Each subset ends up as `dest/<subset>.parquet` -- a directory of shard
    files that pandas/pyarrow read transparently as a single table, so no
    merge step (and no second copy of the data) is needed.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading {subsets} from {repo_id} (refs/convert/parquet) -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="refs/convert/parquet",
        allow_patterns=[f"{subset}/*" for subset in subsets],
        local_dir=dest,
    )

    for subset in subsets:
        src_dir = dest / subset
        out_dir = dest / f"{subset}.parquet"
        if not src_dir.exists():
            logging.warning(f"No files downloaded for subset '{subset}'")
            continue

        shards = sorted(src_dir.rglob("*.parquet"))
        out_dir.mkdir(parents=True, exist_ok=True)
        for shard in shards:
            shutil.move(str(shard), out_dir / shard.name)
        shutil.rmtree(src_dir)

        size_mb = sum(f.stat().st_size for f in out_dir.glob("*")) / (1024 * 1024)
        logging.info(f"  {subset}: {len(shards)} shard(s), {size_mb:.1f} MB -> {out_dir.name}/")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=str, required=True, help="Destination directory (usable as --aidev-dir)")
    parser.add_argument("--repo-id", type=str, default=REPO_ID, help=f"HF dataset repo id (default: {REPO_ID})")
    parser.add_argument(
        "--subsets", nargs="+", default=SUBSETS,
        help=f"Subsets to download (default: {SUBSETS})",
    )
    args = parser.parse_args()

    download_aidev_subsets(Path(args.dest), subsets=args.subsets, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
