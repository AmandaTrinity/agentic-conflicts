"""Fused per-repository mining: PR chronology + merge mining in a single clone.

The original pipeline cloned every repository twice: once in Stage 0.5
(``pr_chronology.extract_pr_commits``, to reconstruct which commits belong to
each PR) and once more in Stage 1 (``analysis_utils.process_single_repository``,
to mine internal merge commits among those commits), deleting the bare clone
after each stage. Since Stage 1 only needs the SHA list Stage 0.5 already
computes, this module does both in one clone-fetch-cleanup cycle per
repository, halving the git network I/O of the mining stage.

The chronology extraction logic (fork-point + commit log) and the merge
mining logic (``_process_one_merge``) are unchanged from ``pr_chronology.py``
and ``analysis_utils.py`` respectively; only the orchestration is fused, so
output records and schemas are identical to the two-stage pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .analysis_utils import _process_one_merge, _normalize_repo_url, cleanup_repo_scratch
from .mining_utils import (
    clone_repo_bare,
    run_git_command,
    _parse_commit_object,
    _PR_INTEGRATION_MERGE_RE,
)

FusedResult = Tuple[str, List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]


def process_repository_fused(repo_info: Tuple[str, pd.DataFrame], scratch_dir: Path = None) -> FusedResult:
    """Clone a repository once, extract PR chronology, and mine internal merges.

    Args:
        repo_info: Tuple of (repo_full_name, repo_df) with one row per PR
            (pre-chronology universe: pr_id, number, repo_url, ...).
        scratch_dir: Directory for the temporary bare clone (required).

    Returns:
        Tuple of (repo_full_name, chronology_records, chronology_errors,
        internal_merges, conflict_chunks, resolved_chunks, classified_chunks,
        extraction_errors) -- same record shapes as the former two-stage
        pipeline (pr_chronology.extract_pr_commits + analysis_utils.process_single_repository).
    """
    repo_full_name, repo_df = repo_info
    repo_url = _normalize_repo_url(repo_df.iloc[0]['repo_url'])
    t0 = time.time()

    if scratch_dir is None:
        raise ValueError("scratch_dir must be provided")

    chronology_records: List[Dict] = []
    chronology_errors: List[Dict] = []
    internal_merges: List[Dict] = []
    conflict_chunks: List[Dict] = []
    resolved_chunks: List[Dict] = []
    classified_chunks: List[Dict] = []
    extraction_errors: List[Dict] = []

    repo_path = clone_repo_bare(repo_url, scratch_dir)
    if not repo_path:
        logging.error(f"Failed to clone/update repository {repo_full_name}")
        chronology_errors.append({"repo_full_name": repo_full_name, "error": "clone_failed"})
        extraction_errors.append({
            "repo_full_name": repo_full_name,
            "pr_id": None,
            "merge_sha": None,
            "error_type": "clone_failed",
            "error_message": "Failed to clone or update repository",
        })
        logging.info(f"[TIMING] {repo_full_name}: {time.time() - t0:.1f}s (clone_failed)")
        return (repo_full_name, chronology_records, chronology_errors,
                internal_merges, conflict_chunks, resolved_chunks, classified_chunks, extraction_errors)

    sha_cache: Dict[str, Tuple[str, Dict]] = {}

    try:
        pr_rows = repo_df.drop_duplicates(subset=["pr_id"])
        pr_numbers = sorted({int(n) for n in pr_rows["number"].dropna().unique()})

        # Fetch only the PR head refs this repo's AIDev rows actually reference,
        # instead of every PR ref the repository has ever had (refs/pull/*/head) --
        # for popular repos with thousands of unrelated historical PRs, the
        # wildcard fetch pulls in commit objects for PRs we never look at.
        if not pr_numbers:
            chronology_errors.append({"repo_full_name": repo_full_name, "error": "no_pr_numbers"})
            return (repo_full_name, chronology_records, chronology_errors,
                    internal_merges, conflict_chunks, resolved_chunks, classified_chunks, extraction_errors)

        refspecs = [f"+refs/pull/{n}/head:refs/pull/{n}/head" for n in pr_numbers]
        fetch_bytes = run_git_command(
            repo_path, "fetch", "origin", *refspecs, check=False
        )
        if fetch_bytes is None:
            chronology_errors.append({"repo_full_name": repo_full_name, "error": "fetch_failed"})
            return (repo_full_name, chronology_records, chronology_errors,
                    internal_merges, conflict_chunks, resolved_chunks, classified_chunks, extraction_errors)

        for _, row in pr_rows.iterrows():
            pr_id = row['pr_id']
            pr_number = row['number']

            try:
                if pd.isna(pr_number):
                    continue
                pr_number = int(pr_number)
                head_ref = f"refs/pull/{pr_number}/head"

                check_ref = run_git_command(repo_path, "show-ref", "--verify", head_ref, check=False)
                if not check_ref:
                    chronology_errors.append({
                        "repo_full_name": repo_full_name,
                        "pr_id": pr_id,
                        "pr_number": pr_number,
                        "error": "pr_ref_not_found",
                    })
                    continue

                base_bytes = run_git_command(repo_path, "merge-base", head_ref, "HEAD", check=False)
                if not base_bytes:
                    chronology_errors.append({
                        "repo_full_name": repo_full_name,
                        "pr_id": pr_id,
                        "pr_number": pr_number,
                        "error": "no_merge_base",
                    })
                    continue
                base_sha = base_bytes.decode("utf-8", "ignore").strip()

                log_bytes = run_git_command(
                    repo_path, "log", f"{base_sha}..{head_ref}", "--reverse", "--format=%H|%aI", check=False
                )
                if not log_bytes:
                    # PR has 0 own commits (identical to base) - nothing to emit or mine
                    continue

                log_lines = [line for line in log_bytes.decode("utf-8", "ignore").splitlines() if line.strip()]

                shas_ordered: List[str] = []
                for commit_index, line in enumerate(log_lines):
                    parts = line.split("|")
                    sha = parts[0]
                    author_date = parts[1] if len(parts) > 1 else None

                    chronology_records.append({
                        "pr_id": pr_id,
                        "repo_full_name": repo_full_name,
                        "pr_number": pr_number,
                        "sha": sha,
                        "author_date": author_date,
                        "commit_index": commit_index,
                        "commit_count": len(log_lines),
                    })
                    shas_ordered.append(sha)

                # --- mining pass over this PR's commits, reusing the same clone ---
                for sha in shas_ordered:
                    if sha in sha_cache:
                        kind, payload = sha_cache[sha]
                        if kind == "merge":
                            internal_merges.append({**payload, "pr_id": pr_id})
                        elif kind == "error":
                            extraction_errors.append({**payload, "pr_id": pr_id})
                        continue

                    blob = run_git_command(repo_path, "cat-file", "-p", sha, check=False)
                    if not blob:
                        err_template = {
                            "repo_full_name": repo_full_name,
                            "merge_sha": sha,
                            "error_type": "sha_not_in_repo",
                            "error_message": "git cat-file returned empty",
                        }
                        extraction_errors.append({**err_template, "pr_id": pr_id})
                        sha_cache[sha] = ("error", err_template)
                        continue

                    parsed = _parse_commit_object(blob)
                    n_parents = len(parsed["parents"])

                    if n_parents < 2:
                        sha_cache[sha] = ("skip", None)
                        continue

                    if n_parents >= 3:
                        err_template = {
                            "repo_full_name": repo_full_name,
                            "merge_sha": sha,
                            "error_type": "octopus_merge",
                            "error_message": f"Commit has {n_parents} parents (excluded)",
                        }
                        extraction_errors.append({**err_template, "pr_id": pr_id})
                        sha_cache[sha] = ("error", err_template)
                        continue

                    if _PR_INTEGRATION_MERGE_RE.search(parsed["message"]):
                        err_template = {
                            "repo_full_name": repo_full_name,
                            "merge_sha": sha,
                            "error_type": "pr_integration_merge",
                            "error_message": "GitHub 'Merge pull request' integration merge (excluded)",
                        }
                        extraction_errors.append({**err_template, "pr_id": pr_id})
                        sha_cache[sha] = ("error", err_template)
                        continue

                    im, cc, rc, ca, errs = _process_one_merge(
                        repo_path, repo_full_name, pr_id, sha, parsed
                    )
                    internal_merges.append(im)
                    conflict_chunks.extend(cc)
                    resolved_chunks.extend(rc)
                    classified_chunks.extend(ca)
                    extraction_errors.extend(errs)

                    im_template = {k: v for k, v in im.items() if k != "pr_id"}
                    sha_cache[sha] = ("merge", im_template)

            except Exception as e:
                extraction_errors.append({
                    "repo_full_name": repo_full_name,
                    "pr_id": pr_id,
                    "merge_sha": None,
                    "error_type": "pr_processing_exception",
                    "error_message": str(e),
                })

    except Exception as e:
        chronology_errors.append({"repo_full_name": repo_full_name, "error": str(e)})
    finally:
        cleanup_repo_scratch(repo_path)
        logging.info(f"[TIMING] {repo_full_name}: {time.time() - t0:.1f}s (prs={len(repo_df)})")

    return (repo_full_name, chronology_records, chronology_errors,
            internal_merges, conflict_chunks, resolved_chunks, classified_chunks, extraction_errors)


__all__ = ["process_repository_fused"]
