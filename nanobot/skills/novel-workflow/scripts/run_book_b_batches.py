#!/usr/bin/env python3
"""Batch runner for long Book-B generation with per-batch verification and resume."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CHAPTER_NO_RE = re.compile(r"(\d+)")


@dataclass
class RunnerConfig:
    repo_root: Path
    generate_script: Path
    world_config: Path
    llm_config: Path
    output_dir: Path
    log_dir: Path
    canon_db_path: Path
    book_id: str
    template_book_id: str
    template_canon_db_path: Path
    template_neo4j_uri: str
    template_neo4j_user: str
    template_neo4j_pass: str
    template_neo4j_database: str
    template_qdrant_url: str
    template_qdrant_collection: str
    target_neo4j_uri: str
    target_neo4j_user: str
    target_neo4j_pass: str
    target_neo4j_database: str
    target_qdrant_url: str
    target_qdrant_collection: str
    reference_top_k: int
    continuity_mode: str
    continuity_retry: int
    continuity_window: int
    continuity_min_entities: int
    continuity_min_open_threads: int
    chapter_summary_style: str
    consistency_policy: str
    llm_max_retries: int
    llm_retry_backoff: float
    llm_backoff_factor: float
    llm_backoff_max: float
    llm_retry_jitter: float


def parse_chapter_int(chapter_no: str) -> int | None:
    match = CHAPTER_NO_RE.search(str(chapter_no or ""))
    if not match:
        return None
    return int(match.group(1))


def chapter_file(output_dir: Path, book_id: str, chapter_no: int) -> Path:
    return output_dir / f"{book_id}_chapter_{chapter_no:04d}.md"


def fetch_all_done_chapters(canon_db_path: Path, book_id: str) -> set[int]:
    if not canon_db_path.exists():
        return set()
    conn = sqlite3.connect(canon_db_path)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT chapter_no
            FROM commit_log
            WHERE book_id = ? AND status = 'ALL_DONE'
            """,
            (book_id,),
        ).fetchall()
    finally:
        conn.close()

    result: set[int] = set()
    for row in rows:
        chapter_int = parse_chapter_int(row[0]) if row else None
        if chapter_int is not None:
            result.add(chapter_int)
    return result


def verify_batch(cfg: RunnerConfig, start_chapter: int, end_chapter: int) -> tuple[list[int], list[int]]:
    expected = set(range(start_chapter, end_chapter + 1))

    missing_files = sorted(
        chapter_no
        for chapter_no in expected
        if not chapter_file(cfg.output_dir, cfg.book_id, chapter_no).exists()
    )

    all_done = fetch_all_done_chapters(cfg.canon_db_path, cfg.book_id)
    missing_db = sorted(chapter_no for chapter_no in expected if chapter_no not in all_done)
    return (missing_files, missing_db)


def write_line(log_fp, message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    log_fp.write(line + "\n")
    log_fp.flush()


def build_batch_command(cfg: RunnerConfig, start_chapter: int, chapter_count: int) -> list[str]:
    return [
        sys.executable,
        str(cfg.generate_script),
        "--target-book-id",
        cfg.book_id,
        "--template-book-id",
        cfg.template_book_id,
        "--world-config",
        str(cfg.world_config),
        "--chapter-count",
        str(chapter_count),
        "--start-chapter",
        str(start_chapter),
        "--output-dir",
        str(cfg.output_dir),
        "--llm-config",
        str(cfg.llm_config),
        "--resume",
        "--commit-memory",
        "--consistency-policy",
        cfg.consistency_policy,
        "--continuity-mode",
        cfg.continuity_mode,
        "--continuity-retry",
        str(cfg.continuity_retry),
        "--continuity-window",
        str(cfg.continuity_window),
        "--continuity-min-entities",
        str(cfg.continuity_min_entities),
        "--continuity-min-open-threads",
        str(cfg.continuity_min_open_threads),
        "--chapter-summary-style",
        cfg.chapter_summary_style,
        "--enforce-isolation",
        "--template-semantic-search",
        "--template-semantic-model",
        "chinese-large",
        "--reference-top-k",
        str(cfg.reference_top_k),
        "--llm-max-retries",
        str(cfg.llm_max_retries),
        "--llm-retry-backoff",
        str(cfg.llm_retry_backoff),
        "--llm-backoff-factor",
        str(cfg.llm_backoff_factor),
        "--llm-backoff-max",
        str(cfg.llm_backoff_max),
        "--llm-retry-jitter",
        str(cfg.llm_retry_jitter),
        "--template-canon-db-path",
        str(cfg.template_canon_db_path),
        "--template-neo4j-uri",
        cfg.template_neo4j_uri,
        "--template-neo4j-user",
        cfg.template_neo4j_user,
        "--template-neo4j-pass",
        cfg.template_neo4j_pass,
        "--template-neo4j-database",
        cfg.template_neo4j_database,
        "--template-qdrant-url",
        cfg.template_qdrant_url,
        "--template-qdrant-collection",
        cfg.template_qdrant_collection,
        "--target-canon-db-path",
        str(cfg.canon_db_path),
        "--target-neo4j-uri",
        cfg.target_neo4j_uri,
        "--target-neo4j-user",
        cfg.target_neo4j_user,
        "--target-neo4j-pass",
        cfg.target_neo4j_pass,
        "--target-neo4j-database",
        cfg.target_neo4j_database,
        "--target-qdrant-url",
        cfg.target_qdrant_url,
        "--target-qdrant-collection",
        cfg.target_qdrant_collection,
        "--enforce-chinese-on-injection",
        "--enforce-chinese-on-commit",
        "--enforce-chinese-fields",
        "rule,status,trait,goal,secret,state",
        "--log-dir",
        str(cfg.log_dir),
        "--log-injections",
    ]


def run_batch_once(
    cfg: RunnerConfig,
    start_chapter: int,
    end_chapter: int,
    *,
    log_fp,
) -> int:
    chapter_count = end_chapter - start_chapter + 1
    cmd = build_batch_command(cfg, start_chapter, chapter_count)
    env = dict(os.environ)
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)

    write_line(log_fp, f"RUN batch {start_chapter:04d}-{end_chapter:04d}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cfg.repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        log_fp.write(line + "\n")
    proc.wait()
    log_fp.flush()
    write_line(log_fp, f"EXIT batch {start_chapter:04d}-{end_chapter:04d}: code={proc.returncode}")
    return int(proc.returncode or 0)


def run_batches(
    cfg: RunnerConfig,
    *,
    start_chapter: int,
    end_chapter: int,
    batch_size: int,
    max_attempts: int,
    retry_wait_seconds: float,
    log_file: Path,
) -> int:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("a", encoding="utf-8") as log_fp:
        write_line(log_fp, "==== Book-B batch runner started ====")
        write_line(
            log_fp,
            f"target={cfg.book_id} chapters={start_chapter}-{end_chapter} batch_size={batch_size} max_attempts={max_attempts}",
        )

        for batch_start in range(start_chapter, end_chapter + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end_chapter)
            success = False

            for attempt in range(1, max_attempts + 1):
                write_line(log_fp, f"BATCH {batch_start:04d}-{batch_end:04d} attempt {attempt}/{max_attempts}")
                code = run_batch_once(cfg, batch_start, batch_end, log_fp=log_fp)
                missing_files, missing_db = verify_batch(cfg, batch_start, batch_end)

                if not missing_files and not missing_db:
                    write_line(log_fp, f"BATCH {batch_start:04d}-{batch_end:04d} verified OK")
                    success = True
                    break

                write_line(
                    log_fp,
                    (
                        f"BATCH {batch_start:04d}-{batch_end:04d} incomplete: "
                        f"missing_files={missing_files[:10]} total={len(missing_files)}; "
                        f"missing_all_done={missing_db[:10]} total={len(missing_db)}; code={code}"
                    ),
                )
                if attempt < max_attempts:
                    write_line(log_fp, f"Retry after {retry_wait_seconds}s...")
                    time.sleep(retry_wait_seconds)

            if not success:
                write_line(log_fp, f"STOP on batch {batch_start:04d}-{batch_end:04d} (exhausted retries)")
                return 1

        write_line(log_fp, "==== Book-B batch runner finished successfully ====")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Book-B generation in batches with verification.")
    parser.add_argument("--start-chapter", type=int, default=1)
    parser.add_argument("--end-chapter", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-wait-seconds", type=float, default=20.0)
    parser.add_argument(
        "--log-file",
        default="/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/run_book_b_full_batches.log",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = RunnerConfig(
        repo_root=Path("/home/chris/Desktop/my_workspace/nanobot"),
        generate_script=Path("/home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/scripts/generate_book_ab.py"),
        world_config=Path("/home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/templates/world_spec.updated.json"),
        llm_config=Path("/home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config_claude.json"),
        output_dir=Path("/home/chris/Desktop/my_workspace/novel_data/04/new_book"),
        log_dir=Path("/home/chris/Desktop/my_workspace/novel_data/04/new_book/log"),
        canon_db_path=Path("/home/chris/Desktop/my_workspace/novel_data/04/new_book/canon_novel_04_b_test5.db"),
        book_id="novel_04_b_test5",
        template_book_id="novel_04_a_full",
        template_canon_db_path=Path("/home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_a_full.db"),
        template_neo4j_uri="bolt://localhost:7689",
        template_neo4j_user="neo4j",
        template_neo4j_pass="novel123",
        template_neo4j_database="neo4j",
        template_qdrant_url="http://localhost:6333",
        template_qdrant_collection="novel_04_a_full_assets",
        target_neo4j_uri="bolt://localhost:7690",
        target_neo4j_user="neo4j",
        target_neo4j_pass="novel123",
        target_neo4j_database="neo4j",
        target_qdrant_url="http://localhost:6333",
        target_qdrant_collection="novel_04_b_test5_assets",
        reference_top_k=12,
        continuity_mode="strict_gate",
        continuity_retry=3,
        continuity_window=12,
        continuity_min_entities=3,
        continuity_min_open_threads=1,
        chapter_summary_style="structured",
        consistency_policy="strict_blocking",
        llm_max_retries=4,
        llm_retry_backoff=3.0,
        llm_backoff_factor=2.0,
        llm_backoff_max=75.0,
        llm_retry_jitter=0.5,
    )

    if args.start_chapter <= 0 or args.end_chapter < args.start_chapter:
        raise SystemExit("Invalid chapter range")
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")

    return run_batches(
        cfg,
        start_chapter=args.start_chapter,
        end_chapter=args.end_chapter,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
        log_file=Path(args.log_file),
    )


if __name__ == "__main__":
    raise SystemExit(main())
