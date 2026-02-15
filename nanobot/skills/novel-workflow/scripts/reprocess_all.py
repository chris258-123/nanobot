#!/usr/bin/env python3
"""Batch chapter processing with delta/llm/replay modes."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chapter_processor import ChapterProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _parse_chapter_no(path: Path) -> str:
    parts = path.stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return path.stem


def _load_llm_config(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_llm_rate_limit(min_interval: float, last_start_at: float | None) -> float:
    """Sleep if needed to keep a minimum interval between LLM chapter starts."""
    now = time.monotonic()
    if min_interval > 0 and last_start_at is not None:
        wait_for = min_interval - (now - last_start_at)
        if wait_for > 0:
            logger.info("Rate limit sleep %.2fs before next chapter", wait_for)
            time.sleep(wait_for)
            now = time.monotonic()
    return now


def _retry_sleep(
    attempt: int,
    retry_backoff: float,
    backoff_factor: float,
    backoff_max: float,
    retry_jitter: float,
):
    base = retry_backoff * (backoff_factor ** max(0, attempt - 1))
    delay = min(base, backoff_max)
    if retry_jitter > 0:
        delay += random.uniform(0, retry_jitter)
    logger.info("Retry sleep %.2fs before attempt %d", delay, attempt + 1)
    time.sleep(delay)


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    """Lightweight terminal progress bar with elapsed/ETA timing."""

    def __init__(self, total: int, enabled: bool = True):
        self.total = max(total, 0)
        self.enabled = enabled and self.total > 0
        self.start = time.monotonic()
        self.done = 0

    def update(self, chapter_no: str, status: str):
        if not self.enabled:
            return
        self.done += 1
        ratio = self.done / self.total if self.total else 1.0
        width = 30
        filled = int(width * ratio)
        bar = "=" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.start
        avg = elapsed / self.done if self.done else 0.0
        eta = avg * (self.total - self.done)
        line = (
            f"\r[{bar}] {self.done}/{self.total} {ratio * 100:5.1f}% "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
            f"last={chapter_no} status={status}"
        )
        print(line, end="", flush=True)
        if self.done >= self.total:
            print("", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Reprocess novel chapters into three memory tiers")
    parser.add_argument("--book-id", default="novel_04")
    parser.add_argument("--mode", choices=["delta", "llm", "replay"], default="delta")
    parser.add_argument("--from-chapter", default="0001")
    parser.add_argument("--asset-dir", default="/home/chris/novel_assets_test100")
    parser.add_argument("--chapter-dir", default="")
    parser.add_argument("--canon-db-path", default=os.path.expanduser("~/.nanobot/workspace/canon_v2_reprocessed.db"))
    parser.add_argument("--llm-config", default="")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="novel123")
    parser.add_argument("--qdrant-url", default="")
    parser.add_argument("--qdrant-collection", default="novel_assets_v2")
    parser.add_argument("--qdrant-api-key", default="")
    parser.add_argument("--reset-neo4j", action="store_true")
    parser.add_argument("--reset-canon", action="store_true")
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=3.0)
    parser.add_argument("--llm-backoff-factor", type=float, default=2.0)
    parser.add_argument("--llm-backoff-max", type=float, default=60.0)
    parser.add_argument("--llm-retry-jitter", type=float, default=0.5)
    parser.add_argument("--llm-min-interval", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--context-state-limit", type=int, default=30)
    parser.add_argument("--context-relation-limit", type=int, default=30)
    parser.add_argument("--context-thread-limit", type=int, default=20)
    parser.add_argument("--max-chapters", type=int, default=0, help="0 means no limit")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output")
    args = parser.parse_args()

    if args.mode == "llm" and not args.llm_config:
        raise ValueError("--llm-config is required in llm mode")
    if args.mode == "llm" and not args.chapter_dir and not args.asset_dir:
        raise ValueError("--chapter-dir or --asset-dir is required in llm mode")
    if args.llm_max_retries < 0:
        raise ValueError("--llm-max-retries must be >= 0")
    if args.llm_retry_backoff < 0 or args.llm_backoff_factor < 1:
        raise ValueError("--llm-retry-backoff must be >= 0 and --llm-backoff-factor must be >= 1")
    if args.llm_backoff_max < 0 or args.llm_retry_jitter < 0 or args.llm_min_interval < 0:
        raise ValueError("--llm-backoff-max/--llm-retry-jitter/--llm-min-interval must be >= 0")
    if args.llm_max_tokens <= 0:
        raise ValueError("--llm-max-tokens must be > 0")
    if args.max_chapters < 0:
        raise ValueError("--max-chapters must be >= 0")
    if min(args.context_state_limit, args.context_relation_limit, args.context_thread_limit) <= 0:
        raise ValueError("--context-*-limit must be > 0")

    if args.reset_canon and os.path.exists(args.canon_db_path):
        os.unlink(args.canon_db_path)
        logger.info("Removed Canon DB: %s", args.canon_db_path)

    processor = ChapterProcessor(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_pass=args.neo4j_pass,
        canon_db_path=args.canon_db_path,
        qdrant_url=args.qdrant_url or None,
        qdrant_collection=args.qdrant_collection,
        qdrant_api_key=args.qdrant_api_key,
        llm_config=_load_llm_config(args.llm_config),
        llm_max_tokens=args.llm_max_tokens,
        context_state_limit=args.context_state_limit,
        context_relation_limit=args.context_relation_limit,
        context_thread_limit=args.context_thread_limit,
    )

    if args.reset_neo4j:
        logger.info("Clearing Neo4j data...")
        processor.neo4j.clear_all()
        processor.neo4j._init_schema()

    total = {"entities": 0, "facts": 0, "relations": 0, "events": 0}
    errors: list[tuple[str, str]] = []

    try:
        if args.mode == "replay":
            commits = processor.canon_db.replay_from_commit(args.book_id, args.from_chapter)
            logger.info("Found %d commits to replay", len(commits))
            progress = ProgressBar(len(commits), enabled=not args.no_progress)
            for idx, commit in enumerate(commits, 1):
                commit_id = commit["commit_id"]
                chapter_no = commit["chapter_no"]
                try:
                    processor.replay_commit(commit_id)
                    logger.info("[%d/%d] replay %s (%s) ok", idx, len(commits), chapter_no, commit_id[:8])
                    progress.update(chapter_no, "ok")
                except Exception as exc:  # pragma: no cover - operational path
                    logger.exception("Replay failed for chapter %s", chapter_no)
                    errors.append((chapter_no, str(exc)))
                    progress.update(chapter_no, "err")
        elif args.mode == "delta":
            asset_files = sorted(Path(args.asset_dir).glob("*.json"))
            asset_files = [path for path in asset_files if _parse_chapter_no(path) >= args.from_chapter]
            if args.max_chapters > 0:
                asset_files = asset_files[: args.max_chapters]
            logger.info("Found %d asset files", len(asset_files))
            progress = ProgressBar(len(asset_files), enabled=not args.no_progress)
            for idx, asset_file in enumerate(asset_files, 1):
                chapter_no = _parse_chapter_no(asset_file)
                try:
                    result = processor.process_from_assets(
                        book_id=args.book_id,
                        asset_path=str(asset_file),
                        chapter_no=chapter_no,
                    )
                    for key in total:
                        total[key] += int(result.get(key, 0))
                    logger.info(
                        "[%d/%d] %s entities=%s facts=%s relations=%s events=%s",
                        idx,
                        len(asset_files),
                        chapter_no,
                        result.get("entities", 0),
                        result.get("facts", 0),
                        result.get("relations", 0),
                        result.get("events", 0),
                    )
                    progress.update(chapter_no, "ok")
                except Exception as exc:  # pragma: no cover - operational path
                    logger.exception("Delta processing failed for chapter %s", chapter_no)
                    errors.append((chapter_no, str(exc)))
                    progress.update(chapter_no, "err")
        else:
            use_asset_inputs = not args.chapter_dir
            if use_asset_inputs:
                source_files = sorted(Path(args.asset_dir).glob("*.json"))
                source_label = "asset files"
            else:
                source_files = sorted(Path(args.chapter_dir).glob("*.md"))
                source_label = "chapter files"
            source_files = [path for path in source_files if _parse_chapter_no(path) >= args.from_chapter]
            if args.max_chapters > 0:
                source_files = source_files[: args.max_chapters]
            logger.info("Found %d %s", len(source_files), source_label)
            logger.info(
                "LLM settings: max_tokens=%d context=(%d,%d,%d) retries=%d backoff=%.2fs factor=%.2f max=%.2fs jitter=%.2fs min_interval=%.2fs",
                args.llm_max_tokens,
                args.context_state_limit,
                args.context_relation_limit,
                args.context_thread_limit,
                args.llm_max_retries,
                args.llm_retry_backoff,
                args.llm_backoff_factor,
                args.llm_backoff_max,
                args.llm_retry_jitter,
                args.llm_min_interval,
            )
            progress = ProgressBar(len(source_files), enabled=not args.no_progress)
            last_llm_start_at: float | None = None
            for idx, source_file in enumerate(source_files, 1):
                chapter_no = _parse_chapter_no(source_file)
                chapter_text = ""
                assets = None
                if use_asset_inputs:
                    with open(source_file, "r", encoding="utf-8") as handle:
                        assets = json.load(handle)
                else:
                    chapter_text = source_file.read_text(encoding="utf-8")
                success = False
                attempts = args.llm_max_retries + 1
                for attempt in range(1, attempts + 1):
                    last_llm_start_at = _apply_llm_rate_limit(args.llm_min_interval, last_llm_start_at)
                    try:
                        result = processor.process_chapter(
                            book_id=args.book_id,
                            chapter_no=chapter_no,
                            chapter_text=chapter_text,
                            assets=assets,
                            mode="llm",
                        )
                        if result.get("status") == "blocked":
                            message = f"blocked by conflicts: {result.get('conflicts')}"
                            logger.error(
                                "[%d/%d] %s %s (attempt %d/%d)",
                                idx,
                                len(source_files),
                                chapter_no,
                                message,
                                attempt,
                                attempts,
                            )
                            errors.append((chapter_no, message))
                            success = False
                            break

                        for key in total:
                            total[key] += int(result.get(key, 0))
                        logger.info(
                            "[%d/%d] %s entities=%s facts=%s relations=%s events=%s (attempt %d/%d)",
                            idx,
                            len(source_files),
                            chapter_no,
                            result.get("entities", 0),
                            result.get("facts", 0),
                            result.get("relations", 0),
                            result.get("events", 0),
                            attempt,
                            attempts,
                        )
                        success = True
                        progress.update(chapter_no, "ok")
                        break
                    except Exception as exc:  # pragma: no cover - operational path
                        is_last_attempt = attempt >= attempts
                        if is_last_attempt:
                            logger.exception(
                                "LLM processing failed for chapter %s after %d/%d attempts",
                                chapter_no,
                                attempt,
                                attempts,
                            )
                            errors.append((chapter_no, str(exc)))
                            break

                        logger.warning(
                            "LLM processing failed for chapter %s attempt %d/%d: %s",
                            chapter_no,
                            attempt,
                            attempts,
                            exc,
                        )
                        _retry_sleep(
                            attempt=attempt,
                            retry_backoff=args.llm_retry_backoff,
                            backoff_factor=args.llm_backoff_factor,
                            backoff_max=args.llm_backoff_max,
                            retry_jitter=args.llm_retry_jitter,
                        )

                if not success and not any(ch == chapter_no for ch, _ in errors):
                    errors.append((chapter_no, "llm chapter failed for unknown reason"))
                    progress.update(chapter_no, "err")
                elif not success:
                    progress.update(chapter_no, "err")

        logger.info("=" * 60)
        logger.info("Batch run complete mode=%s errors=%d", args.mode, len(errors))
        logger.info("Totals: %s", total)
        logger.info("Canon stats: %s", processor.canon_db.get_statistics())
        logger.info("Neo4j stats: %s", processor.neo4j.get_statistics())

        if errors:
            logger.warning("Failed chapters: %s", errors)
            return 1
        return 0
    finally:
        processor.close()


if __name__ == "__main__":
    sys.exit(main())
