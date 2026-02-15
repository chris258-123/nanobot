#!/usr/bin/env python3
"""One-click long-form novel generation from world settings.

Flow:
1) User provides a world setting JSON/text and chapter count.
2) LLM generates a global chapter blueprint.
3) LLM generates chapters sequentially with rolling summaries.
4) Optional: each generated chapter is committed into three-tier memory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


def _strip_markdown_code_blocks(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = _strip_markdown_code_blocks(content)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object")
        return data
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        sliced = cleaned[start : end + 1]
    else:
        sliced = cleaned
    repaired = re.sub(r",\s*([}\]])", r"\1", sliced)
    data = json.loads(repaired)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _load_world_spec(world_text: str, world_config_path: str) -> dict[str, Any]:
    world_text = (world_text or "").strip()
    if world_config_path:
        loaded = json.loads(Path(world_config_path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--world-config JSON must be an object")
        if world_text:
            loaded["world_appendix"] = world_text
        return loaded
    if world_text:
        return {"world": world_text}
    raise ValueError("Provide --world or --world-config")


def _normalize_chapter_plan(
    raw_chapters: list[dict[str, Any]],
    chapter_count: int,
    start_chapter: int,
) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    for idx in range(chapter_count):
        chapter_no = start_chapter + idx
        source = raw_chapters[idx] if idx < len(raw_chapters) and isinstance(raw_chapters[idx], dict) else {}
        title = str(source.get("title") or f"第{chapter_no}章")
        goal = str(source.get("goal") or source.get("core_goal") or "推进主线并制造新冲突")
        conflict = str(source.get("conflict") or source.get("key_conflict") or "角色目标与外部阻力发生碰撞")
        ending_hook = str(source.get("ending_hook") or "留下推动下一章的悬念")
        beat_outline = source.get("beat_outline")
        if not isinstance(beat_outline, list):
            beat_outline = []
        beats = [str(item) for item in beat_outline if str(item).strip()]

        chapters.append(
            {
                "chapter_no": f"{chapter_no:04d}",
                "title": title,
                "goal": goal,
                "conflict": conflict,
                "beat_outline": beats,
                "ending_hook": ending_hook,
            }
        )
    return chapters


@dataclass
class LLMClient:
    llm_config: dict[str, Any]
    timeout: float = 180.0

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        # Mode A: custom endpoint (same style as existing extractors)
        if self.llm_config.get("type") == "custom":
            old_all_proxy = os.environ.pop("ALL_PROXY", None)
            old_all_proxy_lower = os.environ.pop("all_proxy", None)
            try:
                response = httpx.post(
                    self.llm_config["url"],
                    json={
                        "model": self.llm_config["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    headers={"Authorization": f"Bearer {self.llm_config['api_key']}"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                content = content or ""
                if not content.strip():
                    raise RuntimeError("empty LLM response")
                return content
            finally:
                if old_all_proxy:
                    os.environ["ALL_PROXY"] = old_all_proxy
                if old_all_proxy_lower:
                    os.environ["all_proxy"] = old_all_proxy_lower

        # Mode B: providers + model (used in your claude-sonnet-4-5 config)
        if self.llm_config.get("providers") and self.llm_config.get("model"):
            provider_cfg = self.llm_config["providers"].get("anthropic")
            if not provider_cfg:
                raise ValueError("providers.anthropic is required in providers mode")
            api_key = provider_cfg.get("apiKey") or provider_cfg.get("api_key")
            api_base = provider_cfg.get("apiBase") or provider_cfg.get("api_base")
            extra_headers = provider_cfg.get("extraHeaders") or provider_cfg.get("extra_headers")

            from nanobot.providers.litellm_provider import LiteLLMProvider

            provider = LiteLLMProvider(
                api_key=api_key,
                api_base=api_base,
                default_model=self.llm_config["model"],
                extra_headers=extra_headers,
            )
            chat_coro = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_config["model"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    response = pool.submit(lambda: asyncio.run(chat_coro)).result()
            except RuntimeError:
                response = asyncio.run(chat_coro)
            content = (response.content or "").strip()
            if not content:
                raise RuntimeError("empty LLM response")
            if content.lower().startswith("error calling llm:"):
                raise RuntimeError(content)
            return content

        raise ValueError("Unsupported llm_config: expected {type: custom} or {providers, model}")

    def complete_with_retry(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        max_retries: int,
        retry_backoff: float,
        backoff_factor: float,
        backoff_max: float,
        retry_jitter: float,
    ) -> str:
        attempts = max(max_retries, 0) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.complete(prompt, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:  # pragma: no cover - operational retry
                last_error = exc
                if attempt >= attempts:
                    break
                base = retry_backoff * (backoff_factor ** max(0, attempt - 1))
                delay = min(base, backoff_max)
                if retry_jitter > 0:
                    delay += random.uniform(0, retry_jitter)
                time.sleep(delay)
        raise RuntimeError(f"LLM request failed after {attempts} attempts: {last_error}")


class OneClickBookGenerator:
    """Generate a whole book from world settings and chapter count."""

    def __init__(self, llm_config: dict[str, Any]):
        self.llm = LLMClient(llm_config=llm_config)

    def build_blueprint(
        self,
        *,
        book_id: str,
        world_spec: dict[str, Any],
        chapter_count: int,
        start_chapter: int,
        max_tokens: int,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> dict[str, Any]:
        prompt = (
            "你是长篇小说策划编辑。根据世界观，输出整本书的章节蓝图。\n"
            "只返回 JSON，不要 markdown。\n\n"
            f"book_id: {book_id}\n"
            f"start_chapter: {start_chapter}\n"
            f"chapter_count: {chapter_count}\n"
            f"world_spec: {json.dumps(world_spec, ensure_ascii=False)}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "book_title": "string",\n'
            '  "genre": "string",\n'
            '  "global_arc": ["string"],\n'
            '  "chapters": [\n'
            "    {\n"
            '      "chapter_no": "0001",\n'
            '      "title": "string",\n'
            '      "goal": "string",\n'
            '      "conflict": "string",\n'
            '      "beat_outline": ["string"],\n'
            '      "ending_hook": "string"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"要求 chapters 数组长度必须是 {chapter_count}。"
        )
        raw = self.llm.complete_with_retry(
            prompt,
            temperature=0.4,
            max_tokens=max_tokens,
            max_retries=llm_max_retries,
            retry_backoff=llm_retry_backoff,
            backoff_factor=llm_backoff_factor,
            backoff_max=llm_backoff_max,
            retry_jitter=llm_retry_jitter,
        )
        parsed = _parse_json_object(raw)
        chapters_raw = parsed.get("chapters")
        if not isinstance(chapters_raw, list):
            chapters_raw = []
        parsed["chapters"] = _normalize_chapter_plan(chapters_raw, chapter_count, start_chapter)
        return parsed

    def generate_chapter_text(
        self,
        *,
        world_spec: dict[str, Any],
        blueprint: dict[str, Any],
        chapter_plan: dict[str, Any],
        recent_summaries: list[str],
        chapter_min_chars: int,
        chapter_max_chars: int,
        temperature: float,
        max_tokens: int,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> str:
        recents = recent_summaries[-8:]
        prompt = (
            "你是中文长篇小说作者。请生成完整章节正文。\n"
            "必须遵循世界观，不要跳出设定，不要解释写作过程。\n"
            "输出纯正文，不要 JSON，不要代码块。\n\n"
            f"world_spec: {json.dumps(world_spec, ensure_ascii=False)}\n"
            f"book_blueprint: {json.dumps({k: v for k, v in blueprint.items() if k != 'chapters'}, ensure_ascii=False)}\n"
            f"current_chapter_plan: {json.dumps(chapter_plan, ensure_ascii=False)}\n"
            f"recent_chapter_summaries: {json.dumps(recents, ensure_ascii=False)}\n\n"
            f"硬约束：\n- 字数尽量在 {chapter_min_chars} 到 {chapter_max_chars} 中文字符之间\n"
            "- 有明确起承转合\n"
            "- 章节结尾必须呼应 ending_hook\n"
            "- 对话和叙述自然穿插，避免流水账"
        )
        text = self.llm.complete_with_retry(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=llm_max_retries,
            retry_backoff=llm_retry_backoff,
            backoff_factor=llm_backoff_factor,
            backoff_max=llm_backoff_max,
            retry_jitter=llm_retry_jitter,
        )
        return _strip_markdown_code_blocks(text)

    def summarize_chapter(
        self,
        *,
        chapter_title: str,
        chapter_text: str,
        max_tokens: int = 256,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> str:
        prompt = (
            "请将以下章节概括为 1-2 句，用于后续章节连贯性记忆。\n"
            f"chapter_title: {chapter_title}\n"
            f"chapter_text:\n{chapter_text[:4000]}"
        )
        summary = self.llm.complete_with_retry(
            prompt,
            temperature=0.2,
            max_tokens=max_tokens,
            max_retries=llm_max_retries,
            retry_backoff=llm_retry_backoff,
            backoff_factor=llm_backoff_factor,
            backoff_max=llm_backoff_max,
            retry_jitter=llm_retry_jitter,
        ).strip()
        summary = summary.replace("\n", " ")
        return summary[:300]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click novel generation from world settings")
    parser.add_argument("--book-id", required=True, help="Book identifier")
    parser.add_argument("--world", default="", help="Short world setting text")
    parser.add_argument("--world-config", default="", help="Path to world setting JSON")
    parser.add_argument("--chapter-count", type=int, required=True, help="How many chapters to generate")
    parser.add_argument("--start-chapter", type=int, default=1, help="First chapter number")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated markdown")
    parser.add_argument("--llm-config", required=True, help="Path to llm config JSON")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--plan-max-tokens", type=int, default=4096)
    parser.add_argument("--chapter-max-tokens", type=int, default=4096)
    parser.add_argument("--chapter-min-chars", type=int, default=2800)
    parser.add_argument("--chapter-max-chars", type=int, default=4200)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=3.0)
    parser.add_argument("--llm-backoff-factor", type=float, default=2.0)
    parser.add_argument("--llm-backoff-max", type=float, default=60.0)
    parser.add_argument("--llm-retry-jitter", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Skip chapters that already exist")
    parser.add_argument(
        "--commit-memory",
        action="store_true",
        help="Write generated chapters into Canon/Neo4j using ChapterProcessor",
    )
    parser.add_argument("--canon-db-path", default=str(Path("~/.nanobot/workspace/canon_v2_reprocessed.db").expanduser()))
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="novel123")
    parser.add_argument("--qdrant-url", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chapter_count <= 0:
        raise ValueError("--chapter-count must be > 0")
    if args.start_chapter <= 0:
        raise ValueError("--start-chapter must be > 0")
    if args.chapter_min_chars <= 0 or args.chapter_max_chars <= 0:
        raise ValueError("--chapter-min-chars and --chapter-max-chars must be > 0")
    if args.chapter_min_chars > args.chapter_max_chars:
        raise ValueError("--chapter-min-chars cannot be greater than --chapter-max-chars")
    if args.plan_max_tokens <= 0 or args.chapter_max_tokens <= 0:
        raise ValueError("--plan-max-tokens and --chapter-max-tokens must be > 0")
    if args.llm_max_retries < 0:
        raise ValueError("--llm-max-retries must be >= 0")
    if args.llm_retry_backoff < 0 or args.llm_backoff_factor < 1:
        raise ValueError("--llm-retry-backoff must be >= 0 and --llm-backoff-factor must be >= 1")
    if args.llm_backoff_max < 0 or args.llm_retry_jitter < 0:
        raise ValueError("--llm-backoff-max and --llm-retry-jitter must be >= 0")

    llm_config = json.loads(Path(args.llm_config).expanduser().read_text(encoding="utf-8"))
    world_spec = _load_world_spec(args.world, args.world_config)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = OneClickBookGenerator(llm_config=llm_config)
    blueprint = generator.build_blueprint(
        book_id=args.book_id,
        world_spec=world_spec,
        chapter_count=args.chapter_count,
        start_chapter=args.start_chapter,
        max_tokens=args.plan_max_tokens,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff=args.llm_retry_backoff,
        llm_backoff_factor=args.llm_backoff_factor,
        llm_backoff_max=args.llm_backoff_max,
        llm_retry_jitter=args.llm_retry_jitter,
    )

    blueprint_path = output_dir / f"{args.book_id}_blueprint.json"
    blueprint_path.write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Blueprint saved: {blueprint_path}")

    processor = None
    if args.commit_memory:
        from chapter_processor import ChapterProcessor

        processor = ChapterProcessor(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_pass=args.neo4j_pass,
            canon_db_path=str(Path(args.canon_db_path).expanduser()),
            qdrant_url=args.qdrant_url or None,
            llm_config=llm_config,
        )

    chapters = blueprint["chapters"]
    rolling_summaries: list[str] = []
    run_items: list[dict[str, Any]] = []
    started_at = time.monotonic()
    success_count = 0
    fail_count = 0

    try:
        for idx, chapter in enumerate(chapters, 1):
            chapter_no = chapter["chapter_no"]
            title = chapter["title"]
            chapter_path = output_dir / f"{args.book_id}_chapter_{chapter_no}.md"

            if args.resume and chapter_path.exists():
                print(f"[{idx}/{len(chapters)}] skip existing chapter {chapter_no}")
                run_items.append({"chapter_no": chapter_no, "status": "skipped", "path": str(chapter_path)})
                rolling_summaries.append(f"{chapter_no} {title}（已存在，续写时跳过）")
                elapsed = time.monotonic() - started_at
                ratio = idx / len(chapters) if chapters else 1.0
                eta = (elapsed / idx) * (len(chapters) - idx) if idx else 0
                print(
                    f"[{idx}/{len(chapters)} {ratio * 100:5.1f}%] elapsed={_format_duration(elapsed)} "
                    f"eta={_format_duration(eta)} ok={success_count} fail={fail_count} last={chapter_no} skipped",
                    flush=True,
                )
                continue

            print(f"[{idx}/{len(chapters)}] generating chapter {chapter_no} ...", flush=True)
            try:
                chapter_text = generator.generate_chapter_text(
                    world_spec=world_spec,
                    blueprint=blueprint,
                    chapter_plan=chapter,
                    recent_summaries=rolling_summaries,
                    chapter_min_chars=args.chapter_min_chars,
                    chapter_max_chars=args.chapter_max_chars,
                    temperature=args.temperature,
                    max_tokens=args.chapter_max_tokens,
                    llm_max_retries=args.llm_max_retries,
                    llm_retry_backoff=args.llm_retry_backoff,
                    llm_backoff_factor=args.llm_backoff_factor,
                    llm_backoff_max=args.llm_backoff_max,
                    llm_retry_jitter=args.llm_retry_jitter,
                )
                summary = generator.summarize_chapter(
                    chapter_title=title,
                    chapter_text=chapter_text,
                    llm_max_retries=args.llm_max_retries,
                    llm_retry_backoff=args.llm_retry_backoff,
                    llm_backoff_factor=args.llm_backoff_factor,
                    llm_backoff_max=args.llm_backoff_max,
                    llm_retry_jitter=args.llm_retry_jitter,
                )
                rolling_summaries.append(f"{chapter_no} {title}: {summary}")

                rendered = f"# {title}\n\n{chapter_text.strip()}\n"
                chapter_path.write_text(rendered, encoding="utf-8")

                item: dict[str, Any] = {
                    "chapter_no": chapter_no,
                    "title": title,
                    "status": "generated",
                    "summary": summary,
                    "path": str(chapter_path),
                }
                if processor is not None:
                    result = processor.process_chapter(
                        book_id=args.book_id,
                        chapter_no=chapter_no,
                        chapter_title=title,
                        chapter_summary=summary,
                        chapter_text=chapter_text,
                        mode="llm",
                    )
                    item["memory_commit"] = result
                run_items.append(item)
                success_count += 1
            except Exception as exc:  # pragma: no cover - operational path
                fail_count += 1
                item = {
                    "chapter_no": chapter_no,
                    "title": title,
                    "status": "failed",
                    "error": str(exc),
                }
                run_items.append(item)
                print(f"[{idx}/{len(chapters)}] chapter {chapter_no} failed: {exc}", flush=True)

            elapsed = time.monotonic() - started_at
            ratio = idx / len(chapters) if chapters else 1.0
            eta = (elapsed / idx) * (len(chapters) - idx) if idx else 0
            print(
                f"[{idx}/{len(chapters)} {ratio * 100:5.1f}%] elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(eta)} ok={success_count} fail={fail_count} last={chapter_no}",
                flush=True,
            )

        report = {
            "book_id": args.book_id,
            "started_at_chapter": f"{args.start_chapter:04d}",
            "chapter_count": args.chapter_count,
            "generated_at": datetime.now().isoformat(),
            "blueprint_path": str(blueprint_path),
            "items": run_items,
        }
        report_path = output_dir / f"{args.book_id}_run_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Run report saved: {report_path}")
        return 0
    finally:
        if processor is not None:
            processor.close()


if __name__ == "__main__":
    raise SystemExit(main())
