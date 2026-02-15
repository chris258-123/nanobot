#!/usr/bin/env python3
"""Generate Book B from world settings while reading Book A template memory."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None


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
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    sliced = cleaned[start : end + 1] if start != -1 and end > start else cleaned
    repaired = re.sub(r",\s*([}\]])", r"\1", sliced)
    parsed = json.loads(repaired)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


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


def _normalize_chapter_plan(raw_chapters: list[dict[str, Any]], chapter_count: int, start_chapter: int) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    for idx in range(chapter_count):
        chapter_no = start_chapter + idx
        source = raw_chapters[idx] if idx < len(raw_chapters) and isinstance(raw_chapters[idx], dict) else {}
        beats = source.get("beat_outline")
        if not isinstance(beats, list):
            beats = []
        chapters.append(
            {
                "chapter_no": f"{chapter_no:04d}",
                "title": str(source.get("title") or f"第{chapter_no}章"),
                "goal": str(source.get("goal") or source.get("core_goal") or "推进主线并制造新冲突"),
                "conflict": str(source.get("conflict") or source.get("key_conflict") or "角色目标与外部阻力发生碰撞"),
                "beat_outline": [str(item) for item in beats if str(item).strip()],
                "ending_hook": str(source.get("ending_hook") or "留下推动下一章的悬念"),
            }
        )
    return chapters


@dataclass
class MemoryStore:
    canon_db_path: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_pass: str
    neo4j_database: str
    qdrant_url: str
    qdrant_collection: str
    qdrant_api_key: str = ""

    def canon_norm(self) -> str:
        return str(Path(self.canon_db_path).expanduser().resolve())

    def neo4j_norm(self) -> tuple[str, str]:
        return (self.neo4j_uri.strip(), self.neo4j_database.strip())

    def qdrant_norm(self) -> tuple[str, str]:
        return (self.qdrant_url.rstrip("/"), self.qdrant_collection.strip())


def _assert_isolation(template_store: MemoryStore, target_store: MemoryStore) -> None:
    clashes = []
    if template_store.canon_norm() == target_store.canon_norm():
        clashes.append("Canon DB path is identical")
    if template_store.neo4j_norm() == target_store.neo4j_norm():
        clashes.append("Neo4j target (uri+database) is identical")
    if template_store.qdrant_norm() == target_store.qdrant_norm():
        clashes.append("Qdrant target (url+collection) is identical")
    if clashes:
        raise ValueError("Isolation check failed: " + "; ".join(clashes))


@dataclass
class LLMClient:
    llm_config: dict[str, Any]
    timeout: float = 180.0

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 4096) -> str:
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
                content = response.json()["choices"][0]["message"]["content"] or ""
                if not content.strip():
                    raise RuntimeError("empty LLM response")
                return content
            finally:
                if old_all_proxy:
                    os.environ["ALL_PROXY"] = old_all_proxy
                if old_all_proxy_lower:
                    os.environ["all_proxy"] = old_all_proxy_lower

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
            if not content or content.lower().startswith("error calling llm:"):
                raise RuntimeError(content or "empty LLM response")
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


class TemplateMemoryReader:
    """Read-only template extraction from Book A memory stores."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def _qdrant_headers(self) -> dict[str, str]:
        return {"api-key": self.store.qdrant_api_key} if self.store.qdrant_api_key else {}

    def _qdrant_scroll(self, book_id: str, asset_type: str, limit: int) -> list[dict[str, Any]]:
        if not self.store.qdrant_url:
            return []
        try:
            response = httpx.post(
                f"{self.store.qdrant_url.rstrip('/')}/collections/{self.store.qdrant_collection}/points/scroll",
                headers=self._qdrant_headers(),
                json={
                    "filter": {
                        "must": [
                            {"key": "book_id", "match": {"value": book_id}},
                            {"key": "asset_type", "match": {"value": asset_type}},
                        ]
                    },
                    "limit": max(limit, 1),
                    "with_payload": True,
                    "with_vector": False,
                },
                timeout=20.0,
            )
            response.raise_for_status()
            rows = response.json().get("result", {}).get("points", [])
            return [
                {
                    "text": row.get("payload", {}).get("text", ""),
                    "chapter": row.get("payload", {}).get("chapter"),
                    "metadata": row.get("payload", {}).get("metadata", {}),
                }
                for row in rows
            ]
        except Exception:
            return []

    def _canon_recent(self, book_id: str, limit: int) -> dict[str, Any]:
        path = Path(self.store.canon_db_path).expanduser()
        if not path.exists():
            return {"hard_rules": [], "relations": []}
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            hard_rules = conn.execute(
                """
                SELECT fh.chapter_no, er.canonical_name, fh.predicate, fh.object_json, fh.evidence_chunk_id
                FROM fact_history fh
                JOIN commit_log cl ON cl.commit_id = fh.commit_id
                JOIN entity_registry er ON er.entity_id = fh.subject_id
                WHERE cl.book_id = ? AND fh.tier = 'HARD_RULE'
                ORDER BY fh.chapter_no DESC, fh.created_at DESC
                LIMIT ?
                """,
                (book_id, max(limit, 1)),
            ).fetchall()
            relations = conn.execute(
                """
                SELECT rh.chapter_no, fr.canonical_name AS from_name, tr.canonical_name AS to_name,
                       rh.kind, rh.status, rh.evidence_chunk_id
                FROM relationship_history rh
                JOIN commit_log cl ON cl.commit_id = rh.commit_id
                LEFT JOIN entity_registry fr ON fr.entity_id = rh.from_id
                LEFT JOIN entity_registry tr ON tr.entity_id = rh.to_id
                WHERE cl.book_id = ?
                ORDER BY rh.chapter_no DESC, rh.created_at DESC
                LIMIT ?
                """,
                (book_id, max(limit, 1)),
            ).fetchall()
            return {
                "hard_rules": [
                    {
                        "chapter_no": row["chapter_no"],
                        "subject": row["canonical_name"],
                        "predicate": row["predicate"],
                        "value": json.loads(row["object_json"] or "null"),
                        "evidence": row["evidence_chunk_id"],
                    }
                    for row in hard_rules
                ],
                "relations": [dict(row) for row in relations],
            }
        finally:
            conn.close()

    def _neo4j_relation_kinds(self) -> list[dict[str, Any]]:
        if GraphDatabase is None:
            return []
        try:
            driver = GraphDatabase.driver(self.store.neo4j_uri, auth=(self.store.neo4j_user, self.store.neo4j_pass))
            with driver.session(database=self.store.neo4j_database) as session:
                rows = session.run(
                    """
                    MATCH ()-[r:RELATES]->()
                    RETURN r.kind AS kind, count(*) AS cnt
                    ORDER BY cnt DESC
                    LIMIT 20
                    """
                )
                result = [{"kind": row["kind"], "count": row["cnt"]} for row in rows]
            driver.close()
            return result
        except Exception:
            return []

    def build_book_template_profile(self, book_id: str, top_k: int) -> dict[str, Any]:
        return {
            "plot_templates": self._qdrant_scroll(book_id, "plot_beat", top_k),
            "style_templates": self._qdrant_scroll(book_id, "style", top_k),
            "conflict_templates": self._qdrant_scroll(book_id, "conflict", top_k),
            "canon": self._canon_recent(book_id, top_k),
            "neo4j_relation_kinds": self._neo4j_relation_kinds(),
        }

    def build_chapter_template_pack(self, book_id: str, chapter_plan: dict[str, Any], top_k: int) -> dict[str, Any]:
        return {
            "chapter_goal": chapter_plan.get("goal", ""),
            "chapter_conflict": chapter_plan.get("conflict", ""),
            "plot_templates": self._qdrant_scroll(book_id, "plot_beat", top_k),
            "style_templates": self._qdrant_scroll(book_id, "style", top_k),
        }


class TargetMemoryReader:
    """Read Book B hard context before generating each chapter."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def build_hard_pack(self, book_id: str, chapter_no: str, top_k: int) -> dict[str, Any]:
        from canon_db_v2 import CanonDBV2

        db = CanonDBV2(str(Path(self.store.canon_db_path).expanduser()))
        try:
            prev_context = db.get_prev_context_snapshot(
                chapter_no,
                state_limit=max(top_k, 10),
                relation_limit=max(top_k, 10),
                thread_limit=max(top_k // 2, 5),
            )
        finally:
            db.close()

        path = Path(self.store.canon_db_path).expanduser()
        if not path.exists():
            return {"hard_rules": [], "prev_context": prev_context}

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT fh.chapter_no, er.canonical_name, fh.predicate, fh.object_json, fh.evidence_chunk_id
                FROM fact_history fh
                JOIN commit_log cl ON cl.commit_id = fh.commit_id
                JOIN entity_registry er ON er.entity_id = fh.subject_id
                WHERE cl.book_id = ? AND fh.tier = 'HARD_RULE' AND fh.chapter_no < ?
                ORDER BY fh.chapter_no DESC, fh.created_at DESC
                LIMIT ?
                """,
                (book_id, chapter_no, max(top_k, 1)),
            ).fetchall()
            hard_rules = [
                {
                    "chapter_no": row["chapter_no"],
                    "subject": row["canonical_name"],
                    "predicate": row["predicate"],
                    "value": json.loads(row["object_json"] or "null"),
                    "evidence": row["evidence_chunk_id"],
                }
                for row in rows
            ]
        finally:
            conn.close()
        return {"hard_rules": hard_rules, "prev_context": prev_context}


class OneClickBookGenerator:
    def __init__(self, llm_config: dict[str, Any]):
        self.llm = LLMClient(llm_config=llm_config)

    def build_blueprint(
        self,
        *,
        target_book_id: str,
        world_spec: dict[str, Any],
        template_profile: dict[str, Any],
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
            "你是长篇小说策划编辑。根据世界观与模板参考，输出 Book B 的章节蓝图。\n"
            "只返回 JSON，不要 markdown。\n\n"
            f"target_book_id: {target_book_id}\n"
            f"start_chapter: {start_chapter}\n"
            f"chapter_count: {chapter_count}\n"
            f"world_spec: {json.dumps(world_spec, ensure_ascii=False)}\n"
            f"template_profile_from_bookA: {json.dumps(template_profile, ensure_ascii=False)}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "book_title": "string",\n'
            '  "genre": "string",\n'
            '  "global_arc": ["string"],\n'
            '  "chapters": [{"chapter_no":"0001","title":"string","goal":"string","conflict":"string","beat_outline":["string"],"ending_hook":"string"}]\n'
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
        blueprint_meta: dict[str, Any],
        chapter_plan: dict[str, Any],
        template_chapter_pack: dict[str, Any],
        hard_pack_b: dict[str, Any],
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
        prompt = (
            "你是中文长篇小说作者。请生成完整章节正文。\n"
            "Book B 需要保持自身硬一致性，并参考 Book A 模板节拍与风格。\n"
            "输出纯正文，不要 JSON，不要代码块。\n\n"
            f"world_spec: {json.dumps(world_spec, ensure_ascii=False)}\n"
            f"book_blueprint_meta: {json.dumps(blueprint_meta, ensure_ascii=False)}\n"
            f"chapter_plan: {json.dumps(chapter_plan, ensure_ascii=False)}\n"
            f"template_pack_from_bookA: {json.dumps(template_chapter_pack, ensure_ascii=False)}\n"
            f"hard_context_from_bookB: {json.dumps(hard_pack_b, ensure_ascii=False)}\n"
            f"recent_summaries_bookB: {json.dumps(recent_summaries[-8:], ensure_ascii=False)}\n\n"
            f"硬约束：\n- 字数尽量在 {chapter_min_chars} 到 {chapter_max_chars} 中文字符之间\n"
            "- 绝不违背 hard_context_from_bookB 的硬事实\n"
            "- 章节结尾呼应 ending_hook\n"
            "- 可借鉴模板，不可照抄文本"
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
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> str:
        prompt = f"请用1-2句总结章节。标题：{chapter_title}\n正文：\n{chapter_text[:4000]}"
        summary = self.llm.complete_with_retry(
            prompt,
            temperature=0.2,
            max_tokens=256,
            max_retries=llm_max_retries,
            retry_backoff=llm_retry_backoff,
            backoff_factor=llm_backoff_factor,
            backoff_max=llm_backoff_max,
            retry_jitter=llm_retry_jitter,
        )
        return summary.replace("\n", " ").strip()[:300]


def _make_store(args: argparse.Namespace, prefix: str, fallback: str = "") -> MemoryStore:
    def _read(name: str, default: str) -> str:
        value = getattr(args, f"{prefix}_{name}", "")
        if value:
            return value
        if fallback:
            fallback_value = getattr(args, f"{fallback}_{name}", "")
            if fallback_value:
                return fallback_value
        return default

    return MemoryStore(
        canon_db_path=_read("canon_db_path", str(Path("~/.nanobot/workspace/canon_v2_reprocessed.db"))),
        neo4j_uri=_read("neo4j_uri", "bolt://localhost:7687"),
        neo4j_user=_read("neo4j_user", "neo4j"),
        neo4j_pass=_read("neo4j_pass", "novel123"),
        neo4j_database=_read("neo4j_database", "neo4j"),
        qdrant_url=_read("qdrant_url", ""),
        qdrant_collection=_read("qdrant_collection", "novel_assets_v2"),
        qdrant_api_key=_read("qdrant_api_key", ""),
    )


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    target_book_id = (args.target_book_id or args.book_id).strip()
    if not target_book_id:
        raise ValueError("book_id or target_book_id is required")

    llm_config_obj = getattr(args, "llm_config_obj", None)
    if llm_config_obj is not None:
        if not isinstance(llm_config_obj, dict):
            raise ValueError("llm_config_obj must be a JSON object")
        llm_config = llm_config_obj
    else:
        llm_config_path = args.llm_config_path or args.llm_config
        if not llm_config_path:
            raise ValueError("--llm-config or --llm-config-path is required")
        llm_config = json.loads(Path(llm_config_path).expanduser().read_text(encoding="utf-8"))
    world_spec = _load_world_spec(args.world, args.world_config)

    target_store = _make_store(args, "target", fallback="legacy")
    template_store = _make_store(args, "template", fallback="legacy") if args.template_book_id else None
    if args.enforce_isolation and template_store is not None:
        _assert_isolation(template_store, target_store)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = OneClickBookGenerator(llm_config)
    template_reader = TemplateMemoryReader(template_store) if template_store else None
    target_reader = TargetMemoryReader(target_store)
    template_profile = (
        template_reader.build_book_template_profile(args.template_book_id, args.reference_top_k)
        if template_reader
        else {}
    )
    blueprint = generator.build_blueprint(
        target_book_id=target_book_id,
        world_spec=world_spec,
        template_profile=template_profile,
        chapter_count=args.chapter_count,
        start_chapter=args.start_chapter,
        max_tokens=args.plan_max_tokens,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff=args.llm_retry_backoff,
        llm_backoff_factor=args.llm_backoff_factor,
        llm_backoff_max=args.llm_backoff_max,
        llm_retry_jitter=args.llm_retry_jitter,
    )
    blueprint_path = output_dir / f"{target_book_id}_blueprint.json"
    blueprint_path.write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Blueprint saved: {blueprint_path}", flush=True)

    processor = None
    if args.commit_memory:
        from chapter_processor import ChapterProcessor

        processor = ChapterProcessor(
            neo4j_uri=target_store.neo4j_uri,
            neo4j_user=target_store.neo4j_user,
            neo4j_pass=target_store.neo4j_pass,
            canon_db_path=str(Path(target_store.canon_db_path).expanduser()),
            qdrant_url=target_store.qdrant_url or None,
            qdrant_collection=target_store.qdrant_collection,
            qdrant_api_key=target_store.qdrant_api_key,
            llm_config=llm_config,
        )

    run_items: list[dict[str, Any]] = []
    rolling_summaries: list[str] = []
    started = time.monotonic()
    ok_count = 0
    fail_count = 0
    terminated = False
    blueprint_meta = {k: v for k, v in blueprint.items() if k != "chapters"}

    try:
        for idx, chapter in enumerate(blueprint["chapters"], 1):
            chapter_no = chapter["chapter_no"]
            chapter_title = chapter["title"]
            chapter_path = output_dir / f"{target_book_id}_chapter_{chapter_no}.md"

            if args.resume and chapter_path.exists():
                run_items.append({"chapter_no": chapter_no, "status": "skipped", "path": str(chapter_path)})
                elapsed = time.monotonic() - started
                eta = (elapsed / idx) * (len(blueprint["chapters"]) - idx) if idx else 0
                print(
                    f"[{idx}/{len(blueprint['chapters'])}] skip {chapter_no} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                    flush=True,
                )
                continue

            try:
                template_pack = (
                    template_reader.build_chapter_template_pack(args.template_book_id, chapter, args.reference_top_k)
                    if template_reader
                    else {}
                )
                hard_pack_b = target_reader.build_hard_pack(target_book_id, chapter_no, args.reference_top_k)
                chapter_text = generator.generate_chapter_text(
                    world_spec=world_spec,
                    blueprint_meta=blueprint_meta,
                    chapter_plan=chapter,
                    template_chapter_pack=template_pack,
                    hard_pack_b=hard_pack_b,
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
                    chapter_title=chapter_title,
                    chapter_text=chapter_text,
                    llm_max_retries=args.llm_max_retries,
                    llm_retry_backoff=args.llm_retry_backoff,
                    llm_backoff_factor=args.llm_backoff_factor,
                    llm_backoff_max=args.llm_backoff_max,
                    llm_retry_jitter=args.llm_retry_jitter,
                )
                rolling_summaries.append(f"{chapter_no} {chapter_title}: {summary}")

                chapter_path.write_text(f"# {chapter_title}\n\n{chapter_text.strip()}\n", encoding="utf-8")
                item = {
                    "chapter_no": chapter_no,
                    "title": chapter_title,
                    "status": "generated",
                    "summary": summary,
                    "path": str(chapter_path),
                }

                if processor is not None:
                    memory_commit = processor.process_chapter(
                        book_id=target_book_id,
                        chapter_no=chapter_no,
                        chapter_title=chapter_title,
                        chapter_summary=summary,
                        chapter_text=chapter_text,
                        mode="llm",
                    )
                    item["memory_commit"] = memory_commit
                    if memory_commit.get("status") == "blocked":
                        raise RuntimeError(f"blocking conflicts: {memory_commit.get('conflicts')}")

                run_items.append(item)
                ok_count += 1
            except Exception as exc:  # pragma: no cover - operational path
                run_items.append({"chapter_no": chapter_no, "title": chapter_title, "status": "failed", "error": str(exc)})
                fail_count += 1
                if args.consistency_policy == "strict_blocking":
                    terminated = True

            elapsed = time.monotonic() - started
            eta = (elapsed / idx) * (len(blueprint["chapters"]) - idx) if idx else 0
            print(
                f"[{idx}/{len(blueprint['chapters'])} {(idx/len(blueprint['chapters']))*100:5.1f}%] "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} ok={ok_count} fail={fail_count} "
                f"last={chapter_no}",
                flush=True,
            )
            if terminated:
                print("Terminated due to strict_blocking policy.", flush=True)
                break
    finally:
        if processor is not None:
            processor.close()

    report = {
        "target_book_id": target_book_id,
        "template_book_id": args.template_book_id,
        "chapter_count": args.chapter_count,
        "generated_at": datetime.now().isoformat(),
        "target_store": target_store.__dict__,
        "template_store": template_store.__dict__ if template_store else None,
        "ok": ok_count,
        "failed": fail_count,
        "terminated": terminated,
        "items": run_items,
    }
    report_path = output_dir / f"{target_book_id}_run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok" if fail_count == 0 else "partial",
        "target_book_id": target_book_id,
        "template_book_id": args.template_book_id,
        "ok": ok_count,
        "failed": fail_count,
        "terminated": terminated,
        "blueprint_path": str(blueprint_path),
        "report_path": str(report_path),
        "output_dir": str(output_dir),
    }


def default_run_options() -> dict[str, Any]:
    """Programmatic defaults mirroring CLI parse_args."""
    return {
        "book_id": "",
        "target_book_id": "",
        "template_book_id": "",
        "world": "",
        "world_config": "",
        "chapter_count": 0,
        "start_chapter": 1,
        "output_dir": "",
        "llm_config": "",
        "llm_config_path": "",
        "llm_config_obj": None,
        "temperature": 0.8,
        "plan_max_tokens": 4096,
        "chapter_max_tokens": 4096,
        "chapter_min_chars": 2800,
        "chapter_max_chars": 4200,
        "llm_max_retries": 3,
        "llm_retry_backoff": 3.0,
        "llm_backoff_factor": 2.0,
        "llm_backoff_max": 60.0,
        "llm_retry_jitter": 0.5,
        "reference_top_k": 8,
        "consistency_policy": "strict_blocking",
        "enforce_isolation": True,
        "resume": False,
        "commit_memory": False,
        "legacy_canon_db_path": str(Path("~/.nanobot/workspace/canon_v2_reprocessed.db")),
        "legacy_neo4j_uri": "bolt://localhost:7687",
        "legacy_neo4j_user": "neo4j",
        "legacy_neo4j_pass": "novel123",
        "legacy_neo4j_database": "neo4j",
        "legacy_qdrant_url": "",
        "legacy_qdrant_collection": "novel_assets_v2",
        "legacy_qdrant_api_key": "",
        "target_canon_db_path": "",
        "target_neo4j_uri": "",
        "target_neo4j_user": "",
        "target_neo4j_pass": "",
        "target_neo4j_database": "",
        "target_qdrant_url": "",
        "target_qdrant_collection": "",
        "target_qdrant_api_key": "",
        "template_canon_db_path": "",
        "template_neo4j_uri": "",
        "template_neo4j_user": "",
        "template_neo4j_pass": "",
        "template_neo4j_database": "",
        "template_qdrant_url": "",
        "template_qdrant_collection": "",
        "template_qdrant_api_key": "",
    }


def run_generation_with_options(**overrides: Any) -> dict[str, Any]:
    """Programmatic entrypoint for tools/tests (without CLI argv parsing)."""
    options = default_run_options()
    options.update(overrides)
    return run_generation(argparse.Namespace(**options))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A(read)-B(write) one-click novel generation")
    parser.add_argument("--book-id", default="", help="Target Book B id (legacy alias)")
    parser.add_argument("--target-book-id", default="", help="Target Book B id")
    parser.add_argument("--template-book-id", default="", help="Book A template id")
    parser.add_argument("--world", default="")
    parser.add_argument("--world-config", default="")
    parser.add_argument("--chapter-count", type=int, required=True)
    parser.add_argument("--start-chapter", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-config", default="")
    parser.add_argument("--llm-config-path", default="")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--plan-max-tokens", type=int, default=4096)
    parser.add_argument("--chapter-max-tokens", type=int, default=4096)
    parser.add_argument("--chapter-min-chars", type=int, default=2800)
    parser.add_argument("--chapter-max-chars", type=int, default=4200)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=3.0)
    parser.add_argument("--llm-backoff-factor", type=float, default=2.0)
    parser.add_argument("--llm-backoff-max", type=float, default=60.0)
    parser.add_argument("--llm-retry-jitter", type=float, default=0.5)
    parser.add_argument("--reference-top-k", type=int, default=8)
    parser.add_argument("--consistency-policy", choices=["strict_blocking", "warn_only"], default="strict_blocking")
    parser.add_argument("--enforce-isolation", action="store_true", default=True)
    parser.add_argument("--no-enforce-isolation", action="store_false", dest="enforce_isolation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--commit-memory", action="store_true")

    # legacy shared store args
    parser.add_argument("--legacy-canon-db-path", default=str(Path("~/.nanobot/workspace/canon_v2_reprocessed.db")))
    parser.add_argument("--legacy-neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--legacy-neo4j-user", default="neo4j")
    parser.add_argument("--legacy-neo4j-pass", default="novel123")
    parser.add_argument("--legacy-neo4j-database", default="neo4j")
    parser.add_argument("--legacy-qdrant-url", default="")
    parser.add_argument("--legacy-qdrant-collection", default="novel_assets_v2")
    parser.add_argument("--legacy-qdrant-api-key", default="")

    # target Book B store
    parser.add_argument("--target-canon-db-path", default="")
    parser.add_argument("--target-neo4j-uri", default="")
    parser.add_argument("--target-neo4j-user", default="")
    parser.add_argument("--target-neo4j-pass", default="")
    parser.add_argument("--target-neo4j-database", default="")
    parser.add_argument("--target-qdrant-url", default="")
    parser.add_argument("--target-qdrant-collection", default="")
    parser.add_argument("--target-qdrant-api-key", default="")

    # template Book A store
    parser.add_argument("--template-canon-db-path", default="")
    parser.add_argument("--template-neo4j-uri", default="")
    parser.add_argument("--template-neo4j-user", default="")
    parser.add_argument("--template-neo4j-pass", default="")
    parser.add_argument("--template-neo4j-database", default="")
    parser.add_argument("--template-qdrant-url", default="")
    parser.add_argument("--template-qdrant-collection", default="")
    parser.add_argument("--template-qdrant-api-key", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_generation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
