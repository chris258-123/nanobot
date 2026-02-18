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

try:
    from FlagEmbedding import FlagModel
    FLAG_MODEL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FlagModel = None
    FLAG_MODEL_AVAILABLE = False

ENTITY_ID_KEY_TO_NAME_KEY: dict[str, str] = {
    "entity_id": "entity_name",
    "subject_id": "subject_name",
    "from_id": "from_name",
    "to_id": "to_name",
    "character_id": "character_name",
    "actor_id": "actor_name",
    "target_id": "target_name",
}
PROMPT_DROP_KEYS: set[str] = {
    "memory_type",
    "chapter_sort_key",
    "evidence_chunk_id",
}
PROMPT_METADATA_LABELS: dict[str, str] = {
    "characters": "角色",
    "event": "事件",
    "impact": "影响",
    "causality": "因果",
    "chapter_position": "章节定位",
    "subject_name": "主体",
    "predicate": "谓词",
    "value": "事实值",
    "from_name": "关系起点",
    "to_name": "关系终点",
    "kind": "关系类型",
    "status": "关系状态",
    "relation": "关系描述",
    "fact": "事实描述",
    "summary": "摘要",
    "goal": "目标",
    "conflict": "冲突",
    "scene": "场景",
    "location": "地点",
    "time": "时间",
    "emotion": "情绪",
    "action": "动作",
    "result": "结果",
    "notes": "备注",
}


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _is_probable_internal_entity_id(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    return bool(re.fullmatch(r"[a-z]+_[0-9a-f]{6,}", text))


def _replace_inline_entity_ids(text: str, entity_name_index: dict[str, str]) -> str:
    if not text or not entity_name_index:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return entity_name_index.get(token, token)

    return re.sub(r"[a-z]+_[0-9a-f]{6,}", _replace, text)


def _coerce_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or _is_probable_internal_entity_id(text):
        return None
    return text


def _collect_entity_name_index(payload: Any, index: dict[str, str]) -> None:
    if isinstance(payload, dict):
        for id_key, name_key in ENTITY_ID_KEY_TO_NAME_KEY.items():
            raw_id = payload.get(id_key)
            if not isinstance(raw_id, str):
                continue
            entity_id = raw_id.strip()
            if not entity_id:
                continue
            name = _coerce_name(
                payload.get(name_key)
                or payload.get("canonical_name")
                or payload.get("name")
                or payload.get("subject")
            )
            if name:
                index.setdefault(entity_id, name)

        generic_id = payload.get("id")
        generic_name = _coerce_name(payload.get("name") or payload.get("canonical_name"))
        if isinstance(generic_id, str) and _is_probable_internal_entity_id(generic_id) and generic_name:
            index.setdefault(generic_id, generic_name)

        for value in payload.values():
            _collect_entity_name_index(value, index)
        return

    if isinstance(payload, list):
        for value in payload:
            _collect_entity_name_index(value, index)


def _replace_entity_ids_with_names(payload: Any, entity_name_index: dict[str, str]) -> Any:
    if isinstance(payload, dict):
        replaced: dict[str, Any] = {}
        for key, value in payload.items():
            if key in PROMPT_DROP_KEYS:
                continue

            if key in ENTITY_ID_KEY_TO_NAME_KEY:
                if isinstance(value, str):
                    entity_id = value.strip()
                    if entity_id:
                        replaced[ENTITY_ID_KEY_TO_NAME_KEY[key]] = entity_name_index.get(entity_id, entity_id)
                continue

            if key.endswith("_id") and isinstance(value, str) and _is_probable_internal_entity_id(value):
                name_key = f"{key[:-3]}_name"
                entity_id = value.strip()
                if entity_id:
                    replaced[name_key] = entity_name_index.get(entity_id, entity_id)
                continue

            child = _replace_entity_ids_with_names(value, entity_name_index)
            if _is_empty_value(child):
                continue
            replaced[key] = child
        return replaced

    if isinstance(payload, list):
        cleaned_list = []
        for value in payload:
            child = _replace_entity_ids_with_names(value, entity_name_index)
            if _is_empty_value(child):
                continue
            cleaned_list.append(child)
        return cleaned_list

    if isinstance(payload, str):
        text = payload.strip()
        if _is_probable_internal_entity_id(text):
            return entity_name_index.get(text, text)
        return _replace_inline_entity_ids(payload, entity_name_index)

    return payload


def _metadata_to_prompt_elements(metadata: dict[str, Any]) -> dict[str, Any]:
    elements: dict[str, Any] = {}
    for key, value in metadata.items():
        if _is_empty_value(value):
            continue
        label = PROMPT_METADATA_LABELS.get(key)
        if label is None and re.search(r"[\u4e00-\u9fff]", key):
            label = key
        if label is None:
            continue
        existing = elements.get(label)
        if existing is None:
            elements[label] = value
            continue
        if isinstance(existing, list):
            existing.append(value)
            continue
        elements[label] = [existing, value]
    return elements


def _sanitize_recent_summaries_for_prompt(recent_summaries: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in recent_summaries:
        text = str(item or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def _ensure_hard_context_shape(hard_pack: Any) -> dict[str, Any]:
    default_prev = {
        "character_state": [],
        "recent_relations": [],
        "open_threads": [],
    }
    if not isinstance(hard_pack, dict):
        return {
            "hard_rules": [],
            "prev_context": default_prev,
        }

    hard_rules = hard_pack.get("hard_rules")
    if not isinstance(hard_rules, list):
        hard_rules = []

    prev_context_raw = hard_pack.get("prev_context")
    prev_context = prev_context_raw if isinstance(prev_context_raw, dict) else {}
    character_state = prev_context.get("character_state")
    recent_relations = prev_context.get("recent_relations")
    open_threads = prev_context.get("open_threads")
    return {
        "hard_rules": hard_rules,
        "prev_context": {
            "character_state": character_state if isinstance(character_state, list) else [],
            "recent_relations": recent_relations if isinstance(recent_relations, list) else [],
            "open_threads": open_threads if isinstance(open_threads, list) else [],
        },
    }


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


def _default_log_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "logs"


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


@dataclass
class InjectionLogger:
    root_dir: Path
    target_book_id: str

    def __post_init__(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.root_dir / "generate_book_ab" / f"{self.target_book_id}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rel_path: str, payload: dict[str, Any] | list[Any]) -> Path:
        path = self.run_dir / rel_path
        _write_json(path, payload)
        return path

    @staticmethod
    def raw_rel_path(rel_path: str) -> str:
        if rel_path.endswith(".json"):
            return rel_path[:-5] + ".raw.json"
        return rel_path + ".raw"

    def write_with_raw(
        self,
        rel_path: str,
        payload: dict[str, Any] | list[Any],
        *,
        raw_payload: dict[str, Any] | list[Any] | None = None,
    ) -> tuple[Path, Path | None]:
        clean_path = self.write(rel_path, payload)
        raw_path: Path | None = None
        if raw_payload is not None:
            raw_path = self.write(self.raw_rel_path(rel_path), raw_payload)
        return clean_path, raw_path


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

    @staticmethod
    def _pop_proxy_env() -> dict[str, str]:
        backup: dict[str, str] = {}
        for key in (
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
        ):
            value = os.environ.pop(key, None)
            if value is not None:
                backup[key] = value
        return backup

    @staticmethod
    def _restore_proxy_env(backup: dict[str, str]) -> None:
        for key, value in backup.items():
            os.environ[key] = value

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        if self.llm_config.get("type") == "custom":
            proxy_backup = self._pop_proxy_env()
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
                self._restore_proxy_env(proxy_backup)

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
            async def _chat_with_timeout():
                return await asyncio.wait_for(
                    provider.chat(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.llm_config["model"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=self.timeout,
                )

            proxy_backup = self._pop_proxy_env()
            try:
                try:
                    asyncio.get_running_loop()
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        response = pool.submit(lambda: asyncio.run(_chat_with_timeout())).result()
                except RuntimeError:
                    response = asyncio.run(_chat_with_timeout())
            finally:
                self._restore_proxy_env(proxy_backup)
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

    def __init__(
        self,
        store: MemoryStore,
        *,
        semantic_search_enabled: bool = True,
        semantic_model_name: str = "",
    ):
        self.store = store
        self._asset_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.semantic_search_enabled = semantic_search_enabled
        self.semantic_model_name = semantic_model_name.strip()
        self._semantic_model: Any | None = None
        self._semantic_model_loaded = False
        self._vector_config_cache: dict[str, Any] | None = None
        self._semantic_model_warning = ""
        self._entity_name_index_cache: dict[str, str] | None = None

    def _qdrant_headers(self) -> dict[str, str]:
        return {"api-key": self.store.qdrant_api_key} if self.store.qdrant_api_key else {}

    def _load_vector_config(self) -> dict[str, Any]:
        if self._vector_config_cache is not None:
            return self._vector_config_cache
        self._vector_config_cache = {}
        if not self.store.qdrant_url:
            return self._vector_config_cache
        try:
            response = httpx.get(
                f"{self.store.qdrant_url.rstrip('/')}/collections/{self.store.qdrant_collection}",
                headers=self._qdrant_headers(),
                timeout=20.0,
                trust_env=False,
            )
            response.raise_for_status()
            vectors = (
                response.json()
                .get("result", {})
                .get("config", {})
                .get("params", {})
                .get("vectors", {})
            )
            if isinstance(vectors, dict):
                self._vector_config_cache = vectors
        except Exception:
            self._vector_config_cache = {}
        return self._vector_config_cache

    @staticmethod
    def _pop_proxy_env() -> dict[str, str]:
        backup: dict[str, str] = {}
        for key in (
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
        ):
            value = os.environ.pop(key, None)
            if value is not None:
                backup[key] = value
        return backup

    @staticmethod
    def _restore_proxy_env(backup: dict[str, str]) -> None:
        for key, value in backup.items():
            os.environ[key] = value

    def _resolve_vector_target(self) -> tuple[str | None, int | None]:
        vectors = self._load_vector_config()
        if not vectors:
            return (None, None)
        if "size" in vectors:
            try:
                return (None, int(vectors["size"]))
            except Exception:
                return (None, None)
        for name, cfg in vectors.items():
            if isinstance(cfg, dict) and "size" in cfg:
                try:
                    return (str(name), int(cfg["size"]))
                except Exception:
                    continue
        return (None, None)

    @staticmethod
    def _default_semantic_model_for_dim(vector_dim: int | None) -> str:
        if vector_dim == 1024:
            return "BAAI/bge-large-zh-v1.5"
        if vector_dim == 768:
            return "moka-ai/m3e-base"
        if vector_dim == 384:
            return "paraphrase-multilingual-MiniLM-L12-v2"
        return "moka-ai/m3e-base"

    def _get_semantic_model(self, expected_dim: int | None) -> Any | None:
        if self._semantic_model_loaded:
            return self._semantic_model
        self._semantic_model_loaded = True
        if not self.semantic_search_enabled:
            return None
        alias_map = {
            "chinese": "moka-ai/m3e-base",
            "chinese-large": "BAAI/bge-large-zh-v1.5",
            "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
        }
        requested = self.semantic_model_name.strip()
        model_name = alias_map.get(requested, requested) if requested else self._default_semantic_model_for_dim(expected_dim)
        use_flag_model = model_name == "BAAI/bge-large-zh-v1.5" and FLAG_MODEL_AVAILABLE
        proxy_backup = self._pop_proxy_env()
        try:
            model_dim: int | None = None
            if use_flag_model:
                model = FlagModel(
                    model_name,
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True,
                )
                try:
                    model_dim = len(model.encode(["维度探测"])[0])
                except Exception:
                    model_dim = None
                loaded: dict[str, Any] = {"type": "flag", "model": model, "name": model_name, "dim": model_dim}
            else:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name)
                try:
                    model_dim = int(model.get_sentence_embedding_dimension())
                except Exception:
                    model_dim = None
                loaded = {"type": "sentence_transformer", "model": model, "name": model_name, "dim": model_dim}
            if expected_dim is not None and model_dim is not None and model_dim != expected_dim:
                fallback_name = self._default_semantic_model_for_dim(expected_dim)
                # If caller explicitly requested a model alias/name, keep strict behavior:
                # return None so retrieval does not silently switch to another embedding family.
                if requested:
                    self._semantic_model_warning = (
                        "template semantic model dimension mismatch: "
                        f"requested={model_name} ({model_dim}), expected={expected_dim}"
                    )
                    print(f"[warn] {self._semantic_model_warning}", flush=True)
                    self._semantic_model = None
                    return self._semantic_model
                if fallback_name != model_name:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(fallback_name)
                    loaded = {
                        "type": "sentence_transformer",
                        "model": model,
                        "name": fallback_name,
                        "dim": int(model.get_sentence_embedding_dimension()),
                    }
                    self.semantic_model_name = fallback_name
            else:
                self.semantic_model_name = model_name
            self._semantic_model = loaded
        except Exception:
            self._semantic_model = None
        finally:
            self._restore_proxy_env(proxy_backup)
        return self._semantic_model

    @staticmethod
    def _chapter_sort_key(chapter_value: Any) -> tuple[int, str]:
        text = str(chapter_value or "")
        match = re.search(r"(\d+)", text)
        if not match:
            return (10**9, text)
        return (int(match.group(1)), text)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        if not text:
            return set()
        lowered = text.lower()
        tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered))
        return {token for token in tokens if token.strip()}

    @staticmethod
    def _dedupe_templates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for item in items:
            key = f"{item.get('chapter','')}::{item.get('text','')}"
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _load_entity_name_index(self) -> dict[str, str]:
        if self._entity_name_index_cache is not None:
            return self._entity_name_index_cache

        index: dict[str, str] = {}
        path = Path(self.store.canon_db_path).expanduser()
        if not path.exists():
            self._entity_name_index_cache = index
            return self._entity_name_index_cache

        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                """
                SELECT entity_id, canonical_name
                FROM entity_registry
                WHERE canonical_name IS NOT NULL AND canonical_name <> ''
                """
            ).fetchall()
            for entity_id, canonical_name in rows:
                if isinstance(entity_id, str):
                    cleaned_id = entity_id.strip()
                else:
                    cleaned_id = ""
                name = _coerce_name(canonical_name)
                if cleaned_id and name:
                    index[cleaned_id] = name
        except Exception:
            index = {}
        finally:
            conn.close()

        self._entity_name_index_cache = index
        return self._entity_name_index_cache

    def _sanitize_template_item_for_prompt(self, item: dict[str, Any]) -> dict[str, Any]:
        entity_name_index = dict(self._load_entity_name_index())
        _collect_entity_name_index(item, entity_name_index)
        sanitized = _replace_entity_ids_with_names(item, entity_name_index)
        if not isinstance(sanitized, dict):
            return {
                "text": str(item.get("text") or "").strip(),
                "chapter": str(item.get("chapter") or "").strip(),
            }
        sanitized.pop("chapter_sort_key", None)
        if "text" in sanitized:
            sanitized["text"] = str(sanitized.get("text") or "").strip()
        if "chapter" in sanitized:
            sanitized["chapter"] = str(sanitized.get("chapter") or "").strip()
        return sanitized

    def sanitize_template_payload_for_prompt(self, payload: Any) -> Any:
        if isinstance(payload, list):
            sanitized_list = [self.sanitize_template_payload_for_prompt(item) for item in payload]
            return [item for item in sanitized_list if not _is_empty_value(item)]

        if isinstance(payload, dict):
            if "text" in payload and ("metadata" in payload or "chapter" in payload):
                return self._sanitize_template_item_for_prompt(payload)
            sanitized: dict[str, Any] = {}
            preserve_empty_keys = {
                "plot_templates",
                "style_templates",
                "narrative_templates",
                "fact_templates",
                "relation_templates",
            }
            for key, value in payload.items():
                if key == "chapter_sort_key":
                    continue
                child = self.sanitize_template_payload_for_prompt(value)
                if _is_empty_value(child) and key not in preserve_empty_keys:
                    continue
                if key in preserve_empty_keys and not isinstance(child, list):
                    child = []
                sanitized[key] = child
            return sanitized

        return payload

    def _qdrant_scroll_all(self, book_id: str, asset_type: str, limit: int = 5000) -> list[dict[str, Any]]:
        if not self.store.qdrant_url:
            return []
        cache_key = (book_id, asset_type)
        cached = self._asset_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            rows: list[dict[str, Any]] = []
            offset: Any | None = None
            page_limit = 256
            while len(rows) < max(limit, 1):
                request_payload: dict[str, Any] = {
                    "filter": {
                        "must": [
                            {"key": "book_id", "match": {"value": book_id}},
                            {"key": "asset_type", "match": {"value": asset_type}},
                        ]
                    },
                    "limit": min(page_limit, max(limit, 1) - len(rows)),
                    "with_payload": True,
                    "with_vector": False,
                }
                if offset is not None:
                    request_payload["offset"] = offset
                response = httpx.post(
                    f"{self.store.qdrant_url.rstrip('/')}/collections/{self.store.qdrant_collection}/points/scroll",
                    headers=self._qdrant_headers(),
                    json=request_payload,
                    timeout=20.0,
                    trust_env=False,
                )
                response.raise_for_status()
                result = response.json().get("result", {})
                points = result.get("points", [])
                if not points:
                    break
                for row in points:
                    payload = row.get("payload", {}) or {}
                    chapter = payload.get("chapter")
                    rows.append(
                        {
                            "text": payload.get("text", ""),
                            "chapter": chapter,
                            "metadata": payload.get("metadata", {}),
                            "chapter_sort_key": self._chapter_sort_key(chapter),
                        }
                    )
                offset = result.get("next_page_offset")
                if offset is None:
                    break
            rows = self._dedupe_templates(rows)
            self._asset_cache[cache_key] = rows
            return rows
        except Exception:
            self._asset_cache[cache_key] = []
            return []

    @staticmethod
    def _pick_evenly(items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        if len(items) <= top_k:
            return items
        if top_k == 1:
            return [items[len(items) // 2]]
        picks: list[dict[str, Any]] = []
        last_index = len(items) - 1
        for idx in range(top_k):
            position = round((idx / (top_k - 1)) * last_index)
            picks.append(items[position])
        return TemplateMemoryReader._dedupe_templates(picks)[:top_k]

    def _select_book_templates(
        self,
        book_id: str,
        asset_type: str,
        top_k: int,
        *,
        query_text: str = "",
    ) -> list[dict[str, Any]]:
        # Prefer semantic retrieval when we have a profile-level query.
        if query_text.strip():
            semantic_rows = self._qdrant_semantic_search(
                book_id=book_id,
                asset_type=asset_type,
                query_text=query_text,
                limit=max(top_k, 1),
            )
            if semantic_rows:
                return semantic_rows
        pool = self._qdrant_scroll_all(book_id, asset_type)
        if not pool:
            return []
        if query_text.strip():
            token_selected = self._select_by_token_overlap(pool, query_text, top_k)
            if token_selected:
                return token_selected
        sorted_pool = sorted(pool, key=lambda item: item.get("chapter_sort_key", (10**9, "")))
        return self._pick_evenly(sorted_pool, top_k)

    def _select_chapter_templates(
        self,
        book_id: str,
        asset_type: str,
        top_k: int,
        *,
        query_text: str,
    ) -> list[dict[str, Any]]:
        semantic_rows = self._qdrant_semantic_search(
            book_id=book_id,
            asset_type=asset_type,
            query_text=query_text,
            limit=top_k,
        )
        if semantic_rows:
            return semantic_rows

        pool = self._qdrant_scroll_all(book_id, asset_type)
        if not pool:
            return []
        token_selected = self._select_by_token_overlap(pool, query_text, top_k)
        if token_selected:
            return token_selected
        if not self._tokenize(query_text):
            return self._select_book_templates(book_id, asset_type, top_k)
        return self._select_book_templates(book_id, asset_type, top_k)

    def _select_by_token_overlap(
        self,
        pool: list[dict[str, Any]],
        query_text: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []

        scored: list[tuple[int, tuple[int, str], dict[str, Any]]] = []
        for item in pool:
            text = str(item.get("text") or "")
            metadata = item.get("metadata")
            metadata_text = json.dumps(metadata, ensure_ascii=False) if metadata else ""
            item_tokens = self._tokenize(f"{text} {metadata_text}")
            overlap = len(query_tokens & item_tokens)
            if overlap <= 0:
                continue
            scored.append((overlap, item.get("chapter_sort_key", (10**9, "")), item))
        if not scored:
            return []

        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        selected = [item for _, _, item in scored[: max(top_k * 3, top_k)]]
        selected = self._dedupe_templates(selected)
        return selected[:top_k]

    def _qdrant_semantic_search(
        self,
        *,
        book_id: str,
        asset_type: str,
        query_text: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = query_text.strip()
        if not query or not self.store.qdrant_url:
            return []
        vector_name, vector_dim = self._resolve_vector_target()
        if vector_dim is None:
            return []
        model = self._get_semantic_model(vector_dim)
        if model is None:
            return []
        try:
            if isinstance(model, dict) and model.get("type") == "flag":
                query_vector = model["model"].encode([query])[0].tolist()
            elif isinstance(model, dict):
                query_vector = model["model"].encode(query).tolist()
            else:
                query_vector = model.encode(query).tolist()
            payload: dict[str, Any] = {
                "limit": max(limit, 1),
                "with_payload": True,
                "with_vector": False,
                "filter": {
                    "must": [
                        {"key": "book_id", "match": {"value": book_id}},
                        {"key": "asset_type", "match": {"value": asset_type}},
                    ]
                },
            }
            if vector_name:
                payload["vector"] = {"name": vector_name, "vector": query_vector}
            else:
                payload["vector"] = query_vector

            response = httpx.post(
                f"{self.store.qdrant_url.rstrip('/')}/collections/{self.store.qdrant_collection}/points/search",
                headers=self._qdrant_headers(),
                json=payload,
                timeout=20.0,
                trust_env=False,
            )
            response.raise_for_status()
            rows = response.json().get("result", [])
            numeric_scores = [float(row.get("score")) for row in rows if isinstance(row.get("score"), (int, float))]
            if len(numeric_scores) >= 2 and (max(numeric_scores) - min(numeric_scores)) < 1e-9:
                # Usually indicates low-signal vectors (e.g., placeholder zeros); let lexical fallback handle it.
                return []
            result: list[dict[str, Any]] = []
            for row in rows:
                row_payload = row.get("payload", {}) or {}
                chapter = row_payload.get("chapter")
                result.append(
                    {
                        "text": row_payload.get("text", ""),
                        "chapter": chapter,
                        "metadata": row_payload.get("metadata", {}),
                        "score": row.get("score"),
                        "retrieval": "semantic_search",
                        "chapter_sort_key": self._chapter_sort_key(chapter),
                    }
                )
            return self._dedupe_templates(result)[: max(limit, 1)]
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

    def build_book_template_profile(self, book_id: str, top_k: int, *, query_text: str = "") -> dict[str, Any]:
        return {
            "plot_templates": self._select_book_templates(book_id, "plot_beat", top_k, query_text=query_text),
            "style_templates": self._select_book_templates(book_id, "style", top_k, query_text=query_text),
            "conflict_templates": self._select_book_templates(book_id, "conflict", top_k, query_text=query_text),
            # Digest assets usually have much wider chapter coverage than style/plot tags.
            "chapter_digest_templates": self._select_book_templates(
                book_id,
                "chapter_digest",
                top_k,
                query_text=query_text,
            ),
            "fact_digest_templates": self._select_book_templates(
                book_id,
                "fact_digest",
                max(top_k // 2, 2),
                query_text=query_text,
            ),
            "relation_digest_templates": self._select_book_templates(
                book_id,
                "relation_digest",
                max(top_k // 2, 2),
                query_text=query_text,
            ),
            "canon": self._canon_recent(book_id, top_k),
            "neo4j_relation_kinds": self._neo4j_relation_kinds(),
        }

    def build_chapter_template_pack(self, book_id: str, chapter_plan: dict[str, Any], top_k: int) -> dict[str, Any]:
        query_text = " ".join(
            [
                str(chapter_plan.get("goal", "")),
                str(chapter_plan.get("conflict", "")),
                " ".join(str(item) for item in chapter_plan.get("beat_outline", []) if str(item).strip()),
            ]
        ).strip()
        digest_top_k = max(top_k // 2, 2)
        return {
            "chapter_goal": chapter_plan.get("goal", ""),
            "chapter_conflict": chapter_plan.get("conflict", ""),
            "plot_templates": self._select_chapter_templates(
                book_id,
                "plot_beat",
                top_k,
                query_text=query_text,
            ),
            "style_templates": self._select_chapter_templates(
                book_id,
                "style",
                top_k,
                query_text=query_text,
            ),
            "narrative_templates": self._select_chapter_templates(
                book_id,
                "chapter_digest",
                top_k,
                query_text=query_text,
            ),
            "fact_templates": self._select_chapter_templates(
                book_id,
                "fact_digest",
                digest_top_k,
                query_text=query_text,
            ),
            "relation_templates": self._select_chapter_templates(
                book_id,
                "relation_digest",
                digest_top_k,
                query_text=query_text,
            ),
        }


class TargetMemoryReader:
    """Read Book B hard context before generating each chapter."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self._entity_name_index_cache: dict[str, str] | None = None

    def _load_entity_name_index(self) -> dict[str, str]:
        if self._entity_name_index_cache is not None:
            return self._entity_name_index_cache

        index: dict[str, str] = {}
        path = Path(self.store.canon_db_path).expanduser()
        if not path.exists():
            self._entity_name_index_cache = index
            return self._entity_name_index_cache

        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                """
                SELECT entity_id, canonical_name
                FROM entity_registry
                WHERE canonical_name IS NOT NULL AND canonical_name <> ''
                """
            ).fetchall()
            for entity_id, canonical_name in rows:
                if isinstance(entity_id, str):
                    cleaned_id = entity_id.strip()
                else:
                    cleaned_id = ""
                name = _coerce_name(canonical_name)
                if cleaned_id and name:
                    index[cleaned_id] = name
        except Exception:
            index = {}
        finally:
            conn.close()

        self._entity_name_index_cache = index
        return self._entity_name_index_cache

    @staticmethod
    def _sanitize_prev_context_for_prompt(prev_context: dict[str, Any]) -> dict[str, Any]:
        """Keep human-readable context fields only (no internal entity ids)."""
        clean_context: dict[str, Any] = {
            "character_state": [],
            "recent_relations": [],
            "open_threads": [],
        }

        for row in prev_context.get("character_state", []) or []:
            if not isinstance(row, dict):
                continue
            canonical_name = str(row.get("name") or "").strip()
            if not canonical_name:
                continue
            clean_context["character_state"].append(
                {
                    "canonical_name": canonical_name,
                    "name": canonical_name,
                    "state": row.get("state", {}) if isinstance(row.get("state"), dict) else {},
                    "updated_chapter": row.get("updated_chapter"),
                }
            )

        for row in prev_context.get("recent_relations", []) or []:
            if not isinstance(row, dict):
                continue
            from_name = str(row.get("from_name") or "").strip()
            to_name = str(row.get("to_name") or "").strip()
            if not from_name or not to_name:
                continue
            clean_context["recent_relations"].append(
                {
                    "from_name": from_name,
                    "to_name": to_name,
                    "kind": row.get("kind"),
                    "status": row.get("status"),
                    "valid_from": row.get("valid_from"),
                    "valid_to": row.get("valid_to"),
                    "evidence_chunk_id": row.get("evidence_chunk_id"),
                }
            )

        for row in prev_context.get("open_threads", []) or []:
            if not isinstance(row, dict):
                continue
            clean_context["open_threads"].append(
                {
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "priority": row.get("priority"),
                    "planned_window": row.get("planned_window"),
                    "notes": row.get("notes"),
                    "updated_chapter": row.get("updated_chapter"),
                }
            )

        return clean_context

    def sanitize_hard_pack_for_prompt(self, hard_pack: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(hard_pack, dict):
            return {"hard_rules": [], "prev_context": {"character_state": [], "recent_relations": [], "open_threads": []}}

        pack = dict(hard_pack)
        prev_context = pack.get("prev_context")
        if isinstance(prev_context, dict):
            pack["prev_context"] = TargetMemoryReader._sanitize_prev_context_for_prompt(prev_context)

        entity_name_index = dict(self._load_entity_name_index())
        _collect_entity_name_index(pack, entity_name_index)
        cleaned = _replace_entity_ids_with_names(pack, entity_name_index)
        if not isinstance(cleaned, dict):
            return {"hard_rules": [], "prev_context": {"character_state": [], "recent_relations": [], "open_threads": []}}

        hard_rules = cleaned.get("hard_rules")
        if isinstance(hard_rules, list):
            for row in hard_rules:
                if isinstance(row, dict):
                    row.pop("evidence", None)
                    row.pop("evidence_chunk_id", None)

        prev_context_clean = cleaned.get("prev_context")
        if isinstance(prev_context_clean, dict):
            for row in prev_context_clean.get("recent_relations", []) or []:
                if isinstance(row, dict):
                    row.pop("evidence_chunk_id", None)
        return cleaned

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
            prev_context = self._sanitize_prev_context_for_prompt(prev_context)
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


def _load_all_done_chapters(canon_db_path: str, book_id: str) -> set[str]:
    path = Path(canon_db_path).expanduser()
    if not path.exists():
        return set()
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            """
            SELECT chapter_no
            FROM commit_log
            WHERE book_id = ? AND status = 'ALL_DONE'
            """,
            (book_id,),
        ).fetchall()
        return {str(row[0]) for row in rows if row and row[0]}
    finally:
        conn.close()


def _load_run_report_summaries(output_dir: Path, target_book_id: str) -> dict[str, str]:
    report_path = output_dir / f"{target_book_id}_run_report.json"
    if not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = data.get("items", [])
    summary_map: dict[str, str] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            chapter_no = str(item.get("chapter_no") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if chapter_no and summary:
                summary_map[chapter_no] = summary
    return summary_map


def _redact_store_for_log(store: MemoryStore) -> dict[str, Any]:
    return {
        "canon_db_path": store.canon_db_path,
        "neo4j_uri": store.neo4j_uri,
        "neo4j_user": store.neo4j_user,
        "neo4j_database": store.neo4j_database,
        "qdrant_url": store.qdrant_url,
        "qdrant_collection": store.qdrant_collection,
    }


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
    if args.commit_memory:
        if not target_store.neo4j_database.strip():
            raise ValueError("target_neo4j_database is required when --commit-memory is enabled")
        if template_store is not None:
            isolation_violations = []
            if template_store.canon_norm() == target_store.canon_norm():
                isolation_violations.append("Canon DB path is identical")
            if template_store.qdrant_norm() == target_store.qdrant_norm():
                isolation_violations.append("Qdrant target (url+collection) is identical")
            if template_store.neo4j_norm() == target_store.neo4j_norm():
                isolation_violations.append("Neo4j target (uri+database) is identical")
            if isolation_violations:
                raise ValueError(
                    "Commit-memory isolation requires physically separate A/B stores: "
                    + "; ".join(isolation_violations)
                )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(getattr(args, "log_dir", str(_default_log_dir()))).expanduser()
    injection_logger = (
        InjectionLogger(log_root, target_book_id) if getattr(args, "log_injections", True) else None
    )

    generator = OneClickBookGenerator(llm_config)
    template_reader = (
        TemplateMemoryReader(
            template_store,
            semantic_search_enabled=getattr(args, "template_semantic_search", True),
            semantic_model_name=getattr(args, "template_semantic_model", ""),
        )
        if template_store
        else None
    )
    target_reader = TargetMemoryReader(target_store)
    profile_query_text = json.dumps(world_spec, ensure_ascii=False)[:4000]
    raw_template_profile = (
        template_reader.build_book_template_profile(
            args.template_book_id,
            args.reference_top_k,
            query_text=profile_query_text,
        )
        if template_reader
        else {}
    )
    template_profile = (
        template_reader.sanitize_template_payload_for_prompt(raw_template_profile)
        if template_reader
        else raw_template_profile
    )
    if injection_logger is not None:
        clean_payload = {
            "target_book_id": target_book_id,
            "template_book_id": args.template_book_id,
            "world_spec": world_spec,
            "template_profile_from_bookA": template_profile,
            "template_store": _redact_store_for_log(template_store) if template_store else None,
            "target_store": _redact_store_for_log(target_store),
            "reference_top_k": args.reference_top_k,
            "template_semantic_search": getattr(args, "template_semantic_search", True),
            "template_semantic_model": getattr(args, "template_semantic_model", ""),
        }
        raw_payload = {
            **clean_payload,
            "template_profile_from_bookA": raw_template_profile,
        }
        path, raw_path = injection_logger.write_with_raw(
            "injection_book_level_template_profile.json",
            clean_payload,
            raw_payload=raw_payload,
        )
        print(f"[log] book-level injections -> {path}", flush=True)
        if raw_path is not None:
            print(f"[log] book-level injections raw -> {raw_path}", flush=True)
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
    if injection_logger is not None:
        path = injection_logger.write(
            "injection_blueprint_response.json",
            {
                "target_book_id": target_book_id,
                "blueprint": blueprint,
            },
        )
        print(f"[log] blueprint response -> {path}", flush=True)

    processor = None
    if args.commit_memory:
        from chapter_processor import ChapterProcessor

        processor = ChapterProcessor(
            neo4j_uri=target_store.neo4j_uri,
            neo4j_user=target_store.neo4j_user,
            neo4j_pass=target_store.neo4j_pass,
            canon_db_path=str(Path(target_store.canon_db_path).expanduser()),
            neo4j_database=target_store.neo4j_database,
            qdrant_url=target_store.qdrant_url or None,
            qdrant_collection=target_store.qdrant_collection,
            qdrant_api_key=target_store.qdrant_api_key,
            llm_config=llm_config,
        )

    run_items: list[dict[str, Any]] = []
    rolling_summaries: list[str] = []
    existing_summaries = _load_run_report_summaries(output_dir, target_book_id) if args.resume else {}
    all_done_chapters: set[str] = set()
    if args.resume and processor is not None:
        all_done_chapters = _load_all_done_chapters(target_store.canon_db_path, target_book_id)
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
                if processor is None or chapter_no in all_done_chapters:
                    run_items.append({"chapter_no": chapter_no, "status": "skipped", "path": str(chapter_path)})
                    elapsed = time.monotonic() - started
                    eta = (elapsed / idx) * (len(blueprint["chapters"]) - idx) if idx else 0
                    print(
                        f"[{idx}/{len(blueprint['chapters'])}] skip {chapter_no} "
                        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                        flush=True,
                    )
                    continue

                # Resume mode with memory commit enabled: recover missing commit from existing chapter file.
                chapter_markdown = chapter_path.read_text(encoding="utf-8")
                chapter_text = chapter_markdown
                heading = f"# {chapter_title}"
                if chapter_markdown.startswith(heading):
                    chapter_text = chapter_markdown[len(heading) :].lstrip("\n").strip()
                fallback_summary = chapter_text.replace("\n", " ").strip()[:220]
                summary = existing_summaries.get(chapter_no) or fallback_summary or chapter_title
                memory_commit = processor.process_chapter(
                    book_id=target_book_id,
                    chapter_no=chapter_no,
                    chapter_title=chapter_title,
                    chapter_summary=summary,
                    chapter_text=chapter_text,
                    assets=None,
                    mode="llm",
                )
                if memory_commit.get("status") == "blocked":
                    raise RuntimeError(f"blocking conflicts: {memory_commit.get('conflicts')}")
                all_done_chapters.add(chapter_no)
                run_items.append(
                    {
                        "chapter_no": chapter_no,
                        "title": chapter_title,
                        "status": "resumed_commit",
                        "summary": summary,
                        "path": str(chapter_path),
                        "memory_commit": memory_commit,
                    }
                )
                ok_count += 1
                elapsed = time.monotonic() - started
                eta = (elapsed / idx) * (len(blueprint["chapters"]) - idx) if idx else 0
                print(
                    f"[{idx}/{len(blueprint['chapters'])}] resume-commit {chapter_no} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} ok={ok_count} fail={fail_count}",
                    flush=True,
                )
                continue

            try:
                raw_template_pack = (
                    template_reader.build_chapter_template_pack(args.template_book_id, chapter, args.reference_top_k)
                    if template_reader
                    else {}
                )
                template_pack = (
                    template_reader.sanitize_template_payload_for_prompt(raw_template_pack)
                    if template_reader
                    else raw_template_pack
                )
                raw_hard_pack_b = target_reader.build_hard_pack(target_book_id, chapter_no, args.reference_top_k)
                hard_pack_b = target_reader.sanitize_hard_pack_for_prompt(raw_hard_pack_b)
                hard_pack_b = _ensure_hard_context_shape(hard_pack_b)
                recent_summaries_for_prompt = _sanitize_recent_summaries_for_prompt(rolling_summaries[-8:])
                if injection_logger is not None:
                    clean_payload = {
                        "target_book_id": target_book_id,
                        "template_book_id": args.template_book_id,
                        "chapter_no": chapter_no,
                        "world_spec": world_spec,
                        "book_blueprint_meta": blueprint_meta,
                        "chapter_plan": chapter,
                        "template_pack_from_bookA": template_pack,
                        "hard_context_from_bookB": hard_pack_b,
                        "recent_summaries_bookB": recent_summaries_for_prompt,
                    }
                    raw_payload = {
                        **clean_payload,
                        "template_pack_from_bookA": raw_template_pack,
                        "hard_context_from_bookB": raw_hard_pack_b,
                        "recent_summaries_bookB": rolling_summaries[-8:],
                    }
                    path, raw_path = injection_logger.write_with_raw(
                        f"chapters/{chapter_no}_pre_generation_injection.json",
                        clean_payload,
                        raw_payload=raw_payload,
                    )
                    print(f"[log] chapter {chapter_no} injections -> {path}", flush=True)
                    if raw_path is not None:
                        print(f"[log] chapter {chapter_no} injections raw -> {raw_path}", flush=True)
                chapter_text = generator.generate_chapter_text(
                    world_spec=world_spec,
                    blueprint_meta=blueprint_meta,
                    chapter_plan=chapter,
                    template_chapter_pack=template_pack,
                    hard_pack_b=hard_pack_b,
                    recent_summaries=recent_summaries_for_prompt,
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
                        assets=None,
                        mode="llm",
                    )
                    item["memory_commit"] = memory_commit
                    if memory_commit.get("status") == "blocked":
                        raise RuntimeError(f"blocking conflicts: {memory_commit.get('conflicts')}")
                    all_done_chapters.add(chapter_no)

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
        "injection_log_dir": str(injection_logger.run_dir) if injection_logger is not None else None,
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
        "injection_log_dir": str(injection_logger.run_dir) if injection_logger is not None else None,
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
        "template_semantic_search": True,
        "template_semantic_model": "",
        "log_dir": str(_default_log_dir()),
        "log_injections": True,
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
    parser.add_argument("--template-semantic-search", action="store_true", default=True)
    parser.add_argument("--no-template-semantic-search", action="store_false", dest="template_semantic_search")
    parser.add_argument("--template-semantic-model", default="")
    parser.add_argument("--log-dir", default=str(_default_log_dir()))
    parser.add_argument("--log-injections", action="store_true", default=True)
    parser.add_argument("--no-log-injections", action="store_false", dest="log_injections")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_generation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
