"""Chapter processor coordinating Canon DB, Neo4j, and Qdrant memory tiers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import httpx
from canon_db_v2 import CanonDBV2
from delta_converter import convert_assets_to_delta
from delta_extractor_llm import DeltaExtractorLLM
from name_normalizer import VALID_RELATION_TYPES, classify_relation_type, normalize_name
from neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)

GENERIC_CHARACTER_TOKENS = {
    "彼此",
    "双方",
    "对方",
    "众人",
    "大家",
    "前辈们",
    "后辈们",
    "其他人",
    "其余人",
    "全体成员",
    "所有人",
}

GENERIC_CHARACTER_PATTERNS = [
    re.compile(r".*全员$"),
    re.compile(r"^(其他|其余).+"),
    re.compile(r".*（群体）$"),
]

SYMMETRIC_RELATION_KINDS = {
    "ASSOCIATE",
    "CO_PARTICIPANT",
    "ALLY",
    "ENEMY",
    "FAMILY",
    "COLLEAGUE",
    "RIVAL",
    "ROMANTIC",
}
DEFAULT_ENFORCE_CHINESE_FIELDS = ("rule", "status", "trait", "goal", "secret", "state")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


def _strip_markdown_code_blocks(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class ChineseValueNormalizer:
    def __init__(
        self,
        llm_config: dict[str, Any] | None,
        *,
        max_tokens: int = 512,
        max_retries: int = 2,
        retry_backoff: float = 2.0,
        backoff_factor: float = 2.0,
        backoff_max: float = 30.0,
        retry_jitter: float = 0.5,
    ):
        self.llm_config = llm_config or {}
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.backoff_factor = backoff_factor
        self.backoff_max = backoff_max
        self.retry_jitter = retry_jitter
        self._cache: dict[str, str] = {}

    @staticmethod
    def _needs_translation(text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        if not LATIN_CHAR_RE.search(cleaned):
            return False
        if re.fullmatch(r"[A-Z0-9_:-]{2,}", cleaned):
            return False
        if re.fullmatch(r"[a-z]+_[0-9a-f]{6,}", cleaned):
            return False
        return True

    @staticmethod
    def _pop_proxy_env() -> dict[str, str]:
        backup: dict[str, str] = {}
        for key in ("ALL_PROXY", "all_proxy"):
            value = os.environ.pop(key, None)
            if value is not None:
                backup[key] = value
        return backup

    @staticmethod
    def _restore_proxy_env(backup: dict[str, str]) -> None:
        for key, value in backup.items():
            os.environ[key] = value

    def _call_llm(self, prompt: str) -> str:
        if self.llm_config.get("type") == "custom":
            proxy_backup = self._pop_proxy_env()
            try:
                response = httpx.post(
                    self.llm_config["url"],
                    json={
                        "model": self.llm_config["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": self.max_tokens,
                    },
                    headers={"Authorization": f"Bearer {self.llm_config['api_key']}"},
                    timeout=120.0,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"] or ""
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
                        temperature=0.0,
                        max_tokens=self.max_tokens,
                    ),
                    timeout=120.0,
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
            return response.content or ""

        raise ValueError("Unsupported llm_config: expected {type: custom} or {providers, model}")

    def translate(self, text: str) -> str:
        original = str(text or "")
        if not self._needs_translation(original):
            return original
        cached = self._cache.get(original)
        if cached is not None:
            return cached

        prompt = (
            "将下面文本翻译成简体中文，只输出翻译后的文本。\n"
            "保留专有名词、实体ID、关系枚举和技术缩写原样（例如 character_ab12cd、CO_PARTICIPANT、Neo4j）。\n"
            "不要解释，不要 markdown。\n\n"
            f"{original}"
        )
        attempts = max(self.max_retries, 0) + 1
        last_error: Exception | None = None
        translated = original
        for attempt in range(1, attempts + 1):
            try:
                translated = _strip_markdown_code_blocks(self._call_llm(prompt)).strip() or original
                break
            except Exception as exc:  # pragma: no cover - operational retry
                last_error = exc
                if attempt >= attempts:
                    logger.warning("Chinese translation fallback used: %s", exc)
                    translated = original
                    break
                base = self.retry_backoff * (self.backoff_factor ** max(0, attempt - 1))
                delay = min(base, self.backoff_max)
                if self.retry_jitter > 0:
                    delay += random.uniform(0, self.retry_jitter)
                time.sleep(delay)
        if last_error and translated == original:
            logger.debug("Translation kept original text after retries: %s", last_error)
        self._cache[original] = translated
        return translated

    def normalize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.translate(value)
        if isinstance(value, list):
            return [self.normalize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.normalize_value(item) for key, item in value.items()}
        return value


class ChapterProcessor:
    """Orchestrates chapter commits across all three memory tiers."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_pass: str,
        canon_db_path: str,
        neo4j_database: str = "neo4j",
        qdrant_url: Optional[str] = None,
        qdrant_collection: str = "novel_assets_v2",
        qdrant_api_key: str = "",
        llm_config: Optional[dict[str, Any]] = None,
        llm_max_tokens: int = 4096,
        context_state_limit: int = 30,
        context_relation_limit: int = 30,
        context_thread_limit: int = 20,
        embedding_model=None,
        use_flag_model: bool = False,
        vector_size: int = 1024,
        enforce_chinese_on_commit: bool = False,
        enforce_chinese_fields: tuple[str, ...] | None = None,
        chinese_llm_max_retries: int = 2,
        chinese_retry_backoff: float = 2.0,
        chinese_backoff_factor: float = 2.0,
        chinese_backoff_max: float = 30.0,
        chinese_retry_jitter: float = 0.5,
        delta_json_repair_attempts: int = 2,
        delta_parse_debug_log: bool = True,
        delta_parse_debug_dir: str | None = None,
    ):
        self.neo4j = Neo4jManager(neo4j_uri, neo4j_user, neo4j_pass, neo4j_database)
        self.canon_db = CanonDBV2(canon_db_path)
        self.qdrant_url = (qdrant_url or "").rstrip("/")
        self.qdrant_collection = qdrant_collection
        self.qdrant_api_key = qdrant_api_key
        self._qdrant_vector_schema: dict[str, Any] | None = None
        self.llm_config = llm_config
        self.llm_max_tokens = llm_max_tokens
        self.context_state_limit = context_state_limit
        self.context_relation_limit = context_relation_limit
        self.context_thread_limit = context_thread_limit
        self.embedding_model = embedding_model
        self.use_flag_model = use_flag_model
        self.vector_size = vector_size
        self.enforce_chinese_on_commit = enforce_chinese_on_commit
        self.enforce_chinese_fields = {
            str(field).strip().lower()
            for field in (enforce_chinese_fields or DEFAULT_ENFORCE_CHINESE_FIELDS)
            if str(field).strip()
        }
        self.chinese_normalizer = (
            ChineseValueNormalizer(
                llm_config=llm_config or {},
                max_retries=chinese_llm_max_retries,
                retry_backoff=chinese_retry_backoff,
                backoff_factor=chinese_backoff_factor,
                backoff_max=chinese_backoff_max,
                retry_jitter=chinese_retry_jitter,
            )
            if enforce_chinese_on_commit and llm_config
            else None
        )
        self.delta_extractor = (
            DeltaExtractorLLM(
                llm_config,
                max_tokens=llm_max_tokens,
                json_repair_attempts=delta_json_repair_attempts,
                parse_debug_log=delta_parse_debug_log,
                parse_debug_dir=delta_parse_debug_dir,
            )
            if llm_config
            else None
        )

        # Log embedding configuration
        if self.qdrant_url:
            if self.embedding_model is not None:
                logger.info("ChapterProcessor initialized with embedding model")
                logger.info("  - Embedding model type: %s", "FlagModel" if use_flag_model else "SentenceTransformer")
                logger.info("  - Vector size: %d", vector_size)
                logger.info("  - Qdrant collection: %s", qdrant_collection)
            else:
                logger.info("ChapterProcessor initialized without embedding model (zero vectors)")
                logger.info("  - Vector size: %d", vector_size)
                logger.info("  - Qdrant collection: %s", qdrant_collection)
        else:
            logger.info("ChapterProcessor initialized without Qdrant integration")

    def close(self):
        """Close all connections."""
        self.neo4j.close()
        self.canon_db.close()

    def create_chunks(
        self,
        chapter_text: str,
        chapter_no: str,
        max_chars: int = 550,
        book_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Split chapter into punctuation-aware chunks with stable IDs."""
        chapter_text = (chapter_text or "").strip()
        if not chapter_text:
            return []

        units = re.split(r"(?<=[。！？!?；;])", chapter_text)
        chunks: list[dict[str, Any]] = []
        buffer: list[str] = []
        current_len = 0
        cursor = 0
        chunk_index = 0

        chunk_prefix = f"{book_id}:{chapter_no}" if book_id else chapter_no
        for unit in units:
            if not unit:
                continue
            if current_len + len(unit) > max_chars and buffer:
                text = "".join(buffer).strip()
                chunks.append(
                    {
                        "chunk_id": f"{chunk_prefix}#c{chunk_index:02d}",
                        "text": text,
                        "start_pos": cursor - len(text),
                        "end_pos": cursor,
                    }
                )
                chunk_index += 1
                buffer = []
                current_len = 0

            buffer.append(unit)
            current_len += len(unit)
            cursor += len(unit)

        if buffer:
            text = "".join(buffer).strip()
            chunks.append(
                {
                    "chunk_id": f"{chunk_prefix}#c{chunk_index:02d}",
                    "text": text,
                    "start_pos": max(cursor - len(text), 0),
                    "end_pos": cursor,
                }
            )
        return chunks

    def _resolve_name_to_id(
        self,
        name: str,
        entity_type: str,
        chapter_no: str,
        aliases: list[str] | None = None,
    ) -> str | None:
        if not name:
            return None
        name = str(name).strip()
        if not name:
            return None
        # If LLM already returns a stable entity_id, reuse it instead of re-normalizing as a name.
        existing = self.canon_db.get_entity_by_id(name)
        if existing:
            self.canon_db.touch_entity(name, chapter_no)
            return name
        safe_type = (entity_type or "character").lower()
        safe_aliases = [str(alias).strip() for alias in (aliases or []) if str(alias).strip()]
        if safe_type == "character":
            normalized = normalize_name(name)
            if not normalized.is_valid:
                return None
            if self._is_generic_character_name(normalized.canonical_name):
                return None
            alias_set = set(safe_aliases)
            alias_set.update(normalized.aliases)
            if normalized.canonical_name != name:
                alias_set.add(name)
            safe_aliases = [
                alias
                for alias in alias_set
                if alias
                and alias != normalized.canonical_name
                and not self._is_generic_character_name(alias)
            ]
            name = normalized.canonical_name
        return self.canon_db.normalize_entity(name, safe_type, chapter_no, safe_aliases)

    def _is_generic_character_name(self, name: str) -> bool:
        """Filter pronouns/group placeholders that should not become character entities."""
        if not name:
            return True
        candidate = str(name).strip()
        if not candidate:
            return True
        if candidate in GENERIC_CHARACTER_TOKENS:
            return True
        for pattern in GENERIC_CHARACTER_PATTERNS:
            if pattern.match(candidate):
                return True
        return False

    def _normalize_delta(
        self,
        delta: dict[str, Any],
        chapter_no: str,
        default_chunk_id: Optional[str],
        assets: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Normalize multi-source delta payload into one canonical shape."""
        normalized: dict[str, Any] = {
            "entities_new": [],
            "fact_changes": [],
            "relations_delta": [],
            "events": [],
            "address_changes": [],
            "hooks": [],
            "payoffs": [],
            "knows_updates": [],
        }

        for ent in delta.get("entities_new", []):
            normalized["entities_new"].append(
                {
                    "name": ent.get("name") or ent.get("canonical_name") or "",
                    "type": ent.get("type", "character"),
                    "aliases": ent.get("aliases", []),
                    "traits": ent.get("traits", {}),
                    "status": ent.get("status", "active"),
                    "evidence_chunk_id": ent.get("evidence_chunk_id", default_chunk_id),
                }
            )

        for fact in delta.get("fact_changes", []):
            normalized["fact_changes"].append(
                {
                    "subject_name": fact.get("subject_name") or fact.get("subject"),
                    "subject_id": fact.get("subject_id"),
                    "predicate": fact.get("predicate"),
                    "value": fact.get("value"),
                    "op": fact.get("op", "INSERT"),
                    "valid_from": fact.get("valid_from", chapter_no),
                    "valid_to": fact.get("valid_to"),
                    "tier": fact.get("tier", "SOFT_NOTE"),
                    "status": fact.get("status", "confirmed"),
                    "confidence": fact.get("confidence", 1.0),
                    "evidence_chunk_id": fact.get("evidence_chunk_id", default_chunk_id),
                }
            )

        for rel in delta.get("relations_delta", []):
            kind = rel.get("kind", "ASSOCIATE")
            if kind not in VALID_RELATION_TYPES:
                kind = classify_relation_type(str(rel.get("description", "")))
            normalized["relations_delta"].append(
                {
                    "from_name": rel.get("from_name") or rel.get("from"),
                    "to_name": rel.get("to_name") or rel.get("to"),
                    "from_id": rel.get("from_id"),
                    "to_id": rel.get("to_id"),
                    "kind": kind,
                    "op": rel.get("op", "INSERT"),
                    "valid_from": rel.get("valid_from", chapter_no),
                    "valid_to": rel.get("valid_to"),
                    "status": rel.get("status", "confirmed"),
                    "confidence": rel.get("confidence", 1.0),
                    "evidence_chunk_id": rel.get("evidence_chunk_id", default_chunk_id),
                }
            )

        events = delta.get("events", [])
        for i, event in enumerate(events):
            normalized["events"].append(
                {
                    "event_id": event.get("event_id") or f"{chapter_no}_evt_{i:02d}",
                    "type": event.get("type", "plot_beat"),
                    "summary": event.get("summary", ""),
                    "participants": event.get("participants", []),
                    "location": event.get("location"),
                    "effects": event.get("effects", []),
                    "evidence_chunk_id": event.get("evidence_chunk_id", default_chunk_id),
                }
            )

        normalized["address_changes"] = list(delta.get("address_changes", []))
        normalized["hooks"] = list(delta.get("hooks", []))
        normalized["payoffs"] = list(delta.get("payoffs", []))
        normalized["knows_updates"] = list(delta.get("knows_updates", []))
        self._augment_relations_from_assets(normalized, assets or {}, chapter_no, default_chunk_id)
        self._augment_relations_from_events(normalized, chapter_no, default_chunk_id)
        return normalized

    def _augment_relations_from_assets(
        self,
        normalized_delta: dict[str, Any],
        assets: dict[str, Any],
        chapter_no: str,
        default_chunk_id: Optional[str],
    ) -> None:
        """Add low-confidence structural links from 8-element assets to reduce hub-only graphs."""
        if not assets:
            return

        existing = {
            (
                rel.get("from_name"),
                rel.get("to_name"),
                rel.get("kind"),
                rel.get("valid_from", chapter_no),
            )
            for rel in normalized_delta.get("relations_delta", [])
        }

        def add_relation(from_name: str, to_name: str, kind: str, confidence: float = 0.65) -> None:
            if not from_name or not to_name or from_name == to_name:
                return
            if self._is_generic_character_name(from_name) or self._is_generic_character_name(to_name):
                return
            if kind == "CO_PARTICIPANT":
                from_name, to_name = sorted((from_name, to_name))
            key = (from_name, to_name, kind, chapter_no)
            rev_key = (to_name, from_name, kind, chapter_no)
            if key in existing or rev_key in existing:
                return
            normalized_delta["relations_delta"].append(
                {
                    "from_name": from_name,
                    "to_name": to_name,
                    "from_id": None,
                    "to_id": None,
                    "kind": kind,
                    "op": "INSERT",
                    "valid_from": chapter_no,
                    "valid_to": None,
                    "status": "implied",
                    "confidence": confidence,
                    "evidence_chunk_id": default_chunk_id,
                }
            )
            existing.add(key)

        for beat in assets.get("plot_beats", []) or []:
            chars = [c.strip() for c in beat.get("characters", []) if isinstance(c, str) and c.strip()]
            deduped: list[str] = []
            for name in chars:
                if name not in deduped:
                    deduped.append(name)
            for i in range(len(deduped)):
                for j in range(i + 1, len(deduped)):
                    add_relation(deduped[i], deduped[j], "CO_PARTICIPANT", confidence=0.7)

        for card in assets.get("character_cards", []) or []:
            source = (card.get("name") or "").strip()
            if not source:
                continue
            rel_map = card.get("relationships") or {}
            if not isinstance(rel_map, dict):
                continue
            for target, desc in rel_map.items():
                if not isinstance(target, str):
                    continue
                target_name = target.strip()
                if not target_name:
                    continue
                description = desc if isinstance(desc, str) else str(desc)
                kind = classify_relation_type(description)
                add_relation(source, target_name, kind, confidence=0.6)

    def _augment_relations_from_events(
        self,
        normalized_delta: dict[str, Any],
        chapter_no: str,
        default_chunk_id: Optional[str],
    ) -> None:
        """Generate co-participant relations from event participants to reduce hub bias."""
        existing = {
            (
                rel.get("from_name"),
                rel.get("to_name"),
                rel.get("kind"),
                rel.get("valid_from", chapter_no),
            )
            for rel in normalized_delta.get("relations_delta", [])
        }

        for event in normalized_delta.get("events", []):
            participants = [p for p in event.get("participants", []) if p]
            if len(participants) < 2:
                continue
            # Pairwise edges create local clusters instead of only protagonist spokes.
            for i in range(len(participants)):
                for j in range(i + 1, len(participants)):
                    a = participants[i]
                    b = participants[j]
                    key = (a, b, "CO_PARTICIPANT", chapter_no)
                    rev_key = (b, a, "CO_PARTICIPANT", chapter_no)
                    if key in existing or rev_key in existing:
                        continue
                    normalized_delta["relations_delta"].append(
                        {
                            "from_name": a,
                            "to_name": b,
                            "from_id": None,
                            "to_id": None,
                            "kind": "CO_PARTICIPANT",
                            "op": "INSERT",
                            "valid_from": chapter_no,
                            "valid_to": None,
                            "status": "implied",
                            "confidence": 0.7,
                            "evidence_chunk_id": event.get("evidence_chunk_id", default_chunk_id),
                        }
                    )
                    existing.add(key)

    def _augment_resolved_relations_from_events(
        self,
        resolved_delta: dict[str, Any],
        entity_meta: dict[str, dict[str, Any]],
        chapter_no: str,
    ) -> None:
        """Add CO_PARTICIPANT links from resolved event participants (character-character only)."""
        existing = {
            (
                rel.get("from_id"),
                rel.get("to_id"),
                rel.get("kind"),
                rel.get("valid_from", chapter_no),
            )
            for rel in resolved_delta.get("relations_delta", [])
        }

        for event in resolved_delta.get("events", []):
            raw_ids = [pid for pid in event.get("participants", []) if pid]
            participants: list[str] = []
            for pid in raw_ids:
                if pid in participants:
                    continue
                meta = entity_meta.get(pid)
                if not meta:
                    existing_entity = self.canon_db.get_entity_by_id(pid)
                    if existing_entity:
                        meta = {
                            "entity_id": pid,
                            "name": existing_entity["canonical_name"],
                            "type": existing_entity["type"],
                            "aliases": existing_entity["aliases"],
                            "traits": {},
                            "status": "active",
                        }
                        entity_meta[pid] = meta
                if (meta or {}).get("type", "character") == "character":
                    participants.append(pid)

            if len(participants) < 2:
                continue

            for i in range(len(participants)):
                for j in range(i + 1, len(participants)):
                    a, b = sorted((participants[i], participants[j]))
                    key = (a, b, "CO_PARTICIPANT", chapter_no)
                    if key in existing:
                        continue
                    resolved_delta["relations_delta"].append(
                        {
                            "from_id": a,
                            "to_id": b,
                            "from_name": entity_meta.get(a, {}).get("name", a),
                            "to_name": entity_meta.get(b, {}).get("name", b),
                            "kind": "CO_PARTICIPANT",
                            "op": "INSERT",
                            "valid_from": chapter_no,
                            "valid_to": None,
                            "status": "implied",
                            "confidence": 0.75,
                            "evidence_chunk_id": event.get("evidence_chunk_id", f"{chapter_no}#c00"),
                        }
                    )
                    existing.add(key)

    def _resolve_delta_entity_ids(
        self, delta: dict[str, Any], chapter_no: str
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Resolve names to stable entity ids and return enriched delta."""
        name_to_id: dict[str, str] = {}
        entity_meta: dict[str, dict[str, Any]] = {}
        resolved = {
            "entities_new": [],
            "fact_changes": [],
            "relations_delta": [],
            "events": [],
            "address_changes": delta.get("address_changes", []),
            "hooks": delta.get("hooks", []),
            "payoffs": delta.get("payoffs", []),
            "knows_updates": delta.get("knows_updates", []),
        }

        def resolve_ref(
            ref_name: Optional[str],
            ref_id: Optional[str],
            default_type: str = "character",
        ) -> Optional[str]:
            if ref_id:
                return ref_id
            if not ref_name:
                return None
            if ref_name in name_to_id:
                return name_to_id[ref_name]
            inferred_type = typed_name_hints.get(ref_name, default_type)
            return self._resolve_name_to_id(ref_name, inferred_type, chapter_no)

        typed_name_hints: dict[str, str] = {}

        for ent in delta.get("entities_new", []):
            ent_name = ent.get("name", "")
            ent_type = ent.get("type", "character")
            aliases = ent.get("aliases", [])
            entity_id = self._resolve_name_to_id(ent_name, ent_type, chapter_no, aliases)
            if not entity_id:
                continue
            name_to_id[ent_name] = entity_id
            typed_name_hints[ent_name] = ent_type
            for alias in aliases:
                name_to_id[alias] = entity_id
                typed_name_hints[alias] = ent_type
            entity_meta[entity_id] = {
                "entity_id": entity_id,
                "name": ent_name,
                "type": ent_type,
                "aliases": aliases,
                "traits": ent.get("traits", {}),
                "status": ent.get("status", "active"),
            }
            resolved["entities_new"].append({**ent, "entity_id": entity_id})

        for fact in delta.get("fact_changes", []):
            subject_id = resolve_ref(fact.get("subject_name"), fact.get("subject_id"), "character")
            if not subject_id or not fact.get("predicate"):
                continue
            if subject_id not in entity_meta:
                existing = self.canon_db.get_entity_by_id(subject_id)
                entity_meta[subject_id] = {
                    "entity_id": subject_id,
                    "name": (
                        existing["canonical_name"]
                        if existing
                        else (fact.get("subject_name") or subject_id)
                    ),
                    "type": existing["type"] if existing else "character",
                    "aliases": existing["aliases"] if existing else [],
                    "traits": {},
                    "status": "active",
                }
            resolved["fact_changes"].append({
                **fact,
                "subject_id": subject_id,
                "subject_name": entity_meta[subject_id]["name"],
            })

        for rel in delta.get("relations_delta", []):
            from_id = resolve_ref(rel.get("from_name"), rel.get("from_id"), "character")
            to_id = resolve_ref(rel.get("to_name"), rel.get("to_id"), "character")
            if not from_id or not to_id or from_id == to_id:
                continue
            for eid, fallback_name in (
                (from_id, rel.get("from_name") or from_id),
                (to_id, rel.get("to_name") or to_id),
            ):
                if eid not in entity_meta:
                    existing = self.canon_db.get_entity_by_id(eid)
                    entity_meta[eid] = {
                        "entity_id": eid,
                        "name": existing["canonical_name"] if existing else fallback_name,
                        "type": existing["type"] if existing else "character",
                        "aliases": existing["aliases"] if existing else [],
                        "traits": {},
                        "status": "active",
                    }
            resolved["relations_delta"].append({
                **rel,
                "from_id": from_id,
                "to_id": to_id,
                "from_name": entity_meta[from_id]["name"],
                "to_name": entity_meta[to_id]["name"],
            })

        for event in delta.get("events", []):
            participant_ids = []
            for participant in event.get("participants", []):
                participant_id = resolve_ref(participant, None, "character")
                if participant_id:
                    participant_ids.append(participant_id)
                    if participant_id not in entity_meta:
                        existing = self.canon_db.get_entity_by_id(participant_id)
                        entity_meta[participant_id] = {
                            "entity_id": participant_id,
                            "name": existing["canonical_name"] if existing else participant,
                            "type": existing["type"] if existing else "character",
                            "aliases": existing["aliases"] if existing else [],
                            "traits": {},
                            "status": "active",
                        }
            resolved["events"].append({**event, "participants": participant_ids})

        return resolved, entity_meta

    def _prune_relations(self, resolved_delta: dict[str, Any]) -> None:
        """Drop low-value duplicate/weak relations after IDs are resolved."""
        relations = resolved_delta.get("relations_delta", [])
        if not relations:
            return

        specific_symmetric_pairs: set[tuple[str, str]] = set()
        specific_directed_pairs: set[tuple[str, str, str]] = set()
        for rel in relations:
            from_id = rel.get("from_id")
            to_id = rel.get("to_id")
            kind = rel.get("kind")
            if not from_id or not to_id or not kind or kind == "ASSOCIATE":
                continue
            if kind in SYMMETRIC_RELATION_KINDS:
                specific_symmetric_pairs.add(tuple(sorted((from_id, to_id))))
            else:
                specific_directed_pairs.add((from_id, to_id, kind))

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for rel in relations:
            from_id = rel.get("from_id")
            to_id = rel.get("to_id")
            kind = rel.get("kind")
            if not from_id or not to_id or not kind:
                continue

            if kind == "ASSOCIATE":
                sym_key = tuple(sorted((from_id, to_id)))
                if sym_key in specific_symmetric_pairs:
                    continue

            dedupe_key = (
                from_id,
                to_id,
                kind,
                str(rel.get("valid_from") or ""),
                str(rel.get("status") or ""),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append(rel)

        resolved_delta["relations_delta"] = deduped

    def _enforce_chinese_fields_on_delta(self, resolved_delta: dict[str, Any]) -> None:
        if not self.enforce_chinese_on_commit or not self.chinese_normalizer:
            return
        facts = resolved_delta.get("fact_changes", [])
        if not isinstance(facts, list):
            return
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            predicate = str(fact.get("predicate") or "").strip().lower()
            if predicate not in self.enforce_chinese_fields:
                continue
            fact["value"] = self.chinese_normalizer.normalize_value(fact.get("value"))

    def _extract_assets(self, chapter_text: str) -> dict[str, Any]:
        from asset_extractor_parallel import extract_all_assets

        if not self.llm_config:
            raise ValueError("llm_config is required for asset extraction")
        try:
            return extract_all_assets(chapter_text, self.llm_config)
        except Exception:
            logger.exception("asset extraction failed; fallback to empty assets")
            return {}

    def _qdrant_headers(self) -> dict[str, str]:
        return {"api-key": self.qdrant_api_key} if self.qdrant_api_key else {}

    def _load_qdrant_vector_schema(self) -> dict[str, Any]:
        if self._qdrant_vector_schema is not None:
            return self._qdrant_vector_schema
        response = httpx.get(
            f"{self.qdrant_url}/collections/{self.qdrant_collection}",
            headers=self._qdrant_headers(),
            timeout=20.0,
            trust_env=False,
        )
        response.raise_for_status()
        vectors = response.json().get("result", {}).get("config", {}).get("params", {}).get("vectors")
        if not vectors:
            vectors = {"size": 384}
        self._qdrant_vector_schema = vectors
        return vectors

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using the configured model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if self.embedding_model is None:
            # Return zero vector if no model configured
            logger.debug("No embedding model configured, returning zero vector (size=%d)", self.vector_size)
            return [0.0] * self.vector_size

        try:
            if self.use_flag_model:
                # FlagModel returns numpy array, need to convert to list
                logger.debug("Generating embedding with FlagModel (text length: %d chars)", len(text))
                embedding = self.embedding_model.encode([text])[0].tolist()
            else:
                logger.debug("Generating embedding with SentenceTransformer (text length: %d chars)", len(text))
                embedding = self.embedding_model.encode(text).tolist()

            logger.debug("✓ Embedding generated (vector size: %d)", len(embedding))
            return embedding
        except Exception as e:
            logger.error("Failed to generate embedding: %s", e)
            logger.warning("Falling back to zero vector")
            return [0.0] * self.vector_size

    def _build_qdrant_vector_payload(self) -> list[float] | dict[str, list[float]]:
        vectors = self._load_qdrant_vector_schema()
        if isinstance(vectors, dict) and "size" in vectors:
            return [0.0] * int(vectors["size"])
        if isinstance(vectors, dict):
            payload: dict[str, list[float]] = {}
            for name, spec in vectors.items():
                size = int(spec.get("size", 384)) if isinstance(spec, dict) else 384
                payload[name] = [0.0] * size
            return payload
        return [0.0] * self.vector_size

    def _stable_point_id(self, *parts: str) -> int:
        digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _upsert_qdrant_points(self, points: list[dict[str, Any]]) -> None:
        response = httpx.put(
            f"{self.qdrant_url}/collections/{self.qdrant_collection}/points",
            headers=self._qdrant_headers(),
            json={"points": points},
            timeout=30.0,
            trust_env=False,
        )
        response.raise_for_status()

    def _write_qdrant(
        self,
        book_id: str,
        chapter_no: str,
        commit_id: str,
        chapter_summary: str,
        assets: dict[str, Any],
        delta: dict[str, Any],
    ):
        """Write chapter/fact/relation digest points into Qdrant for recall + hard-fact supplements."""
        if not self.qdrant_url:
            return

        logger.info("[%s] Starting Qdrant write with embedding generation", chapter_no)
        points: list[dict[str, Any]] = []

        # Generate chapter digest point
        digest_text = chapter_summary or (delta.get("events", [{}])[0].get("summary", "") if delta.get("events") else "")
        if digest_text:
            logger.debug("[%s] Generating embedding for chapter digest", chapter_no)
            points.append(
                {
                    "id": self._stable_point_id(book_id, chapter_no, "chapter_digest"),
                    "vector": self._generate_embedding(digest_text),
                    "payload": {
                        "book_id": book_id,
                        "chapter": chapter_no,
                        "commit_id": commit_id,
                        "asset_type": "chapter_digest",
                        "memory_type": "chapter_digest",
                        "text": digest_text,
                        "metadata": {"source": "chapter_processor"},
                    },
                }
            )
            logger.debug("[%s] ✓ Chapter digest embedding generated", chapter_no)

        # Generate fact digest points
        fact_count = len(delta.get("fact_changes", []))
        if fact_count > 0:
            logger.debug("[%s] Generating embeddings for %d fact digests", chapter_no, fact_count)

        for idx, fact in enumerate(delta.get("fact_changes", []), 1):
            value = json.dumps(fact.get("value"), ensure_ascii=False)
            fact_text = (
                f"{fact.get('subject_name') or fact.get('subject_id')} "
                f"{fact.get('predicate')} -> {value} "
                f"({fact.get('tier')}/{fact.get('status')})"
            )
            points.append(
                {
                    "id": self._stable_point_id(book_id, chapter_no, "fact_digest", str(idx)),
                    "vector": self._generate_embedding(fact_text),
                    "payload": {
                        "book_id": book_id,
                        "chapter": chapter_no,
                        "commit_id": commit_id,
                        "asset_type": "fact_digest",
                        "memory_type": "fact_digest",
                        "text": fact_text,
                        "metadata": {
                            "evidence_chunk_id": fact.get("evidence_chunk_id"),
                            "subject_id": fact.get("subject_id"),
                        },
                    },
                }
            )

        if fact_count > 0:
            logger.debug("[%s] ✓ %d fact digest embeddings generated", chapter_no, fact_count)

        # Generate relation digest points
        relation_count = len(delta.get("relations_delta", []))
        if relation_count > 0:
            logger.debug("[%s] Generating embeddings for %d relation digests", chapter_no, relation_count)

        for idx, rel in enumerate(delta.get("relations_delta", []), 1):
            rel_text = (
                f"{rel.get('from_name') or rel.get('from_id')} -{rel.get('kind')}/"
                f"{rel.get('status')}-> {rel.get('to_name') or rel.get('to_id')}"
            )
            points.append(
                {
                    "id": self._stable_point_id(book_id, chapter_no, "relation_digest", str(idx)),
                    "vector": self._generate_embedding(rel_text),
                    "payload": {
                        "book_id": book_id,
                        "chapter": chapter_no,
                        "commit_id": commit_id,
                        "asset_type": "relation_digest",
                        "memory_type": "relation_digest",
                        "text": rel_text,
                        "metadata": {
                            "evidence_chunk_id": rel.get("evidence_chunk_id"),
                            "from_id": rel.get("from_id"),
                            "to_id": rel.get("to_id"),
                            "kind": rel.get("kind"),
                        },
                    },
                }
            )

        if relation_count > 0:
            logger.debug("[%s] ✓ %d relation digest embeddings generated", chapter_no, relation_count)

        # Generate 8-element narrative asset points
        asset_counts = {}

        # 1. plot_beat assets
        plot_beats = assets.get("plot_beats", []) or []
        for idx, beat in enumerate(plot_beats, 1):
            if isinstance(beat, dict):
                # Build comprehensive text: event + impact (matching embedder_parallel.py)
                event = beat.get("event", "") or beat.get("beat", "") or beat.get("description", "")
                impact = beat.get("impact", "")
                beat_text = f"{event} {impact}".strip()
                characters = beat.get("characters", [])
            else:
                beat_text = str(beat)
                characters = []
            if beat_text:
                points.append({
                    "id": self._stable_point_id(book_id, chapter_no, "plot_beat", str(idx)),
                    "vector": self._generate_embedding(beat_text),
                    "payload": {
                        "book_id": book_id,
                        "chapter": chapter_no,
                        "commit_id": commit_id,
                        "asset_type": "plot_beat",
                        "memory_type": "plot_beat",
                        "text": beat_text,
                        "characters": characters if isinstance(characters, list) else [],
                        "metadata": beat if isinstance(beat, dict) else {},
                    },
                })
        asset_counts["plot_beat"] = len(plot_beats)

        # 2. character_card assets
        character_cards = assets.get("character_cards", []) or []
        for idx, card in enumerate(character_cards, 1):
            if isinstance(card, dict):
                # Build comprehensive text: name + traits + state (matching embedder_parallel.py)
                name = card.get("name", "")
                traits = card.get("traits", [])
                traits_text = " ".join(traits) if isinstance(traits, list) else str(traits)
                state = card.get("state", "")
                card_text = f"{name}: {traits_text} {state}".strip()
                characters = [name] if name else []
            else:
                card_text = str(card)
                characters = []
            if card_text and card_text != ":":
                points.append({
                    "id": self._stable_point_id(book_id, chapter_no, "character_card", str(idx)),
                    "vector": self._generate_embedding(card_text),
                    "payload": {
                        "book_id": book_id,
                        "chapter": chapter_no,
                        "commit_id": commit_id,
                        "asset_type": "character_card",
                        "memory_type": "character_card",
                        "text": card_text,
                        "characters": characters,
                        "metadata": card if isinstance(card, dict) else {},
                    },
                })
        asset_counts["character_card"] = len(character_cards)

        # 3. conflict assets
        conflicts = assets.get("conflicts", []) or []
        for idx, conflict in enumerate(conflicts, 1):
            if isinstance(conflict, dict):
                # Build comprehensive text: type + description (matching embedder_parallel.py)
                conflict_type = conflict.get("type", "")
                description = conflict.get("conflict", "") or conflict.get("description", "")
                conflict_text = f"{conflict_type} {description}".strip()
                characters = conflict.get("parties", [])
            else:
                conflict_text = str(conflict)
                characters = []
            if conflict_text:
                points.append({
                    "id": self._stable_point_id(book_id, chapter_no, "conflict", str(idx)),
                    "vector": self._generate_embedding(conflict_text),
                    "payload": {
                        "book_id": book_id,
                        "chapter": chapter_no,
                        "commit_id": commit_id,
                        "asset_type": "conflict",
                        "memory_type": "conflict",
                        "text": conflict_text,
                        "characters": characters if isinstance(characters, list) else [],
                        "metadata": conflict if isinstance(conflict, dict) else {},
                    },
                })
        asset_counts["conflict"] = len(conflicts)

        # 4. setting assets
        settings = assets.get("settings", []) or []
        for idx, setting in enumerate(settings, 1):
            if isinstance(setting, dict):
                # Build comprehensive text: location + time + atmosphere (matching embedder_parallel.py)
                location = setting.get("location", "")
                time = setting.get("time", "")
                atmosphere = setting.get("atmosphere", "")
                setting_text = f"{location} {time} {atmosphere}".strip()
            else:
                setting_text = str(setting)
            if setting_text:
                points.append({
                    "id": self._stable_point_id(book_id, chapter_no, "setting", str(idx)),
                    "vector": self._generate_embedding(setting_text),
                    "payload": {
                        "book_id": book_id,
                        "chapter": chapter_no,
                        "commit_id": commit_id,
                        "asset_type": "setting",
                        "memory_type": "setting",
                        "text": setting_text,
                        "characters": [],
                        "metadata": setting if isinstance(setting, dict) else {},
                    },
                })
        asset_counts["setting"] = len(settings)

        # 5. theme assets
        themes = assets.get("themes", []) or []
        for idx, theme in enumerate(themes, 1):
            if isinstance(theme, dict):
                # Build comprehensive text: theme + manifestation (matching embedder_parallel.py)
                theme_text = theme.get("theme", "")
                manifestation = theme.get("manifestation", "")
                theme_full_text = f"{theme_text} {manifestation}".strip()
            else:
                theme_full_text = str(theme)
            if theme_full_text:
                points.append({
                    "id": self._stable_point_id(book_id, chapter_no, "theme", str(idx)),
                    "vector": self._generate_embedding(theme_full_text),
                    "payload": {
                        "book_id": book_id,
                        "chapter": chapter_no,
                        "commit_id": commit_id,
                        "asset_type": "theme",
                        "memory_type": "theme",
                        "text": theme_full_text,
                        "characters": [],
                        "metadata": theme if isinstance(theme, dict) else {},
                    },
                })
        asset_counts["theme"] = len(themes)

        # 6. pov asset (single value)
        pov = assets.get("pov", {}) or assets.get("point_of_view", {})
        if isinstance(pov, dict):
            # Build text from all dict values (matching embedder_parallel.py)
            pov_text = " ".join(str(v) for v in pov.values() if v)
        else:
            pov_text = str(pov) if pov else ""
        if pov_text:
            points.append({
                "id": self._stable_point_id(book_id, chapter_no, "pov"),
                "vector": self._generate_embedding(pov_text),
                "payload": {
                    "book_id": book_id,
                    "chapter": chapter_no,
                    "commit_id": commit_id,
                    "asset_type": "pov",
                    "memory_type": "pov",
                    "text": pov_text,
                    "characters": [],
                    "metadata": pov if isinstance(pov, dict) else {},
                },
            })
            asset_counts["pov"] = 1

        # 7. tone asset (single value)
        tone = assets.get("tone", {})
        if isinstance(tone, dict):
            # Build text from all dict values (matching embedder_parallel.py)
            tone_text = " ".join(str(v) for v in tone.values() if v)
        else:
            tone_text = str(tone) if tone else ""
        if tone_text:
            points.append({
                "id": self._stable_point_id(book_id, chapter_no, "tone"),
                "vector": self._generate_embedding(tone_text),
                "payload": {
                    "book_id": book_id,
                    "chapter": chapter_no,
                    "commit_id": commit_id,
                    "asset_type": "tone",
                    "memory_type": "tone",
                    "text": tone_text,
                    "characters": [],
                    "metadata": tone if isinstance(tone, dict) else {},
                },
            })
            asset_counts["tone"] = 1

        # 8. style asset (single value)
        style = assets.get("style", {})
        if isinstance(style, dict):
            # Build text from all dict values (matching embedder_parallel.py)
            style_text = " ".join(str(v) for v in style.values() if v)
        else:
            style_text = str(style) if style else ""
        if style_text:
            points.append({
                "id": self._stable_point_id(book_id, chapter_no, "style"),
                "vector": self._generate_embedding(style_text),
                "payload": {
                    "book_id": book_id,
                    "chapter": chapter_no,
                    "commit_id": commit_id,
                    "asset_type": "style",
                    "memory_type": "style",
                    "text": style_text,
                    "characters": [],
                    "metadata": style if isinstance(style, dict) else {},
                },
            })
            asset_counts["style"] = 1

        if not points:
            logger.debug("[%s] No points to write to Qdrant", chapter_no)
            return

        logger.info(
            "[%s] Upserting %d points to Qdrant (digests: chapter=%d fact=%d relation=%d, assets: plot=%d char=%d conflict=%d setting=%d theme=%d pov=%d tone=%d style=%d)",
            chapter_no,
            len(points),
            1 if digest_text else 0,
            fact_count,
            relation_count,
            asset_counts.get("plot_beat", 0),
            asset_counts.get("character_card", 0),
            asset_counts.get("conflict", 0),
            asset_counts.get("setting", 0),
            asset_counts.get("theme", 0),
            asset_counts.get("pov", 0),
            asset_counts.get("tone", 0),
            asset_counts.get("style", 0),
        )

        self._upsert_qdrant_points(points)

        logger.info(
            "[%s] ✓ Qdrant write complete: %d points (commit=%s collection=%s)",
            chapter_no,
            len(points),
            commit_id[:8],
            self.qdrant_collection,
        )

    def _apply_to_neo4j(
        self,
        book_id: str,
        chapter_no: str,
        chapter_title: str,
        chapter_summary: str,
        chunks: list[dict[str, Any]],
        delta: dict[str, Any],
        entity_meta: dict[str, dict[str, Any]],
        commit_id: str,
    ) -> dict[str, int]:
        stats = {"entities": 0, "relations": 0, "events": 0}
        self.neo4j.create_chapter(book_id, chapter_no, chapter_title, None, chapter_summary)
        if chunks:
            self.neo4j.create_chunks(book_id, chapter_no, chunks)

        for entity_id, meta in entity_meta.items():
            self.neo4j.upsert_entity(
                entity_id=entity_id,
                entity_type=meta.get("type", "character"),
                canonical_name=meta.get("name", entity_id),
                aliases=meta.get("aliases", []),
                traits=meta.get("traits", {}),
                status=meta.get("status", "active"),
                commit_id=commit_id,
            )
            stats["entities"] += 1

        for rel in delta.get("relations_delta", []):
            self.neo4j.upsert_relation(
                from_id=rel["from_id"],
                to_id=rel["to_id"],
                kind=rel["kind"],
                status=rel.get("status", "confirmed"),
                valid_from=rel.get("valid_from", chapter_no),
                valid_to=rel.get("valid_to"),
                evidence_chunk_id=rel.get("evidence_chunk_id"),
                commit_id=commit_id,
            )
            stats["relations"] += 1

        for event in delta.get("events", []):
            self.neo4j.create_event(
                event_id=event["event_id"],
                event_type=event.get("type", "plot_beat"),
                summary=event.get("summary", ""),
                book_id=book_id,
                chapter_no=chapter_no,
                participants=event.get("participants", []),
                location_id=None,
                commit_id=commit_id,
            )
            stats["events"] += 1

        for hook in delta.get("hooks", []):
            thread_id = f"thread_{uuid.uuid4().hex[:8]}"
            hook_id = f"hook_{uuid.uuid4().hex[:8]}"
            self.neo4j.create_thread(
                thread_id,
                hook.get("thread_name") or hook.get("name", f"thread_{chapter_no}"),
                "open",
                hook.get("priority", 1),
                None,
            )
            self.neo4j.create_hook(
                hook_id,
                hook.get("summary", ""),
                book_id,
                chapter_no,
                thread_id,
                hook.get("evidence_chunk_id"),
            )

        return stats

    def process_chapter(
        self,
        book_id: str,
        chapter_no: str,
        delta: Optional[dict[str, Any]] = None,
        chapter_title: str = "",
        chapter_summary: str = "",
        chapter_text: str = "",
        assets: Optional[dict[str, Any]] = None,
        mode: str = "delta",
    ) -> dict[str, Any]:
        """Process a chapter in `delta` mode or `llm` mode.

        - `mode="delta"`: use provided delta; if missing and assets exist, derive from assets.
        - `mode="llm"`: chapter text -> assets(optional) -> LLM delta extractor.
        """
        chunks = self.create_chunks(chapter_text, chapter_no, book_id=book_id) if chapter_text else []
        default_chunk_id = chunks[0]["chunk_id"] if chunks else f"{chapter_no}#c00"

        if mode == "llm":
            if not chapter_text and not assets:
                raise ValueError("chapter_text or assets is required in llm mode")
            if not self.delta_extractor:
                raise ValueError("llm_config is required in ChapterProcessor for llm mode")
            if delta is None:
                # In llm mode, empty assets means "extract assets first", not "use empty delta".
                if not assets:
                    assets = self._extract_assets(chapter_text)
                prev_context = self.canon_db.get_prev_context_snapshot(
                    chapter_no,
                    state_limit=self.context_state_limit,
                    relation_limit=self.context_relation_limit,
                    thread_limit=self.context_thread_limit,
                )
                delta_raw = self.delta_extractor.extract(
                    chapter_no=chapter_no,
                    chapter_text=chapter_text,
                    chunks=chunks,
                    assets=assets or {},
                    prev_context=prev_context,
                )
                delta = self._normalize_delta(delta_raw, chapter_no, default_chunk_id, assets=assets)
            else:
                delta = self._normalize_delta(delta, chapter_no, default_chunk_id, assets=assets)
        else:
            if delta is None:
                if assets is None:
                    raise ValueError("delta or assets must be provided in delta mode")
                delta = convert_assets_to_delta(assets, chapter_no)
            delta = self._normalize_delta(delta, chapter_no, default_chunk_id, assets=assets)

        resolved_delta, entity_meta = self._resolve_delta_entity_ids(delta, chapter_no)
        self._augment_resolved_relations_from_events(resolved_delta, entity_meta, chapter_no)
        self._prune_relations(resolved_delta)
        self._enforce_chinese_fields_on_delta(resolved_delta)
        proposed_facts = resolved_delta.get("fact_changes", [])
        proposed_rels = resolved_delta.get("relations_delta", [])

        conflicts = self.canon_db.detect_conflicts(chapter_no, proposed_facts, proposed_rels)
        if conflicts["blocking"]:
            return {"status": "blocked", "conflicts": conflicts, "chapter_no": chapter_no}

        commit_payload = {"assets": assets or {}, "delta": resolved_delta, "chunks": chunks}
        commit_id = self.canon_db.begin_commit(book_id, chapter_no, commit_payload)
        results = {
            "status": "success",
            "commit_id": commit_id,
            "chapter_no": chapter_no,
            "entities": len(entity_meta),
            "facts": 0,
            "relations": 0,
            "events": 0,
            "warnings": conflicts["warnings"],
            "errors": [],
        }

        try:
            if proposed_facts:
                self.canon_db.append_fact_history(
                    commit_id,
                    [
                        {
                            "chapter_no": chapter_no,
                            "subject_id": fact["subject_id"],
                            "predicate": fact["predicate"],
                            "value": fact["value"],
                            "op": fact.get("op", "INSERT"),
                            "valid_from": fact.get("valid_from", chapter_no),
                            "valid_to": fact.get("valid_to"),
                            "tier": fact.get("tier", "SOFT_NOTE"),
                            "status": fact.get("status", "confirmed"),
                            "confidence": fact.get("confidence", 1.0),
                            "evidence_chunk_id": fact.get("evidence_chunk_id"),
                            "source": "llm_delta" if mode == "llm" else "asset_delta",
                        }
                        for fact in proposed_facts
                    ],
                )
                results["facts"] = len(proposed_facts)

            if proposed_rels:
                self.canon_db.append_relationship_history(
                    commit_id,
                    [
                        {
                            "chapter_no": chapter_no,
                            "from_id": rel["from_id"],
                            "to_id": rel["to_id"],
                            "kind": rel["kind"],
                            "op": rel.get("op", "INSERT"),
                            "valid_from": rel.get("valid_from", chapter_no),
                            "valid_to": rel.get("valid_to"),
                            "status": rel.get("status", "confirmed"),
                            "confidence": rel.get("confidence", 1.0),
                            "evidence_chunk_id": rel.get("evidence_chunk_id"),
                            "source": "llm_delta" if mode == "llm" else "asset_delta",
                        }
                        for rel in proposed_rels
                    ],
                )
                results["relations"] = len(proposed_rels)

            self.canon_db.update_current_snapshots(commit_id)
            self.canon_db.mark_commit_status(commit_id, "CANON_DONE")

            neo4j_stats = self._apply_to_neo4j(
                book_id=book_id,
                chapter_no=chapter_no,
                chapter_title=chapter_title,
                chapter_summary=chapter_summary,
                chunks=chunks,
                delta=resolved_delta,
                entity_meta=entity_meta,
                commit_id=commit_id,
            )
            results["events"] = neo4j_stats["events"]
            self.canon_db.mark_commit_status(commit_id, "NEO4J_DONE")

            self._write_qdrant(
                book_id=book_id,
                chapter_no=chapter_no,
                commit_id=commit_id,
                chapter_summary=chapter_summary,
                assets=assets or {},
                delta=resolved_delta,
            )
            self.canon_db.mark_commit_status(commit_id, "ALL_DONE")
        except Exception as exc:
            logger.exception("[%s] chapter processing failed", chapter_no)
            self.canon_db.mark_commit_status(commit_id, "FAILED")
            results["status"] = "failed"
            results["errors"].append(str(exc))
            raise

        return results

    def replay_commit(self, commit_id: str) -> dict[str, Any]:
        """Replay an existing commit payload into Neo4j/Qdrant without LLM."""
        commit_record = self.canon_db.get_commit_payload(commit_id)
        if not commit_record:
            raise ValueError(f"Commit not found: {commit_id}")

        payload = commit_record["payload"]
        chapter_no = commit_record["chapter_no"]
        book_id = commit_record["book_id"]
        delta = payload.get("delta", {})
        chunks = payload.get("chunks", [])
        assets = payload.get("assets", {})

        # Backward compatibility for old compact payloads.
        if not delta and payload.get("entities") is not None:
            return {
                "status": "skipped",
                "reason": "commit payload has no delta/chunks to replay",
                "commit_id": commit_id,
            }

        normalized_delta = self._normalize_delta(
            delta,
            chapter_no,
            f"{chapter_no}#c00",
            assets=payload.get("assets", {}),
        )
        if normalized_delta.get("fact_changes") and "subject_id" not in normalized_delta["fact_changes"][0]:
            normalized_delta, entity_meta = self._resolve_delta_entity_ids(normalized_delta, chapter_no)
        else:
            entity_meta = {
                ent["entity_id"]: {
                    "entity_id": ent["entity_id"],
                    "name": ent.get("name", ent["entity_id"]),
                    "type": ent.get("type", "character"),
                    "aliases": ent.get("aliases", []),
                    "traits": ent.get("traits", {}),
                    "status": ent.get("status", "active"),
                }
                for ent in normalized_delta.get("entities_new", [])
                if ent.get("entity_id")
            }
            for rel in normalized_delta.get("relations_delta", []):
                for eid, n in ((rel.get("from_id"), rel.get("from_name")), (rel.get("to_id"), rel.get("to_name"))):
                    if eid and eid not in entity_meta:
                        entity_meta[eid] = {
                            "entity_id": eid,
                            "name": n or eid,
                            "type": "character",
                            "aliases": [],
                            "traits": {},
                            "status": "active",
                        }

        self._augment_resolved_relations_from_events(normalized_delta, entity_meta, chapter_no)
        self._prune_relations(normalized_delta)

        self.canon_db.mark_commit_status(commit_id, "CANON_DONE")
        self._apply_to_neo4j(
            book_id=book_id,
            chapter_no=chapter_no,
            chapter_title="",
            chapter_summary="",
            chunks=chunks,
            delta=normalized_delta,
            entity_meta=entity_meta,
            commit_id=commit_id,
        )
        self.canon_db.mark_commit_status(commit_id, "NEO4J_DONE")
        self._write_qdrant(
            book_id=book_id,
            chapter_no=chapter_no,
            commit_id=commit_id,
            chapter_summary="",
            assets=assets,
            delta=normalized_delta,
        )
        self.canon_db.mark_commit_status(commit_id, "ALL_DONE")
        return {"status": "success", "commit_id": commit_id, "chapter_no": chapter_no}

    def process_from_assets(
        self,
        book_id: str,
        asset_path: str | Path,
        chapter_no: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Load an asset file, convert to delta, then process."""
        # Load the original assets from file
        with open(asset_path, 'r', encoding='utf-8') as f:
            assets = json.load(f)

        # Convert assets to delta format
        delta = convert_assets_to_delta(assets, chapter_no)

        # Pass both assets and delta to process_chapter
        # assets are needed for _write_qdrant() to store the 8 asset types
        return self.process_chapter(book_id, chapter_no, delta=delta, assets=assets, mode="delta", **kwargs)
