"""Chapter processor coordinating Canon DB, Neo4j, and Qdrant memory tiers."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from canon_db_v2 import CanonDBV2
from delta_converter import convert_assets_to_delta, load_and_convert
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


class ChapterProcessor:
    """Orchestrates chapter commits across all three memory tiers."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_pass: str,
        canon_db_path: str,
        qdrant_url: Optional[str] = None,
        llm_config: Optional[dict[str, Any]] = None,
        llm_max_tokens: int = 4096,
        context_state_limit: int = 30,
        context_relation_limit: int = 30,
        context_thread_limit: int = 20,
    ):
        self.neo4j = Neo4jManager(neo4j_uri, neo4j_user, neo4j_pass)
        self.canon_db = CanonDBV2(canon_db_path)
        self.qdrant_url = qdrant_url
        self.llm_config = llm_config
        self.llm_max_tokens = llm_max_tokens
        self.context_state_limit = context_state_limit
        self.context_relation_limit = context_relation_limit
        self.context_thread_limit = context_thread_limit
        self.delta_extractor = (
            DeltaExtractorLLM(llm_config, max_tokens=llm_max_tokens) if llm_config else None
        )

    def close(self):
        """Close all connections."""
        self.neo4j.close()
        self.canon_db.close()

    def create_chunks(
        self, chapter_text: str, chapter_no: str, max_chars: int = 550
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

        for unit in units:
            if not unit:
                continue
            if current_len + len(unit) > max_chars and buffer:
                text = "".join(buffer).strip()
                chunks.append(
                    {
                        "chunk_id": f"{chapter_no}#c{chunk_index:02d}",
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
                    "chunk_id": f"{chapter_no}#c{chunk_index:02d}",
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
            resolved["fact_changes"].append({**fact, "subject_id": subject_id})

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
            resolved["relations_delta"].append({**rel, "from_id": from_id, "to_id": to_id})

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

    def _extract_assets(self, chapter_text: str) -> dict[str, Any]:
        from asset_extractor_parallel import extract_all_assets

        if not self.llm_config:
            raise ValueError("llm_config is required for asset extraction")
        return extract_all_assets(chapter_text, self.llm_config)

    def _write_qdrant(self, chapter_no: str, commit_id: str, assets: dict[str, Any], delta: dict[str, Any]):
        """Placeholder for Qdrant writes; keeps commit state flow stable."""
        if not self.qdrant_url:
            return
        fact_digest_count = len(delta.get("fact_changes", [])) + len(delta.get("relations_delta", []))
        logger.info(
            "[%s] Qdrant write placeholder (commit=%s assets=%d fact_digests=%d)",
            chapter_no,
            commit_id[:8],
            len(assets or {}),
            fact_digest_count,
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
            self.neo4j.create_chunks(chapter_no, chunks)

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
        chunks = self.create_chunks(chapter_text, chapter_no) if chapter_text else []
        default_chunk_id = chunks[0]["chunk_id"] if chunks else f"{chapter_no}#c00"

        if mode == "llm":
            if not chapter_text and assets is None:
                raise ValueError("chapter_text or assets is required in llm mode")
            if not self.delta_extractor:
                raise ValueError("llm_config is required in ChapterProcessor for llm mode")
            assets = assets or self._extract_assets(chapter_text)
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
            if delta is None:
                if assets is None:
                    raise ValueError("delta or assets must be provided in delta mode")
                delta = convert_assets_to_delta(assets, chapter_no)
            delta = self._normalize_delta(delta, chapter_no, default_chunk_id, assets=assets)

        resolved_delta, entity_meta = self._resolve_delta_entity_ids(delta, chapter_no)
        self._augment_resolved_relations_from_events(resolved_delta, entity_meta, chapter_no)
        self._prune_relations(resolved_delta)
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

            self._write_qdrant(chapter_no, commit_id, assets or {}, resolved_delta)
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
        self._write_qdrant(chapter_no, commit_id, assets, normalized_delta)
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
        delta = load_and_convert(asset_path, chapter_no)
        return self.process_chapter(book_id, chapter_no, delta=delta, mode="delta", **kwargs)
