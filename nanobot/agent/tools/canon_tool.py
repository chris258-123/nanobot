"""Canon DB authoritative memory tool."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


def _load_canon_db_v2():
    scripts_dir = Path(__file__).resolve().parents[2] / "skills" / "novel-workflow" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from canon_db_v2 import CanonDBV2  # type: ignore

    return CanonDBV2


class CanonTool(Tool):
    """Read/query authoritative memory and conflict checks from Canon DB."""

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser())
        self._canon_cls = _load_canon_db_v2()

    @property
    def name(self) -> str:
        return "canon"

    @property
    def description(self) -> str:
        return """Canon DB authoritative memory operations.

Actions:
- stats: database counters and distributions
- get_state: get current state for one entity (requires: entity_id)
- get_history: fact/relationship history for one entity (requires: entity_id)
- get_rules: list HARD_RULE entries up to chapter (optional: chapter_no, limit)
- detect_conflicts: run preflight conflict check (requires: chapter_no; optional: proposed_facts, proposed_relations)
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["stats", "get_state", "get_history", "get_rules", "detect_conflicts"],
                },
                "entity_id": {"type": "string"},
                "chapter_no": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
                "proposed_facts": {"type": "array", "items": {"type": "object"}},
                "proposed_relations": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        try:
            if action == "stats":
                return self._stats()
            if action == "get_state":
                return self._get_state(kwargs.get("entity_id", ""))
            if action == "get_history":
                return self._get_history(kwargs.get("entity_id", ""), kwargs.get("limit", 20))
            if action == "get_rules":
                return self._get_rules(kwargs.get("chapter_no", "9999"), kwargs.get("limit", 20))
            if action == "detect_conflicts":
                return self._detect_conflicts(
                    kwargs.get("chapter_no", ""),
                    kwargs.get("proposed_facts") or [],
                    kwargs.get("proposed_relations") or [],
                )
            return f"Unknown action: {action}"
        except Exception as exc:
            return f"Error: {exc}"

    def _stats(self) -> str:
        db = self._canon_cls(self.db_path)
        try:
            payload = db.get_statistics()
            return json.dumps(payload, ensure_ascii=False, indent=2)
        finally:
            db.close()

    def _get_state(self, entity_id: str) -> str:
        if not entity_id:
            return "Error: entity_id is required"
        db = self._canon_cls(self.db_path)
        try:
            entity = db.get_entity_by_id(entity_id)
            state = db.get_character_state(entity_id)
            payload = {"entity": entity, "state": state}
            return json.dumps(payload, ensure_ascii=False, indent=2)
        finally:
            db.close()

    def _get_history(self, entity_id: str, limit: int) -> str:
        if not entity_id:
            return "Error: entity_id is required"
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            facts = conn.execute(
                """
                SELECT chapter_no, predicate, object_json, op, tier, status, evidence_chunk_id
                FROM fact_history
                WHERE subject_id = ?
                ORDER BY chapter_no DESC, created_at DESC
                LIMIT ?
                """,
                (entity_id, max(limit, 1)),
            ).fetchall()
            rels = conn.execute(
                """
                SELECT chapter_no, from_id, to_id, kind, op, status, evidence_chunk_id
                FROM relationship_history
                WHERE from_id = ? OR to_id = ?
                ORDER BY chapter_no DESC, created_at DESC
                LIMIT ?
                """,
                (entity_id, entity_id, max(limit, 1)),
            ).fetchall()
            payload = {
                "facts": [
                    {
                        **dict(row),
                        "object_json": json.loads(row["object_json"] or "null"),
                    }
                    for row in facts
                ],
                "relations": [dict(row) for row in rels],
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)
        finally:
            conn.close()

    def _get_rules(self, chapter_no: str, limit: int) -> str:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT fh.chapter_no, fh.subject_id, er.canonical_name, fh.predicate, fh.object_json,
                       fh.evidence_chunk_id
                FROM fact_history fh
                JOIN entity_registry er ON er.entity_id = fh.subject_id
                WHERE fh.tier = 'HARD_RULE' AND fh.chapter_no <= ?
                ORDER BY fh.chapter_no DESC, fh.created_at DESC
                LIMIT ?
                """,
                (chapter_no, max(limit, 1)),
            ).fetchall()
            payload = []
            for row in rows:
                item = dict(row)
                item["object_json"] = json.loads(item["object_json"] or "null")
                payload.append(item)
            return json.dumps(payload, ensure_ascii=False, indent=2)
        finally:
            conn.close()

    def _detect_conflicts(
        self, chapter_no: str, proposed_facts: list[dict[str, Any]], proposed_relations: list[dict[str, Any]]
    ) -> str:
        if not chapter_no:
            return "Error: chapter_no is required"
        db = self._canon_cls(self.db_path)
        try:
            result = db.detect_conflicts(chapter_no, proposed_facts, proposed_relations)
            return json.dumps(result, ensure_ascii=False, indent=2)
        finally:
            db.close()
