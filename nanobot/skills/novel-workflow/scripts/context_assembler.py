"""Context pack assembler for three-tier novel memory.

Builds a writing context from Canon DB (authoritative), Neo4j (structural),
and Qdrant (recall) for a target chapter.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
from canon_db_v2 import CanonDBV2
from neo4j_manager import Neo4jManager


class ContextAssembler:
    """Assemble writing context packs from all memory tiers."""

    def __init__(
        self,
        canon_db_path: str,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_pass: str = "novel123",
        qdrant_url: str = "",
        qdrant_collection: str = "novel_assets_v2",
    ):
        self.canon_db_path = str(Path(canon_db_path).expanduser())
        self.canon_db = CanonDBV2(self.canon_db_path)
        self.neo4j = Neo4jManager(neo4j_uri, neo4j_user, neo4j_pass)
        self.qdrant_url = qdrant_url.rstrip("/")
        self.qdrant_collection = qdrant_collection

    def close(self) -> None:
        self.neo4j.close()
        self.canon_db.close()

    def assemble_context_pack(
        self,
        book_id: str,
        chapter_no: str,
        outline: str = "",
        top_n: int = 15,
        recall_k: int = 8,
    ) -> dict[str, Any]:
        """Build a structured context pack for chapter writing."""
        focus_entities = self._select_focus_entities(outline, top_n=top_n)
        entity_ids = [item["entity_id"] for item in focus_entities]

        context = {
            "hard_rules": self._get_hard_rules(chapter_no, limit=top_n),
            "hard_state": {
                "characters": self._get_character_states(entity_ids, chapter_no, top_n=top_n),
                "items": self._get_item_states(entity_ids, chapter_no, top_n=top_n),
            },
            "relations": self._get_relations(entity_ids, chapter_no, top_n=top_n),
            "threads": self._get_threads(book_id, top_n=top_n),
            "pov_knowledge": [],
            "recall": self._get_recall(book_id, outline, recall_k=recall_k),
            "outline": outline,
            "meta": {
                "book_id": book_id,
                "chapter_no": chapter_no,
                "focus_entities": focus_entities,
            },
        }
        return context

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.canon_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _chapter_index(self, chapter_no: str) -> int:
        digits = "".join(ch for ch in str(chapter_no) if ch.isdigit())
        return int(digits) if digits else 0

    def _select_focus_entities(self, outline: str, top_n: int = 15) -> list[dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT er.entity_id, er.canonical_name, er.type, cc.updated_chapter
                FROM entity_registry er
                LEFT JOIN character_current cc ON cc.entity_id = er.entity_id
                WHERE er.type = 'character' AND er.merged_into IS NULL
                ORDER BY cc.updated_chapter DESC, er.last_seen_chapter DESC
                LIMIT 120
                """
            ).fetchall()
            outlined = []
            fallback = []
            for row in rows:
                item = {
                    "entity_id": row["entity_id"],
                    "name": row["canonical_name"],
                    "type": row["type"],
                    "updated_chapter": row["updated_chapter"],
                }
                if outline and row["canonical_name"] and row["canonical_name"] in outline:
                    outlined.append(item)
                else:
                    fallback.append(item)
            merged = outlined + [x for x in fallback if x["entity_id"] not in {i["entity_id"] for i in outlined}]
            return merged[:top_n]
        finally:
            conn.close()

    def _get_hard_rules(self, chapter_no: str, limit: int = 20) -> list[dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT fh.subject_id, er.canonical_name, fh.predicate, fh.object_json,
                       fh.evidence_chunk_id, fh.chapter_no
                FROM fact_history fh
                JOIN entity_registry er ON er.entity_id = fh.subject_id
                WHERE fh.tier = 'HARD_RULE'
                  AND fh.chapter_no <= ?
                ORDER BY fh.chapter_no DESC, fh.created_at DESC
                LIMIT ?
                """,
                (chapter_no, limit),
            ).fetchall()
            result = []
            for row in rows:
                result.append(
                    {
                        "subject_id": row["subject_id"],
                        "subject_name": row["canonical_name"],
                        "predicate": row["predicate"],
                        "value": json.loads(row["object_json"] or "null"),
                        "evidence": row["evidence_chunk_id"],
                        "chapter_no": row["chapter_no"],
                    }
                )
            return result
        finally:
            conn.close()

    def _get_character_states(
        self, entity_ids: list[str], chapter_no: str, top_n: int = 15
    ) -> dict[str, dict[str, Any]]:
        conn = self._get_conn()
        try:
            if not entity_ids:
                rows = conn.execute(
                    """
                    SELECT er.entity_id, er.canonical_name, cc.state_json, cc.status_tags_json,
                           cc.updated_chapter
                    FROM character_current cc
                    JOIN entity_registry er ON er.entity_id = cc.entity_id
                    WHERE er.type = 'character'
                    ORDER BY cc.updated_chapter DESC
                    LIMIT ?
                    """,
                    (top_n,),
                ).fetchall()
            else:
                placeholders = ",".join("?" for _ in entity_ids)
                rows = conn.execute(
                    f"""
                    SELECT er.entity_id, er.canonical_name, cc.state_json, cc.status_tags_json,
                           cc.updated_chapter
                    FROM character_current cc
                    JOIN entity_registry er ON er.entity_id = cc.entity_id
                    WHERE cc.entity_id IN ({placeholders})
                    ORDER BY cc.updated_chapter DESC
                    """,
                    entity_ids,
                ).fetchall()
            payload: dict[str, dict[str, Any]] = {}
            for row in rows:
                payload[row["entity_id"]] = {
                    "name": row["canonical_name"],
                    "state": json.loads(row["state_json"] or "{}"),
                    "status_tags": json.loads(row["status_tags_json"] or "[]"),
                    "updated_chapter": row["updated_chapter"] or chapter_no,
                }
            return payload
        finally:
            conn.close()

    def _get_item_states(
        self, focus_entity_ids: list[str], chapter_no: str, top_n: int = 15
    ) -> dict[str, dict[str, Any]]:
        conn = self._get_conn()
        try:
            if focus_entity_ids:
                placeholders = ",".join("?" for _ in focus_entity_ids)
                rows = conn.execute(
                    f"""
                    SELECT ic.entity_id, er.canonical_name, ic.owner_id, ic.status, ic.props_json,
                           ic.updated_chapter
                    FROM item_current ic
                    JOIN entity_registry er ON er.entity_id = ic.entity_id
                    WHERE ic.owner_id IN ({placeholders})
                    ORDER BY ic.updated_chapter DESC
                    LIMIT ?
                    """,
                    [*focus_entity_ids, top_n],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT ic.entity_id, er.canonical_name, ic.owner_id, ic.status, ic.props_json,
                           ic.updated_chapter
                    FROM item_current ic
                    JOIN entity_registry er ON er.entity_id = ic.entity_id
                    ORDER BY ic.updated_chapter DESC
                    LIMIT ?
                    """,
                    (top_n,),
                ).fetchall()
            payload: dict[str, dict[str, Any]] = {}
            for row in rows:
                payload[row["entity_id"]] = {
                    "name": row["canonical_name"],
                    "owner_id": row["owner_id"],
                    "status": row["status"],
                    "props": json.loads(row["props_json"] or "{}"),
                    "updated_chapter": row["updated_chapter"] or chapter_no,
                }
            return payload
        finally:
            conn.close()

    def _get_relations(self, entity_ids: list[str], chapter_no: str, top_n: int = 20) -> dict[str, Any]:
        active = []
        seen = set()
        for entity_id in entity_ids:
            for rel in self.neo4j.get_active_relations(entity_id, chapter_no):
                pair = tuple(sorted((entity_id, rel.get("to_id", ""))))
                key = (pair, rel.get("kind"))
                if key in seen:
                    continue
                seen.add(key)
                active.append(
                    {
                        "from_id": entity_id,
                        "to_id": rel.get("to_id"),
                        "to_name": rel.get("to_name"),
                        "kind": rel.get("kind"),
                        "status": rel.get("status"),
                        "since": rel.get("since"),
                    }
                )

        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT rh.from_id, fr.canonical_name AS from_name,
                       rh.to_id, tr.canonical_name AS to_name,
                       rh.kind, rh.status, rh.chapter_no
                FROM relationship_history rh
                LEFT JOIN entity_registry fr ON fr.entity_id = rh.from_id
                LEFT JOIN entity_registry tr ON tr.entity_id = rh.to_id
                WHERE rh.chapter_no <= ?
                ORDER BY rh.chapter_no DESC, rh.created_at DESC
                LIMIT ?
                """,
                (chapter_no, top_n),
            ).fetchall()
            recent_changes = [dict(row) for row in rows]
        finally:
            conn.close()

        return {"active": active[:top_n], "recent_changes": recent_changes}

    def _get_threads(self, book_id: str, top_n: int = 15) -> dict[str, Any]:
        canon_open = []
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT thread_id, name, status, priority, planned_window, notes, updated_chapter
                FROM thread_current
                WHERE status != 'resolved'
                ORDER BY priority DESC, updated_chapter DESC
                LIMIT ?
                """,
                (top_n,),
            ).fetchall()
            canon_open = [dict(row) for row in rows]
        finally:
            conn.close()
        graph_open = self.neo4j.get_unresolved_threads(book_id)
        return {"unresolved": canon_open, "graph_unresolved": graph_open[:top_n]}

    def _get_recall(self, book_id: str, outline: str, recall_k: int = 8) -> dict[str, Any]:
        return {
            "similar_beats": self._qdrant_scroll(book_id, "plot_beat", recall_k),
            "similar_conflicts": self._qdrant_scroll(book_id, "conflict", recall_k),
            "style_templates": self._qdrant_scroll(book_id, "style", recall_k),
            "query": outline,
        }

    def _qdrant_scroll(self, book_id: str, asset_type: str, limit: int) -> list[dict[str, Any]]:
        if not self.qdrant_url:
            return []
        response = httpx.post(
            f"{self.qdrant_url}/collections/{self.qdrant_collection}/points/scroll",
            json={
                "filter": {
                    "must": [
                        {"key": "book_id", "match": {"value": book_id}},
                        {"key": "asset_type", "match": {"value": asset_type}},
                    ]
                },
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        rows = response.json().get("result", {}).get("points", [])
        payload = []
        for item in rows:
            p = item.get("payload", {})
            payload.append(
                {
                    "id": item.get("id"),
                    "text": p.get("text", ""),
                    "chapter": p.get("chapter"),
                    "asset_type": p.get("asset_type"),
                    "metadata": p.get("metadata", {}),
                }
            )
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble context pack from three-tier memory")
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--chapter-no", required=True)
    parser.add_argument("--outline", default="")
    parser.add_argument("--canon-db-path", required=True)
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="novel123")
    parser.add_argument("--qdrant-url", default="")
    parser.add_argument("--qdrant-collection", default="novel_assets_v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--recall-k", type=int, default=8)
    args = parser.parse_args()

    assembler = ContextAssembler(
        canon_db_path=args.canon_db_path,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_pass=args.neo4j_pass,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
    )
    try:
        pack = assembler.assemble_context_pack(
            book_id=args.book_id,
            chapter_no=args.chapter_no,
            outline=args.outline,
            top_n=args.top_n,
            recall_k=args.recall_k,
        )
    finally:
        assembler.close()

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Context pack written to: {output_path}")


if __name__ == "__main__":
    main()
