"""Enhanced Canon database with history tracking and conflict detection.

SQLite database for authoritative novel facts with event sourcing.
"""

import sqlite3
import json
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CanonDBV2:
    """Enhanced Canon DB with history tracking, commits, and conflict detection."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize enhanced database schema."""
        self.conn.executescript("""
            -- Entity Registry (single source of truth)
            CREATE TABLE IF NOT EXISTS entity_registry (
                entity_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                aliases_json TEXT,
                first_seen_chapter TEXT,
                last_seen_chapter TEXT,
                merged_into TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(type, canonical_name)
            );

            -- Current State Snapshots
            CREATE TABLE IF NOT EXISTS character_current (
                entity_id TEXT PRIMARY KEY,
                state_json TEXT,
                voice_json TEXT,
                faction TEXT,
                status_tags_json TEXT,
                updated_chapter TEXT,
                updated_commit TEXT,
                FOREIGN KEY(entity_id) REFERENCES entity_registry(entity_id)
            );

            CREATE TABLE IF NOT EXISTS item_current (
                entity_id TEXT PRIMARY KEY,
                owner_id TEXT,
                status TEXT,
                props_json TEXT,
                updated_chapter TEXT,
                updated_commit TEXT,
                FOREIGN KEY(entity_id) REFERENCES entity_registry(entity_id)
            );

            CREATE TABLE IF NOT EXISTS rule_current (
                entity_id TEXT PRIMARY KEY,
                hard_level TEXT,
                rule_json TEXT,
                exceptions_json TEXT,
                updated_chapter TEXT,
                updated_commit TEXT,
                FOREIGN KEY(entity_id) REFERENCES entity_registry(entity_id)
            );

            -- Fact History (event sourcing)
            CREATE TABLE IF NOT EXISTS fact_history (
                fact_id TEXT PRIMARY KEY,
                commit_id TEXT NOT NULL,
                chapter_no TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_json TEXT NOT NULL,
                op TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                tier TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                evidence_chunk_id TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(subject_id) REFERENCES entity_registry(entity_id)
            );

            -- Relationship History
            CREATE TABLE IF NOT EXISTS relationship_history (
                rel_hist_id TEXT PRIMARY KEY,
                commit_id TEXT NOT NULL,
                chapter_no TEXT NOT NULL,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                op TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                status TEXT NOT NULL,
                confidence REAL,
                evidence_chunk_id TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(from_id) REFERENCES entity_registry(entity_id),
                FOREIGN KEY(to_id) REFERENCES entity_registry(entity_id)
            );

            -- Address Book (称呼关系)
            CREATE TABLE IF NOT EXISTS address_book (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                name_used TEXT NOT NULL,
                context TEXT,
                valid_from TEXT,
                valid_to TEXT,
                evidence_chunk_id TEXT,
                commit_id TEXT,
                PRIMARY KEY(from_id, to_id, name_used, valid_from),
                FOREIGN KEY(from_id) REFERENCES entity_registry(entity_id),
                FOREIGN KEY(to_id) REFERENCES entity_registry(entity_id)
            );

            -- Thread Tracking
            CREATE TABLE IF NOT EXISTS thread_current (
                thread_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER,
                planned_window TEXT,
                notes TEXT,
                updated_chapter TEXT,
                updated_commit TEXT
            );

            -- Commit Log
            CREATE TABLE IF NOT EXISTS commit_log (
                commit_id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL,
                chapter_no TEXT NOT NULL,
                commit_type TEXT NOT NULL,
                payload_json TEXT,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_entity_type ON entity_registry(type);
            CREATE INDEX IF NOT EXISTS idx_fact_subject ON fact_history(subject_id);
            CREATE INDEX IF NOT EXISTS idx_fact_chapter ON fact_history(chapter_no);
            CREATE INDEX IF NOT EXISTS idx_fact_commit ON fact_history(commit_id);
            CREATE INDEX IF NOT EXISTS idx_rel_from ON relationship_history(from_id);
            CREATE INDEX IF NOT EXISTS idx_rel_to ON relationship_history(to_id);
            CREATE INDEX IF NOT EXISTS idx_rel_chapter ON relationship_history(chapter_no);
            CREATE INDEX IF NOT EXISTS idx_commit_book ON commit_log(book_id);
            CREATE INDEX IF NOT EXISTS idx_commit_chapter ON commit_log(chapter_no);
        """)
        self.conn.commit()

    # ===== Commit Management =====

    def generate_commit_id(self, book_id: str, chapter_no: str) -> str:
        """Generate deterministic commit ID."""
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(namespace, f"{book_id}:{chapter_no}"))

    def begin_commit(self, book_id: str, chapter_no: str, payload: dict) -> str:
        """Start a new commit transaction."""
        commit_id = self.generate_commit_id(book_id, chapter_no)
        self.conn.execute("""
            INSERT OR REPLACE INTO commit_log (commit_id, book_id, chapter_no, commit_type, payload_json, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (commit_id, book_id, chapter_no, "CHAPTER_PROCESS", json.dumps(payload), "STARTED"))
        self.conn.commit()
        return commit_id

    def mark_commit_status(self, commit_id: str, status: str):
        """Update commit status."""
        completed_at = datetime.now().isoformat() if status in ["ALL_DONE", "FAILED"] else None
        self.conn.execute("""
            UPDATE commit_log SET status = ?, completed_at = ? WHERE commit_id = ?
        """, (status, completed_at, commit_id))
        self.conn.commit()

    # ===== Entity Management =====

    def normalize_entity(self, name: str, entity_type: str, chapter_no: str,
                        aliases: list[str] | None = None) -> str:
        """Normalize entity name to entity_id (alias resolution).

        Checks canonical_name, then aliases. Creates new entity if not found.
        If aliases provided, merges them into existing entity.
        """
        # Check if name matches canonical name
        cursor = self.conn.execute("""
            SELECT entity_id FROM entity_registry
            WHERE type = ? AND canonical_name = ?
        """, (entity_type, name))
        row = cursor.fetchone()
        if row:
            entity_id = row["entity_id"]
            if aliases:
                self._merge_aliases(entity_id, aliases)
            self._update_last_seen(entity_id, chapter_no)
            return entity_id

        # Check if name is in aliases
        cursor = self.conn.execute("""
            SELECT entity_id, aliases_json FROM entity_registry WHERE type = ?
        """, (entity_type,))
        for row in cursor:
            existing_aliases = json.loads(row["aliases_json"] or "[]")
            if name in existing_aliases:
                entity_id = row["entity_id"]
                if aliases:
                    self._merge_aliases(entity_id, aliases)
                self._update_last_seen(entity_id, chapter_no)
                return entity_id

        # Check if any provided alias matches an existing entity
        if aliases:
            for alias in aliases:
                cursor = self.conn.execute("""
                    SELECT entity_id FROM entity_registry
                    WHERE type = ? AND (canonical_name = ? OR aliases_json LIKE ?)
                """, (entity_type, alias, f'%"{alias}"%'))
                row = cursor.fetchone()
                if row:
                    entity_id = row["entity_id"]
                    self._merge_aliases(entity_id, [name] + aliases)
                    self._update_last_seen(entity_id, chapter_no)
                    return entity_id

        # Create new entity
        entity_id = f"{entity_type}_{uuid.uuid4().hex[:8]}"
        self.conn.execute("""
            INSERT INTO entity_registry (entity_id, type, canonical_name, aliases_json, first_seen_chapter, last_seen_chapter)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (entity_id, entity_type, name, json.dumps(aliases or []), chapter_no, chapter_no))
        self.conn.commit()
        return entity_id

    def _merge_aliases(self, entity_id: str, new_aliases: list[str]):
        """Merge new aliases into existing entity."""
        cursor = self.conn.execute("""
            SELECT canonical_name, aliases_json FROM entity_registry WHERE entity_id = ?
        """, (entity_id,))
        row = cursor.fetchone()
        if not row:
            return
        canonical = row["canonical_name"]
        existing = set(json.loads(row["aliases_json"] or "[]"))
        for alias in new_aliases:
            if alias != canonical and alias not in existing:
                existing.add(alias)
        self.conn.execute("""
            UPDATE entity_registry SET aliases_json = ? WHERE entity_id = ?
        """, (json.dumps(list(existing)), entity_id))
        self.conn.commit()

    def _update_last_seen(self, entity_id: str, chapter_no: str):
        """Update last_seen_chapter for entity."""
        self.conn.execute("""
            UPDATE entity_registry SET last_seen_chapter = ?
            WHERE entity_id = ? AND (last_seen_chapter IS NULL OR last_seen_chapter < ?)
        """, (chapter_no, entity_id, chapter_no))
        self.conn.commit()

    def merge_entities(self, keep_id: str, merge_id: str):
        """Merge two entities: keep one, mark other as merged.

        Moves all aliases from merge_id to keep_id, updates merged_into.
        """
        # Get merge entity info
        cursor = self.conn.execute("""
            SELECT canonical_name, aliases_json FROM entity_registry WHERE entity_id = ?
        """, (merge_id,))
        merge_row = cursor.fetchone()
        if not merge_row:
            return

        # Add merge entity's name and aliases to keep entity
        merge_aliases = [merge_row["canonical_name"]] + json.loads(merge_row["aliases_json"] or "[]")
        self._merge_aliases(keep_id, merge_aliases)

        # Mark as merged
        self.conn.execute("""
            UPDATE entity_registry SET merged_into = ? WHERE entity_id = ?
        """, (keep_id, merge_id))

        # Update all references in relationship_history
        self.conn.execute("""
            UPDATE relationship_history SET from_id = ? WHERE from_id = ?
        """, (keep_id, merge_id))
        self.conn.execute("""
            UPDATE relationship_history SET to_id = ? WHERE to_id = ?
        """, (keep_id, merge_id))

        # Update all references in fact_history
        self.conn.execute("""
            UPDATE fact_history SET subject_id = ? WHERE subject_id = ?
        """, (keep_id, merge_id))

        self.conn.commit()

    def register_entity(self, entity_id: str, entity_type: str, canonical_name: str,
                       aliases: list[str], chapter_no: str):
        """Register new entity in registry."""
        self.conn.execute("""
            INSERT OR REPLACE INTO entity_registry
            (entity_id, type, canonical_name, aliases_json, first_seen_chapter, last_seen_chapter)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (entity_id, entity_type, canonical_name, json.dumps(aliases), chapter_no, chapter_no))
        self.conn.commit()

    # ===== Fact History =====

    def append_fact_history(self, commit_id: str, facts: list[dict]):
        """Append facts to history."""
        for fact in facts:
            fact_id = f"fact_{uuid.uuid4().hex[:12]}"
            self.conn.execute("""
                INSERT INTO fact_history
                (fact_id, commit_id, chapter_no, subject_id, predicate, object_json,
                 op, valid_from, valid_to, tier, status, confidence, evidence_chunk_id, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact_id, commit_id, fact["chapter_no"], fact["subject_id"],
                fact["predicate"], json.dumps(fact["value"]), fact["op"],
                fact.get("valid_from"), fact.get("valid_to"),
                fact.get("tier", "SOFT_NOTE"), fact.get("status", "confirmed"),
                fact.get("confidence", 1.0), fact.get("evidence_chunk_id"),
                fact.get("source", "extractor")
            ))
        self.conn.commit()

    def append_relationship_history(self, commit_id: str, relations: list[dict]):
        """Append relationships to history."""
        for rel in relations:
            rel_hist_id = f"rel_{uuid.uuid4().hex[:12]}"
            self.conn.execute("""
                INSERT INTO relationship_history
                (rel_hist_id, commit_id, chapter_no, from_id, to_id, kind,
                 op, valid_from, valid_to, status, confidence, evidence_chunk_id, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rel_hist_id, commit_id, rel["chapter_no"], rel["from_id"],
                rel["to_id"], rel["kind"], rel["op"],
                rel.get("valid_from"), rel.get("valid_to"),
                rel.get("status", "confirmed"), rel.get("confidence", 1.0),
                rel.get("evidence_chunk_id"), rel.get("source", "extractor")
            ))
        self.conn.commit()

    # ===== Current State Updates =====

    def update_current_snapshots(self, commit_id: str):
        """Update current state tables from ALL fact history (not just this commit).

        Aggregates the latest value for each (subject_id, predicate) pair
        across all commits up to and including this one.
        """
        # Get commit info
        cursor = self.conn.execute("""
            SELECT chapter_no FROM commit_log WHERE commit_id = ?
        """, (commit_id,))
        row = cursor.fetchone()
        if not row:
            return
        chapter_no = row["chapter_no"]

        # Get all entities that have facts in this commit
        cursor = self.conn.execute("""
            SELECT DISTINCT subject_id FROM fact_history WHERE commit_id = ?
        """, (commit_id,))
        entity_ids = [row["subject_id"] for row in cursor]

        for entity_id in entity_ids:
            # Get latest value for each predicate (across ALL commits)
            facts_cursor = self.conn.execute("""
                SELECT predicate, object_json, tier, chapter_no
                FROM fact_history
                WHERE subject_id = ?
                  AND op != 'DELETE'
                ORDER BY created_at ASC
            """, (entity_id,))

            state = {}
            status_tags = []
            for fact_row in facts_cursor:
                pred = fact_row["predicate"]
                val = json.loads(fact_row["object_json"])
                state[pred] = val

                # Track status tags for HARD_STATE facts
                if fact_row["tier"] == "HARD_STATE" and pred == "status":
                    status_tags = [val] if isinstance(val, str) else val

            self.conn.execute("""
                INSERT OR REPLACE INTO character_current
                (entity_id, state_json, status_tags_json, updated_chapter, updated_commit)
                VALUES (?, ?, ?, ?, ?)
            """, (entity_id, json.dumps(state, ensure_ascii=False),
                  json.dumps(status_tags, ensure_ascii=False), chapter_no, commit_id))

        self.conn.commit()

    # ===== Conflict Detection =====

    def detect_conflicts(self, chapter_no: str, proposed_facts: list[dict],
                        proposed_relations: list[dict]) -> dict:
        """Detect conflicts in proposed changes."""
        blocking = []
        warnings = []

        for fact in proposed_facts:
            subject_id = fact["subject_id"]
            predicate = fact["predicate"]
            value = fact["value"]

            # Check if character is dead
            if predicate == "appears":
                cursor = self.conn.execute("""
                    SELECT object_json FROM fact_history
                    WHERE subject_id = ? AND predicate = 'status'
                    ORDER BY created_at DESC LIMIT 1
                """, (subject_id,))
                row = cursor.fetchone()
                if row:
                    status = json.loads(row["object_json"])
                    if status == "dead":
                        blocking.append({
                            "type": "dead_character_appears",
                            "entity_id": subject_id,
                            "message": f"Character {subject_id} is dead but appears in chapter {chapter_no}"
                        })

            # Check item ownership conflicts
            if predicate == "owner":
                cursor = self.conn.execute("""
                    SELECT owner_id FROM item_current WHERE entity_id = ?
                """, (subject_id,))
                row = cursor.fetchone()
                if row and row["owner_id"] and row["owner_id"] != value:
                    warnings.append({
                        "type": "item_ownership_change",
                        "entity_id": subject_id,
                        "old_owner": row["owner_id"],
                        "new_owner": value,
                        "message": f"Item {subject_id} ownership changed from {row['owner_id']} to {value}"
                    })

        # Check relationship contradictions
        for rel in proposed_relations:
            if rel["kind"] in ["ENEMY", "ALLY"]:
                cursor = self.conn.execute("""
                    SELECT kind FROM relationship_history
                    WHERE from_id = ? AND to_id = ? AND valid_to IS NULL
                    ORDER BY created_at DESC LIMIT 1
                """, (rel["from_id"], rel["to_id"]))
                row = cursor.fetchone()
                if row and row["kind"] != rel["kind"]:
                    warnings.append({
                        "type": "relationship_change",
                        "from_id": rel["from_id"],
                        "to_id": rel["to_id"],
                        "old_kind": row["kind"],
                        "new_kind": rel["kind"],
                        "message": f"Relationship changed from {row['kind']} to {rel['kind']}"
                    })

        return {"blocking": blocking, "warnings": warnings}

    # ===== Query Methods =====

    def get_character_state(self, entity_id: str) -> Optional[dict]:
        """Get current character state."""
        cursor = self.conn.execute("""
            SELECT state_json, updated_chapter FROM character_current WHERE entity_id = ?
        """, (entity_id,))
        row = cursor.fetchone()
        if row:
            return {
                "state": json.loads(row["state_json"]),
                "updated_chapter": row["updated_chapter"]
            }
        return None

    def get_all_entities(self, entity_type: Optional[str] = None) -> list[dict]:
        """Get all entities of a type."""
        if entity_type:
            cursor = self.conn.execute("""
                SELECT entity_id, canonical_name, aliases_json FROM entity_registry WHERE type = ?
            """, (entity_type,))
        else:
            cursor = self.conn.execute("""
                SELECT entity_id, type, canonical_name, aliases_json FROM entity_registry
            """)
        return [
            {
                "entity_id": row["entity_id"],
                "type": row["type"] if "type" in row.keys() else None,
                "canonical_name": row["canonical_name"],
                "aliases": json.loads(row["aliases_json"] or "[]")
            }
            for row in cursor
        ]

    def get_commit_status(self, commit_id: str) -> Optional[dict]:
        """Get commit status."""
        cursor = self.conn.execute("""
            SELECT book_id, chapter_no, status, created_at, completed_at
            FROM commit_log WHERE commit_id = ?
        """, (commit_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_statistics(self) -> dict:
        """Get database statistics for validation."""
        stats = {}
        for table in ["entity_registry", "character_current", "fact_history",
                       "relationship_history", "address_book", "thread_current", "commit_log"]:
            cursor = self.conn.execute(f"SELECT COUNT(*) as cnt FROM {table}")
            stats[table] = cursor.fetchone()["cnt"]

        # Entity type breakdown
        cursor = self.conn.execute("""
            SELECT type, COUNT(*) as cnt FROM entity_registry
            WHERE merged_into IS NULL GROUP BY type
        """)
        stats["entity_types"] = {row["type"]: row["cnt"] for row in cursor}

        # Relationship kind breakdown
        cursor = self.conn.execute("""
            SELECT kind, COUNT(*) as cnt FROM relationship_history GROUP BY kind
        """)
        stats["relation_kinds"] = {row["kind"]: row["cnt"] for row in cursor}

        # Entities with aliases
        cursor = self.conn.execute("""
            SELECT COUNT(*) as cnt FROM entity_registry
            WHERE aliases_json != '[]' AND aliases_json IS NOT NULL AND merged_into IS NULL
        """)
        stats["entities_with_aliases"] = cursor.fetchone()["cnt"]

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Example usage
    db = CanonDBV2("~/.nanobot/workspace/canon_v2.db")

    # Start commit
    commit_id = db.begin_commit("book1", "ch001", {"test": "data"})
    print(f"Commit ID: {commit_id}")

    # Register entity
    db.register_entity("char_001", "character", "Alice", ["小艾"], "ch001")

    # Add facts
    db.append_fact_history(commit_id, [
        {
            "chapter_no": "ch001",
            "subject_id": "char_001",
            "predicate": "status",
            "value": "active",
            "op": "INSERT",
            "valid_from": "ch001"
        }
    ])

    # Update snapshots
    db.update_current_snapshots(commit_id)
    db.mark_commit_status(commit_id, "ALL_DONE")

    # Query
    state = db.get_character_state("char_001")
    print(f"Character state: {state}")

    db.close()

