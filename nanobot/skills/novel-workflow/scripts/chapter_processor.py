"""Chapter processor that coordinates all three memory tiers.

Processes chapters and updates Qdrant (recall), Neo4j (structural), and Canon DB (authoritative).
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Optional
from neo4j_manager import Neo4jManager
from canon_db_v2 import CanonDBV2

logger = logging.getLogger(__name__)


class ChapterProcessor:
    """Orchestrates chapter processing across all three memory tiers."""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pass: str,
                 canon_db_path: str, qdrant_url: Optional[str] = None):
        self.neo4j = Neo4jManager(neo4j_uri, neo4j_user, neo4j_pass)
        self.canon_db = CanonDBV2(canon_db_path)
        self.qdrant_url = qdrant_url

    def close(self):
        """Close all connections."""
        self.neo4j.close()
        self.canon_db.close()

    def create_chunks(self, chapter_text: str, chapter_no: str, chunk_size: int = 500) -> list[dict]:
        """Split chapter into chunks with stable IDs."""
        chunks = []
        words = chapter_text.split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunk_id = f"{chapter_no}#c{i // chunk_size:02d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_pos": i,
                "end_pos": min(i + chunk_size, len(words))
            })
        return chunks

    def process_chapter(self, book_id: str, chapter_no: str, chapter_text: str,
                       title: str, pov: Optional[str], delta: dict) -> dict:
        """Process a chapter and update all three memory tiers.

        Args:
            book_id: Book identifier
            chapter_no: Chapter number (e.g., "ch001")
            chapter_text: Full chapter text
            title: Chapter title
            pov: Point of view character
            delta: Delta JSON with entities, events, relations, facts

        Returns:
            Result dict with status and commit_id
        """
        logger.info(f"Processing {book_id} {chapter_no}")

        # Step 1: Create chunks
        chunks = self.create_chunks(chapter_text, chapter_no)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 2: Normalize entities in delta and build ID mapping
        entity_id_map = {}  # old_id -> new_id mapping
        for entity in delta.get("entities_new", []):
            entity_id = self.canon_db.normalize_entity(
                entity["name"], entity["type"], chapter_no
            )
            # Store mapping if entity had an old ID
            if "entity_id" in entity:
                entity_id_map[entity["entity_id"]] = entity_id
            entity["entity_id"] = entity_id
            # Also map by name for convenience
            entity_id_map[entity["name"]] = entity_id

        # Step 2.5: Build mapping for all referenced entities (not just new ones)
        # Collect all entity references from relations, facts, events
        referenced_ids = set()
        for rel in delta.get("relations_delta", []):
            if rel.get("from_id"):
                referenced_ids.add(rel["from_id"])
            if rel.get("to_id"):
                referenced_ids.add(rel["to_id"])
        for fact in delta.get("fact_changes", []):
            if fact.get("subject_id"):
                referenced_ids.add(fact["subject_id"])
        for event in delta.get("events", []):
            for p in event.get("participants", []):
                referenced_ids.add(p)

        # For each referenced ID, try to find the actual entity_id
        for ref_id in referenced_ids:
            if ref_id not in entity_id_map:
                # Try to find by name (assuming ref_id might be a name)
                # This is a heuristic - in real usage, the delta should use proper IDs
                all_entities = self.canon_db.get_all_entities()
                for ent in all_entities:
                    if ent["canonical_name"].lower() == ref_id.lower().replace("char_", "").replace("_", " "):
                        entity_id_map[ref_id] = ent["entity_id"]
                        break
                    # Also check aliases
                    for alias in ent.get("aliases", []):
                        if alias.lower() == ref_id.lower():
                            entity_id_map[ref_id] = ent["entity_id"]
                            break

        # Step 2.6: Update all entity references in delta using the mapping
        for fact in delta.get("fact_changes", []):
            if fact.get("subject_id") in entity_id_map:
                fact["subject_id"] = entity_id_map[fact["subject_id"]]

        for rel in delta.get("relations_delta", []):
            if rel.get("from_id") in entity_id_map:
                rel["from_id"] = entity_id_map[rel["from_id"]]
            if rel.get("to_id") in entity_id_map:
                rel["to_id"] = entity_id_map[rel["to_id"]]

        for event in delta.get("events", []):
            if event.get("participants"):
                event["participants"] = [
                    entity_id_map.get(p, p) for p in event["participants"]
                ]
            if event.get("location_id") in entity_id_map:
                event["location_id"] = entity_id_map[event["location_id"]]

        # Step 3: Generate commit
        commit_id = self.canon_db.generate_commit_id(book_id, chapter_no)
        commit_payload = {
            "delta": delta,
            "chunks": [{"chunk_id": c["chunk_id"], "start": c["start_pos"], "end": c["end_pos"]}
                      for c in chunks],
            "title": title,
            "pov": pov
        }

        # Step 4: Conflict detection (preflight)
        conflicts = self.canon_db.detect_conflicts(
            chapter_no,
            delta.get("fact_changes", []),
            delta.get("relations_delta", [])
        )

        if conflicts["blocking"]:
            logger.error(f"Blocking conflicts detected: {conflicts['blocking']}")
            return {
                "status": "blocked",
                "conflicts": conflicts,
                "commit_id": commit_id
            }

        if conflicts["warnings"]:
            logger.warning(f"Warnings detected: {conflicts['warnings']}")

        # Step 5: Write to Canon DB (authoritative)
        try:
            self.canon_db.begin_commit(book_id, chapter_no, commit_payload)

            # Register new entities
            for entity in delta.get("entities_new", []):
                self.canon_db.register_entity(
                    entity["entity_id"],
                    entity["type"],
                    entity["name"],
                    entity.get("aliases", []),
                    chapter_no
                )

            # Append fact history
            fact_changes = []
            for fact in delta.get("fact_changes", []):
                fact_changes.append({
                    "chapter_no": chapter_no,
                    "subject_id": fact["subject_id"],
                    "predicate": fact["predicate"],
                    "value": fact["value"],
                    "op": fact.get("op", "INSERT"),
                    "valid_from": fact.get("valid_from", chapter_no),
                    "valid_to": fact.get("valid_to"),
                    "tier": fact.get("tier", "SOFT_NOTE"),
                    "status": fact.get("status", "confirmed"),
                    "evidence_chunk_id": fact.get("evidence_chunk_id")
                })
            self.canon_db.append_fact_history(commit_id, fact_changes)

            # Append relationship history
            relation_changes = []
            for rel in delta.get("relations_delta", []):
                relation_changes.append({
                    "chapter_no": chapter_no,
                    "from_id": rel["from_id"],
                    "to_id": rel["to_id"],
                    "kind": rel["kind"],
                    "op": rel.get("op", "INSERT"),
                    "valid_from": rel.get("valid_from", chapter_no),
                    "valid_to": rel.get("valid_to"),
                    "status": rel.get("status", "confirmed"),
                    "evidence_chunk_id": rel.get("evidence_chunk_id")
                })
            self.canon_db.append_relationship_history(commit_id, relation_changes)

            # Update current snapshots
            self.canon_db.update_current_snapshots(commit_id)
            self.canon_db.mark_commit_status(commit_id, "CANON_DONE")
            logger.info("Canon DB updated successfully")

        except Exception as e:
            logger.error(f"Canon DB update failed: {e}")
            self.canon_db.mark_commit_status(commit_id, "FAILED")
            return {"status": "failed", "error": str(e), "commit_id": commit_id}

        # Step 6: Write to Neo4j (structural)
        try:
            # Create chapter
            self.neo4j.create_chapter(book_id, chapter_no, title, pov, "")

            # Create chunks
            self.neo4j.create_chunks(chapter_no, chunks)

            # Upsert entities
            for entity in delta.get("entities_new", []):
                if entity["type"] == "character":
                    self.neo4j.upsert_character(
                        entity["entity_id"],
                        entity["name"],
                        entity.get("aliases", []),
                        entity.get("traits", {}),
                        entity.get("status", "active"),
                        commit_id
                    )
                elif entity["type"] == "location":
                    self.neo4j.upsert_location(
                        entity["entity_id"],
                        entity["name"],
                        entity.get("level", "scene"),
                        entity.get("parent_id"),
                        entity.get("description", ""),
                        commit_id
                    )
                elif entity["type"] == "item":
                    self.neo4j.upsert_item(
                        entity["entity_id"],
                        entity["name"],
                        entity.get("owner_id"),
                        entity.get("powers", []),
                        entity.get("limits", []),
                        commit_id
                    )

            # Create events
            for event in delta.get("events", []):
                event_id = event.get("event_id", f"evt_{uuid.uuid4().hex[:8]}")
                self.neo4j.create_event(
                    event_id,
                    event.get("type", "general"),
                    event.get("summary", ""),
                    chapter_no,
                    event.get("participants", []),
                    event.get("location_id"),
                    commit_id
                )

            # Upsert relations
            for rel in delta.get("relations_delta", []):
                self.neo4j.upsert_relation(
                    rel["from_id"],
                    rel["to_id"],
                    rel["kind"],
                    rel.get("status", "confirmed"),
                    rel.get("valid_from", chapter_no),
                    rel.get("valid_to"),
                    rel.get("evidence_chunk_id"),
                    commit_id
                )

            # Create threads/hooks
            for hook in delta.get("hooks", []):
                thread_id = hook.get("thread_id", f"thread_{uuid.uuid4().hex[:8]}")
                self.neo4j.create_thread(
                    thread_id,
                    hook.get("name", "Unnamed thread"),
                    "open",
                    hook.get("priority", 1),
                    hook.get("planned_window")
                )
                hook_id = f"hook_{uuid.uuid4().hex[:8]}"
                self.neo4j.create_hook(
                    hook_id,
                    hook.get("summary", ""),
                    chapter_no,
                    thread_id,
                    hook.get("evidence_chunk_id")
                )

            self.canon_db.mark_commit_status(commit_id, "NEO4J_DONE")
            logger.info("Neo4j updated successfully")

        except Exception as e:
            logger.error(f"Neo4j update failed: {e}")
            self.canon_db.mark_commit_status(commit_id, "FAILED")
            return {"status": "failed", "error": str(e), "commit_id": commit_id}

        # Step 7: Write to Qdrant (if configured)
        if self.qdrant_url:
            # TODO: Integrate with existing asset extraction and embedding
            logger.info("Qdrant integration pending (use existing embedder)")
            self.canon_db.mark_commit_status(commit_id, "ALL_DONE")
        else:
            self.canon_db.mark_commit_status(commit_id, "ALL_DONE")

        return {
            "status": "success",
            "commit_id": commit_id,
            "warnings": conflicts["warnings"],
            "chunks_created": len(chunks)
        }


if __name__ == "__main__":
    # Example usage
    processor = ChapterProcessor(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pass="novel123",
        canon_db_path="~/.nanobot/workspace/canon_v2.db"
    )

    # Test delta (manually created for Phase 1)
    test_delta = {
        "entities_new": [
            {
                "name": "Alice",
                "type": "character",
                "aliases": ["小艾"],
                "traits": {"personality": "brave"},
                "status": "active"
            }
        ],
        "events": [
            {
                "type": "meeting",
                "summary": "Alice meets Bob",
                "participants": ["char_001", "char_002"]
            }
        ],
        "relations_delta": [
            {
                "from_id": "char_001",
                "to_id": "char_002",
                "kind": "ALLY",
                "op": "INSERT"
            }
        ],
        "fact_changes": [
            {
                "subject_id": "char_001",
                "predicate": "status",
                "value": "active",
                "op": "INSERT"
            }
        ],
        "hooks": []
    }

    result = processor.process_chapter(
        book_id="test_book",
        chapter_no="ch001",
        chapter_text="This is a test chapter. Alice meets Bob in the forest.",
        title="Chapter 1: The Meeting",
        pov="Alice",
        delta=test_delta
    )

    print(json.dumps(result, indent=2))
    processor.close()
