"""Chapter processor that coordinates all three memory tiers.

Processes chapters and updates Qdrant (recall), Neo4j (structural), and Canon DB (authoritative).
Uses name_normalizer for entity cleaning and delta_converter for structured delta extraction.
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Optional
from neo4j_manager import Neo4jManager
from canon_db_v2 import CanonDBV2
from delta_converter import convert_assets_to_delta, load_and_convert

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

    def _resolve_name_to_id(self, name: str, entity_type: str, chapter_no: str,
                            aliases: list[str] | None = None) -> str | None:
        """Resolve entity name to entity_id via Canon DB registry."""
        if not name:
            return None
        return self.canon_db.normalize_entity(name, entity_type, chapter_no, aliases)

    def process_chapter(self, book_id: str, chapter_no: str, delta: dict,
                        chapter_title: str = "", chapter_summary: str = "",
                        chapter_text: str = "") -> dict:
        """Process a chapter delta through all three memory tiers.

        Args:
            book_id: Book identifier
            chapter_no: Chapter number string (e.g., "0001")
            delta: Structured delta from delta_converter
            chapter_title: Optional chapter title
            chapter_summary: Optional chapter summary
            chapter_text: Optional full chapter text (for chunking)

        Returns:
            dict with processing results and statistics
        """
        results = {"entities": 0, "facts": 0, "relations": 0, "events": 0, "errors": []}

        # --- Step 1: Begin Canon DB commit ---
        compact_payload = {
            "entities": len(delta.get("entities_new", [])),
            "facts": len(delta.get("fact_changes", [])),
            "relations": len(delta.get("relations_delta", [])),
            "events": len(delta.get("events", [])),
        }
        commit_id = self.canon_db.begin_commit(book_id, chapter_no, compact_payload)
        logger.info(f"[{chapter_no}] Commit {commit_id[:8]}... started")

        try:
            # --- Step 2: Register entities and build name→id lookup ---
            name_to_id = {}
            for ent in delta.get("entities_new", []):
                ent_name = ent["name"]
                ent_type = ent.get("type", "character")
                aliases = ent.get("aliases", [])
                entity_id = self._resolve_name_to_id(ent_name, ent_type, chapter_no, aliases)
                if entity_id:
                    name_to_id[ent_name] = entity_id
                    for alias in aliases:
                        name_to_id[alias] = entity_id
                    results["entities"] += 1

                    # Write to Neo4j
                    if ent_type == "character":
                        self.neo4j.upsert_character(
                            entity_id=entity_id,
                            canonical_name=ent_name,
                            aliases=aliases,
                            traits=ent.get("traits", {}),
                            status=ent.get("status", "active"),
                            commit_id=commit_id,
                        )

            logger.info(f"[{chapter_no}] Registered {results['entities']} entities")

            # --- Step 3: Write fact_changes to Canon DB ---
            resolved_facts = []
            for fact in delta.get("fact_changes", []):
                subject_name = fact.get("subject_name", "")
                subject_id = name_to_id.get(subject_name)
                if not subject_id:
                    subject_id = self._resolve_name_to_id(subject_name, "character", chapter_no)
                if not subject_id:
                    continue
                resolved_facts.append({
                    "chapter_no": chapter_no,
                    "subject_id": subject_id,
                    "predicate": fact["predicate"],
                    "value": fact["value"],
                    "op": fact.get("op", "INSERT"),
                    "valid_from": fact.get("valid_from", chapter_no),
                    "tier": fact.get("tier", "SOFT_NOTE"),
                    "status": fact.get("status", "confirmed"),
                })

            if resolved_facts:
                self.canon_db.append_fact_history(commit_id, resolved_facts)
                results["facts"] = len(resolved_facts)
            logger.info(f"[{chapter_no}] Wrote {results['facts']} facts")

            # --- Step 4: Write relations to Canon DB and Neo4j ---
            resolved_rels = []
            for rel in delta.get("relations_delta", []):
                from_name = rel.get("from_name", "")
                to_name = rel.get("to_name", "")
                from_id = name_to_id.get(from_name)
                to_id = name_to_id.get(to_name)
                if not from_id:
                    from_id = self._resolve_name_to_id(from_name, "character", chapter_no)
                if not to_id:
                    to_id = self._resolve_name_to_id(to_name, "character", chapter_no)
                if not from_id or not to_id or from_id == to_id:
                    continue

                resolved_rels.append({
                    "chapter_no": chapter_no,
                    "from_id": from_id,
                    "to_id": to_id,
                    "kind": rel["kind"],
                    "op": rel.get("op", "INSERT"),
                    "valid_from": rel.get("valid_from", chapter_no),
                    "status": rel.get("status", "confirmed"),
                })

                # Write to Neo4j
                try:
                    self.neo4j.upsert_relation(
                        from_id=from_id, to_id=to_id,
                        kind=rel["kind"], status=rel.get("status", "confirmed"),
                        valid_from=rel.get("valid_from", chapter_no),
                        valid_to=None, evidence_chunk_id=None,
                        commit_id=commit_id,
                    )
                except Exception as e:
                    logger.warning(f"[{chapter_no}] Neo4j relation error: {e}")

            if resolved_rels:
                self.canon_db.append_relationship_history(commit_id, resolved_rels)
                results["relations"] = len(resolved_rels)
            logger.info(f"[{chapter_no}] Wrote {results['relations']} relations")

            # --- Step 5: Write events to Neo4j ---
            # Create chapter node first
            self.neo4j.create_chapter(book_id, chapter_no, chapter_title, None, chapter_summary)

            for event in delta.get("events", []):
                participant_ids = []
                for p_name in event.get("participants", []):
                    p_id = name_to_id.get(p_name)
                    if not p_id:
                        p_id = self._resolve_name_to_id(p_name, "character", chapter_no)
                    if p_id:
                        participant_ids.append(p_id)

                try:
                    self.neo4j.create_event(
                        event_id=event["event_id"],
                        event_type=event.get("type", "plot_beat"),
                        summary=event.get("summary", ""),
                        chapter_no=chapter_no,
                        participants=participant_ids,
                        location_id=None,
                        commit_id=commit_id,
                    )
                    results["events"] += 1
                except Exception as e:
                    logger.warning(f"[{chapter_no}] Neo4j event error: {e}")

            logger.info(f"[{chapter_no}] Wrote {results['events']} events")

            # --- Step 6: Write hooks/threads ---
            for hook in delta.get("hooks", []):
                thread_id = f"thread_{uuid.uuid4().hex[:8]}"
                hook_id = f"hook_{uuid.uuid4().hex[:8]}"
                try:
                    self.neo4j.create_thread(thread_id, hook["name"], "open",
                                            hook.get("priority", 1), None)
                    self.neo4j.create_hook(hook_id, hook["summary"], chapter_no,
                                          thread_id, None)
                except Exception as e:
                    logger.warning(f"[{chapter_no}] Neo4j hook error: {e}")

            # --- Step 7: Update current snapshots and finalize ---
            self.canon_db.update_current_snapshots(commit_id)
            self.canon_db.mark_commit_status(commit_id, "ALL_DONE")
            logger.info(f"[{chapter_no}] Commit {commit_id[:8]}... ALL_DONE")

        except Exception as e:
            logger.error(f"[{chapter_no}] Processing failed: {e}")
            self.canon_db.mark_commit_status(commit_id, "FAILED")
            results["errors"].append(str(e))
            raise

        return results

    def process_from_assets(self, book_id: str, asset_path: str | Path,
                           chapter_no: str, **kwargs) -> dict:
        """Convenience: load asset file, convert to delta, and process.

        Args:
            book_id: Book identifier
            asset_path: Path to asset JSON file
            chapter_no: Chapter number string
            **kwargs: Passed to process_chapter (chapter_title, chapter_summary, etc.)
        """
        delta = load_and_convert(asset_path, chapter_no)
        return self.process_chapter(book_id, chapter_no, delta, **kwargs)
