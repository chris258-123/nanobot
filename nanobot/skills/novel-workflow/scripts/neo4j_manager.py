"""Neo4j graph database manager for novel structural memory.

Manages entities, relationships, events, and foreshadowing threads.
"""

from neo4j import GraphDatabase
from typing import Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class Neo4jManager:
    """Neo4j manager for novel structural memory (graph layer)."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self._init_schema()

    def close(self):
        """Close Neo4j driver."""
        self.driver.close()

    def _init_schema(self):
        """Initialize constraints and indexes."""
        with self.driver.session(database=self.database) as session:
            # Constraints for uniqueness
            constraints = [
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
                "CREATE CONSTRAINT chapter_id_unique IF NOT EXISTS FOR (c:Chapter) REQUIRE (c.book_id, c.chapter_no) IS UNIQUE",
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (ck:Chunk) REQUIRE ck.chunk_id IS UNIQUE",
                "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (ev:Event) REQUIRE ev.event_id IS UNIQUE",
                "CREATE CONSTRAINT thread_id_unique IF NOT EXISTS FOR (t:Thread) REQUIRE t.thread_id IS UNIQUE"
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")

            # Indexes for performance
            indexes = [
                "CREATE INDEX chapter_no_idx IF NOT EXISTS FOR (c:Chapter) ON (c.chapter_no)",
                "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX valid_from_idx IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.valid_from)",
                "CREATE INDEX valid_to_idx IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.valid_to)"
            ]

            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

    # ===== Entity Layer =====

    def upsert_character(self, entity_id: str, canonical_name: str, aliases: list[str],
                        traits: dict, status: str, commit_id: str) -> None:
        """Create or update character entity."""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (e:Entity:Character {entity_id: $entity_id})
                SET e.canonical_name = $canonical_name,
                    e.aliases = $aliases,
                    e.traits_json = $traits_json,
                    e.status = $status,
                    e.type = 'character',
                    e.updated_commit = $commit_id,
                    e.updated_at = datetime()
            """, entity_id=entity_id, canonical_name=canonical_name, aliases=aliases,
                traits_json=json.dumps(traits), status=status, commit_id=commit_id)

    def upsert_location(self, entity_id: str, name: str, level: str,
                       parent_id: Optional[str], description: str, commit_id: str) -> None:
        """Create or update location entity."""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (e:Entity:Location {entity_id: $entity_id})
                SET e.canonical_name = $name,
                    e.level = $level,
                    e.description = $description,
                    e.type = 'location',
                    e.updated_commit = $commit_id,
                    e.updated_at = datetime()
            """, entity_id=entity_id, name=name, level=level,
                description=description, commit_id=commit_id)

            if parent_id:
                session.run("""
                    MATCH (child:Location {entity_id: $entity_id})
                    MATCH (parent:Location {entity_id: $parent_id})
                    MERGE (child)-[:PART_OF]->(parent)
                """, entity_id=entity_id, parent_id=parent_id)

    def upsert_item(self, entity_id: str, name: str, owner_id: Optional[str],
                   powers: list[str], limits: list[str], commit_id: str) -> None:
        """Create or update item entity."""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (e:Entity:Item {entity_id: $entity_id})
                SET e.canonical_name = $name,
                    e.powers = $powers,
                    e.limits = $limits,
                    e.type = 'item',
                    e.updated_commit = $commit_id,
                    e.updated_at = datetime()
            """, entity_id=entity_id, name=name, powers=powers,
                limits=limits, commit_id=commit_id)

            if owner_id:
                session.run("""
                    MATCH (item:Item {entity_id: $entity_id})
                    MATCH (owner:Character {entity_id: $owner_id})
                    MERGE (owner)-[:OWNS]->(item)
                """, entity_id=entity_id, owner_id=owner_id)

    # ===== Chapter and Chunk Layer =====

    def create_chapter(self, book_id: str, chapter_no: str, title: str,
                      pov: Optional[str], summary: str) -> None:
        """Create chapter node."""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (c:Chapter {book_id: $book_id, chapter_no: $chapter_no})
                SET c.title = $title,
                    c.pov = $pov,
                    c.summary = $summary,
                    c.created_at = datetime()
            """, book_id=book_id, chapter_no=chapter_no, title=title,
                pov=pov, summary=summary)

    def create_chunks(self, chapter_no: str, chunks: list[dict]) -> None:
        """Create chunk nodes and link to chapter."""
        with self.driver.session(database=self.database) as session:
            for chunk in chunks:
                session.run("""
                    MATCH (c:Chapter {chapter_no: $chapter_no})
                    CREATE (ck:Chunk {
                        chunk_id: $chunk_id,
                        text: $text,
                        start_pos: $start_pos,
                        end_pos: $end_pos
                    })
                    CREATE (c)-[:HAS_CHUNK]->(ck)
                """, chapter_no=chapter_no, chunk_id=chunk["chunk_id"],
                    text=chunk["text"], start_pos=chunk["start_pos"],
                    end_pos=chunk["end_pos"])

    # ===== Relationship Layer =====

    def upsert_relation(self, from_id: str, to_id: str, kind: str, status: str,
                       valid_from: str, valid_to: Optional[str],
                       evidence_chunk_id: Optional[str], commit_id: str) -> None:
        """Create or update relationship with validity period."""
        with self.driver.session(database=self.database) as session:
            # Close old relationships if new one starts
            if valid_to is None:
                session.run("""
                    MATCH (a:Entity {entity_id: $from_id})-[r:RELATES]->(b:Entity {entity_id: $to_id})
                    WHERE r.kind = $kind AND r.valid_to IS NULL
                    SET r.valid_to = $valid_from
                """, from_id=from_id, to_id=to_id, kind=kind, valid_from=valid_from)

            # Create new relationship
            session.run("""
                MATCH (a:Entity {entity_id: $from_id})
                MATCH (b:Entity {entity_id: $to_id})
                CREATE (a)-[r:RELATES {
                    kind: $kind,
                    status: $status,
                    valid_from: $valid_from,
                    valid_to: $valid_to,
                    evidence_chunk_id: $evidence_chunk_id,
                    commit_id: $commit_id,
                    created_at: datetime()
                }]->(b)
            """, from_id=from_id, to_id=to_id, kind=kind, status=status,
                valid_from=valid_from, valid_to=valid_to,
                evidence_chunk_id=evidence_chunk_id, commit_id=commit_id)

    # ===== Event Layer =====

    def create_event(self, event_id: str, event_type: str, summary: str,
                    chapter_no: str, participants: list[str],
                    location_id: Optional[str], commit_id: str) -> None:
        """Create event node with relationships."""
        with self.driver.session(database=self.database) as session:
            # Create event
            session.run("""
                CREATE (ev:Event {
                    event_id: $event_id,
                    type: $event_type,
                    summary: $summary,
                    commit_id: $commit_id,
                    created_at: datetime()
                })
            """, event_id=event_id, event_type=event_type, summary=summary,
                commit_id=commit_id)

            # Link to chapter
            session.run("""
                MATCH (ev:Event {event_id: $event_id})
                MATCH (c:Chapter {chapter_no: $chapter_no})
                CREATE (ev)-[:OCCURS_IN]->(c)
            """, event_id=event_id, chapter_no=chapter_no)

            # Link participants
            for participant_id in participants:
                session.run("""
                    MATCH (ev:Event {event_id: $event_id})
                    MATCH (p:Character {entity_id: $participant_id})
                    CREATE (p)-[:PARTICIPATES_IN]->(ev)
                """, event_id=event_id, participant_id=participant_id)

            # Link location
            if location_id:
                session.run("""
                    MATCH (ev:Event {event_id: $event_id})
                    MATCH (loc:Location {entity_id: $location_id})
                    CREATE (ev)-[:HAPPENS_AT]->(loc)
                """, event_id=event_id, location_id=location_id)

    # ===== Thread Layer =====

    def create_thread(self, thread_id: str, name: str, status: str,
                     priority: int, planned_window: Optional[str]) -> None:
        """Create foreshadowing thread."""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (t:Thread {thread_id: $thread_id})
                SET t.name = $name,
                    t.status = $status,
                    t.priority = $priority,
                    t.planned_window = $planned_window,
                    t.created_at = datetime()
            """, thread_id=thread_id, name=name, status=status,
                priority=priority, planned_window=planned_window)

    def create_hook(self, hook_id: str, summary: str, chapter_no: str,
                   thread_id: str, evidence_chunk_id: Optional[str]) -> None:
        """Create hook and link to thread."""
        with self.driver.session(database=self.database) as session:
            session.run("""
                CREATE (h:Hook {
                    hook_id: $hook_id,
                    summary: $summary,
                    evidence_chunk_id: $evidence_chunk_id,
                    created_at: datetime()
                })
            """, hook_id=hook_id, summary=summary, evidence_chunk_id=evidence_chunk_id)

            session.run("""
                MATCH (h:Hook {hook_id: $hook_id})
                MATCH (c:Chapter {chapter_no: $chapter_no})
                MATCH (t:Thread {thread_id: $thread_id})
                CREATE (h)-[:OCCURS_IN]->(c)
                CREATE (h)-[:SETS_UP]->(t)
            """, hook_id=hook_id, chapter_no=chapter_no, thread_id=thread_id)

    # ===== Query Methods =====

    def get_all_entities(self, entity_type: str | None = None) -> list[dict]:
        """Get all entity nodes."""
        with self.driver.session(database=self.database) as session:
            if entity_type:
                result = session.run("""
                    MATCH (e:Entity {type: $type})
                    RETURN e.entity_id as entity_id, e.canonical_name as name,
                           e.type as type, e.aliases as aliases
                """, type=entity_type)
            else:
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.entity_id as entity_id, e.canonical_name as name,
                           e.type as type, e.aliases as aliases
                """)
            return [dict(record) for record in result]

    def get_relationship_stats(self) -> dict:
        """Get relationship statistics by type."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH ()-[r:RELATES]->()
                RETURN r.kind as kind, count(r) as count
                ORDER BY count DESC
            """)
            return {record["kind"]: record["count"] for record in result}

    def get_statistics(self) -> dict:
        """Get full graph statistics."""
        with self.driver.session(database=self.database) as session:
            stats = {}

            result = session.run("MATCH (e:Entity) RETURN count(e) as cnt")
            stats["entities"] = result.single()["cnt"]

            result = session.run("MATCH (c:Chapter) RETURN count(c) as cnt")
            stats["chapters"] = result.single()["cnt"]

            result = session.run("MATCH (ck:Chunk) RETURN count(ck) as cnt")
            stats["chunks"] = result.single()["cnt"]

            result = session.run("MATCH (ev:Event) RETURN count(ev) as cnt")
            stats["events"] = result.single()["cnt"]

            result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as cnt")
            stats["relations"] = result.single()["cnt"]

            result = session.run("MATCH (t:Thread) RETURN count(t) as cnt")
            stats["threads"] = result.single()["cnt"]

            # Relationship type distribution
            stats["relation_kinds"] = self.get_relationship_stats()

            # Entity type distribution
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.type as type, count(e) as cnt
            """)
            stats["entity_types"] = {r["type"]: r["cnt"] for r in result}

            return stats

    def clear_all(self):
        """Clear all data from Neo4j (for re-processing)."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def get_character_state(self, entity_id: str, as_of_chapter: Optional[str] = None) -> Optional[dict]:
        """Get character state at specific chapter."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Character {entity_id: $entity_id})
                RETURN e.canonical_name as name, e.traits_json as traits_json,
                       e.status as status, e.aliases as aliases
            """, entity_id=entity_id)
            record = result.single()
            if record:
                data = dict(record)
                # Parse traits_json back to dict
                if data.get("traits_json"):
                    data["traits"] = json.loads(data["traits_json"])
                    del data["traits_json"]
                return data
            return None

    def get_active_relations(self, entity_id: str, as_of_chapter: str) -> list[dict]:
        """Get active relationships at specific chapter."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (a:Entity {entity_id: $entity_id})-[r:RELATES]->(b:Entity)
                WHERE r.valid_from <= $as_of_chapter
                  AND (r.valid_to IS NULL OR r.valid_to > $as_of_chapter)
                RETURN b.entity_id as to_id, b.canonical_name as to_name,
                       r.kind as kind, r.status as status, r.valid_from as since
            """, entity_id=entity_id, as_of_chapter=as_of_chapter)
            return [dict(record) for record in result]

    def get_unresolved_threads(self, book_id: str) -> list[dict]:
        """Get open foreshadowing threads."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (t:Thread)
                WHERE t.status = 'open'
                OPTIONAL MATCH (h:Hook)-[:SETS_UP]->(t)
                RETURN t.thread_id as thread_id, t.name as name,
                       t.priority as priority, count(h) as hook_count
                ORDER BY t.priority DESC
            """)
            return [dict(record) for record in result]


if __name__ == "__main__":
    # Example usage
    manager = Neo4jManager(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="novel123"
    )

    # Create a test character
    manager.upsert_character(
        entity_id="char_001",
        canonical_name="Alice",
        aliases=["小艾", "艾丽丝"],
        traits={"personality": "brave", "age": 25},
        status="active",
        commit_id="commit_001"
    )

    # Query character
    state = manager.get_character_state("char_001")
    print(f"Character state: {state}")

    manager.close()

