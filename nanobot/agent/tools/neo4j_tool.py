"""Neo4j structural memory tool."""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.tools.base import Tool

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None


class Neo4jTool(Tool):
    """Query structural memory from Neo4j."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        if GraphDatabase is None:
            raise RuntimeError("neo4j package is not installed; run pip install neo4j")
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    @property
    def name(self) -> str:
        return "neo4j"

    @property
    def description(self) -> str:
        return """Neo4j structural memory queries.

Actions:
- stats: get node/edge counts
- character_state: fetch character node properties (requires: entity_id)
- active_relations: relations active at a chapter (requires: entity_id, as_of_chapter)
- unresolved_threads: list open threads
- character_subgraph: one-hop neighborhood (requires: entity_id)
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "stats",
                        "character_state",
                        "active_relations",
                        "unresolved_threads",
                        "character_subgraph",
                    ],
                },
                "entity_id": {"type": "string"},
                "as_of_chapter": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        try:
            if action == "stats":
                return self._stats()
            if action == "character_state":
                return self._character_state(kwargs.get("entity_id", ""))
            if action == "active_relations":
                return self._active_relations(
                    kwargs.get("entity_id", ""), kwargs.get("as_of_chapter", ""), kwargs.get("limit", 20)
                )
            if action == "unresolved_threads":
                return self._unresolved_threads(kwargs.get("limit", 20))
            if action == "character_subgraph":
                return self._character_subgraph(kwargs.get("entity_id", ""), kwargs.get("limit", 20))
            return f"Unknown action: {action}"
        except Exception as exc:
            return f"Error: {exc}"

    def _stats(self) -> str:
        with self.driver.session(database=self.database) as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            chars = session.run("MATCH (c:Character) RETURN count(c) AS c").single()["c"]
            events = session.run("MATCH (e:Event) RETURN count(e) AS c").single()["c"]
        return json.dumps(
            {"nodes": nodes, "relations": rels, "characters": chars, "events": events},
            ensure_ascii=False,
            indent=2,
        )

    def _character_state(self, entity_id: str) -> str:
        if not entity_id:
            return "Error: entity_id is required"
        with self.driver.session(database=self.database) as session:
            row = session.run(
                """
                MATCH (c:Character {entity_id: $entity_id})
                RETURN c.entity_id AS entity_id, c.canonical_name AS name, c.status AS status,
                       c.aliases AS aliases, c.type AS type
                """,
                entity_id=entity_id,
            ).single()
        if not row:
            return "Not found"
        return json.dumps(dict(row), ensure_ascii=False, indent=2)

    def _active_relations(self, entity_id: str, as_of_chapter: str, limit: int) -> str:
        if not entity_id or not as_of_chapter:
            return "Error: entity_id and as_of_chapter are required"
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                """
                MATCH (a:Entity {entity_id: $entity_id})-[r:RELATES]->(b:Entity)
                WHERE r.valid_from <= $as_of_chapter
                  AND (r.valid_to IS NULL OR r.valid_to > $as_of_chapter)
                RETURN b.entity_id AS to_id, b.canonical_name AS to_name,
                       r.kind AS kind, r.status AS status, r.valid_from AS since
                ORDER BY r.valid_from DESC
                LIMIT $limit
                """,
                entity_id=entity_id,
                as_of_chapter=as_of_chapter,
                limit=max(limit, 1),
            )
            payload = [dict(r) for r in rows]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _unresolved_threads(self, limit: int) -> str:
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                """
                MATCH (t:Thread)
                WHERE t.status = 'open'
                OPTIONAL MATCH (h:Hook)-[:SETS_UP]->(t)
                RETURN t.thread_id AS thread_id, t.name AS name, t.priority AS priority,
                       count(h) AS hook_count
                ORDER BY t.priority DESC
                LIMIT $limit
                """,
                limit=max(limit, 1),
            )
            payload = [dict(r) for r in rows]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _character_subgraph(self, entity_id: str, limit: int) -> str:
        if not entity_id:
            return "Error: entity_id is required"
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                """
                MATCH (c:Character {entity_id: $entity_id})-[r:RELATES]-(n:Character)
                RETURN c.canonical_name AS center, n.entity_id AS neighbor_id,
                       n.canonical_name AS neighbor_name, r.kind AS kind, r.status AS status
                LIMIT $limit
                """,
                entity_id=entity_id,
                limit=max(limit, 1),
            )
            payload = [dict(r) for r in rows]
        return json.dumps(payload, ensure_ascii=False, indent=2)
