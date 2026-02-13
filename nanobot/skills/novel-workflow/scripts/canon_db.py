"""Canon database manager for novel character states.

SQLite database for tracking character states, world rules, and plot threads.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional


class CanonDB:
    """SQLite database for novel canon (character states, world rules)."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL,
                name TEXT NOT NULL,
                traits TEXT,
                current_state TEXT,
                goals TEXT,
                secrets TEXT,
                relationships TEXT,
                last_updated_chapter TEXT,
                UNIQUE(book_id, name)
            );

            CREATE TABLE IF NOT EXISTS world_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL,
                rule_type TEXT NOT NULL,
                rule_name TEXT NOT NULL,
                description TEXT,
                constraints TEXT,
                introduced_chapter TEXT,
                UNIQUE(book_id, rule_type, rule_name)
            );

            CREATE TABLE IF NOT EXISTS plot_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL,
                thread_name TEXT NOT NULL,
                description TEXT,
                status TEXT,
                introduced_chapter TEXT,
                resolved_chapter TEXT,
                UNIQUE(book_id, thread_name)
            );

            CREATE INDEX IF NOT EXISTS idx_characters_book ON characters(book_id);
            CREATE INDEX IF NOT EXISTS idx_world_rules_book ON world_rules(book_id);
            CREATE INDEX IF NOT EXISTS idx_plot_threads_book ON plot_threads(book_id);
        """)
        self.conn.commit()

    def upsert_character(self, book_id: str, name: str, traits: list[str],
                        current_state: str, goals: list[str], secrets: list[str],
                        relationships: dict, chapter: str):
        """Insert or update character."""
        self.conn.execute("""
            INSERT INTO characters (book_id, name, traits, current_state, goals, secrets, relationships, last_updated_chapter)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id, name) DO UPDATE SET
                traits = excluded.traits,
                current_state = excluded.current_state,
                goals = excluded.goals,
                secrets = excluded.secrets,
                relationships = excluded.relationships,
                last_updated_chapter = excluded.last_updated_chapter
        """, (book_id, name, json.dumps(traits), current_state, json.dumps(goals),
              json.dumps(secrets), json.dumps(relationships), chapter))
        self.conn.commit()

    def get_character(self, book_id: str, name: str) -> Optional[dict]:
        """Get character by name."""
        cursor = self.conn.execute("""
            SELECT name, traits, current_state, goals, secrets, relationships, last_updated_chapter
            FROM characters
            WHERE book_id = ? AND name = ?
        """, (book_id, name))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "name": row[0],
            "traits": json.loads(row[1]),
            "current_state": row[2],
            "goals": json.loads(row[3]),
            "secrets": json.loads(row[4]),
            "relationships": json.loads(row[5]),
            "last_updated_chapter": row[6]
        }

    def get_all_characters(self, book_id: str) -> list[dict]:
        """Get all characters for a book."""
        cursor = self.conn.execute("""
            SELECT name, traits, current_state, goals, secrets, relationships, last_updated_chapter
            FROM characters
            WHERE book_id = ?
            ORDER BY name
        """, (book_id,))
        return [
            {
                "name": row[0],
                "traits": json.loads(row[1]),
                "current_state": row[2],
                "goals": json.loads(row[3]),
                "secrets": json.loads(row[4]),
                "relationships": json.loads(row[5]),
                "last_updated_chapter": row[6]
            }
            for row in cursor.fetchall()
        ]

    def upsert_world_rule(self, book_id: str, rule_type: str, rule_name: str,
                         description: str, constraints: list[str], chapter: str):
        """Insert or update world rule."""
        self.conn.execute("""
            INSERT INTO world_rules (book_id, rule_type, rule_name, description, constraints, introduced_chapter)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id, rule_type, rule_name) DO UPDATE SET
                description = excluded.description,
                constraints = excluded.constraints
        """, (book_id, rule_type, rule_name, description, json.dumps(constraints), chapter))
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Example usage
    db = CanonDB("~/.nanobot/workspace/canon.db")

    # Add character
    db.upsert_character(
        book_id="book1",
        name="Alice",
        traits=["brave", "curious"],
        current_state="searching for answers",
        goals=["find the truth", "protect her friends"],
        secrets=["knows about the prophecy"],
        relationships={"Bob": "trusted ally", "Eve": "rival"},
        chapter="chapter_01"
    )

    # Get character
    alice = db.get_character("book1", "Alice")
    print(json.dumps(alice, indent=2))

    db.close()

