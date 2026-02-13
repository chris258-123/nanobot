#!/usr/bin/env python3
"""Test script for Phase 1: Foundation.

Tests basic pipeline with manual delta JSON.
"""

import sys
import json
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from chapter_processor import ChapterProcessor


def test_phase1():
    """Test Phase 1 implementation."""
    print("=" * 60)
    print("Phase 1: Foundation Test")
    print("=" * 60)

    # Initialize processor
    processor = ChapterProcessor(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pass="novel123",
        canon_db_path="~/.nanobot/workspace/canon_v2_test.db"
    )

    # Test data: 5 chapters with manual delta
    test_chapters = [
        {
            "book_id": "test_novel",
            "chapter_no": "ch001",
            "title": "Chapter 1: The Beginning",
            "pov": "Alice",
            "text": "Alice woke up in a strange forest. She had no memory of how she got there.",
            "delta": {
                "entities_new": [
                    {
                        "name": "Alice",
                        "type": "character",
                        "aliases": ["小艾", "艾丽丝"],
                        "traits": {"personality": "curious", "age": 25},
                        "status": "active"
                    },
                    {
                        "name": "Strange Forest",
                        "type": "location",
                        "level": "region",
                        "description": "A mysterious forest"
                    }
                ],
                "events": [
                    {
                        "type": "awakening",
                        "summary": "Alice wakes up in the forest",
                        "participants": []
                    }
                ],
                "relations_delta": [],
                "fact_changes": [
                    {
                        "subject_id": "char_alice",
                        "predicate": "location",
                        "value": "Strange Forest",
                        "op": "INSERT"
                    }
                ],
                "hooks": [
                    {
                        "name": "Memory Loss Mystery",
                        "summary": "Why does Alice have no memory?",
                        "priority": 1
                    }
                ]
            }
        },
        {
            "book_id": "test_novel",
            "chapter_no": "ch002",
            "title": "Chapter 2: The Stranger",
            "pov": "Alice",
            "text": "Alice met a mysterious stranger named Bob who offered to help her.",
            "delta": {
                "entities_new": [
                    {
                        "name": "Bob",
                        "type": "character",
                        "aliases": ["鲍勃", "神秘人"],
                        "traits": {"personality": "mysterious", "age": 30},
                        "status": "active"
                    }
                ],
                "events": [
                    {
                        "type": "meeting",
                        "summary": "Alice meets Bob",
                        "participants": ["char_alice", "char_bob"]
                    }
                ],
                "relations_delta": [
                    {
                        "from_id": "char_alice",
                        "to_id": "char_bob",
                        "kind": "ACQUAINTANCE",
                        "op": "INSERT"
                    }
                ],
                "fact_changes": [],
                "hooks": []
            }
        },
        {
            "book_id": "test_novel",
            "chapter_no": "ch003",
            "title": "Chapter 3: The Alliance",
            "pov": "Alice",
            "text": "Alice and Bob decided to work together to find answers.",
            "delta": {
                "entities_new": [],
                "events": [
                    {
                        "type": "alliance",
                        "summary": "Alice and Bob form an alliance",
                        "participants": ["char_alice", "char_bob"]
                    }
                ],
                "relations_delta": [
                    {
                        "from_id": "char_alice",
                        "to_id": "char_bob",
                        "kind": "ALLY",
                        "op": "UPDATE"
                    }
                ],
                "fact_changes": [],
                "hooks": []
            }
        },
        {
            "book_id": "test_novel",
            "chapter_no": "ch004",
            "title": "Chapter 4: The Betrayal",
            "pov": "Alice",
            "text": "Bob revealed he was working for the enemy all along.",
            "delta": {
                "entities_new": [],
                "events": [
                    {
                        "type": "betrayal",
                        "summary": "Bob betrays Alice",
                        "participants": ["char_alice", "char_bob"]
                    }
                ],
                "relations_delta": [
                    {
                        "from_id": "char_alice",
                        "to_id": "char_bob",
                        "kind": "ENEMY",
                        "op": "UPDATE"
                    }
                ],
                "fact_changes": [],
                "hooks": []
            }
        },
        {
            "book_id": "test_novel",
            "chapter_no": "ch005",
            "title": "Chapter 5: The Injury",
            "pov": "Alice",
            "text": "Alice was injured in the fight with Bob.",
            "delta": {
                "entities_new": [],
                "events": [
                    {
                        "type": "combat",
                        "summary": "Alice fights Bob and gets injured",
                        "participants": ["char_alice", "char_bob"]
                    }
                ],
                "relations_delta": [],
                "fact_changes": [
                    {
                        "subject_id": "char_alice",
                        "predicate": "status",
                        "value": "injured",
                        "op": "UPDATE"
                    }
                ],
                "hooks": []
            }
        }
    ]

    # Process each chapter
    results = []
    for chapter in test_chapters:
        print(f"\nProcessing {chapter['chapter_no']}: {chapter['title']}")
        result = processor.process_chapter(
            book_id=chapter["book_id"],
            chapter_no=chapter["chapter_no"],
            chapter_text=chapter["text"],
            title=chapter["title"],
            pov=chapter["pov"],
            delta=chapter["delta"]
        )
        results.append(result)
        print(f"Status: {result['status']}")
        if result.get("warnings"):
            print(f"Warnings: {result['warnings']}")
        if result.get("error"):
            print(f"Error: {result['error']}")

    # Verify data in Canon DB
    print("\n" + "=" * 60)
    print("Verification: Canon DB")
    print("=" * 60)

    entities = processor.canon_db.get_all_entities("character")
    print(f"\nCharacters registered: {len(entities)}")
    for entity in entities:
        print(f"  - {entity['canonical_name']} ({entity['entity_id']})")
        state = processor.canon_db.get_character_state(entity['entity_id'])
        if state:
            print(f"    State: {state}")

    # Verify data in Neo4j
    print("\n" + "=" * 60)
    print("Verification: Neo4j")
    print("=" * 60)

    # Get actual entity IDs from Canon DB
    alice_entity = next((e for e in entities if e["canonical_name"] == "Alice"), None)
    bob_entity = next((e for e in entities if e["canonical_name"] == "Bob"), None)

    if alice_entity:
        alice_state = processor.neo4j.get_character_state(alice_entity["entity_id"])
        if alice_state:
            print(f"\nAlice state: {alice_state}")

        relations = processor.neo4j.get_active_relations(alice_entity["entity_id"], "ch005")
        print(f"\nAlice's relationships (as of ch005): {len(relations)}")
        for rel in relations:
            print(f"  - {rel['kind']} with {rel['to_name']} (since {rel['since']})")

    threads = processor.neo4j.get_unresolved_threads("test_novel")
    print(f"\nUnresolved threads: {len(threads)}")
    for thread in threads:
        print(f"  - {thread['name']} (priority: {thread['priority']}, hooks: {thread['hook_count']})")

    # Summary
    print("\n" + "=" * 60)
    print("Phase 1 Test Summary")
    print("=" * 60)
    successful = sum(1 for r in results if r["status"] == "success")
    print(f"Chapters processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")

    processor.close()
    print("\nPhase 1 test complete!")


if __name__ == "__main__":
    test_phase1()
