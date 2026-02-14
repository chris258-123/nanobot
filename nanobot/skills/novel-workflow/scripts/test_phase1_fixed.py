#!/usr/bin/env python3
"""Integration test for Phase 1 fixes.

Validates all acceptance criteria:
1. No dirty entity names (no brackets, ∈, generic roles)
2. fact_history > 0 (facts actually written)
3. Structured relationship types (ALLY/ENEMY/FAMILY, not free-text)
4. Entity registry with aliases working
5. Pairwise relationships (not protagonist-centered star)
6. Event-driven CO_PARTICIPANT relationships from plot_beats
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from name_normalizer import (
    normalize_name, normalize_entity_list, classify_relation_type,
    is_noise_entity, VALID_RELATION_TYPES,
)
from delta_converter import convert_assets_to_delta, load_and_convert
from canon_db_v2 import CanonDBV2

ASSET_DIR = Path("/home/chris/novel_assets_test100")
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def test_name_normalizer():
    """Test 1: Name normalizer handles all known dirty patterns."""
    print("\n" + "=" * 60)
    print("Test 1: Name Normalizer")
    print("=" * 60)
    errors = []

    # Bracket splitting
    r = normalize_name("荆璜（玄虹）")
    if r.canonical_name != "荆璜" or "玄虹" not in r.aliases:
        errors.append(f"Bracket split failed: {r}")

    # Noise filtering
    for noise in ["∈先生", "∈", "母亲", "出租车师傅", "村民", "修女们"]:
        r = normalize_name(noise)
        if r.is_valid:
            errors.append(f"Should be noise: '{noise}' → valid={r.is_valid}")

    # Valid names pass through
    for valid in ["荆璜", "周雨", "林默", "苏瑶"]:
        r = normalize_name(valid)
        if not r.is_valid:
            errors.append(f"Should be valid: '{valid}' → valid={r.is_valid}")

    # Slash splitting
    r = normalize_name("张三/李四")
    if r.canonical_name != "张三" or "李四" not in r.aliases:
        errors.append(f"Slash split failed: {r}")

    # Batch normalization with dedup
    raw = [
        {"name": "荆璜（玄虹）", "type": "character"},
        {"name": "玄虹", "type": "character"},
        {"name": "∈先生", "type": "character"},
        {"name": "母亲", "type": "character"},
        {"name": "周雨", "type": "character"},
    ]
    valid, filtered = normalize_entity_list(raw)
    valid_names = [e["name"] for e in valid]
    if "荆璜" not in valid_names:
        errors.append(f"荆璜 missing from valid: {valid_names}")
    if "玄虹" in valid_names:
        errors.append(f"玄虹 should be merged into 荆璜, not separate: {valid_names}")
    if "∈先生" in valid_names or "母亲" in valid_names:
        errors.append(f"Noise entities not filtered: {valid_names}")

    if errors:
        for e in errors:
            print(f"  {FAIL} {e}")
    else:
        print(f"  {PASS} All name normalizer tests passed")
    return len(errors) == 0


def test_relation_classifier():
    """Test 2: Relationship type classification."""
    print("\n" + "=" * 60)
    print("Test 2: Relation Type Classifier")
    print("=" * 60)
    errors = []

    cases = [
        ("多年好友，关系密切", "ALLY"),
        ("师父", "MENTOR"),
        ("父亲", "FAMILY"),
        ("母女关系", "FAMILY"),
        ("恋人", "ROMANTIC"),
        ("死敌", "ENEMY"),
        ("同事", "COLLEAGUE"),
        ("竞争对手", "RIVAL"),
        ("上司", "HIERARCHY"),
        ("领导", "HIERARCHY"),
    ]
    for desc, expected in cases:
        result = classify_relation_type(desc)
        if result != expected:
            errors.append(f"'{desc}' → {result}, expected {expected}")

    # All results must be in VALID_RELATION_TYPES
    for desc, _ in cases:
        result = classify_relation_type(desc)
        if result not in VALID_RELATION_TYPES:
            errors.append(f"'{desc}' → '{result}' not in VALID_RELATION_TYPES")

    if errors:
        for e in errors:
            print(f"  {FAIL} {e}")
    else:
        print(f"  {PASS} All relation classifier tests passed ({len(cases)} cases)")
    return len(errors) == 0


def test_delta_converter_real_assets():
    """Test 3: Delta converter with real asset files."""
    print("\n" + "=" * 60)
    print("Test 3: Delta Converter (Real Assets)")
    print("=" * 60)
    errors = []

    asset_files = sorted(ASSET_DIR.glob("novel_04_0*.json"))
    if not asset_files:
        print(f"  {FAIL} No asset files found in {ASSET_DIR}")
        return False

    total_entities = 0
    total_facts = 0
    total_relations = 0
    total_events = 0
    all_entity_names = set()
    all_relation_kinds = set()
    dirty_names = []

    for af in asset_files:
        chapter_no = af.stem.split("_")[2]
        delta = load_and_convert(af, chapter_no)

        # Check entities
        for ent in delta["entities_new"]:
            name = ent["name"]
            all_entity_names.add(name)
            total_entities += 1

            # Check for dirty patterns
            if "（" in name or "）" in name or "(" in name or ")" in name:
                dirty_names.append(name)
            if "∈" in name:
                dirty_names.append(name)
            if is_noise_entity(name)[0]:
                dirty_names.append(name)

        # Check facts
        total_facts += len(delta["fact_changes"])
        for fact in delta["fact_changes"]:
            if "subject_name" not in fact:
                errors.append(f"Fact missing subject_name: {fact}")

        # Check relations
        total_relations += len(delta["relations_delta"])
        for rel in delta["relations_delta"]:
            all_relation_kinds.add(rel["kind"])
            if "from_name" not in rel or "to_name" not in rel:
                errors.append(f"Relation missing from_name/to_name: {rel}")
            if rel["kind"] not in VALID_RELATION_TYPES:
                errors.append(f"Invalid relation type: {rel['kind']}")

        total_events += len(delta["events"])

    # Acceptance criteria checks
    if dirty_names:
        errors.append(f"Dirty names found: {dirty_names[:10]}")

    if total_facts == 0:
        errors.append("fact_changes = 0 (should be > 0)")

    invalid_kinds = all_relation_kinds - VALID_RELATION_TYPES
    if invalid_kinds:
        errors.append(f"Invalid relation kinds: {invalid_kinds}")

    has_co_participant = "CO_PARTICIPANT" in all_relation_kinds
    if not has_co_participant:
        errors.append("No CO_PARTICIPANT relations (event-driven extraction missing)")

    print(f"  Processed {len(asset_files)} chapters")
    print(f"  Unique entities: {len(all_entity_names)}")
    print(f"  Total facts: {total_facts}")
    print(f"  Total relations: {total_relations}")
    print(f"  Total events: {total_events}")
    print(f"  Relation kinds: {sorted(all_relation_kinds)}")

    if errors:
        for e in errors:
            print(f"  {FAIL} {e}")
    else:
        print(f"  {PASS} All delta converter tests passed")
    return len(errors) == 0


def test_canon_db_pipeline():
    """Test 4: Full Canon DB pipeline with real data."""
    print("\n" + "=" * 60)
    print("Test 4: Canon DB Pipeline (Real Assets)")
    print("=" * 60)
    errors = []

    # Use temp DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = CanonDBV2(db_path)

        asset_files = sorted(ASSET_DIR.glob("novel_04_0*.json"))[:10]  # First 10 chapters
        if not asset_files:
            print(f"  {FAIL} No asset files found")
            return False

        book_id = "novel_04"
        for af in asset_files:
            chapter_no = af.stem.split("_")[2]
            delta = load_and_convert(af, chapter_no)

            # Begin commit
            commit_id = db.begin_commit(book_id, chapter_no, {"test": True})

            # Register entities
            name_to_id = {}
            for ent in delta["entities_new"]:
                entity_id = db.normalize_entity(
                    ent["name"], ent.get("type", "character"),
                    chapter_no, ent.get("aliases", [])
                )
                name_to_id[ent["name"]] = entity_id
                for alias in ent.get("aliases", []):
                    name_to_id[alias] = entity_id

            # Resolve and write facts
            resolved_facts = []
            for fact in delta["fact_changes"]:
                subject_name = fact.get("subject_name", "")
                subject_id = name_to_id.get(subject_name)
                if not subject_id:
                    subject_id = db.normalize_entity(subject_name, "character", chapter_no)
                if subject_id:
                    resolved_facts.append({
                        "chapter_no": chapter_no,
                        "subject_id": subject_id,
                        "predicate": fact["predicate"],
                        "value": fact["value"],
                        "op": fact.get("op", "INSERT"),
                        "valid_from": chapter_no,
                        "tier": fact.get("tier", "SOFT_NOTE"),
                        "status": fact.get("status", "confirmed"),
                    })
            if resolved_facts:
                db.append_fact_history(commit_id, resolved_facts)

            # Resolve and write relations
            resolved_rels = []
            for rel in delta["relations_delta"]:
                from_id = name_to_id.get(rel["from_name"])
                to_id = name_to_id.get(rel["to_name"])
                if not from_id:
                    from_id = db.normalize_entity(rel["from_name"], "character", chapter_no)
                if not to_id:
                    to_id = db.normalize_entity(rel["to_name"], "character", chapter_no)
                if from_id and to_id and from_id != to_id:
                    resolved_rels.append({
                        "chapter_no": chapter_no,
                        "from_id": from_id,
                        "to_id": to_id,
                        "kind": rel["kind"],
                        "op": rel.get("op", "INSERT"),
                        "valid_from": chapter_no,
                        "status": rel.get("status", "confirmed"),
                    })
            if resolved_rels:
                db.append_relationship_history(commit_id, resolved_rels)

            # Update snapshots and finalize
            db.update_current_snapshots(commit_id)
            db.mark_commit_status(commit_id, "ALL_DONE")

        # --- Validate acceptance criteria ---
        stats = db.get_statistics()
        print(f"  Entity registry: {stats['entity_registry']} entities")
        print(f"  Entity types: {stats.get('entity_types', {})}")
        print(f"  Fact history: {stats['fact_history']} facts")
        print(f"  Relationship history: {stats['relationship_history']} relations")
        print(f"  Relation kinds: {stats.get('relation_kinds', {})}")
        print(f"  Entities with aliases: {stats.get('entities_with_aliases', 0)}")
        print(f"  Commits: {stats['commit_log']}")

        # Check: fact_history > 0
        if stats["fact_history"] == 0:
            errors.append("fact_history = 0 (CRITICAL: no facts written)")

        # Check: entities have proper names (no dirty data)
        entities = db.get_all_entities()
        for ent in entities:
            name = ent["canonical_name"]
            if "（" in name or "）" in name or "∈" in name:
                errors.append(f"Dirty entity name in DB: '{name}'")

        # Check: relationship kinds are structured
        rel_kinds = stats.get("relation_kinds", {})
        for kind in rel_kinds:
            if kind not in VALID_RELATION_TYPES:
                errors.append(f"Invalid relation kind in DB: '{kind}'")

        # Check: aliases working
        if stats.get("entities_with_aliases", 0) == 0:
            errors.append("No entities have aliases (alias system not working)")

        # Check: entity count is reasonable (deduplication working)
        if stats["entity_registry"] > 50:
            errors.append(f"Too many entities ({stats['entity_registry']}) for 10 chapters - dedup may be broken")

        # Check: not all relationships are from one entity (star-shape check)
        cursor = db.conn.execute("""
            SELECT from_id, COUNT(*) as cnt FROM relationship_history
            GROUP BY from_id ORDER BY cnt DESC LIMIT 1
        """)
        top_row = cursor.fetchone()
        if top_row:
            total_rels = stats["relationship_history"]
            top_count = top_row["cnt"]
            if total_rels > 10 and top_count / total_rels > 0.5:
                errors.append(f"Star-shaped graph: top entity has {top_count}/{total_rels} relations ({top_count/total_rels:.0%})")

        db.close()

    finally:
        os.unlink(db_path)

    if errors:
        for e in errors:
            print(f"  {FAIL} {e}")
    else:
        print(f"  {PASS} All Canon DB pipeline tests passed")
    return len(errors) == 0


def test_entity_merge():
    """Test 5: Entity merge and alias resolution."""
    print("\n" + "=" * 60)
    print("Test 5: Entity Merge & Alias Resolution")
    print("=" * 60)
    errors = []

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = CanonDBV2(db_path)

        # Register entity with alias
        id1 = db.normalize_entity("荆璜", "character", "0001", aliases=["玄虹"])

        # Look up by alias should return same entity
        id2 = db.normalize_entity("玄虹", "character", "0002")
        if id1 != id2:
            errors.append(f"Alias lookup failed: {id1} != {id2}")

        # Look up by canonical should return same entity
        id3 = db.normalize_entity("荆璜", "character", "0003")
        if id1 != id3:
            errors.append(f"Canonical lookup failed: {id1} != {id3}")

        # New entity should get different ID
        id4 = db.normalize_entity("周雨", "character", "0001")
        if id4 == id1:
            errors.append(f"Different entity got same ID: {id4} == {id1}")

        # Verify last_seen updated
        cursor = db.conn.execute(
            "SELECT last_seen_chapter FROM entity_registry WHERE entity_id = ?", (id1,)
        )
        row = cursor.fetchone()
        if row["last_seen_chapter"] != "0003":
            errors.append(f"last_seen not updated: {row['last_seen_chapter']}")

        db.close()
    finally:
        os.unlink(db_path)

    if errors:
        for e in errors:
            print(f"  {FAIL} {e}")
    else:
        print(f"  {PASS} All entity merge tests passed")
    return len(errors) == 0


def main():
    print("=" * 60)
    print("Phase 1 Fix Integration Tests")
    print("=" * 60)

    results = {
        "Name Normalizer": test_name_normalizer(),
        "Relation Classifier": test_relation_classifier(),
        "Delta Converter": test_delta_converter_real_assets(),
        "Canon DB Pipeline": test_canon_db_pipeline(),
        "Entity Merge": test_entity_merge(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  All {len(results)} tests passed!")
    else:
        failed = sum(1 for v in results.values() if not v)
        print(f"\n  {failed}/{len(results)} tests FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
