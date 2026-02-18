"""Delta converter: transforms raw LLM asset JSON into structured delta format.

Replaces the broken load_asset_file() with proper:
- Entity normalization (clean names, split aliases, filter noise)
- Structured relationship types (ALLY/ENEMY/FAMILY instead of free-text)
- Fact extraction from character state/traits
- Event-driven pairwise relationship extraction from plot_beats
"""

import json
import uuid
from pathlib import Path
from name_normalizer import (
    normalize_entity_list, normalize_name, classify_relation_type, NormalizedName
)


def convert_assets_to_delta(assets: dict, chapter_no: str) -> dict:
    """Convert raw LLM-extracted assets to structured delta.

    Args:
        assets: Raw asset dict from asset_extractor (has character_cards, plot_beats, etc.)
        chapter_no: Chapter number string (e.g., "0001")

    Returns:
        Structured delta dict with entities_new, events, relations_delta, fact_changes, hooks
    """
    delta = {
        "entities_new": [],
        "events": [],
        "relations_delta": [],
        "fact_changes": [],
        "hooks": [],
    }

    # --- Step 1: Extract and normalize entities from character_cards ---
    raw_entities = []
    for idx, char in enumerate(assets.get("character_cards", [])):
        raw_entities.append({
            "name": char.get("name", f"unknown_{idx}"),
            "type": "character",
            "traits": {k: v for k, v in char.items()
                       if k not in ("name", "relationships", "traits")},
            "trait_list": char.get("traits", []),
            "status": char.get("state", "active"),
            "goals": char.get("goals", []),
            "secrets": char.get("secrets", []),
            "_raw_relationships": char.get("relationships", {}),
        })

    valid_entities, filtered = normalize_entity_list(raw_entities)
    delta["entities_new"] = valid_entities

    # Build name→entity lookup for relationship resolution
    name_lookup = {}  # name/alias → entity dict
    for ent in valid_entities:
        name_lookup[ent["name"]] = ent
        for alias in ent.get("aliases", []):
            name_lookup[alias] = ent

    # --- Step 2: Extract fact_changes from character state ---
    for ent in valid_entities:
        entity_name = ent["name"]

        # Status fact (HARD_STATE)
        status = ent.get("status", "active")
        if status:
            delta["fact_changes"].append({
                "subject_name": entity_name,
                "predicate": "status",
                "value": status,
                "op": "INSERT",
                "valid_from": chapter_no,
                "tier": "HARD_STATE",
                "status": "confirmed",
            })

        # Traits fact (SOFT_NOTE)
        traits = ent.get("trait_list", [])
        if traits:
            delta["fact_changes"].append({
                "subject_name": entity_name,
                "predicate": "traits",
                "value": traits,
                "op": "INSERT",
                "valid_from": chapter_no,
                "tier": "SOFT_NOTE",
                "status": "confirmed",
            })

        # Goals fact (SOFT_NOTE)
        goals = ent.get("goals", [])
        if goals:
            delta["fact_changes"].append({
                "subject_name": entity_name,
                "predicate": "goals",
                "value": goals,
                "op": "INSERT",
                "valid_from": chapter_no,
                "tier": "SOFT_NOTE",
                "status": "confirmed",
            })

        # Secrets fact (SOFT_NOTE)
        secrets = ent.get("secrets", [])
        if secrets:
            delta["fact_changes"].append({
                "subject_name": entity_name,
                "predicate": "secrets",
                "value": secrets,
                "op": "INSERT",
                "valid_from": chapter_no,
                "tier": "SOFT_NOTE",
                "status": "implied",
            })

    # --- Step 3: Extract structured relationships from character_cards ---
    seen_relations = set()  # (from_name, to_name, kind) to deduplicate

    for ent in valid_entities:
        from_name = ent["name"]
        raw_rels = ent.get("_raw_relationships", {})

        for target_name, rel_description in raw_rels.items():
            # Normalize target name
            target_result = normalize_name(target_name)

            # Skip if target is noise (generic role like 母亲)
            if not target_result.is_valid:
                continue

            # Try to resolve target to a known entity
            resolved_target = target_result.canonical_name
            if resolved_target not in name_lookup:
                # Check aliases
                for alias in target_result.aliases:
                    if alias in name_lookup:
                        resolved_target = name_lookup[alias]["name"]
                        break

            # Skip self-relationships
            if resolved_target == from_name:
                continue

            # Classify relationship type
            kind = classify_relation_type(rel_description)

            # Deduplicate (A→B ALLY and B→A ALLY should both exist but not duplicate)
            rel_key = (from_name, resolved_target, kind)
            if rel_key in seen_relations:
                continue
            seen_relations.add(rel_key)

            delta["relations_delta"].append({
                "from_name": from_name,
                "to_name": resolved_target,
                "kind": kind,
                "op": "INSERT",
                "valid_from": chapter_no,
                "description": rel_description,  # Keep original for reference
                "status": "confirmed",
            })

    # --- Step 4: Extract events from plot_beats ---
    for idx, beat in enumerate(assets.get("plot_beats", [])):
        # Normalize participant names
        raw_participants = beat.get("characters", [])
        clean_participants = []
        for p in raw_participants:
            p_result = normalize_name(p)
            if p_result.is_valid:
                # Resolve to known entity name
                resolved = p_result.canonical_name
                if resolved in name_lookup:
                    clean_participants.append(resolved)
                else:
                    # Check aliases
                    for alias in p_result.aliases:
                        if alias in name_lookup:
                            clean_participants.append(name_lookup[alias]["name"])
                            break
                    else:
                        clean_participants.append(resolved)

        event = {
            "event_id": f"{chapter_no}_evt_{idx:02d}",
            "type": "plot_beat",
            "summary": beat.get("event", ""),
            "participants": clean_participants,
            "impact": beat.get("impact", ""),
            "causality": beat.get("causality", ""),
            "chapter_position": beat.get("chapter_position", ""),
        }
        delta["events"].append(event)

    # --- Step 5: Event-driven pairwise relationships ---
    # For events with 2+ participants, extract implicit relationships
    for event in delta["events"]:
        participants = event.get("participants", [])
        if len(participants) < 2:
            continue

        # Generate pairwise relationships for co-participants
        for i in range(len(participants)):
            for j in range(i + 1, len(participants)):
                a, b = participants[i], participants[j]
                pair_key = (a, b, "CO_PARTICIPANT")
                reverse_key = (b, a, "CO_PARTICIPANT")

                if pair_key not in seen_relations and reverse_key not in seen_relations:
                    seen_relations.add(pair_key)
                    delta["relations_delta"].append({
                        "from_name": a,
                        "to_name": b,
                        "kind": "CO_PARTICIPANT",
                        "op": "INSERT",
                        "valid_from": chapter_no,
                        "description": f"共同参与事件: {event['summary'][:50]}",
                        "status": "implied",
                        "event_id": event["event_id"],
                    })

    # --- Step 6: Extract hooks (foreshadowing) ---
    for idx, beat in enumerate(assets.get("plot_beats", [])):
        impact = beat.get("impact", "").lower()
        if any(kw in impact for kw in ["伏笔", "暗示", "铺垫", "预示", "悬念", "线索"]):
            delta["hooks"].append({
                "name": f"Hook from {chapter_no}",
                "summary": beat.get("event", ""),
                "priority": 1,
            })

    # Clean up internal fields from entities
    for ent in delta["entities_new"]:
        ent.pop("_raw_relationships", None)
        ent.pop("raw_name", None)

    # Preserve the 8 asset types for Qdrant storage
    # These are needed for semantic search and injection into Book B generation
    delta["plot_beats"] = assets.get("plot_beats", [])
    delta["character_cards"] = assets.get("character_cards", [])
    delta["conflicts"] = assets.get("conflicts", [])
    delta["settings"] = assets.get("settings", [])
    delta["themes"] = assets.get("themes", [])
    # Support both "pov" and "point_of_view" keys
    delta["pov"] = assets.get("pov", {}) or assets.get("point_of_view", {})
    delta["tone"] = assets.get("tone", {})
    delta["style"] = assets.get("style", {})

    return delta


def load_and_convert(asset_path: str | Path, chapter_no: str) -> dict:
    """Load asset file and convert to delta."""
    with open(asset_path, 'r', encoding='utf-8') as f:
        assets = json.load(f)
    return convert_assets_to_delta(assets, chapter_no)


if __name__ == "__main__":
    import sys

    # Test with first asset file
    asset_dir = Path("/home/chris/novel_assets_test100")
    asset_files = sorted(asset_dir.glob("novel_04_0*.json"))

    if not asset_files:
        print("No asset files found")
        sys.exit(1)

    # Process first 3 chapters
    for af in asset_files[:3]:
        chapter_no = af.stem.split("_")[2]  # "0001"
        print(f"\n{'='*60}")
        print(f"Chapter {chapter_no}: {af.name}")
        print(f"{'='*60}")

        delta = load_and_convert(af, chapter_no)

        print(f"\nEntities: {len(delta['entities_new'])}")
        for e in delta["entities_new"]:
            print(f"  {e['name']} aliases={e.get('aliases', [])}")

        print(f"\nFact changes: {len(delta['fact_changes'])}")
        for f in delta["fact_changes"][:5]:
            print(f"  {f['subject_name']}.{f['predicate']} = {f['value']}")

        print(f"\nRelations: {len(delta['relations_delta'])}")
        for r in delta["relations_delta"][:5]:
            print(f"  {r['from_name']} --[{r['kind']}]--> {r['to_name']}")

        print(f"\nEvents: {len(delta['events'])}")
        for ev in delta["events"][:3]:
            print(f"  [{ev['type']}] {ev['summary'][:60]}... participants={ev['participants']}")
