# Phase 1 Fix Design: Three-Tier Memory System

## Problem Summary

Phase 1 processed 100 chapters but produced unusable data:
- 80 entities, all `character` type, zero aliases populated
- 0 fact_history records (state changes never written)
- 951 relationship records with free-text Chinese as `kind` and raw names as `to_id`
- Duplicate entities: `荆璜（玄虹）` and `玄虹（荆璜）` are same person
- Garbage entities: `∈先生`, `∈`, `科研人员（未具名）`
- Star-shaped relationship graph (protagonist-centered extraction)

## Root Causes

1. **No name cleaning** — LLM outputs `荆璜（玄虹）` as a single name string
2. **No alias extraction** — Parenthetical aliases never split out
3. **Unstructured relationship types** — `"多年好友，关系密切"` instead of `ALLY`
4. **Raw name targets** — `to_id = "母亲"` instead of entity_id
5. **Empty fact_changes** — `load_asset_file()` never extracts state from character_cards
6. **Per-character relationships** — Not event-driven pairwise extraction

## Design: Three-Layer Fix

### Layer 1: Name Normalizer (`name_normalizer.py`)

New standalone module. Runs BEFORE any DB writes.

**Operations (in order):**
1. Unicode normalization (NFKC) — fullwidth→halfwidth
2. Strip invisible chars (`\u200b`, `\u200c`, `\u200d`, `\ufeff`)
3. Strip leading/trailing whitespace and quotes
4. Split parenthetical aliases: `荆璜（玄虹）` → canonical=`荆璜`, aliases=[`玄虹`]
   - Handles `（）`, `()`, `【】`, `「」`
5. Split slash-separated: `村民/老人` → canonical=`村民`, aliases=[`老人`]
6. Filter noise entities:
   - Pure symbols: `∈`, `√`, `≈`
   - Symbol-prefixed: `∈先生`
   - Generic roles: `母亲`, `妹妹`, `父亲`, `青梅竹马`, `出租车师傅`
   - Unnamed: contains `未具名`, `未知`, `不明`
   - Group entities: contains `们（群体）`, `群体`
   - Single-char non-surname names
7. Return: `{canonical_name, aliases, is_valid, filter_reason}`

**Configurable:**
- `NOISE_PATTERNS`: regex list for filtering
- `GENERIC_ROLES`: set of generic role words to filter
- `VALID_SINGLE_CHARS`: set of valid single-char names (common Chinese surnames used as names)

### Layer 2: Delta Converter (`delta_converter.py`)

Replaces the broken `load_asset_file()` in test_phase1_novel.py.

**Converts raw asset JSON → structured delta JSON:**

```python
def convert_assets_to_delta(assets: dict, chapter_no: str) -> dict:
    """Convert raw LLM assets to structured delta."""
    # Returns:
    # {
    #   "entities_new": [...],      # cleaned, with aliases
    #   "events": [...],            # from plot_beats, with participant entity_ids
    #   "relations_delta": [...],   # structured kind (ALLY/ENEMY/FAMILY/etc)
    #   "fact_changes": [...],      # from character state/traits
    #   "hooks": [...]
    # }
```

**Relationship type mapping:**
Free-text Chinese → structured enum via keyword matching:

| Keywords | Mapped Type |
|----------|-------------|
| 好友/朋友/信任/伙伴 | ALLY |
| 敌对/对手/仇人/敌人 | ENEMY |
| 师父/师傅/弟子/徒弟 | MENTOR |
| 父/母/兄/弟/姐/妹/家人 | FAMILY |
| 恋人/爱人/情侣/夫妻 | ROMANTIC |
| 上级/下属/部下/手下 | HIERARCHY |
| 同事/同僚/同门 | COLLEAGUE |
| Default | ASSOCIATE |

**Fact extraction from character_cards:**
For each character, extract:
- `status` → HARD_STATE fact (active/injured/dead/missing)
- `traits` → SOFT_NOTE facts
- `goals` → SOFT_NOTE facts
- `secrets` → SOFT_NOTE facts
- `state` → HARD_STATE fact

**Event-driven relationship extraction:**
For each `plot_beat`, extract pairwise relationships from `participants`:
- If 2+ characters participate in same event → check if relationship exists
- Only create relationship if the beat's `impact` or `event` text implies interaction

### Layer 3: Enhanced Chapter Processor

**Changes to `chapter_processor.py`:**

1. **Add name normalizer step** — Before entity registration, run all names through normalizer
2. **Skip invalid entities** — Don't register noise entities (∈, 母亲, etc.)
3. **Merge duplicate entities** — If normalized canonical_name matches existing, merge aliases
4. **Write fact_changes** — Actually populate fact_history from character state
5. **Use structured relationship types** — Map free-text to enum before writing

**Changes to `canon_db_v2.py`:**

1. **Add `merge_entity()`** — Merge two entities (update merged_into, move aliases)
2. **Add `find_entity_by_alias()`** — Efficient alias lookup with index
3. **Add `update_entity_aliases()`** — Append new aliases to existing entity
4. **Fix `update_current_snapshots()`** — Actually aggregate facts across commits, not just current commit

**Changes to `neo4j_manager.py`:**

1. **Add `merge_entities()`** — Merge two Neo4j nodes
2. **Add `get_all_entities()`** — For validation/debugging
3. **Add `get_relationship_stats()`** — Count relationships by type for validation

### Data Migration

For the existing 100-chapter dataset:
1. Run name normalizer on all 80 entities → produce `entity_migration.json`
2. Rebuild entity_registry with clean names and aliases
3. Re-map all 951 relationships to use clean entity_ids
4. Extract fact_changes from existing asset files
5. Rebuild Neo4j graph from clean data

This is a one-time migration script: `migrate_phase1_data.py`

## Acceptance Criteria

1. **No dirty names**: All entities in registry have clean canonical_name, no parens/symbols
2. **Aliases populated**: Entities like 荆璜 have aliases=[玄虹]
3. **No duplicate entities**: 荆璜（玄虹）and 玄虹（荆璜）merged into one entity
4. **Structured relationship types**: All `kind` values are from enum (ALLY/ENEMY/FAMILY/etc)
5. **All relationship targets are entity_ids**: No raw names like "母亲" in from_id/to_id
6. **fact_history > 0**: Character states written as facts
7. **Non-star graph**: Relationship graph has multiple clusters, not just protagonist hub
8. **Noise filtered**: No entities like ∈, 出租车师傅, 科研人员（未具名）
9. **100 commits ALL_DONE**: All chapters processed successfully

## File Changes

### New Files
1. `scripts/name_normalizer.py` — Name cleaning and alias extraction
2. `scripts/delta_converter.py` — Asset→delta conversion with structured types
3. `scripts/migrate_phase1_data.py` — One-time data migration

### Modified Files
1. `scripts/canon_db_v2.py` — Add merge_entity, find_entity_by_alias, fix snapshots
2. `scripts/neo4j_manager.py` — Add merge_entities, get_all_entities, stats
3. `scripts/chapter_processor.py` — Integrate normalizer, fact extraction, structured types
4. `scripts/test_phase1_novel.py` — Use delta_converter instead of load_asset_file

## Implementation Order

1. `name_normalizer.py` (standalone, testable independently)
2. `delta_converter.py` (depends on normalizer)
3. `canon_db_v2.py` fixes (entity merge, alias lookup)
4. `neo4j_manager.py` fixes (merge, stats)
5. `chapter_processor.py` rebuild (integrates all above)
6. `migrate_phase1_data.py` (uses all above to fix existing data)
7. Re-run test_phase1_novel.py with fixes
8. Validate against acceptance criteria
