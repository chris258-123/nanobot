# Three-Tier Memory System - Phase 1: Foundation

## Overview

Phase 1 implements the basic infrastructure for the three-tier memory system:

1. **Recall Memory (Qdrant)** - Already implemented, stores embedded assets
2. **Structural Memory (Neo4j)** - NEW: Graph database for entities, relationships, events
3. **Canonical Memory (Canon DB v2)** - NEW: Enhanced SQLite with history tracking and conflict detection

## What's Implemented

### Infrastructure
- ✅ Neo4j added to docker-compose.yml (port 7474 browser, 7687 bolt)
- ✅ Neo4j Python driver added to requirements-novel.txt
- ✅ Config schema updated with Neo4j and Canon DB settings

### Neo4j Manager (`neo4j_manager.py`)
- ✅ Connection management and schema initialization
- ✅ Entity layer: Character, Location, Item CRUD
- ✅ Chapter and Chunk layer
- ✅ Relationship layer with validity periods
- ✅ Event layer with participants and locations
- ✅ Thread/Hook layer for foreshadowing
- ✅ Query methods: get_character_state, get_active_relations, get_unresolved_threads

### Canon DB V2 (`canon_db_v2.py`)
- ✅ Enhanced schema with 10 tables:
  - entity_registry (single source of truth)
  - character_current, item_current, rule_current (fast lookup)
  - fact_history, relationship_history (event sourcing)
  - address_book (称呼关系)
  - thread_current (foreshadowing)
  - commit_log (version control)
- ✅ Commit management: begin_commit, mark_commit_status
- ✅ Entity normalization with alias resolution
- ✅ Fact and relationship history tracking
- ✅ Current state snapshot updates
- ✅ Conflict detection (dead character, item ownership, relationship changes)
- ✅ Query methods: get_character_state, get_all_entities, get_commit_status

### Chapter Processor (`chapter_processor.py`)
- ✅ Orchestrates all three memory tiers
- ✅ Chunk creation with stable IDs
- ✅ Entity normalization
- ✅ Conflict detection (preflight checks)
- ✅ Canon DB updates (authoritative)
- ✅ Neo4j updates (structural)
- ✅ Commit status tracking
- ✅ Error handling and rollback support

## Setup

### 1. Start Docker Services

```bash
cd /home/chris/Desktop/my_workspace/nanobot
docker-compose up -d
```

This starts:
- Qdrant (port 6333)
- Letta (port 8283)
- Postgres (port 5432)
- Neo4j (port 7474 browser, 7687 bolt)

### 2. Install Dependencies

```bash
pip install -r requirements-novel.txt
```

This installs:
- qdrant-client
- sentence-transformers
- httpx
- neo4j (NEW)
- FlagEmbedding (NEW)

### 3. Configure nanobot

Add to `~/.nanobot/config.json`:

```json
{
  "integrations": {
    "neo4j": {
      "enabled": true,
      "uri": "bolt://localhost:7687",
      "username": "neo4j",
      "password": "novel123",
      "database": "neo4j"
    },
    "canon_db": {
      "enabled": true,
      "db_path": "~/.nanobot/workspace/canon_v2.db"
    },
    "qdrant": {
      "enabled": true,
      "url": "http://localhost:6333",
      "collection_name": "novel_assets_v2"
    }
  }
}
```

## Testing Phase 1

### Run the Test Script

```bash
cd /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/scripts
python test_phase1.py
```

This will:
1. Process 5 test chapters with manual delta JSON
2. Write to Canon DB (entity registry, fact history, current snapshots)
3. Write to Neo4j (entities, relationships, events, threads)
4. Verify data in both databases
5. Print summary report

### Expected Output

```
==============================================================
Phase 1: Foundation Test
==============================================================

Processing ch001: Chapter 1: The Beginning
Status: success

Processing ch002: Chapter 2: The Stranger
Status: success

Processing ch003: Chapter 3: The Alliance
Status: success
Warnings: [{'type': 'relationship_change', ...}]

Processing ch004: Chapter 4: The Betrayal
Status: success
Warnings: [{'type': 'relationship_change', ...}]

Processing ch005: Chapter 5: The Injury
Status: success

==============================================================
Verification: Canon DB
==============================================================

Characters registered: 2
  - Alice (char_alice)
    State: {'state': {'status': 'injured', 'location': 'Strange Forest'}, 'updated_chapter': 'ch005'}
  - Bob (char_bob)
    State: {'state': {}, 'updated_chapter': 'ch002'}

==============================================================
Verification: Neo4j
==============================================================

Alice state: {'name': 'Alice', 'traits': {'personality': 'curious', 'age': 25}, 'status': 'active', 'aliases': ['小艾', '艾丽丝']}

Alice's relationships (as of ch005): 1
  - ENEMY with Bob (since ch004)

Unresolved threads: 1
  - Memory Loss Mystery (priority: 1, hooks: 1)

==============================================================
Phase 1 Test Summary
==============================================================
Chapters processed: 5
Successful: 5
Failed: 0

Phase 1 test complete!
```

### Verify in Neo4j Browser

1. Open http://localhost:7474 in your browser
2. Login with username: `neo4j`, password: `novel123`
3. Run queries:

```cypher
// View all entities
MATCH (e:Entity) RETURN e LIMIT 25

// View character relationships
MATCH (a:Character)-[r:RELATES]->(b:Character)
RETURN a.canonical_name, r.kind, b.canonical_name, r.valid_from, r.valid_to

// View events
MATCH (ev:Event)-[:OCCURS_IN]->(c:Chapter)
RETURN ev.type, ev.summary, c.chapter_no

// View foreshadowing threads
MATCH (h:Hook)-[:SETS_UP]->(t:Thread)
RETURN t.name, t.status, t.priority, h.summary
```

### Verify Canon DB

```bash
sqlite3 ~/.nanobot/workspace/canon_v2_test.db

-- View entities
SELECT * FROM entity_registry;

-- View fact history
SELECT * FROM fact_history ORDER BY created_at;

-- View relationship history
SELECT * FROM relationship_history ORDER BY created_at;

-- View commit log
SELECT commit_id, chapter_no, status FROM commit_log;
```

## What's Next: Phase 2

Phase 2 will add:
- LLM-based delta extraction (automated)
- Entity normalization improvements
- Relationship tracking enhancements
- Address book (称呼关系) tracking
- Batch processing script for full books

## Files Created

1. `docker-compose.yml` - Added Neo4j service
2. `requirements-novel.txt` - Added neo4j and FlagEmbedding
3. `nanobot/config/schema.py` - Added Neo4j and Canon DB config
4. `nanobot/skills/novel-workflow/scripts/neo4j_manager.py` - Neo4j operations (350 lines)
5. `nanobot/skills/novel-workflow/scripts/canon_db_v2.py` - Enhanced Canon DB (450 lines)
6. `nanobot/skills/novel-workflow/scripts/chapter_processor.py` - Main orchestrator (300 lines)
7. `nanobot/skills/novel-workflow/scripts/test_phase1.py` - Test script (250 lines)
8. `PHASE1_README.md` - This file

## Troubleshooting

### Neo4j Connection Error

If you get "ServiceUnavailable" error:
```bash
docker-compose logs neo4j
docker-compose restart neo4j
```

Wait 30 seconds for Neo4j to fully start, then retry.

### SQLite Lock Error

If you get "database is locked":
```bash
rm ~/.nanobot/workspace/canon_v2_test.db
```

Then re-run the test.

### Import Errors

Make sure you're in the scripts directory:
```bash
cd /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/scripts
python test_phase1.py
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Chapter Processor                         │
│  (Orchestrates all three memory tiers)                      │
└────────────┬────────────┬────────────┬──────────────────────┘
             │            │            │
             ▼            ▼            ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐
    │  Canon DB  │ │   Neo4j    │ │  Qdrant    │
    │  (SQLite)  │ │  (Graph)   │ │ (Vector)   │
    └────────────┘ └────────────┘ └────────────┘
         │              │              │
         │              │              │
    Authoritative   Structural     Recall
    Facts with      Relationships  Semantic
    History         & Events       Search
```

## Phase 1 Validation Checklist

- [x] Docker services start successfully
- [x] Neo4j browser accessible at http://localhost:7474
- [x] Python dependencies install without errors
- [x] Test script runs without errors
- [x] 5 chapters processed successfully
- [x] Entities registered in Canon DB
- [x] Fact history recorded
- [x] Relationship history recorded
- [x] Commit log shows ALL_DONE status
- [x] Entities created in Neo4j
- [x] Relationships created with validity periods
- [x] Events linked to chapters
- [x] Threads/hooks created
- [x] Query methods return correct data
- [x] Conflict detection works (warnings shown)

Phase 1 is complete when all items above are checked! ✅
