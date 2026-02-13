---
name: novel-workflow
description: Novel writing workflow with 8-element asset extraction, vector search (BGE-large/m3e-base), and hybrid retrieval.
---

# Novel Workflow Skill

Extract narrative assets from novels, embed them with Chinese-optimized models, and search with hybrid vector+keyword matching.

## Quick Start

```bash
# 1. Start services
docker-compose up -d  # Qdrant (6333), Letta (8283), Postgres (5432)

# 2. Extract assets (parallel, 8 narrative elements)
python nanobot/skills/novel-workflow/scripts/asset_extractor_parallel.py \
  --book-id novel_01 \
  --chapter-dir ~/novel_data/novel_01 \
  --output-dir ~/novel_assets_enhanced \
  --llm-config llm_config.json \
  --workers 8

# 3. Embed with BGE-large (recommended)
python nanobot/skills/novel-workflow/scripts/embedder_parallel.py \
  --assets-dir ~/novel_assets_enhanced \
  --book-id novel_01 \
  --model chinese-large \
  --workers 5

# 4. Search
python test_hybrid_search_bge.py
```

## Prerequisites

```bash
# Install dependencies
pip install -r requirements-novel.txt

# Required packages
# - qdrant-client>=1.7.0
# - sentence-transformers>=2.2.0
# - FlagEmbedding>=1.3.0
# - httpx>=0.25.0
```

## Configuration

`~/.nanobot/config.json`:
```json
{
  "integrations": {
    "qdrant": {
      "enabled": true,
      "url": "http://localhost:6333",
      "collection_name": "novel_assets_v2"
    },
    "letta": {
      "enabled": true,
      "url": "http://localhost:8283"
    },
    "beads": {
      "enabled": true,
      "workspace_path": "~/.beads"
    }
  }
}
```

## Embedding Models

```bash
# BGE-large (best quality, 1024-dim)
--model chinese-large  # BAAI/bge-large-zh-v1.5

# m3e-base (good balance, 768-dim)
--model chinese  # moka-ai/m3e-base

# Multilingual (384-dim)
--model multilingual  # paraphrase-multilingual-MiniLM-L12-v2
```

BGE-large uses optimized FlagModel loading automatically.

## Asset Types (8 Elements)

Extracted per chapter:
- `plot_beat`: Events, causality, character involvement
- `character_card`: Traits, state, relationships, goals
- `conflict`: 6 types (人vs人/环境/社会/自我/命运/超自然)
- `setting`: Location, time, atmosphere, world rules
- `theme`: Themes, manifestations, symbolism
- `point_of_view`: Person, knowledge level, focalization
- `tone`: Emotional arc, mood, tension
- `style`: Sentence structure, vocabulary, rhetoric

## Search Examples

### Python API

```python
from sentence_transformers import SentenceTransformer
import httpx

# Load model
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
query_vector = model.encode("冷静的角色").tolist()

# Vector search
response = httpx.post(
    "http://localhost:6333/collections/novel_assets_v2/points/search",
    json={
        "vector": query_vector,
        "limit": 5,
        "with_payload": True,
        "filter": {
            "must": [{"key": "asset_type", "match": {"value": "character_card"}}]
        }
    }
)

results = response.json()["result"]
```

### Hybrid Search

```python
# Use test script
python test_hybrid_search_bge.py

# Or use HybridSearcherBGE class
from test_hybrid_search_bge import HybridSearcherBGE

searcher = HybridSearcherBGE()
results = searcher.hybrid_search("冷静的，飞船", limit=10)

for r in results:
    print(f"Score: {r['score']:.4f}")
    print(f"Text: {r['payload']['text'][:100]}...")
```

## Tools Available

Use via nanobot agent:

```python
# Qdrant operations
qdrant(action="search", query="冷静的角色", asset_type="character_card", limit=5)
qdrant(action="scroll", book_id="novel_01", asset_type="plot_beat", limit=20)
qdrant(action="info")

# Letta agent memory
letta(action="create_agent", agent_type="writer")
letta(action="send_message", agent_id="writer_id", message="Generate chapter 1")

# Beads task management
beads(action="add", title="Extract chapter 1-10", description="Batch extraction")
beads(action="list", doable=True)

# Novel orchestrator (requires all 3 tools enabled)
novel_orchestrator(action="init_library", book_id="novel_01")
```

## Performance

| Metric | Value |
|--------|-------|
| Extraction speed | ~5 chapters/min (8 workers) |
| Embedding speed | ~2.5 files/sec (5 workers) |
| Search latency | <150ms |
| Assets per chapter | ~23 |

## Upgrade Existing Collection

```bash
# Delete old collection
curl -X DELETE http://localhost:6333/collections/novel_assets_v2

# Re-embed with BGE-large
python nanobot/skills/novel-workflow/scripts/embedder_parallel.py \
  --assets-dir ~/novel_assets_enhanced \
  --book-id novel_01 \
  --model chinese-large \
  --workers 5
```

## Troubleshooting

```bash
# Check Qdrant
curl http://localhost:6333/collections/novel_assets_v2

# Check Letta
curl http://localhost:8283/v1/agents

# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"

# Restart services
docker-compose restart
```

## Scripts

- `asset_extractor_parallel.py`: Parallel extraction (8 elements)
- `embedder_parallel.py`: Parallel embedding (multi-model support)
- `hybrid_search.py`: Hybrid search library
- `test_hybrid_search_bge.py`: Search test script
- `context_pack.py`: Assemble context for generation
- `canon_db.py`: Character state tracking

---

**Version**: 2.1 (BGE-large Support)
**Last Updated**: 2026-02-12
**Test Dataset**: novel_04 (985 chapters, 20636 assets)
