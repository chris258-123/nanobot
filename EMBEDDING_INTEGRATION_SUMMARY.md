# Embedding Integration Summary

## Overview

Successfully integrated embedding generation into the `reprocess_all.py` pipeline, eliminating the need for a separate `embedder_parallel.py` step in most workflows.

## Changes Made

### 1. reprocess_all.py

**Added imports:**
- `sentence_transformers.SentenceTransformer`
- `FlagEmbedding.FlagModel` (optional)
- Availability flags for graceful degradation

**New command-line arguments:**
- `--embedding-model`: Choose embedding model (chinese, chinese-large, multilingual, multilingual-large)
- `--skip-embedding`: Skip embedding generation (write zero vectors for backward compatibility)

**Embedding model initialization:**
- Model selection map with 4 options:
  - `chinese`: moka-ai/m3e-base (768-dim)
  - `chinese-large`: BAAI/bge-large-zh-v1.5 (1024-dim, default)
  - `multilingual`: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
  - `multilingual-large`: distiluse-base-multilingual-cased-v2 (512-dim)
- Automatic fallback to SentenceTransformer if FlagModel unavailable
- Graceful degradation to zero vectors if sentence-transformers unavailable

**ChapterProcessor initialization:**
- Pass `embedding_model`, `use_flag_model`, and `vector_size` parameters

### 2. chapter_processor.py

**Updated __init__ method:**
- Added `embedding_model` parameter (default: None)
- Added `use_flag_model` parameter (default: False)
- Added `vector_size` parameter (default: 1024)
- Store parameters as instance variables

**New _generate_embedding method:**
- Generate real embeddings when model is configured
- Return zero vectors when no model configured (backward compatibility)
- Handle both FlagModel and SentenceTransformer APIs

**Updated _write_qdrant method:**
- Replace `vector_payload = self._build_qdrant_vector_payload()` with per-point `self._generate_embedding(text)`
- Generate embeddings for:
  - Chapter digest (from chapter summary)
  - Fact digests (from fact changes)
  - Relation digests (from relationship changes)

**Updated _build_qdrant_vector_payload method:**
- Use `self.vector_size` instead of hardcoded 384

## Benefits

1. **Immediate Searchability**: Qdrant points are searchable immediately after chapter processing
2. **Simplified Workflow**: One command instead of two separate steps
3. **Consistency**: Embeddings generated with same model configuration across all points
4. **Backward Compatibility**: `--skip-embedding` flag preserves old behavior
5. **Graceful Degradation**: Falls back to zero vectors if embedding libraries unavailable

## Usage Examples

### New Integrated Workflow (Recommended)

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_a \
  --chapter-dir /path/to/chapters \
  --llm-config llm_config_claude.json \
  --canon-db-path /tmp/canon_novel_a.db \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass novel123 \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_a_assets \
  --embedding-model chinese-large \
  --reset-canon \
  --reset-neo4j
```

### Old Two-Step Workflow (Still Supported)

```bash
# Step 1: Process chapters with zero vectors
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_a \
  --chapter-dir /path/to/chapters \
  --llm-config llm_config_claude.json \
  --canon-db-path /tmp/canon_novel_a.db \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass novel123 \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_a_assets \
  --skip-embedding \
  --reset-canon \
  --reset-neo4j

# Step 2: Add embeddings separately
python nanobot/skills/novel-workflow/scripts/embedder_parallel.py \
  --assets-dir /path/to/novel_a_assets \
  --book-id novel_a \
  --qdrant-url http://localhost:6333 \
  --collection novel_a_assets \
  --model chinese-large \
  --workers 5
```

## Testing

Created `test_embedding_integration.py` to verify:
- ✓ Import availability checks
- ✓ ChapterProcessor initialization with/without embedding model
- ✓ _generate_embedding method returns correct vector sizes
- ✓ reprocess_all.py argument parsing

All tests pass successfully.

## Backward Compatibility

- Existing code using ChapterProcessor without embedding parameters continues to work
- `--skip-embedding` flag preserves old zero-vector behavior
- `embedder_parallel.py` still available for re-embedding existing points
- No breaking changes to existing workflows

## Dependencies

**Required:**
- `sentence-transformers` (for embedding generation)

**Optional:**
- `FlagEmbedding` (for optimized BGE model support)

Install with:
```bash
pip install sentence-transformers
pip install FlagEmbedding  # Optional, for better BGE performance
```

## Files Modified

1. `nanobot/skills/novel-workflow/scripts/reprocess_all.py`
2. `nanobot/skills/novel-workflow/scripts/chapter_processor.py`
3. `nanobot/skills/novel-workflow/SKILL.md`

## Files Created

1. `test_embedding_integration.py` - Test suite for verification
2. `EMBEDDING_INTEGRATION_SUMMARY.md` - This document

## Next Steps

1. Update requirements-novel.txt to include sentence-transformers
2. Consider adding embedding model caching to avoid reloading for each chapter
3. Add progress logging for embedding generation (e.g., "Generating embeddings for chapter X...")
4. Consider batch embedding generation for better performance (embed multiple texts at once)
