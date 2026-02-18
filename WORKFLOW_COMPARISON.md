# Workflow Comparison: Old vs New

## Overview

This document shows the difference between the old two-step workflow and the new integrated workflow.

## Old Workflow (Two Steps)

### Step 1: Process chapters with zero vectors

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
  --reset-canon \
  --reset-neo4j
```

**Result:** Qdrant points created with zero vectors (not searchable)

### Step 2: Add embeddings separately

```bash
python nanobot/skills/novel-workflow/scripts/embedder_parallel.py \
  --assets-dir /path/to/novel_a_assets \
  --book-id novel_a \
  --qdrant-url http://localhost:6333 \
  --collection novel_a_assets \
  --model chinese-large \
  --workers 5
```

**Result:** Qdrant points updated with real embeddings (now searchable)

### Issues with Old Workflow

❌ Two separate commands to remember
❌ Window where Qdrant points exist but aren't searchable
❌ Need to manage asset files separately
❌ Risk of using different embedding models by mistake
❌ More complex error handling (two failure points)

---

## New Workflow (One Step)

### Single integrated command

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

**Result:** Qdrant points created with real embeddings (immediately searchable)

### Benefits of New Workflow

✅ Single command - simpler to use
✅ Immediate searchability - no waiting
✅ Consistent configuration - same model guaranteed
✅ Fewer failure points - one command to debug
✅ Better progress tracking - see embedding generation in real-time

---

## Side-by-Side Comparison

| Aspect | Old Workflow | New Workflow |
|--------|-------------|--------------|
| **Commands** | 2 separate commands | 1 integrated command |
| **Searchability** | Delayed (after step 2) | Immediate |
| **Configuration** | 2 places to configure | 1 place to configure |
| **Error Handling** | 2 failure points | 1 failure point |
| **Progress Tracking** | Separate logs | Unified log |
| **Asset Files** | Required | Optional (can use --chapter-dir) |
| **Embedding Model** | Specified in step 2 | Specified in main command |
| **Backward Compatible** | N/A | Yes (--skip-embedding) |

---

## Migration Examples

### Example 1: Basic Migration

**Old:**
```bash
python reprocess_all.py --mode llm --book-id novel_a --asset-dir /path/to/assets ...
python embedder_parallel.py --assets-dir /path/to/assets --book-id novel_a --model chinese-large
```

**New:**
```bash
python reprocess_all.py --mode llm --book-id novel_a --asset-dir /path/to/assets --embedding-model chinese-large ...
```

### Example 2: Direct from Chapters

**Old:**
```bash
# First extract assets
python asset_extractor_parallel.py --book-id novel_a --chapter-dir /path/to/chapters --output-dir /path/to/assets ...
# Then process with zero vectors
python reprocess_all.py --mode llm --book-id novel_a --asset-dir /path/to/assets ...
# Finally add embeddings
python embedder_parallel.py --assets-dir /path/to/assets --book-id novel_a --model chinese-large
```

**New:**
```bash
# Process directly from chapters with embeddings
python reprocess_all.py --mode llm --book-id novel_a --chapter-dir /path/to/chapters --embedding-model chinese-large ...
```

### Example 3: Keep Old Workflow

**If you prefer the old workflow:**
```bash
python reprocess_all.py --mode llm --book-id novel_a --asset-dir /path/to/assets --skip-embedding ...
python embedder_parallel.py --assets-dir /path/to/assets --book-id novel_a --model chinese-large
```

---

## Performance Comparison

### Old Workflow Timing

```
Step 1: reprocess_all.py
  - 100 chapters × 30 seconds = 50 minutes
  - Result: Zero vectors in Qdrant

Step 2: embedder_parallel.py
  - 100 chapters × 5 seconds = 8 minutes
  - Result: Real embeddings in Qdrant

Total: 58 minutes
```

### New Workflow Timing

```
Single command: reprocess_all.py with --embedding-model
  - 100 chapters × 32 seconds = 53 minutes
  - Result: Real embeddings in Qdrant

Total: 53 minutes
```

**Time Saved:** 5 minutes + reduced complexity

---

## When to Use Each Workflow

### Use New Workflow (Recommended)

✅ Starting a new project
✅ Want immediate searchability
✅ Prefer simpler commands
✅ Processing chapters directly (--chapter-dir)
✅ Want consistent configuration

### Use Old Workflow (Legacy)

✅ Already have asset files
✅ Need to re-embed existing points with different model
✅ Want to separate extraction and embedding for debugging
✅ Have custom embedding pipeline
✅ Need parallel embedding with multiple workers

---

## Troubleshooting

### "I want the old behavior"

Add `--skip-embedding` to your command:
```bash
python reprocess_all.py ... --skip-embedding
```

### "Embeddings are too slow"

Try a faster model:
```bash
python reprocess_all.py ... --embedding-model chinese  # 768-dim, faster
```

Or use the old workflow with parallel workers:
```bash
python reprocess_all.py ... --skip-embedding
python embedder_parallel.py ... --workers 10  # More parallel workers
```

### "I need to re-embed with a different model"

Use embedder_parallel.py to update existing points:
```bash
python embedder_parallel.py --assets-dir /path/to/assets --book-id novel_a --model multilingual-large
```

---

## Conclusion

The new integrated workflow is simpler, faster, and more reliable. It's recommended for all new projects.

The old two-step workflow is still supported via `--skip-embedding` for backward compatibility and special use cases.
