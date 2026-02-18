# Quick Reference: Embedding Integration

## TL;DR

`reprocess_all.py` now generates embeddings automatically. No need to run `embedder_parallel.py` separately.

## Quick Start

```bash
# Old workflow (2 steps):
python reprocess_all.py --mode llm ... --skip-embedding
python embedder_parallel.py --assets-dir ... --model chinese-large

# New workflow (1 step):
python reprocess_all.py --mode llm ... --embedding-model chinese-large --log-file /path/to/logs/reprocess.log
```

## Command-Line Options

### --embedding-model

Choose embedding model:
- `chinese` - moka-ai/m3e-base (768-dim)
- `chinese-large` - BAAI/bge-large-zh-v1.5 (1024-dim) **[DEFAULT]**
- `multilingual` - paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
- `multilingual-large` - distiluse-base-multilingual-cased-v2 (512-dim)

### --skip-embedding

Skip embedding generation and write zero vectors (for backward compatibility).

### --log-file

Specify log file location for detailed logging output. Logs include:
- Embedding model initialization details
- Per-chapter embedding generation progress
- Vector dimensions and model type
- Error messages and warnings
- Performance metrics

Example: `--log-file /path/to/logs/reprocess_$(date +%Y%m%d_%H%M%S).log`

## Examples

### Process chapters with integrated embeddings (recommended)

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
  --embedding-model chinese-large
```

### Use different embedding model

```bash
# For multilingual novels
python reprocess_all.py ... --embedding-model multilingual-large

# For Chinese novels with smaller vectors
python reprocess_all.py ... --embedding-model chinese
```

### Skip embedding (old behavior)

```bash
# Write zero vectors, embed later with embedder_parallel.py
python reprocess_all.py ... --skip-embedding
```

## Dependencies

```bash
# Required for embedding generation
pip install sentence-transformers

# Optional, for better BGE model performance
pip install FlagEmbedding
```

## Benefits

✓ Immediate searchability - no waiting for separate embedding step
✓ Consistent configuration - same model for all points
✓ Simplified workflow - one command instead of two
✓ Backward compatible - old workflows still work

## Troubleshooting

### "sentence-transformers not available"

Install the required package:
```bash
pip install sentence-transformers
```

### "FlagModel requested but not available"

This is a warning, not an error. The system will fall back to SentenceTransformer.

For better performance with BGE models, install:
```bash
pip install FlagEmbedding
```

### Qdrant points not searchable

Make sure you didn't use `--skip-embedding`. Check that embeddings are not zero vectors:

```python
import httpx
response = httpx.get("http://localhost:6333/collections/novel_a_assets/points/scroll",
                     json={"limit": 1})
point = response.json()["result"]["points"][0]
print(f"Vector sum: {sum(point['vector'])}")  # Should not be 0.0
```

## Migration Guide

### From old two-step workflow

**Before:**
```bash
python reprocess_all.py --mode llm ... --qdrant-url http://localhost:6333
python embedder_parallel.py --assets-dir ... --model chinese-large
```

**After:**
```bash
python reprocess_all.py --mode llm ... --qdrant-url http://localhost:6333 --embedding-model chinese-large
```

### Keep old workflow

Add `--skip-embedding` to preserve old behavior:
```bash
python reprocess_all.py --mode llm ... --qdrant-url http://localhost:6333 --skip-embedding
python embedder_parallel.py --assets-dir ... --model chinese-large
```

## Performance Notes

- Embedding generation adds ~1-2 seconds per chapter (depends on model and hardware)
- BGE models (chinese-large) are slower but more accurate
- Use `chinese` model for faster processing with slightly lower quality
- FlagEmbedding package provides ~2x speedup for BGE models

## See Also

- `EMBEDDING_INTEGRATION_SUMMARY.md` - Full technical details
- `nanobot/skills/novel-workflow/SKILL.md` - Complete workflow documentation
- `test_embedding_integration.py` - Test suite
