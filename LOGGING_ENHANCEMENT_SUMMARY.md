# Logging Enhancement Summary

## Overview

Enhanced the embedding integration with comprehensive loguru-based logging. All operations are now logged with detailed information for debugging and monitoring.

## Changes Made

### 1. reprocess_all.py - Enhanced Embedding Model Initialization Logging

**Location:** Lines 206-268

**Added logging for:**
- Embedding model selection and configuration
- Model loading process (FlagModel vs SentenceTransformer)
- Success/failure status with clear indicators (✓/✗)
- Fallback scenarios with warnings
- Vector dimensions and model details

**Example output:**
```
============================================================
Initializing embedding model for integrated embedding generation
============================================================
Selected embedding model: chinese-large
Model name: BAAI/bge-large-zh-v1.5
Vector dimension: 1024
Use FlagModel optimization: True
Loading FlagModel (optimized for BGE models)...
✓ FlagModel loaded successfully
Embedding model initialization complete
============================================================
```

### 2. chapter_processor.py - Enhanced ChapterProcessor Initialization Logging

**Location:** Lines 95-106

**Added logging for:**
- Embedding model configuration status
- Model type (FlagModel vs SentenceTransformer)
- Vector size
- Qdrant collection name

**Example output:**
```
ChapterProcessor initialized with embedding model
  - Embedding model type: FlagModel
  - Vector size: 1024
  - Qdrant collection: novel_a_assets
```

### 3. chapter_processor.py - Enhanced _generate_embedding Logging

**Location:** Lines 676-703

**Added logging for:**
- Embedding generation attempts (DEBUG level)
- Text length being embedded
- Model type used (FlagModel vs SentenceTransformer)
- Success indicators
- Error handling with fallback to zero vectors

**Example output:**
```
Generating embedding with FlagModel (text length: 156 chars)
✓ Embedding generated (vector size: 1024)
```

**Error handling:**
```
Failed to generate embedding: CUDA out of memory
Falling back to zero vector
```

### 4. chapter_processor.py - Enhanced _write_qdrant Logging

**Location:** Lines 720-840

**Added logging for:**
- Start of Qdrant write operation
- Per-type embedding generation (chapter_digest, fact_digest, relation_digest)
- Count of embeddings generated per type
- Summary of points being upserted
- Completion status with point counts

**Example output:**
```
[0001] Starting Qdrant write with embedding generation
[0001] Generating embedding for chapter digest
[0001] ✓ Chapter digest embedding generated
[0001] Generating embeddings for 15 fact digests
[0001] ✓ 15 fact digest embeddings generated
[0001] Generating embeddings for 8 relation digests
[0001] ✓ 8 relation digest embeddings generated
[0001] Upserting 24 points to Qdrant (chapter_digest=1, fact_digest=15, relation_digest=8)
[0001] ✓ Qdrant write complete: 24 points (commit=a1b2c3d4 collection=novel_a_assets)
```

## Log Levels Used

- **INFO**: Normal operation messages (model initialization, chapter processing)
- **DEBUG**: Detailed per-embedding messages (text length, vector size)
- **WARNING**: Non-critical issues (fallbacks, missing dependencies)
- **ERROR**: Critical errors (embedding generation failures)

## Log File Configuration

### Existing --log-file Parameter

The `--log-file` parameter was already present in reprocess_all.py. The enhancements ensure that all embedding-related operations are properly logged to this file.

**Usage:**
```bash
python reprocess_all.py \
  --mode llm \
  --book-id novel_a \
  --chapter-dir /path/to/chapters \
  --embedding-model chinese-large \
  --log-file /path/to/logs/reprocess.log \
  ...
```

### Log File Features

- **Automatic directory creation**: Parent directories created automatically
- **UTF-8 encoding**: Supports Chinese and Unicode characters
- **Rotation**: Logs rotate at 50 MB
- **Dual output**: Console + file simultaneously
- **Structured format**: Timestamp | Level | Message

## Logging Architecture

### How It Works

1. **reprocess_all.py** configures loguru with `configure_logger()`
2. **InterceptHandler** bridges standard Python logging to loguru
3. **chapter_processor.py** uses standard `logging.getLogger(__name__)`
4. All logs from chapter_processor.py are automatically captured by loguru
5. Logs are written to both console and file (if --log-file specified)

### Code Flow

```
chapter_processor.py
  └─> logger.info("message")
       └─> Python logging module
            └─> InterceptHandler (in reprocess_all.py)
                 └─> loguru
                      ├─> Console output
                      └─> File output (if --log-file specified)
```

## Benefits

1. **Comprehensive Visibility**: Every embedding operation is logged
2. **Easy Debugging**: Detailed error messages with context
3. **Progress Monitoring**: Real-time progress tracking
4. **Performance Analysis**: Identify slow operations
5. **Audit Trail**: Complete record of all operations
6. **Chinese Support**: UTF-8 encoding for Chinese text

## Usage Examples

### Basic Usage with Logging

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_a \
  --chapter-dir /path/to/chapters \
  --llm-config llm_config_claude.json \
  --embedding-model chinese-large \
  --log-file /var/log/nanobot/reprocess_$(date +%Y%m%d_%H%M%S).log \
  --canon-db-path /tmp/canon_novel_a.db \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass novel123 \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_a_assets
```

### Monitor Progress in Real-Time

```bash
# In one terminal, run the processing
python reprocess_all.py ... --log-file /tmp/reprocess.log

# In another terminal, monitor progress
tail -f /tmp/reprocess.log | grep "Qdrant write complete"
```

### Extract Statistics from Logs

```bash
# Count total embeddings generated
grep "embeddings generated" reprocess.log | \
  awk '{sum += $NF} END {print "Total embeddings:", sum}'

# Count chapters processed
grep "Qdrant write complete" reprocess.log | wc -l

# Find errors
grep "ERROR" reprocess.log

# Check embedding model used
grep "Embedding model type" reprocess.log | head -1
```

## Log Analysis Examples

### Example 1: Verify Embedding Model

```bash
$ grep "Selected embedding model" reprocess.log
Selected embedding model: chinese-large

$ grep "Embedding model type" reprocess.log
  - Embedding model type: FlagModel
```

### Example 2: Count Embeddings per Chapter

```bash
$ grep "Qdrant write complete" reprocess.log | head -5
[0001] ✓ Qdrant write complete: 24 points (commit=a1b2c3d4 collection=novel_a_assets)
[0002] ✓ Qdrant write complete: 31 points (commit=b2c3d4e5 collection=novel_a_assets)
[0003] ✓ Qdrant write complete: 18 points (commit=c3d4e5f6 collection=novel_a_assets)
[0004] ✓ Qdrant write complete: 27 points (commit=d4e5f6g7 collection=novel_a_assets)
[0005] ✓ Qdrant write complete: 22 points (commit=e5f6g7h8 collection=novel_a_assets)
```

### Example 3: Identify Slow Chapters

```bash
# Extract timestamps and chapter numbers
grep "Qdrant write complete" reprocess.log | \
  awk '{print $1, $2, $4}' | \
  head -10
```

## Troubleshooting with Logs

### Problem: No embeddings generated

**Check:**
```bash
grep "skip-embedding\|zero vector" reprocess.log
```

**Expected:** Should not see "skip-embedding" or "zero vector" messages

### Problem: Wrong embedding model

**Check:**
```bash
grep "Selected embedding model\|Embedding model type" reprocess.log
```

**Expected:** Should match your --embedding-model parameter

### Problem: Slow processing

**Check:**
```bash
grep "Generating embedding" reprocess.log | head -20
```

**Look for:** Text lengths, model type (FlagModel is faster for BGE models)

## Documentation Created

1. **LOGGING_GUIDE.md** - Comprehensive logging guide
   - Log file configuration
   - Log levels and formats
   - Analysis examples
   - Best practices

2. **EMBEDDING_QUICK_REFERENCE.md** - Updated with --log-file documentation

3. **test_enhanced_logging.py** - Test script for logging functionality

## Testing

Created `test_enhanced_logging.py` to verify:
- ✓ ChapterProcessor initialization logging
- ✓ Embedding generation logging
- ✓ Zero vector fallback logging
- ✓ Log message format

All tests pass successfully.

## Backward Compatibility

- No breaking changes
- Logging is optional (console-only if --log-file not specified)
- Existing code continues to work without modifications
- Log format is consistent with existing loguru configuration

## Performance Impact

- Minimal overhead (logging is asynchronous)
- DEBUG-level logs only generated when needed
- File I/O is buffered and rotated automatically
- No impact on embedding generation speed

## Next Steps

1. Consider adding metrics export (Prometheus format)
2. Add log aggregation support (JSON format option)
3. Create log analysis dashboard (Grafana)
4. Add performance metrics (embeddings/second)

## See Also

- `LOGGING_GUIDE.md` - Comprehensive logging documentation
- `EMBEDDING_INTEGRATION_SUMMARY.md` - Technical implementation details
- `EMBEDDING_QUICK_REFERENCE.md` - Quick start guide
