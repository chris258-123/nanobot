# Logging Guide: Embedding Integration

## Overview

The embedding integration includes comprehensive logging using loguru. All operations are logged with detailed information for debugging and monitoring.

## Log File Configuration

### Specifying Log File Location

Use the `--log-file` parameter to specify where logs should be written:

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_a \
  --chapter-dir /path/to/chapters \
  --llm-config llm_config_claude.json \
  --embedding-model chinese-large \
  --log-file /path/to/logs/reprocess.log \
  ...
```

### Log File Features

- **Automatic directory creation**: Parent directories are created automatically
- **UTF-8 encoding**: Supports Chinese and other Unicode characters
- **Rotation**: Logs rotate at 50 MB to prevent excessive file sizes
- **Dual output**: Logs are written to both console and file simultaneously

### Dynamic Log File Names

Use shell commands to create timestamped log files:

```bash
# With timestamp
--log-file /path/to/logs/reprocess_$(date +%Y%m%d_%H%M%S).log

# With book ID
--log-file /path/to/logs/reprocess_${BOOK_ID}_$(date +%Y%m%d).log

# Example output: reprocess_20260218_143022.log
```

## Log Levels

The logging system uses the following levels:

- **INFO**: Normal operation messages (default console level)
- **DEBUG**: Detailed operation messages (per-embedding generation)
- **WARNING**: Non-critical issues (fallbacks, missing dependencies)
- **ERROR**: Critical errors (embedding generation failures)

## What Gets Logged

### 1. Embedding Model Initialization

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

### 2. ChapterProcessor Initialization

```
ChapterProcessor initialized with embedding model
  - Embedding model type: FlagModel
  - Vector size: 1024
  - Qdrant collection: novel_a_assets
```

### 3. Per-Chapter Processing

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

### 4. Embedding Generation Details (DEBUG level)

```
No embedding model configured, returning zero vector (size=1024)
Generating embedding with FlagModel (text length: 156 chars)
✓ Embedding generated (vector size: 1024)
```

### 5. Error Handling

```
Failed to generate embedding: CUDA out of memory
Falling back to zero vector
```

### 6. Warnings

```
sentence-transformers not available, falling back to zero vectors
Install with: pip install sentence-transformers
```

```
FlagModel requested but not available, falling back to SentenceTransformer
Install with: pip install FlagEmbedding
```

## Log Format

### Console Output Format

```
2026-02-18 14:30:22 | INFO     | Selected embedding model: chinese-large
2026-02-18 14:30:22 | INFO     | Model name: BAAI/bge-large-zh-v1.5
2026-02-18 14:30:22 | INFO     | Vector dimension: 1024
```

### File Output Format

Same as console, with UTF-8 encoding for Chinese characters:

```
2026-02-18 14:30:22 | INFO     | [0001] 生成章节摘要嵌入向量
2026-02-18 14:30:22 | DEBUG    | Generating embedding with FlagModel (text length: 156 chars)
2026-02-18 14:30:22 | DEBUG    | ✓ Embedding generated (vector size: 1024)
```

## Enabling DEBUG Logging

To see detailed per-embedding logs, modify the log level in `reprocess_all.py`:

```python
# In configure_logger function, change:
logger.add(
    sys.stdout,
    level="DEBUG",  # Changed from "INFO"
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)
```

Or set via environment variable:

```bash
export LOGURU_LEVEL=DEBUG
python reprocess_all.py ...
```

## Log Analysis Examples

### Count Embeddings Generated

```bash
grep "✓.*embeddings generated" reprocess.log | wc -l
```

### Find Errors

```bash
grep "ERROR" reprocess.log
```

### Check Embedding Model Used

```bash
grep "Embedding model type" reprocess.log
```

### Monitor Progress

```bash
tail -f reprocess.log | grep "Qdrant write complete"
```

### Extract Chapter Processing Times

```bash
grep "Qdrant write complete" reprocess.log | awk '{print $1, $2, $4}'
```

## Log File Locations

### Recommended Structure

```
/path/to/logs/
├── reprocess_20260218_143022.log
├── reprocess_20260218_150315.log
└── embedder_20260218_143022.log  # If using separate embedder
```

### Default Locations

If `--log-file` is not specified:
- Console output only (no file logging)
- Standard Python logging to stderr

## Troubleshooting with Logs

### Problem: Embeddings not generated

**Check logs for:**
```bash
grep "sentence-transformers not available" reprocess.log
grep "No embedding model configured" reprocess.log
```

**Solution:** Install required packages

### Problem: Slow embedding generation

**Check logs for:**
```bash
grep "Generating embedding" reprocess.log | head -20
```

**Look for:** Model type (FlagModel vs SentenceTransformer), text lengths

### Problem: Memory errors

**Check logs for:**
```bash
grep "CUDA out of memory\|Failed to generate embedding" reprocess.log
```

**Solution:** Use smaller model or reduce batch size

### Problem: Zero vectors written

**Check logs for:**
```bash
grep "skip-embedding\|zero vector" reprocess.log
```

**Verify:** `--skip-embedding` flag not used, embedding model loaded successfully

## Log Retention

### Automatic Rotation

Logs automatically rotate at 50 MB. Old logs are renamed with `.1`, `.2`, etc. suffixes:

```
reprocess.log
reprocess.log.1
reprocess.log.2
```

### Manual Cleanup

```bash
# Keep only last 7 days
find /path/to/logs -name "reprocess_*.log" -mtime +7 -delete

# Keep only last 10 files
ls -t /path/to/logs/reprocess_*.log | tail -n +11 | xargs rm
```

## Integration with Monitoring Tools

### Logstash/Elasticsearch

Parse logs with grok pattern:

```
%{TIMESTAMP_ISO8601:timestamp} \| %{LOGLEVEL:level}\s+\| %{GREEDYDATA:message}
```

### Prometheus

Export metrics from logs:

```bash
# Count errors
grep -c "ERROR" reprocess.log

# Count embeddings generated
grep -c "embeddings generated" reprocess.log
```

### Grafana

Create dashboard with:
- Embedding generation rate (embeddings/minute)
- Error rate
- Processing time per chapter

## Best Practices

1. **Always use --log-file in production**
   ```bash
   --log-file /var/log/nanobot/reprocess_$(date +%Y%m%d_%H%M%S).log
   ```

2. **Use timestamped log files**
   - Prevents overwriting previous runs
   - Easier to track historical issues

3. **Monitor log file size**
   - Check disk space regularly
   - Set up log rotation/cleanup

4. **Keep logs for debugging**
   - Retain logs for at least 7 days
   - Archive important runs

5. **Use grep/awk for analysis**
   - Extract specific information
   - Monitor progress in real-time

## Example: Complete Logging Setup

```bash
#!/bin/bash

# Configuration
BOOK_ID="novel_a"
LOG_DIR="/var/log/nanobot"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/reprocess_${BOOK_ID}_${TIMESTAMP}.log"

# Create log directory
mkdir -p "${LOG_DIR}"

# Run with logging
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id "${BOOK_ID}" \
  --chapter-dir /path/to/chapters \
  --llm-config llm_config_claude.json \
  --embedding-model chinese-large \
  --log-file "${LOG_FILE}" \
  --canon-db-path /tmp/canon_${BOOK_ID}.db \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass novel123 \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection ${BOOK_ID}_assets

# Check exit code
if [ $? -eq 0 ]; then
    echo "✓ Processing complete. Log: ${LOG_FILE}"
else
    echo "✗ Processing failed. Check log: ${LOG_FILE}"
    tail -50 "${LOG_FILE}"
fi

# Cleanup old logs (keep last 7 days)
find "${LOG_DIR}" -name "reprocess_*.log" -mtime +7 -delete
```

## See Also

- `EMBEDDING_INTEGRATION_SUMMARY.md` - Technical implementation details
- `EMBEDDING_QUICK_REFERENCE.md` - Quick start guide
- `nanobot/skills/novel-workflow/SKILL.md` - Complete workflow documentation
