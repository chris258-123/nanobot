---
name: novel-workflow
description: Novel memory pipeline: Book A asset/memory build + strict A/B isolated Book B generation.
metadata: {"nanobot":{"emoji":"📚","os":["darwin","linux"],"requires":{"bins":["python3"]}}}
---

# Novel Workflow Skill

Use this skill after chapters are already crawled/cleaned. It focuses on Book A memory build and strict A(read)-B(write) generation.

For crawling and chapter cleanup, use `nanobot/skills/novel-crawler/SKILL.md`.

## Workflow Map

1) Extract 8-element assets + embed into Qdrant
2) Build Book A three-tier memory (Canon + Neo4j + Qdrant)
3) Generate Book B with strict physical isolation and memory commits
4) Resume/visualize/verify

## Prerequisites

- Python env ready (`pip install -e ".[dev]"`)
- Qdrant running (default `http://localhost:6333`)
- Neo4j-A running (example `bolt://localhost:7687`)
- Neo4j-B running on a different URI/instance (example `bolt://localhost:7689`)
- LLM config file ready (recommended: `/home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json`)

### Connectivity fix for `chinese-large` embedding (important)

When using `--embedding-model chinese-large` (`BAAI/bge-large-zh-v1.5`), model download/load goes through HuggingFace.
If your shell has `ALL_PROXY`/`all_proxy` set to `socks://...`, `httpx` may fail with:

- `ValueError: Unknown scheme for proxy URL URL('socks://...')`

Use this fix before running workflow commands:

```bash
unset ALL_PROXY all_proxy
```

Then run commands normally (keep `HTTP_PROXY`/`HTTPS_PROXY` if you need them), for example:

```bash
unset ALL_PROXY all_proxy
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --embedding-model chinese-large \
  ...
```

LLM config example (custom endpoint mode):

```json
{
  "type": "custom",
  "url": "https://api.deepseek.com/v1/chat/completions",
  "model": "deepseek-chat",
  "api_key": "YOUR_API_KEY"
}
```

## 1) Extract 8 Elements (Optional - for separate asset extraction)

**Note:** With the new integrated embedding in `reprocess_all.py`, you can skip this step and use `--mode llm` with `--chapter-dir` directly. This section is for advanced workflows that need separate asset extraction.

```bash
python nanobot/skills/novel-workflow/scripts/asset_extractor_parallel.py \
  --book-id novel_a \
  --chapter-dir /path/to/novel_a_chapters \
  --output-dir /path/to/novel_a_assets \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json \
  --workers 8 \
  --log-file /path/to/logs/asset_extract.log

# Optional: align asset filenames to chapter IDs
python nanobot/skills/novel-workflow/scripts/rename_assets_by_chapter_id.py \
  /path/to/novel_a_chapters /path/to/novel_a_assets

# Optional: separate embedding step (only if using --skip-embedding in reprocess_all.py)
python nanobot/skills/novel-workflow/scripts/embedder_parallel.py \
  --assets-dir /path/to/novel_a_assets \
  --book-id novel_a \
  --qdrant-url http://localhost:6333 \
  --collection novel_a_assets \
  --model chinese-large \
  --workers 5
```

## 2) Build Book A Three-Tier Memory Warehouse

`reprocess_all.py` is the canonical batch entry with integrated embedding generation.

Recommended mode (quality-first): use `--chapter-dir` (better chunk evidence chain).

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_a \
  --chapter-dir /path/to/novel_a_chapters \
  --from-chapter 0001 \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json \
  --canon-db-path /tmp/canon_novel_a.db \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass novel123 \
  --neo4j-database neo4j \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_a_assets \
  --embedding-model chinese-large \
  --llm-max-retries 3 \
  --llm-retry-backoff 3 \
  --llm-backoff-factor 2 \
  --llm-backoff-max 60 \
  --llm-retry-jitter 0.5 \
  --llm-min-interval 1.0 \
  --reset-canon \
  --reset-neo4j \
  --reset-qdrant \
  --log-file /path/to/logs/reprocess.log
```

Speed-first alternative (if you already have high-quality assets): replace `--chapter-dir` with `--asset-dir /path/to/novel_a_assets`.

**New Embedding Integration Features:**

- `--embedding-model`: Choose embedding model (chinese, chinese-large, multilingual, multilingual-large)
  - `chinese`: moka-ai/m3e-base (768-dim)
  - `chinese-large`: BAAI/bge-large-zh-v1.5 (1024-dim, default)
  - `multilingual`: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
  - `multilingual-large`: distiluse-base-multilingual-cased-v2 (512-dim)
- `--skip-embedding`: Skip embedding generation and write zero vectors (old behavior)

**Benefits:**
- Qdrant points are immediately searchable (no separate embedder step needed)
- Consistent embedding model configuration
- Simplified workflow (one command instead of two)

**Backward Compatibility:**
- Use `--skip-embedding` to preserve old two-step workflow
- Separate `embedder_parallel.py` still available for re-embedding existing points

Resume from breakpoint:

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_a \
  --chapter-dir /path/to/novel_a_chapters \
  --from-chapter 0121 \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json \
  --canon-db-path /tmp/canon_novel_a.db \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass novel123 \
  --neo4j-database neo4j \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_a_assets \
  --embedding-model chinese-large \
  --log-file /path/to/logs/reprocess_resume.log
```

## 3) Generate Book B with Strict A/B Isolation

Use `generate_book_ab.py` for A-read/B-write one-click generation + commit-memory.

```bash
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
  --target-book-id novel_b \
  --template-book-id novel_a \
  --world-config nanobot/skills/novel-workflow/templates/world_spec.example.json \
  --chapter-count 20 \
  --start-chapter 1 \
  --output-dir /tmp/novel_b \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json \
  --commit-memory \
  --consistency-policy warn_only \
  --enforce-isolation \
  --template-semantic-search \
  --template-semantic-model chinese-large \
  --reference-top-k 8 \
  --llm-max-retries 3 \
  --llm-retry-backoff 3 \
  --llm-backoff-factor 2 \
  --llm-backoff-max 60 \
  --llm-retry-jitter 0.5 \
  --template-canon-db-path /tmp/canon_novel_a.db \
  --template-neo4j-uri bolt://localhost:7687 \
  --template-neo4j-user neo4j \
  --template-neo4j-pass novel123 \
  --template-neo4j-database neo4j \
  --template-qdrant-url http://localhost:6333 \
  --template-qdrant-collection novel_a_assets \
  --target-canon-db-path /tmp/canon_novel_b.db \
  --target-neo4j-uri bolt://localhost:7689 \
  --target-neo4j-user neo4j \
  --target-neo4j-pass novel123 \
  --target-neo4j-database neo4j \
  --target-qdrant-url http://localhost:6333 \
  --target-qdrant-collection novel_b_assets \
  --log-dir /home/chris/Desktop/my_workspace/nanobot/logs \
  --log-injections
```

Resume generation:

```bash
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py ... --resume
```

## Isolation Rules (Critical)

For `--commit-memory`, A and B must be physically isolated:

- Canon: different `.db` files
- Neo4j: different URI/instance (recommended) and/or dedicated database
- Qdrant: different collections

Do not reuse A targets as B write targets.

## 4) Visualization & Validation

Canon stats/charts:

```bash
python nanobot/skills/novel-workflow/scripts/visualize_canon_db.py \
  --db-path /tmp/canon_novel_b.db
```

Neo4j Book-B-only charts:

```bash
python nanobot/skills/novel-workflow/scripts/visualize_neo4j.py \
  --uri bolt://localhost:7689 \
  --username neo4j \
  --password novel123 \
  --book-id novel_b \
  --canon-db-path /tmp/canon_novel_b.db \
  --protagonist-name 主角名
```

## Logs and Outputs

`generate_book_ab.py` writes:

- Blueprint: `<output_dir>/<target_book_id>_blueprint.json`
- Chapters: `<output_dir>/<target_book_id>_chapter_0001.md` ...
- Run report: `<output_dir>/<target_book_id>_run_report.json`
- Injection logs: `<log_dir>/generate_book_ab/<target_book_id>_<timestamp>/`

## If Book A Neo4j Is Polluted

Rebuild a clean Book A warehouse to a fresh target set:

- New Canon DB path
- New Qdrant collection
- New Neo4j instance/database
- Re-run `reprocess_all.py --mode llm --reset-canon --reset-neo4j`

Then point future Book B generations to this clean Book A template store.

## Security

- Never commit API keys, DB files, or runtime logs with secrets.
- Keep secrets in local files/env vars.
