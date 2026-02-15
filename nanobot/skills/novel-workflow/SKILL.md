---
name: novel-workflow
description: Run the novel three-tier memory workflow (LLM delta -> Canon -> Neo4j -> Qdrant), plus A-read/B-write one-click generation.
metadata: {"nanobot":{"emoji":"📚","os":["darwin","linux"],"requires":{"bins":["python3"]}}}
---

# Novel Workflow Skill

Use this skill for two production paths:
- Build/maintain the external memory warehouse from an existing novel (Book A)
- One-click generate a new novel (Book B) with isolated memory writes

## Quickstart: build Book A memory warehouse

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_04_memory \
  --asset-dir /home/chris/novel_assets_enhanced \
  --from-chapter 0001 \
  --llm-config /tmp/llm_config_claude.json \
  --canon-db-path /tmp/canon_v2_novel_04_memory.db \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_assets_v2 \
  --llm-max-retries 3 \
  --llm-retry-backoff 3 \
  --llm-backoff-factor 2 \
  --llm-backoff-max 60 \
  --llm-retry-jitter 0.5 \
  --llm-min-interval 1.0
```

Progress bar is built in:

```text
[==========--------------------] 120/985  12.2% elapsed=00:48:12 eta=05:46:03 last=0120 status=ok
```

Resume after interruption (no reset flags):

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_04_memory \
  --asset-dir /home/chris/novel_assets_enhanced \
  --from-chapter 0121 \
  --llm-config /tmp/llm_config_claude.json \
  --canon-db-path /tmp/canon_v2_novel_04_memory.db \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_assets_v2
```

## Quickstart: one-click Book B generation (A-read/B-write)

Recommended: strict isolation (Book A read-only, Book B write-only).

```bash
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
  --target-book-id novel_new_01 \
  --template-book-id novel_04_memory \
  --world-config nanobot/skills/novel-workflow/templates/world_spec.example.json \
  --chapter-count 20 \
  --start-chapter 1 \
  --output-dir /tmp/novel_new_01 \
  --llm-config /tmp/llm_config_claude.json \
  --commit-memory \
  --enforce-isolation \
  --template-canon-db-path /tmp/canon_v2_novel_04_memory.db \
  --target-canon-db-path /tmp/canon_v2_novel_new_01.db \
  --template-qdrant-url http://localhost:6333 \
  --template-qdrant-collection novel_assets_v2 \
  --target-qdrant-url http://localhost:6333 \
  --target-qdrant-collection novel_new_01_assets_v2
```

Outputs:
- `<output_dir>/<target_book_id>_blueprint.json`
- `<output_dir>/<target_book_id>_chapter_0001.md` ... chapter files
- `<output_dir>/<target_book_id>_run_report.json`

## Modes and core scripts

- `scripts/reprocess_all.py`: batch processing (`--mode llm|delta|replay`)
- `scripts/chapter_processor.py`: chapter commit orchestration
- `scripts/delta_extractor_llm.py`: LLM delta extraction
- `scripts/canon_db_v2.py`: authoritative Canon DB + commit log + replay
- `scripts/neo4j_manager.py`: structural graph memory
- `scripts/generate_book_ab.py`: one-click A-read/B-write generation
- `scripts/visualize_canon_db.py`: Canon stats and charts
- `scripts/visualize_neo4j.py`: Neo4j network/timeline charts

Legacy asset/retrieval scripts (still supported):
- `scripts/asset_extractor_parallel.py`: 8-element extraction
- `scripts/embedder_parallel.py`: embedding pipeline
- `scripts/hybrid_search.py`: hybrid retrieval helper

## Visualize current state

```bash
python nanobot/skills/novel-workflow/scripts/visualize_canon_db.py \
  --db-path /tmp/canon_v2_novel_04_memory.db
```

```bash
python nanobot/skills/novel-workflow/scripts/visualize_neo4j.py \
  --uri bolt://localhost:7687 \
  --username neo4j \
  --password novel123
```

## LLM config formats

Providers mode (Claude proxy):

```json
{
  "providers": {
    "anthropic": {
      "apiKey": "YOUR_API_KEY",
      "apiBase": "https://your-proxy-base",
      "extraHeaders": null
    }
  },
  "model": "claude-sonnet-4-5",
  "max_tokens": 4096
}
```

Custom mode:

```json
{
  "type": "custom",
  "url": "https://your-endpoint/v1/chat/completions",
  "model": "claude-sonnet-4-5",
  "api_key": "YOUR_API_KEY",
  "max_tokens": 4096
}
```

## Troubleshooting

- `chapter_xxxx.md` missing: check `--llm-config` path and API key validity
- `Isolation check failed`: set separate Book A/Book B Canon/Neo4j/Qdrant targets
- Graph names garbled: ensure UTF-8 data source and font support in visualization environment
- Batch run interrupted: resume with `--from-chapter` and the same target Canon DB

## Safety

- Never commit API keys or DB runtime files
- Keep secrets in local files or environment variables
- Redact tokens and endpoints before sharing logs
