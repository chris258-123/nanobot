---
name: novel-workflow
description: Run novel pipelines for 8-element assets, embedding/hybrid retrieval, and the current three-tier memory commit workflow (LLM delta -> Canon -> Neo4j -> replay).
metadata: {"nanobot":{"emoji":"📚","os":["darwin","linux"],"requires":{"bins":["python3"]}}}
---

# Novel Workflow Skill

Use this skill for two related paths:
- Asset/Retrieval path: extraction -> embedding -> hybrid search
- Memory path (current main): LLM delta -> Canon -> Neo4j -> replay -> visualization

Prefer the memory path for long-running chapter consistency work.

## One-click generation (user specifies world + chapter count)

This path does not require per-chapter manual outline. You provide world settings once, then generate N chapters in one run.

```bash
python nanobot/skills/novel-workflow/scripts/generate_book_one_click.py \
  --book-id novel_new_01 \
  --world-config nanobot/skills/novel-workflow/templates/world_spec.example.json \
  --chapter-count 20 \
  --start-chapter 1 \
  --output-dir /tmp/novel_new_01 \
  --llm-config /tmp/llm_config_claude.json
```

Optional memory commit while generating:

```bash
python nanobot/skills/novel-workflow/scripts/generate_book_one_click.py \
  --book-id novel_new_01 \
  --world-config nanobot/skills/novel-workflow/templates/world_spec.example.json \
  --chapter-count 20 \
  --output-dir /tmp/novel_new_01 \
  --llm-config /tmp/llm_config_claude.json \
  --commit-memory \
  --canon-db-path ~/.nanobot/workspace/canon_v2_reprocessed.db
```

World-setting template:
- `nanobot/skills/novel-workflow/templates/world_spec.example.json`

Outputs:
- `<output_dir>/<book_id>_blueprint.json`
- `<output_dir>/<book_id>_chapter_0001.md` ... chapter files
- `<output_dir>/<book_id>_run_report.json`

## Quickstart (memory path, default settings)

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_04_llm_assets_full \
  --asset-dir /home/chris/novel_assets_enhanced \
  --from-chapter 0001 \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/llm_config.json \
  --canon-db-path /tmp/canon_v2_llm_assets_full.db \
  --reset-canon --reset-neo4j
```

After start, you should see a live progress bar:

```text
[=====-------------------------] 120/985  12.2% elapsed=00:48:12 eta=05:46:03 last=0120 status=ok
```

## Resume safely (no reset)

When interrupted, continue from the next chapter and reuse the same Canon DB file:

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_04_llm_assets_full \
  --asset-dir /home/chris/novel_assets_enhanced \
  --from-chapter 0121 \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/llm_config.json \
  --canon-db-path /tmp/canon_v2_llm_assets_full.db
```

Do not pass `--reset-canon` or `--reset-neo4j` when resuming.

## Modes

`reprocess_all.py` supports:
- `--mode llm`: chapter assets/text -> LLM delta -> Canon/Neo4j commit
- `--mode delta`: deterministic assets -> delta (no LLM extraction)
- `--mode replay`: replay commit payloads without LLM

## Visualize current progress

```bash
python nanobot/skills/novel-workflow/scripts/visualize_canon_db.py \
  --db-path /tmp/canon_v2_llm_assets_full.db
```

```bash
python nanobot/skills/novel-workflow/scripts/visualize_neo4j.py \
  --uri bolt://localhost:7687 \
  --username neo4j \
  --password novel123 \
  --protagonist-name 罗彬瀚
```

Generated files (cwd):
- Canon: `canon_entity_distribution.png`, `canon_fact_timeline.png`, `canon_relationship_changes.png`, `canon_commit_status.png`, `canon_top_characters.png`, `canon_summary.txt`
- Neo4j: `neo4j_character_network.png`, `neo4j_character_network_no_protagonist.png`, `neo4j_events_timeline.png`

## Asset/Retrieval quickstart (legacy path, still valid)

```bash
# 1) Start services
docker compose up -d

# 2) Extract 8-element assets
python nanobot/skills/novel-workflow/scripts/asset_extractor_parallel.py \
  --book-id novel_01 \
  --chapter-dir ~/novel_data/novel_01 \
  --output-dir ~/novel_assets_enhanced \
  --llm-config llm_config.json \
  --workers 8

# 3) Embedding
python nanobot/skills/novel-workflow/scripts/embedder_parallel.py \
  --assets-dir ~/novel_assets_enhanced \
  --book-id novel_01 \
  --model chinese-large \
  --workers 5

# 4) Hybrid search
python test_hybrid_search_bge.py
```

## Embedding model options

```bash
--model chinese-large  # BAAI/bge-large-zh-v1.5
--model chinese        # moka-ai/m3e-base
--model multilingual   # paraphrase-multilingual-MiniLM-L12-v2
```

## LLM config formats

### custom

```json
{
  "type": "custom",
  "url": "https://your-endpoint/v1/chat/completions",
  "model": "claude-sonnet-4-5",
  "api_key": "YOUR_API_KEY",
  "max_tokens": 4096
}
```

### providers

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

## Script map

- `scripts/reprocess_all.py`: batch runner (`llm`, `delta`, `replay`) + timed progress bar
- `scripts/generate_book_one_click.py`: world-setting + chapter-count one-click generator
- `scripts/chapter_processor.py`: chapter commit orchestration (Canon + Neo4j + Qdrant hook)
- `scripts/delta_extractor_llm.py`: LLM delta extraction
- `scripts/canon_db_v2.py`: authoritative commit/fact/relationship store
- `scripts/neo4j_manager.py`: graph schema and writes
- `scripts/visualize_canon_db.py`: Canon reports/charts
- `scripts/visualize_neo4j.py`: Neo4j network/timeline charts
- `scripts/asset_extractor_parallel.py`: 8-element extraction
- `scripts/embedder_parallel.py`: embeddings
- `scripts/hybrid_search.py`: hybrid retrieval

## Troubleshooting

- `Connection refused localhost:7687`: run `docker compose up -d neo4j`
- Progress reset unexpectedly: check whether reset flags were used
- JSON parse failures in LLM mode: keep `max_tokens` sufficiently high
- If interrupted: resume with `--from-chapter` and existing Canon DB

## Safety

- Never commit API keys or runtime DB files
- Keep secrets in local config/env
- Redact credentials before sharing logs
