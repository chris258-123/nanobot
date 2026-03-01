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

### Connectivity fix for proxy + `chinese-large` embedding (important)

When using `--embedding-model chinese-large` (`BAAI/bge-large-zh-v1.5`) or calling LLM endpoints:

- `ALL_PROXY`/`all_proxy=socks://...` can break some `httpx` calls (`Unknown scheme for proxy URL`).
- Clearing `HTTP_PROXY`/`HTTPS_PROXY` may cause slow/no-response LLM calls on networks that require HTTP proxy.
- `chinese-large` may still try remote HuggingFace probe calls during model init if offline mode is not enabled.

Use this standard preflight before workflow commands:

```bash
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Keep `HTTP_PROXY`/`HTTPS_PROXY` if your network requires them.
Only clear `ALL_PROXY/all_proxy`.

Example (`reprocess_all.py` + `chinese-large`):

```bash
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --embedding-model chinese-large \
  ...
```

For Book-B generation, use the same proxy preflight:

```bash
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py ...
```

Note: `generate_book_ab.py` is aligned with this rule and only strips `ALL_PROXY/all_proxy` for LLM calls.
LiteLLM `RuntimeWarning: ... async_success_handler was never awaited` is a logging warning, not the root cause of proxy/connectivity failures.

#### Local model cache check (`chinese-large`)

`chinese-large` maps to `BAAI/bge-large-zh-v1.5`.
Typical local cache path:

```bash
~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5
```

Quick check:

```bash
ls -1 ~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5
```

Important: `Qdrant 404` and `blueprint/JSON parse failure` are separate issues from proxy setup.
- `Qdrant 404`: usually wrong collection name / route / service state.
- `JSON parse failure`: usually model output format drift; use retry/backoff and repair logic.

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
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
  --target-book-id novel_b \
  --template-book-id novel_a \
  --world-config nanobot/skills/novel-workflow/templates/world_spec.example.json \
  --chapter-count 20 \
  --start-chapter 1 \
  --output-dir /tmp/novel_b \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json \
  --commit-memory \
  --consistency-policy strict_blocking \
  --continuity-mode strict_gate \
  --continuity-retry 3 \
  --continuity-window 12 \
  --continuity-min-entities 3 \
  --continuity-min-open-threads 1 \
  --chapter-summary-style structured \
  --enforce-isolation \
  --template-semantic-search \
  --template-semantic-model chinese-large \
  --reference-top-k 12 \
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
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py ... --resume
```

### Continuity strict-gate (recommended)

`generate_book_ab.py` now supports chapter-to-chapter continuity gating:

- `--continuity-mode strict_gate`: retry chapter generation if continuity checks fail
- `--continuity-retry`: max retries per chapter in strict mode
- `--continuity-window`: number of recent chapter capsules injected into prompt
- `--continuity-min-entities`: minimum carried entities expected in new chapter text
- `--continuity-min-open-threads`: minimum open-thread progress expected
- `--continuity-min-chars`: hard minimum chapter body chars (retry if below threshold)
- `--chapter-summary-style structured`: use structured continuity capsule instead of plain 1-2 sentence summary

The script also sanitizes model output to remove duplicated heading lines inside body text
(for example both `# 第三章...` and `第三章...` appearing together).

Blueprint is also gated before chapter writing:

- rejects generic hooks (for example `留下推动下一章的悬念`)
- requires per-chapter `carry_over_to_next` and (from chapter 2) `open_with`
- enforces cross-chapter link: chapter N `carry_over_to_next` must be reflected in chapter N+1 `open_with`

The gate report is written to:

- `<log_dir>/generate_book_ab/<target_book_id>_<timestamp>/blueprint_continuity_report.json`

Recommended rollout:

1) Run `0001-0010` first and verify continuity logs/injection files.  
2) If stable, continue long-run batches (`--resume`) for full chapters.

### Hierarchical blueprint mode (new default)

To reduce cross-batch drift, `generate_book_ab.py` now supports hierarchical planning:

- `--blueprint-mode hierarchical` (default): `master_arc -> stage_arc -> batch_blueprint`
- `--stage-size 100`: stage granularity (recommended)
- `--batch-size 20`: expected batch chunk metadata
- `--freeze-published-blueprint` (default): lock already-published ranges
- `--blueprint-root <dir>`: persist reusable blueprint artifacts
- `--book-total-chapters 1350`: optional full-book target for long-run planning

Resume example (from chapter 341):

```bash
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
  ... \
  --start-chapter 341 \
  --chapter-count 20 \
  --resume \
  --blueprint-mode hierarchical \
  --stage-size 100 \
  --batch-size 20 \
  --book-total-chapters 1350 \
  --freeze-published-blueprint
```

Batch-boundary verification (341):

```bash
python - <<'PY'
import json
p = "/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/generate_book_ab/novel_04_b_full_1350_v2_<timestamp>/chapters/0341_pre_generation_injection.json"
o = json.load(open(p, encoding="utf-8"))
print("carry:", o.get("previous_chapter_carry_over"))
print("open :", o.get("current_chapter_open_with"))
print("capsules:", len(o.get("recent_continuity_capsules_bookB") or []))
print("range:", (o.get("recent_continuity_capsules_bookB") or [{}])[0].get("chapter_no"), "->", (o.get("recent_continuity_capsules_bookB") or [{}])[-1].get("chapter_no"))
print("source:", o.get("resolved_previous_carry_source"), o.get("resolved_open_with_source"))
PY
```

### Enforce Chinese memory fields for Book B (A-like context style)

`generate_book_ab.py` now supports Chinese enforcement for Book-B memory context and commit path.

Default behavior (already enabled):

- `--enforce-chinese-on-injection`
- `--enforce-chinese-on-commit`
- `--enforce-chinese-fields rule,status,trait,goal,secret,state`

Example command (explicit):

```bash
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
  ... \
  --enforce-chinese-on-injection \
  --enforce-chinese-on-commit \
  --enforce-chinese-fields rule,status,trait,goal,secret,state
```

If you only want strict Chinese for `rule/status`:

```bash
unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
  ... \
  --enforce-chinese-fields rule,status
```

### Rebuild Book B from scratch (delete then regenerate)

When old B memory is polluted by mixed language, rebuild B before rerun:

1) Stop running Book-B generation process.
2) Delete Book-B outputs (`chapter_*.md`, blueprint, run_report, target Canon DB).
3) Clear Book-B Neo4j target (book-specific data or whole B instance).
4) Recreate Book-B Qdrant collection.
5) Re-run generation from chapter `0001` with `--commit-memory`.

### Full-volume batch rerun (v4 recommended script)

Current recommended full run entry is the batch shell script:

- Script: `/home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/sh/b_full_1350_v4_rebuild_from0139_batch10.sh`
- Scope: chapters `0139-1350` (rebuild continuation on existing v4 base)
- Batch size: `10`
- Retry policy: per-batch max `3` attempts, `20s` backoff
- Log: `/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/b_full_1350_v4_rebuild_from0139_batch10.log`
- tmux session: `b_full_v4_rebuild0139`

Preflight checklist (before launch):

1) Ensure no old full-run tmux/process is still running.
2) Ensure template A read stores are reachable (`canon_novel_04_a_full.db`, `bolt://localhost:7689`, `novel_04_a_full_assets`).
3) Ensure target B write stores are isolated (`canon_novel_04_b_full_1350_v4.db`, `bolt://localhost:7695`, `novel_04_b_full_1350_v4_assets`).

Launch in tmux:

```bash
tmux new-session -d -s b_full_v4_rebuild0139 \
  "bash /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/sh/b_full_1350_v4_rebuild_from0139_batch10.sh"
```

Key quality settings in this script:

- Hierarchical blueprint: `--blueprint-mode hierarchical --stage-size 100 --batch-size 10 --book-total-chapters 1350 --no-freeze-published-blueprint`
- Continuity gate: `--continuity-mode strict_gate --continuity-retry 4 --continuity-window 12`
- Keep relaxed thread gate: `--continuity-min-open-threads 0`
- Hard min length gate: `--continuity-min-chars 2600`
- Chinese enforcement: `--enforce-chinese-on-injection --enforce-chinese-on-commit`
- **LLM-only opening continuity repair** (no rule sentence injection): `--opening-rewrite-by-llm --opening-rewrite-max-attempts 3 --opening-rewrite-max-chars 900`

Live monitoring:

```bash
tmux capture-pane -pt b_full_v4_rebuild0139 -S -120
tail -f /home/chris/Desktop/my_workspace/novel_data/04/new_book/log/b_full_1350_v4_rebuild_from0139_batch10.log
```

Failure handling / resume:

- If one batch fails 3 attempts, the script stops at that batch and keeps logs.
- Fix root cause, then rerun the same script; it resumes by batch and existing chapter files.
- For true clean rerun, clear v3 outputs and v3 target stores first, then relaunch.

## Isolation Rules (Critical)

For `--commit-memory`, A and B must be physically isolated:

- Canon: different `.db` files
- Neo4j: different URI/instance (required for Community Edition)
- Qdrant: different collections

Do not reuse A targets as B write targets.

For Neo4j **Community Edition** (`neo4j:*-community`), only the default `neo4j` database is available.
So A/B isolation must use **different Neo4j instances** (different port/container + separate `/data` volume), not just different database names.

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

### Live progress checks (safe read-only)

These checks are read-only (`SELECT` / `MATCH ... RETURN` / Qdrant `scroll|count`) and do not reset or modify running ingestion.

1) Tail batch log:

```bash
tail -n 20 /path/to/reprocess.log
```

2) Canon progress (chapters done / latest chapter):

```bash
python -c "import sqlite3; c=sqlite3.connect('/path/to/canon.db'); \
print('done', c.execute(\"select count(*) from commit_log where book_id=? and commit_type='CHAPTER_PROCESS' and status='ALL_DONE'\",('novel_a',)).fetchone()[0]); \
print('last', c.execute(\"select max(chapter_no) from commit_log where book_id=?\",('novel_a',)).fetchone()[0]); c.close()"
```

3) Qdrant 8-asset coverage (per chapter completeness):

```bash
python - <<'PY'
import sqlite3, requests, collections
book='novel_a'; db='/path/to/canon.db'; col='novel_a_assets'; q='http://localhost:6333'
req=['plot_beat','character_card','conflict','setting','theme','pov','tone','style']
conn=sqlite3.connect(db)
chapters=[r[0] for r in conn.execute(
    "select chapter_no from commit_log where book_id=? and commit_type='CHAPTER_PROCESS' and status='ALL_DONE' order by chapter_no",
    (book,)
)]
conn.close()
chapter_types=collections.defaultdict(set); counts=collections.Counter(); offset=None
while True:
    payload={'limit':1000,'with_payload':True,'with_vector':False,'filter':{'must':[{'key':'book_id','match':{'value':book}}]}}
    if offset is not None: payload['offset']=offset
    data=requests.post(f'{q}/collections/{col}/points/scroll',json=payload,timeout=60).json()['result']
    for p in data.get('points',[]):
        pl=p.get('payload',{}) or {}
        ch=str(pl.get('chapter') or pl.get('chapter_no') or '').zfill(4)
        t=pl.get('asset_type') or pl.get('memory_type') or pl.get('type')
        if not t: continue
        counts[t]+=1
        if ch and t in req: chapter_types[ch].add(t)
    offset=data.get('next_page_offset')
    if not offset: break
missing=[(ch,[t for t in req if t not in chapter_types.get(ch,set())]) for ch in chapters]
missing=[x for x in missing if x[1]]
print('qdrant_points',sum(counts.values()))
print('qdrant_type_counts',{k:counts.get(k,0) for k in req})
print('chapters_missing_any_8_assets',len(missing))
print('missing_examples',missing[:20])
PY
```

4) Neo4j progress (book-scoped via Canon commit_id filter):

```bash
/home/chris/miniforge3/envs/nanobot/bin/python - <<'PY'
import sqlite3
from neo4j import GraphDatabase
book='novel_a'; db='/path/to/canon.db'
conn=sqlite3.connect(db)
ids=[r[0] for r in conn.execute("select commit_id from commit_log where book_id=? and status='ALL_DONE' order by chapter_no",(book,))]
conn.close()
d=GraphDatabase.driver('bolt://localhost:7689',auth=('neo4j','novel123'))
with d.session(database='neo4j') as s:
    ch=s.run("MATCH (c:Chapter {book_id: $b}) RETURN count(c) AS n",b=book).single()['n']
    ev=s.run("MATCH (e:Event)-[:OCCURS_IN]->(:Chapter {book_id: $b}) RETURN count(DISTINCT e) AS n",b=book).single()['n']
    rel=s.run("MATCH ()-[r:RELATES]->() WHERE r.commit_id IN $ids RETURN count(r) AS n",ids=ids).single()['n']
print('neo4j_chapters',ch)
print('neo4j_events',ev)
print('neo4j_relations_by_commit',rel)
d.close()
PY
```

5) Canon 8-asset completeness (payload assets empty check):

```bash
python - <<'PY'
import sqlite3, json
book='novel_a'; db='/path/to/canon.db'
conn=sqlite3.connect(db)
rows=conn.execute("select chapter_no,payload_json from commit_log where book_id=? and commit_type='CHAPTER_PROCESS' and status='ALL_DONE' order by chapter_no",(book,)).fetchall()
conn.close()
bad=[]
for ch,p in rows:
    d=json.loads(p) if p else {}
    a=d.get('assets') if isinstance(d,dict) else {}
    miss=[]
    if not a.get('plot_beats'): miss.append('plot_beats')
    if not a.get('character_cards'): miss.append('character_cards')
    if not a.get('conflicts'): miss.append('conflicts')
    if not a.get('settings'): miss.append('settings')
    if not a.get('themes'): miss.append('themes')
    if not (a.get('pov') or a.get('point_of_view')): miss.append('pov')
    if not a.get('tone'): miss.append('tone')
    if not a.get('style'): miss.append('style')
    if miss: bad.append((ch,miss))
print('canon_chapters_missing_any_8_assets',len(bad))
print('missing_examples',bad[:20])
PY
```

6) Book-B Chinese-field check (`rule/status/trait/goal/secret/state`):

```bash
python - <<'PY'
import sqlite3, json, re
book='novel_b'; db='/path/to/canon_novel_b.db'
fields={'rule','status','trait','goal','secret','state'}
latin=re.compile(r'[A-Za-z]')
conn=sqlite3.connect(db)
rows=conn.execute("""
SELECT fh.chapter_no, fh.predicate, fh.object_json
FROM fact_history fh
JOIN commit_log cl ON cl.commit_id=fh.commit_id
WHERE cl.book_id=?
""",(book,)).fetchall()
conn.close()
bad=[]
def walk(v, chapter, pred):
    if isinstance(v,str):
        if latin.search(v): bad.append((chapter,pred,v[:120]))
    elif isinstance(v,list):
        for x in v: walk(x,chapter,pred)
    elif isinstance(v,dict):
        for x in v.values(): walk(x,chapter,pred)
for ch,pred,obj in rows:
    if (pred or '').lower() not in fields: continue
    try: val=json.loads(obj or 'null')
    except Exception: val=obj
    walk(val,ch,pred)
print('target_fields_with_english',len(bad))
print('examples',bad[:20])
PY
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
