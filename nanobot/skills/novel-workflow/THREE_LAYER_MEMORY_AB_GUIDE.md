# 三层记忆与 A/B 分离架构说明（默认 Large Chinese）

本文说明 `novel-workflow` 当前的三层记忆体系、A 书重建方式、B 书生成与写入流程，以及每章 injection 的注入机制。

## 1. 三层记忆总体架构

- **Canon（SQLite）**：权威事实账本（事实历史、关系历史、当前快照、commit 状态）。
- **Neo4j**：结构化图谱层（Chapter/Chunk/Entity/RELATES/Event/Thread）。
- **Qdrant**：检索层（语义向量 + 文本 payload），用于模板召回与章节上下文补充。

每章写入顺序（统一在 `ChapterProcessor.process_chapter`）：
1. 写 Canon 历史与快照，commit 标记 `CANON_DONE`。
2. 写 Neo4j 图谱，commit 标记 `NEO4J_DONE`。
3. 写 Qdrant，commit 标记 `ALL_DONE`。

## 2. A 书三层记忆如何写入（默认 Large Chinese）

A 书推荐通过 `reprocess_all.py` 全量重建，默认 embedding 已是 `chinese-large`：
- `--embedding-model` 默认值：`chinese-large`
- 模型：`BAAI/bge-large-zh-v1.5`
- 向量维度：`1024`

### 推荐命令（A 书全量重建）

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_04 \
  --asset-dir /home/chris/Desktop/my_workspace/novel_data/04/novel_assets \
  --from-chapter 0001 \
  --llm-config /tmp/llm_config_claude.json \
  --canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_v2.db \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass novel123 \
  --neo4j-database neo4j \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_04_assets_v2 \
  --reset-canon --reset-neo4j --reset-qdrant
```

### A 书写入到 Qdrant 的内容

每章会写入两大类：
- **Digest 类**：`chapter_digest/fact_digest/relation_digest`
- **模板资产类**：`plot_beat/character_card/conflict/setting/theme/pov/tone/style`

因此 A 既有“检索模板”，也有“事实摘要回忆”。

## 3. A/B 分离架构

`generate_book_ab.py` 在运行前会做隔离校验（开启 `--enforce-isolation` 时）：
- Canon 路径不能相同
- Neo4j `(uri + database)` 不能相同
- Qdrant `(url + collection)` 不能相同

当开启 `--commit-memory` 时，还会再次强制检查物理隔离，防止 A/B 混写。

## 4. B 书如何生成、如何更新三层记忆

### 4.1 生成入口

B 书用 `generate_book_ab.py`：
- 不带 `--commit-memory`：只生成章节文件，不写 B 三层记忆。
- 带 `--commit-memory`：每章生成后立即写入 B 的 Canon/Neo4j/Qdrant。

### 4.2 强制 Large Chinese 语义检索（推荐）

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
  --target-book-id novel_04_book_b_xxx \
  --template-book-id novel_04_test \
  --world-config nanobot/skills/novel-workflow/templates/world_spec.example.json \
  --chapter-count 5 \
  --output-dir /home/chris/Desktop/my_workspace/novel_data/04/book_b_output \
  --llm-config /tmp/llm_config_claude.json \
  --commit-memory \
  --template-semantic-model chinese-large \
  --target-canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_book_b_xxx.db \
  --target-neo4j-uri bolt://localhost:7689 \
  --target-neo4j-user neo4j \
  --target-neo4j-pass novel123 \
  --target-neo4j-database neo4j \
  --target-qdrant-url http://localhost:6333 \
  --target-qdrant-collection novel_04_book_b_xxx \
  --template-canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_v2_test.db \
  --template-neo4j-uri bolt://localhost:7687 \
  --template-neo4j-user neo4j \
  --template-neo4j-pass novel123 \
  --template-neo4j-database neo4j \
  --template-qdrant-url http://localhost:6333 \
  --template-qdrant-collection novel_04_assets_v2_test
```

## 5. B 书每章 injection 如何注入

每章生成前会写日志并把同一份结构喂给 LLM：
- `template_pack_from_bookA`
- `hard_context_from_bookB`
- `recent_summaries_bookB`

日志路径：
- `logs/generate_book_ab/<target_book_id>_<timestamp>/chapters/<chapter_no>_pre_generation_injection.json`

**当前实现已改为 name-only 注入**：`hard_context_from_bookB.prev_context` 仅保留可读字段（`canonical_name/name`），不再注入 `entity_id/from_id/to_id` 到生成模型。

## 6. B 书蓝图与 A 书模板如何生成

### 6.1 A 书模板画像（Book-level）

从 A 的三层记忆抽取：
- Qdrant：`plot/style/conflict` + digest 模板（语义检索优先，失败回退 token/scroll）
- Canon：最近 `HARD_RULE` 与关系历史
- Neo4j：关系类型分布

### 6.2 B 书蓝图

`build_blueprint` 使用：
- `world_spec`
- `template_profile_from_bookA`

产出：
- `book_title/genre/global_arc`
- `chapters[]`，每章至少规范为：
  `chapter_no/title/goal/conflict/beat_outline/ending_hook`

## 7. B 书数据库怎么写入（逐章）

当 `--commit-memory` 打开后，B 每章写入流程：
1. 章节正文 + 摘要生成。
2. `process_chapter(mode="llm")` 提取 delta（含 `prev_context`）。
3. Canon 写 `fact_history/relationship_history` 并刷新快照。
4. Neo4j 写 Chapter/Chunk/Entity/Relations/Event/Thread。
5. Qdrant 写 digest 与模板资产向量点。
6. commit 状态推进到 `ALL_DONE`。

## 8. 快速核验（建议）

- Canon：`commit_log` 中目标 `book_id` 的 `ALL_DONE` 章数是否等于目标章数。
- Neo4j：`MATCH (c:Chapter {book_id:$book_id}) RETURN count(c)`。
- Qdrant：按 `book_id` 过滤计数 points。
- Injection：抽查 `*_pre_generation_injection.json`，确认 B 侧硬上下文为中文 name 字段。
