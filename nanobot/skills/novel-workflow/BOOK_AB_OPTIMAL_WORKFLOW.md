# Book A/B 最优工作流（生产参数版）

本文给出一套可直接执行的流程：  
1) 用最优参数构建 **Book A 三层记忆**（Canon + Neo4j + Qdrant）  
2) 在 **A/B 物理隔离** 前提下生成 **Book B** 并持续写入三层记忆  
3) 提供核验与常见问题处理建议

---

## 0. 前置约束（必须满足）

- Python 环境：已安装项目与依赖（建议 `pip install -e ".[dev]"`）
- 服务：
  - Neo4j-A：`bolt://localhost:7687`
  - Neo4j-B：`bolt://localhost:7689`（必须与 A 不同）
  - Qdrant：`http://localhost:6333`
- LLM 配置文件（建议）：`/home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json`
- 推荐离线 embedding 环境变量：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 1. Book A 三层记忆构建（推荐参数）

### 1.1 质量优先（推荐）——使用 `--chapter-dir`

如果你的目标是「三层记忆质量 + 可追溯证据链」最优，优先用正文输入。  
该模式会保留更完整的 chunk 结构（Neo4j `Chunk` 层通常更完整）。

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_04_test \
  --chapter-dir /home/chris/Desktop/my_workspace/novel_data/04/new_book \
  --from-chapter 0001 \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json \
  --canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_v2_test.db \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass novel123 \
  --neo4j-database neo4j \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_04_assets_v2_test \
  --embedding-model chinese-large \
  --llm-max-retries 3 \
  --llm-retry-backoff 3 \
  --llm-backoff-factor 2 \
  --llm-backoff-max 60 \
  --llm-retry-jitter 0.5 \
  --llm-min-interval 1.0 \
  --reset-canon --reset-neo4j --reset-qdrant
```

### 1.2 速度优先——使用 `--asset-dir`

如果你已有高质量 8 要素资产，且更重视吞吐速度，可用 `--asset-dir`。  
注意：该模式在 chunk 证据链上通常弱于 `--chapter-dir`。

```bash
python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_04_test \
  --asset-dir /home/chris/Desktop/my_workspace/novel_data/04/novel_assets \
  --from-chapter 0001 \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json \
  --canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_v2_test.db \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass novel123 \
  --neo4j-database neo4j \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection novel_04_assets_v2_test \
  --embedding-model chinese-large \
  --llm-max-retries 3 \
  --llm-retry-backoff 3 \
  --llm-backoff-factor 2 \
  --llm-backoff-max 60 \
  --llm-retry-jitter 0.5 \
  --llm-min-interval 1.0 \
  --reset-canon --reset-neo4j --reset-qdrant
```

### A 侧核验

1. Canon：`commit_log` 中 `book_id=novel_04_test` 的 `ALL_DONE` 章数应等于目标章数  
2. Neo4j：`MATCH (c:Chapter {book_id:'novel_04_test'}) RETURN count(c);`  
3. Qdrant：按 `book_id=novel_04_test` 统计 points，并检查 8 要素类型均存在：
   - `plot_beat, character_card, conflict, setting, theme, pov, tone, style`

---

## 2. A/B 隔离生成 Book B（推荐参数）

`generate_book_ab.py` 默认支持模板语义召回；生产建议开启语义检索并指定 `chinese-large`。

```bash
python nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
  --target-book-id novel_04_book_b_prod \
  --template-book-id novel_04_test \
  --world-config nanobot/skills/novel-workflow/templates/world_spec.example.json \
  --chapter-count 100 \
  --start-chapter 1 \
  --output-dir /home/chris/Desktop/my_workspace/novel_data/04/book_b_output \
  --llm-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config.json \
  --commit-memory \
  --enforce-isolation \
  --template-semantic-search \
  --template-semantic-model chinese-large \
  --reference-top-k 8 \
  --consistency-policy warn_only \
  --llm-max-retries 3 \
  --llm-retry-backoff 3 \
  --llm-backoff-factor 2 \
  --llm-backoff-max 60 \
  --llm-retry-jitter 0.5 \
  --target-canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_book_b_prod.db \
  --target-neo4j-uri bolt://localhost:7689 \
  --target-neo4j-user neo4j \
  --target-neo4j-pass novel123 \
  --target-neo4j-database neo4j \
  --target-qdrant-url http://localhost:6333 \
  --target-qdrant-collection novel_04_book_b_prod \
  --template-canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_v2_test.db \
  --template-neo4j-uri bolt://localhost:7687 \
  --template-neo4j-user neo4j \
  --template-neo4j-pass novel123 \
  --template-neo4j-database neo4j \
  --template-qdrant-url http://localhost:6333 \
  --template-qdrant-collection novel_04_assets_v2_test \
  --log-dir /home/chris/Desktop/my_workspace/nanobot/logs \
  --log-injections
```

### A/B 隔离强规则

- Canon：A/B 必须不同 `.db` 路径  
- Neo4j：A/B 必须不同 `(uri + database)`  
- Qdrant：A/B 必须不同 collection

---

## 3. 注入机制说明（当前推荐）

每章生成前会注入三块：

1. `template_pack_from_bookA`  
   - 来自 A 的 Qdrant 模板召回（优先语义检索，失败回退 token/scroll）
2. `hard_context_from_bookB`  
   - 来自 B 的 Canon 历史：`HARD_RULE` + `prev_context`（角色状态/近期关系/open_threads）
3. `recent_summaries_bookB`  
   - 最近章节摘要滚动窗口

注入日志路径：

`logs/generate_book_ab/<target_book_id>_<timestamp>/chapters/*_pre_generation_injection.json`

当前实现已做 **name-only 注入清洗**（给 LLM 的 clean 日志中不再出现内部 `entity_id/from_id/to_id/subject_id`）。

---

## 4. B 侧核验清单

1. Canon：

```sql
SELECT status, COUNT(*) FROM commit_log
WHERE book_id='novel_04_book_b_prod'
GROUP BY status;
```

2. Neo4j：

```cypher
MATCH (c:Chapter {book_id:'novel_04_book_b_prod'}) RETURN count(c);
```

3. Qdrant：
- 按 `book_id=novel_04_book_b_prod` 统计 points
- 抽查 8 要素与 digest 类型（`chapter/fact/relation_digest`）

4. 注入：
- clean 注入日志不应出现 `entity_id/from_id/to_id/subject_id`
- raw 注入日志允许保留内部 ID（用于排障）

---

## 5. 常见问题与建议

- **8要素偶发为空**：多为 LLM 返回 JSON 不稳定，触发资产抽取回退为空结构。  
  建议提高重试和最小间隔，必要时对失败章 `--resume` 重新提交。
- **语义召回不理想**：确认 A/B Qdrant 向量维度一致（`chinese-large`=1024），并开启 `--template-semantic-search`。
- **运行中断**：直接加 `--resume` 续跑。
