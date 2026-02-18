# 生产版运行手册（A/B 三层记忆）

本手册用于长任务稳定执行：A 书全量重建、B 书生成与三层记忆提交、失败恢复与核验。

## 1. 运行前检查（一次性）

```bash
conda activate nanobot
cd /home/chris/Desktop/my_workspace/nanobot
```

确认服务：
- Neo4j(A)：`bolt://localhost:7687`
- Neo4j(B)：`bolt://localhost:7689`
- Qdrant：`http://localhost:6333`

建议固定环境变量：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 2. A 书全量重建（默认 large chinese）

后台运行（tmux）：

```bash
tmux new -s a_rebuild -d \
"conda run -n nanobot python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id novel_04 \
  --asset-dir /home/chris/Desktop/my_workspace/novel_data/04/novel_assets \
  --from-chapter 0001 \
  --llm-config /tmp/llm_config_claude.json \
  --canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_v2.db \
  --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass novel123 --neo4j-database neo4j \
  --qdrant-url http://localhost:6333 --qdrant-collection novel_04_assets_v2 \
  --reset-canon --reset-neo4j --reset-qdrant \
  --embedding-model chinese-large \
  --log-file /home/chris/Desktop/my_workspace/novel_data/04/log/reprocess_full.log"
```

查看进度：
- `tmux attach -t a_rebuild`
- `tail -f /home/chris/Desktop/my_workspace/novel_data/04/log/reprocess_full.log`

## 3. B 书生产生成（A/B 强隔离 + 提交记忆）

### 3.1 先建独立 Qdrant collection（避免 404）

```bash
python -c "import httpx; r=httpx.put('http://localhost:6333/collections/novel_04_book_b_prod', json={'vectors': {'size': 1024, 'distance': 'Cosine'}}, timeout=20.0, trust_env=False); print(r.status_code, r.text)"
```

### 3.2 后台生成（tmux）

```bash
tmux new -s b_gen -d \
"cd /home/chris/Desktop/my_workspace/nanobot && \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 conda run -n nanobot python nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
  --target-book-id novel_04_book_b_prod \
  --template-book-id novel_04_test \
  --world-config nanobot/skills/novel-workflow/templates/world_spec.example.json \
  --chapter-count 1000 \
  --start-chapter 1 \
  --output-dir /home/chris/Desktop/my_workspace/novel_data/04/book_b_output \
  --llm-config /tmp/llm_config_claude.json \
  --commit-memory \
  --template-semantic-model chinese-large \
  --target-canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_book_b_prod.db \
  --target-neo4j-uri bolt://localhost:7689 --target-neo4j-user neo4j --target-neo4j-pass novel123 --target-neo4j-database neo4j \
  --target-qdrant-url http://localhost:6333 --target-qdrant-collection novel_04_book_b_prod \
  --template-canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_v2_test.db \
  --template-neo4j-uri bolt://localhost:7687 --template-neo4j-user neo4j --template-neo4j-pass novel123 --template-neo4j-database neo4j \
  --template-qdrant-url http://localhost:6333 --template-qdrant-collection novel_04_assets_v2_test"
```

断点续跑：在同命令末尾加 `--resume`。

## 4. 生产核验（必须做）

### 4.1 Canon（SQLite）

```bash
python -c "import sqlite3; conn=sqlite3.connect('/home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_book_b_prod.db'); cur=conn.cursor(); cur.execute(\"select status,count(*) from commit_log where book_id='novel_04_book_b_prod' group by status\"); print(cur.fetchall())"
```

### 4.2 Neo4j

```cypher
MATCH (c:Chapter {book_id:'novel_04_book_b_prod'}) RETURN count(c);
```

### 4.3 Qdrant

按 `book_id=novel_04_book_b_prod` 统计 points，确认持续增长。

### 4.4 Injection 格式

抽查最新 `*_pre_generation_injection.json`：
- `hard_context_from_bookB.prev_context` 仅应有 `name/canonical_name`
- 不应出现 `entity_id/from_id/to_id`

## 5. 常见故障处理

- **HuggingFace 超时**：确保 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
- **Qdrant 404**：先创建 collection（第 3.1 步）。
- **strict_blocking 终止**：保留数据，使用 `--resume` 重启。
- **资产抽取 JSON 解析警告**：单章偶发可继续；若连续出现，先检查 LLM 稳定性与 token 设置。

## 6. 推荐日志路径

- A 重建日志：`/home/chris/Desktop/my_workspace/novel_data/04/log/reprocess_full.log`
- B 运行报告：`/home/chris/Desktop/my_workspace/novel_data/04/book_b_output/<book_id>_run_report.json`
- B injection：`/home/chris/Desktop/my_workspace/nanobot/logs/generate_book_ab/<book_id>_<timestamp>/`
