# Phase 1 三层记忆系统修复总结

## 修复前状态（问题诊断）

| 指标 | 修复前 | 问题 |
|------|--------|------|
| 实体数 | 80 | 脏名（括号别名如"荆璜（玄虹）"、垃圾如"∈先生"、泛称如"母亲"） |
| fact_history | **0** | 从未写入任何事实 |
| relationship_history | 951 | 全部自由文本中文（"多年好友，关系密切"），非结构化 |
| 关系图形态 | 星形 | 主角中心化提取，非成对关系 |
| 实体注册表 | 无 | 无别名解析，无entity_id稳定引用 |

### 根因分析

1. `load_asset_file()` 生成空 `fact_changes: []`，事实从未进入数据库
2. 关系直接传入LLM原始描述作为 `kind`，无分类映射
3. 关系 `from_id`/`to_id` 使用原始名字字符串而非entity_id
4. 无名称清洗层，括号、Unicode垃圾、泛称角色直接入库
5. 关系提取以主角为中心辐射，缺少事件驱动的成对关系

## 修复方案与实现

### 新增模块

| 文件 | 职责 |
|------|------|
| `name_normalizer.py` | Unicode NFKC标准化、括号/斜杠拆分、别名提取、噪声过滤、关系类型分类 |
| `delta_converter.py` | 将LLM原始资产JSON转为结构化delta（实体、事实、关系、事件、伏笔） |
| `test_phase1_fixed.py` | 5项集成测试，覆盖全部验收标准 |
| `reprocess_all.py` | 100章全量重处理脚本 |

### 增强模块

| 文件 | 改动 |
|------|------|
| `canon_db_v2.py` | 别名感知的`normalize_entity()`、`merge_entities()`、聚合式`update_current_snapshots()`、`get_statistics()` |
| `neo4j_manager.py` | `get_statistics()`、`clear_all()`、`get_all_entities()` |
| `chapter_processor.py` | 完全重写：名称→ID解析、commit驱动三库一致性、事件写入、伏笔线程 |

### 结构化关系类型（10种）

```
ALLY | ENEMY | FAMILY | MENTOR | ROMANTIC
HIERARCHY | COLLEAGUE | RIVAL | CO_PARTICIPANT | ASSOCIATE
```

关键词映射覆盖常见中文关系描述，`ASSOCIATE` 为兜底类型，`CO_PARTICIPANT` 由事件参与者成对生成。

## 修复后数据（100章全量重处理）

### Canon DB

| 指标 | 数值 | 对比 |
|------|------|------|
| 实体注册表 | 149 | 去重后（原始379次出现） |
| 角色数（character_current） | 54 | 有状态快照的角色 |
| fact_history | **1,446** | 0 → 1,446 |
| relationship_history | **1,371** | 951脏 → 1,371结构化 |
| 提交数 | 100 | 100/100 ALL_DONE |
| 有别名的实体 | 7 | — |

### Neo4j

| 指标 | 数值 |
|------|------|
| 角色节点 | 54 |
| 章节节点 | 100 |
| 事件节点 | 509 |
| 关系边 | 1,210 |
| 伏笔线程 | 144 |

### 关系类型分布

| 类型 | Canon DB | Neo4j | 占比 |
|------|----------|-------|------|
| ASSOCIATE | 554 | 469 | 40.4% |
| CO_PARTICIPANT | 513 | 462 | 37.4% |
| COLLEAGUE | 138 | 134 | 10.1% |
| ALLY | 89 | 84 | 6.5% |
| ENEMY | 32 | 27 | 2.3% |
| HIERARCHY | 20 | 15 | 1.5% |
| MENTOR | 11 | 9 | 0.8% |
| FAMILY | 9 | 6 | 0.7% |
| ROMANTIC | 4 | 3 | 0.3% |
| RIVAL | 1 | 1 | 0.1% |

### 事实类型分布（4维追踪）

- `status`: HARD_STATE层，每章每角色记录状态
- `traits`: SOFT_NOTE层，性格特征
- `goals`: SOFT_NOTE层，角色目标
- `secrets`: SOFT_NOTE层（implied），角色秘密

### Top 5 活跃角色（按事实记录数）

1. 罗彬瀚 — 350 facts（第1章首次出现）
2. 荆璜 — 210 facts（第2章首次出现）
3. 莫莫罗 — 185 facts
4. 雅莱丽伽 — 135 facts
5. 宓谷拉 — 70 facts

## 可视化产出

| 文件 | 内容 |
|------|------|
| `canon_entity_distribution.png` | 实体类型分布（当前全部为character） |
| `canon_fact_timeline.png` | 事实历史时间线（4维按章节） |
| `canon_relationship_changes.png` | 关系变更统计（INSERT/UPDATE/DELETE按章节） |
| `canon_commit_status.png` | 提交状态分布（100% ALL_DONE） |
| `canon_top_characters.png` | Top 20活跃角色排行 |
| `neo4j_character_network.png` | 角色关系网络图 |
| `neo4j_events_timeline.png` | 事件时间线 |

## 集成测试结果

```
PASS  Name Normalizer        — 括号拆分、噪声过滤、批量去重
PASS  Relation Classifier    — 10种关系类型分类（10个测试用例）
PASS  Delta Converter        — 100章真实资产，无脏名，1446事实，10种关系类型
PASS  Canon DB Pipeline      — 10章端到端，事实>0，结构化关系，别名工作，非星形图
PASS  Entity Merge           — 别名查找、last_seen更新、不同实体不同ID
```

## 遗留问题与Phase 2改进方向

### P0: 噪声实体仍有漏网

当前GENERIC_ROLES集合未覆盖的模式：
- 描述性短语："皮肤最白的修女"、"荆璜的仇人"
- 泛称变体："其他修女"、"修女"
- **建议**：增加长度>4且含"的"的名称过滤规则；增加更多泛称模式

### P1: ASSOCIATE兜底比例过高（40%）

554条关系落入ASSOCIATE，说明关键词映射覆盖不足。
- **建议**：分析ASSOCIATE关系的原始description，扩充RELATION_TYPE_MAP关键词
- 考虑引入LLM二次分类（对ASSOCIATE关系做batch reclassification）

### P1: 实体去重不完整

- "雅莱"和"雅莱丽伽"可能是同一角色（子串关系），但未合并
- Canon DB 149实体 vs Neo4j 54实体的差距（95个实体只在关系目标中出现）
- **建议**：增加子串匹配的别名候选；对只出现在关系目标中的实体做二次验证

### P2: 关系图仍偏星形

主角荆璜仍是最大hub节点，这部分是小说结构决定的，但可以改善：
- **建议**：对CO_PARTICIPANT关系增加权重衰减（同一事件参与者过多时降权）
- 增加章节间的关系演化追踪（UPDATE/DELETE操作）

### P2: 缺少location/item/rule实体类型

当前所有149个实体都是character类型，缺少：
- 地点实体（location）
- 物品实体（item）
- 世界规则实体（rule）
- **建议**：扩展asset_extractor的提取prompt，增加非角色实体提取

### P3: Canon DB ↔ Neo4j实体数不一致

Canon DB有149个实体，Neo4j只有54个。差距原因：关系目标名称在Canon DB中被`normalize_entity()`自动创建，但不会写入Neo4j（因为不在`entities_new`中）。
- **建议**：在关系解析阶段，对新创建的实体也同步写入Neo4j
