# 8大核心叙事要素 - 增强版资产提取系统

## 概述

基于叙事写作理论的8大核心要素，我们扩展了资产提取系统，能够从小说章节中提取全面的叙事资产：

### 8大核心要素

1. **人物 (Character)** - 角色的目标、欲望、缺陷、成长弧线
2. **情节 (Plot)** - 事件的因果链和故事发展
3. **冲突 (Conflict)** - 推动故事的对抗与阻力
4. **背景/设定 (Setting)** - 时空、世界观、社会规则
5. **主题 (Theme)** - 核心观念与命题
6. **叙事视角 (Point of View)** - 人称、知识范围、聚焦
7. **语气/氛围 (Tone)** - 整体气质与情绪曲线
8. **文体/风格 (Style)** - 句式、词汇、修辞习惯

## 资产类型详解

### 1. 情节节拍 (plot_beat)

```json
{
  "event": "发生了什么事件",
  "characters": ["涉及的角色名"],
  "impact": "对故事的影响",
  "chapter_position": "beginning|middle|end",
  "causality": "因果关系说明"
}
```

**用途**: 追踪故事发展、理解事件因果、构建情节大纲

### 2. 人物卡片 (character_card)

```json
{
  "name": "角色名",
  "traits": ["性格特质"],
  "state": "当前状态",
  "relationships": {"其他角色": "关系"},
  "goals": ["目标"],
  "flaws": ["缺陷"],
  "growth": "成长变化",
  "arc_type": "flat|positive|negative"
}
```

**用途**: 角色一致性检查、成长弧线追踪、关系网络分析

### 3. 冲突 (conflict)

```json
{
  "type": "人vs人|人vs自我|人vs环境|人vs社会|人vs命运",
  "parties": ["冲突双方"],
  "description": "冲突描述",
  "intensity": "low|medium|high",
  "resolution_status": "unresolved|partially_resolved|resolved",
  "stakes": "赌注/后果"
}
```

**用途**: 张力管理、冲突升级追踪、故事动力分析

### 4. 背景设定 (setting)

```json
{
  "location": "地点",
  "time_period": "时间",
  "atmosphere": "氛围",
  "world_rules": ["规则/限制"],
  "social_context": "社会背景",
  "significance": "对情节的意义"
}
```

**用途**: 世界观一致性、场景设计、环境对情节的影响

### 5. 主题 (theme)

```json
{
  "theme": "主题名称",
  "manifestation": "如何体现",
  "symbols": ["象征"],
  "questions_raised": ["提出的问题"],
  "moral_stance": "道德立场"
}
```

**用途**: 深度分析、主题一致性、象征系统追踪

### 6. 叙事视角 (point_of_view)

```json
{
  "person": "第一/第三/第二人称",
  "knowledge": "全知|限知|客观",
  "focalization": "聚焦角色",
  "distance": "亲密|中等|疏离",
  "reliability": "可靠|不可靠",
  "shifts": ["视角转换"]
}
```

**用途**: 视角一致性检查、信息控制、读者共情管理

### 7. 语气/氛围 (tone)

```json
{
  "overall_tone": "整体语气",
  "emotional_arc": "情绪曲线",
  "mood_keywords": ["氛围关键词"],
  "tension_level": "low|medium|high",
  "pacing": "slow|moderate|fast"
}
```

**用途**: 情绪节奏管理、氛围一致性、节奏控制

### 8. 文体/风格 (style)

```json
{
  "sentence_structure": "句式特点",
  "vocabulary_level": "词汇水平",
  "rhetoric_devices": ["修辞手法"],
  "dialogue_ratio": "对话比例",
  "description_style": "描写风格",
  "distinctive_features": ["独特特征"]
}
```

**用途**: 风格一致性、作者声音识别、模仿学习

## 使用方法

### 基础提取（仅情节和人物）

```bash
python nanobot/skills/novel-workflow/scripts/asset_extractor_enhanced.py \
  --book-id my_novel \
  --chapter-dir ~/novel_chapters \
  --output-dir ~/novel_assets \
  --llm-config llm_config.json
```

### 完整提取（全部8大要素）

```bash
python nanobot/skills/novel-workflow/scripts/asset_extractor_enhanced.py \
  --book-id my_novel \
  --chapter-dir ~/novel_chapters \
  --output-dir ~/novel_assets \
  --llm-config llm_config.json \
  --extract-all
```

**注意**: `--extract-all` 会调用更多 LLM API，处理时间更长，成本更高。

### 嵌入和上传

```bash
python nanobot/skills/novel-workflow/scripts/embedder_enhanced.py \
  --assets ~/novel_assets/my_novel_chapter01_assets.json \
  --qdrant-url http://localhost:6333 \
  --collection novel_assets
```

## 搜索示例

### 按资产类型搜索

```python
# 搜索冲突
response = httpx.post(
    "http://localhost:6333/collections/novel_assets/points/search",
    json={
        "vector": query_vector,
        "filter": {
            "must": [
                {"key": "asset_type", "match": {"value": "conflict"}},
                {"key": "book_id", "match": {"value": "my_novel"}}
            ]
        },
        "limit": 10
    }
)

# 搜索主题
response = httpx.post(
    "http://localhost:6333/collections/novel_assets/points/search",
    json={
        "vector": query_vector,
        "filter": {
            "must": [
                {"key": "asset_type", "match": {"value": "theme"}}
            ]
        },
        "limit": 5
    }
)
```

## 应用场景

### 1. 模板书分析

提取模板书的全部8大要素，用于：
- 理解叙事结构
- 学习风格特征
- 识别成功模式

### 2. 写作一致性检查

- **人物一致性**: 追踪角色特质和成长弧线
- **冲突升级**: 确保冲突强度合理递增
- **主题连贯**: 检查主题是否贯穿始终
- **风格统一**: 保持文体风格一致

### 3. 智能写作辅助

基于提取的资产：
- 生成符合风格的新章节
- 保持角色行为一致
- 延续主题和氛围
- 匹配叙事视角

### 4. 对比分析

比较不同书籍的：
- 冲突类型分布
- 主题表达方式
- 风格特征差异
- 节奏控制策略

## 性能考虑

### API 调用次数

- **基础模式**: 每章 2 次 LLM 调用（情节 + 人物）
- **完整模式**: 每章 8 次 LLM 调用（全部要素）

### 建议策略

1. **模板书**: 使用完整模式，深度分析
2. **新书生成**: 使用基础模式，快速迭代
3. **质量检查**: 针对性提取特定要素（如冲突、主题）

### 成本优化

```bash
# 只提取特定章节的完整要素
for chapter in chapter_001.md chapter_050.md chapter_100.md; do
  python asset_extractor_enhanced.py --extract-all ...
done

# 其他章节使用基础模式
for chapter in chapter_*.md; do
  python asset_extractor_enhanced.py ...  # 不加 --extract-all
done
```

## 测试结果

使用测试小说第100章的提取结果：

- ✅ 情节节拍: 6个
- ✅ 人物卡片: 2个
- ✅ 冲突: 6个（类型：人vs人）
- ✅ 背景设定: 1个
- ✅ 主题: 4个（如：责任与后果）
- ✅ 叙事视角: 第三人称限知
- ✅ 语气氛围: 阴郁
- ✅ 文体风格: 完整提取

## 下一步

1. 处理完整小说，建立全面的资产库
2. 实现资产可视化（冲突图、主题网络、角色关系图）
3. 开发基于资产的智能写作建议系统
4. 创建资产对比分析工具
