# Novel Workflow Quick Start Guide

## 快速开始 (Quick Start)

### 1. 安装依赖 (Install Dependencies)

```bash
# 激活 conda 环境
conda activate nanobot

# 安装 Python 依赖
pip install -r requirements-novel.txt

# 安装 Beads (需要 Rust/Cargo)
git clone https://github.com/steveyegge/beads.git
cd beads
cargo build --release
sudo cp target/release/bd /usr/local/bin/
bd --version
```

### 2. 启动服务 (Start Services)

```bash
# 启动 Docker 服务
docker-compose up -d

# 验证服务状态
docker-compose ps
curl http://localhost:6333/collections  # Qdrant
curl http://localhost:8283/v1/agents    # Letta
```

### 3. 配置 nanobot (Configure)

编辑 `~/.nanobot/config.json`，添加:

```json
{
  "integrations": {
    "qdrant": {
      "enabled": true,
      "url": "http://localhost:6333",
      "collection_name": "novel_assets"
    },
    "letta": {
      "enabled": true,
      "url": "http://localhost:8283"
    },
    "beads": {
      "enabled": true,
      "workspace_path": "~/.beads"
    }
  }
}
```

### 4. 创建 LLM 配置 (Create LLM Config)

创建 `llm_config.json`:

```json
{
  "type": "custom",
  "url": "https://api.deepseek.com/v1/chat/completions",
  "model": "deepseek-chat",
  "api_key": "sk-your-api-key"
}
```

### 5. 测试工具 (Test Tools)

```bash
# 测试 Qdrant
nanobot agent -m "Use qdrant tool to create_collection"

# 测试 Letta
nanobot agent -m "Use letta tool to create a writer agent"

# 测试 Beads
nanobot agent -m "Use beads tool to add a task titled 'Test'"
```

## 工作流程 (Workflows)

### A. 构建小说库 (Build Novel Library)

```bash
# 1. 爬取小说章节
nanobot agent -m "Use novel-crawler to crawl chapters from [URL] to ~/novel_data/book1"

# 2. 提取资产
python nanobot/skills/novel-workflow/scripts/asset_extractor.py \
  --book-id book1 \
  --chapter-dir ~/novel_data/book1 \
  --output-dir ~/novel_assets \
  --llm-config llm_config.json

# 3. 嵌入并上传到 Qdrant
for asset in ~/novel_assets/book1_*.json; do
  python nanobot/skills/novel-workflow/scripts/embedder.py --assets "$asset"
done

# 4. 验证
nanobot agent -m "Use qdrant tool to scroll book_id book1"
```

### B. 生成新章节 (Generate New Chapter)

```bash
# 1. 组装上下文包
python nanobot/skills/novel-workflow/scripts/context_pack.py \
  --template-book-id book1 \
  --new-book-context new_book.json \
  --output context_pack.json

# 2. 使用 Writer agent 生成章节
nanobot agent -m "Use letta tool to send the context pack to writer agent"

# 3. 提取新章节的资产
python nanobot/skills/novel-workflow/scripts/asset_extractor.py \
  --book-id book2 \
  --chapter-dir ~/generated_chapters \
  --output-dir ~/novel_assets \
  --llm-config llm_config.json
```

## 可用工具 (Available Tools)

- `qdrant` - 向量数据库操作
- `letta` - Agent 记忆管理
- `beads` - 任务跟踪
- `novel_orchestrator` - 高级工作流协调

## 故障排除 (Troubleshooting)

### Qdrant 连接错误
```bash
docker-compose ps
docker-compose logs qdrant
```

### Letta 无响应
```bash
docker-compose logs letta
curl http://localhost:8283/v1/agents
```

### Beads 命令未找到
```bash
which bd
# 如果没有，重新安装: cd beads && cargo install --path .
```

### Python 依赖缺失
```bash
conda activate nanobot
pip install -r requirements-novel.txt
```

## 文件位置 (File Locations)

- 配置: `~/.nanobot/config.json`
- 技能: `nanobot/skills/novel-workflow/`
- 脚本: `nanobot/skills/novel-workflow/scripts/`
- 工具: `nanobot/agent/tools/`
- Docker: `docker-compose.yml`
- 依赖: `requirements-novel.txt`

## 下一步 (Next Steps)

1. 完成安装和配置
2. 运行测试验证
3. 爬取模板小说
4. 处理模板小说构建资产库
5. 使用模板生成新章节

详细文档请参考: `NOVEL_WORKFLOW_IMPLEMENTATION.md`
