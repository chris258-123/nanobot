#!/bin/bash
# Supermemory集成快速设置脚本

set -e

echo "🚀 Supermemory集成设置"
echo "====================="
echo ""

# 检查是否在nanobot目录
if [ ! -f "nanobot/agent/loop.py" ]; then
    echo "❌ 错误：请在nanobot项目根目录运行此脚本"
    exit 1
fi

# 1. 检查supermemory工具文件
echo "📝 检查supermemory工具文件..."
if [ -f "nanobot/agent/tools/supermemory.py" ]; then
    echo "✅ supermemory.py 已存在"
else
    echo "❌ supermemory.py 不存在，请先创建"
    exit 1
fi

# 2. 安装依赖
echo ""
echo "📦 安装依赖..."
pip install httpx

# 3. 注册工具
echo ""
echo "🔧 注册supermemory工具..."

# 检查是否已注册
if grep -q "SupermemoryTool" nanobot/agent/loop.py; then
    echo "✅ SupermemoryTool 已注册"
else
    echo "📝 添加导入语句..."
    # 在导入部分添加
    sed -i '/from nanobot.agent.tools.browser import BrowserTool/a from nanobot.agent.tools.supermemory import SupermemoryTool' nanobot/agent/loop.py

    echo "📝 注册工具..."
    # 在_register_default_tools方法中添加
    sed -i '/self.tools.register(BrowserTool())/a \        \n        # Supermemory tool (for semantic memory)\n        self.tools.register(SupermemoryTool())' nanobot/agent/loop.py

    echo "✅ SupermemoryTool 已注册"
fi

# 4. 配置API密钥
echo ""
echo "🔑 配置API密钥..."
echo ""
echo "请输入你的Supermemory API密钥（留空跳过）："
read -r API_KEY

if [ -n "$API_KEY" ]; then
    # 添加到环境变量
    if ! grep -q "SUPERMEMORY_API_KEY" ~/.bashrc; then
        echo "export SUPERMEMORY_API_KEY=\"$API_KEY\"" >> ~/.bashrc
        echo "✅ API密钥已添加到 ~/.bashrc"
    fi

    # 立即设置环境变量
    export SUPERMEMORY_API_KEY="$API_KEY"
    echo "✅ API密钥已设置"
else
    echo "⚠️  跳过API密钥配置"
    echo "   稍后可以通过以下方式设置："
    echo "   export SUPERMEMORY_API_KEY=\"your-key\""
fi

# 5. 更新TOOLS.md
echo ""
echo "📚 更新TOOLS.md文档..."

if grep -q "### supermemory" workspace/TOOLS.md 2>/dev/null; then
    echo "✅ TOOLS.md 已包含supermemory文档"
else
    cat >> workspace/TOOLS.md << 'EOF'

## Semantic Memory (Supermemory)

### supermemory
Store and retrieve memories using semantic search powered by Supermemory.

```
supermemory(
    action: str,           # "store", "search", or "recall"
    content: str = None,   # Content to store (for store action)
    query: str = None,     # Search query (for search action)
    tags: list = None,     # Tags for categorization
    limit: int = 5         # Number of results
) -> str
```

**Actions:**
- `store`: Store a memory with optional tags
- `search`: Search memories semantically using vector similarity
- `recall`: Get recent memories

**Examples:**
```python
# Store a memory
supermemory(action="store", content="User prefers dark mode", tags=["preferences", "ui"])

# Search semantically
supermemory(action="search", query="what are user's UI preferences?", limit=5)

# Get recent memories
supermemory(action="recall", limit=10)
```

**Notes:**
- Requires `SUPERMEMORY_API_KEY` environment variable
- Supports semantic search using vector embeddings
- Tags help organize and categorize memories
- Search uses natural language queries

EOF
    echo "✅ TOOLS.md 已更新"
fi

# 6. 测试集成
echo ""
echo "🧪 测试集成..."
echo ""

if [ -n "$API_KEY" ]; then
    echo "运行测试命令："
    echo "  nanobot agent -m \"使用supermemory存储：测试记忆集成\""
    echo ""
    read -p "是否现在运行测试？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        nanobot agent -m "使用supermemory存储：测试记忆集成成功"
    fi
else
    echo "⚠️  未配置API密钥，跳过测试"
fi

# 完成
echo ""
echo "✅ Supermemory集成设置完成！"
echo ""
echo "📖 查看完整文档："
echo "   cat docs/SUPERMEMORY_INTEGRATION.md"
echo ""
echo "🚀 开始使用："
echo "   nanobot agent -m \"使用supermemory存储：我喜欢Python编程\""
echo "   nanobot agent -m \"使用supermemory搜索：我的编程偏好\""
echo ""
