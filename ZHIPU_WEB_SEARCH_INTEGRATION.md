# 智谱网络搜索工具集成文档

## 概述

成功将智谱 AI 的网络搜索功能集成到 nanobot 中，新增 `zhipu_web_search` 工具。

## 修改文件清单

### 1. `/nanobot/agent/tools/web.py`

**修改内容**：添加 `ZhipuWebSearchTool` 类

```python
class ZhipuWebSearchTool(Tool):
    """Search the web using Zhipu AI Web Search API."""

    name = "zhipu_web_search"
    description = "Search the web using Zhipu AI. Returns titles, URLs, and content snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-50)", "minimum": 1, "maximum": 50},
            "search_engine": {
                "type": "string",
                "enum": ["search_std", "search_pro", "search_pro_sogou", "search_pro_quark"],
                "description": "Search engine type (default: search_pro)"
            }
        },
        "required": ["query"]
    }
```

**功能特点**：
- 支持 1-50 条搜索结果（比 Brave Search 的 1-10 更灵活）
- 支持 4 种搜索引擎：search_std, search_pro, search_pro_sogou, search_pro_quark
- 自动处理代理问题（禁用 socks 代理）
- 格式化输出包含标题、链接、内容摘要和发布日期

### 2. `/nanobot/agent/loop.py`

**修改 1**：导入新工具类
```python
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool, ZhipuWebSearchTool
```

**修改 2**：添加 `zhipu_api_key` 参数到 `__init__` 方法
```python
def __init__(
    self,
    ...
    brave_api_key: str | None = None,
    zhipu_api_key: str | None = None,  # 新增
    ...
):
    ...
    self.zhipu_api_key = zhipu_api_key  # 新增
```

**修改 3**：注册智谱搜索工具
```python
def _register_default_tools(self) -> None:
    ...
    # Web tools
    self.tools.register(WebSearchTool(api_key=self.brave_api_key))
    self.tools.register(ZhipuWebSearchTool(api_key=self.zhipu_api_key))  # 新增
    self.tools.register(WebFetchTool())
```

### 3. `/nanobot/providers/litellm_provider.py`

**修改内容**：添加调试日志功能

**位置 1**：保存请求调试信息（第 150-162 行）
```python
# Save debug info to file
debug_file = Path("/tmp/nanobot_request_debug.json")
try:
    with open(debug_file, "w") as f:
        json_module.dump({
            "model": kwargs.get("model"),
            "messages_count": len(kwargs.get("messages", [])),
            "first_message": kwargs.get("messages", [{}])[0] if kwargs.get("messages") else None,
            "tools_count": len(tools),
            "first_tool": tools[0] if tools else None,
        }, f, indent=2, ensure_ascii=False)
    logger.debug(f"Request debug saved to {debug_file}")
except Exception as e:
    logger.warning(f"Failed to save debug info: {e}")
```

**位置 2**：记录工具调用日志（第 167-175 行）
```python
# Debug: log tool calls from response
from loguru import logger
if hasattr(response, 'choices') and len(response.choices) > 0:
    choice = response.choices[0]
    if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls'):
        if choice.message.tool_calls:
            logger.debug(f"Received {len(choice.message.tool_calls)} tool calls")
            for tc in choice.message.tool_calls:
                logger.debug(f"Tool: {tc.function.name}, Args: {tc.function.arguments}")
```

**功能说明**：
- 每次 LLM 请求都会保存调试信息到 `/tmp/nanobot_request_debug.json`
- 包含模型名称、消息数量、第一条消息内容、工具数量和第一个工具定义
- 响应中的工具调用会记录到日志中，方便调试
- 调试信息以 JSON 格式保存，便于查看和分析

### 4. `/nanobot/cli/commands.py`

**修改位置 1**：gateway 命令（第 209 行）
```python
agent = AgentLoop(
    ...
    brave_api_key=config.tools.web.search.api_key or None,
    zhipu_api_key=config.providers.zhipu.api_key or None,  # 新增
    ...
)
```

**修改位置 2**：agent 命令（第 306 行）
```python
agent_loop = AgentLoop(
    ...
    brave_api_key=config.tools.web.search.api_key or None,
    zhipu_api_key=config.providers.zhipu.api_key or None,  # 新增
    ...
)
```

## 使用方法

### 1. 配置要求

确保 `~/.nanobot/config.json` 中配置了智谱 API 密钥：

```json
{
  "providers": {
    "zhipu": {
      "apiKey": "your-zhipu-api-key-here"
    }
  }
}
```

### 2. 命令行使用

```bash
# 禁用代理（如果有 socks 代理）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 使用智谱搜索
nanobot agent -m "使用智谱搜索工具搜索：2026年春节假期安排"
```

### 3. Agent 自动调用

Agent 会根据用户需求自动选择合适的搜索工具：
- `web_search`：Brave Search（需要 Brave API key）
- `zhipu_web_search`：智谱搜索（使用智谱 API key）

## 工具参数说明

### zhipu_web_search

| 参数 | 类型 | 必需 | 说明 | 默认值 |
|------|------|------|------|--------|
| query | string | 是 | 搜索查询词 | - |
| count | integer | 否 | 返回结果数量（1-50） | 5 |
| search_engine | string | 否 | 搜索引擎类型 | search_pro |

**search_engine 选项**：
- `search_std`：标准搜索（¥0.01/次）
- `search_pro`：专业搜索，多引擎聚合（¥0.03/次）
- `search_pro_sogou`：搜狗专业搜索（¥0.05/次）
- `search_pro_quark`：夸克专业搜索（¥0.05/次）

## 测试结果

### 测试 1：基础搜索
```bash
nanobot agent -m "使用智谱搜索工具搜索：2026年春节假期安排"
```

**结果**：✅ 成功
- Agent 正确调用 `zhipu_web_search` 工具
- 参数：`{"count": 5, "query": "2026年春节假期安排", "search_engine": "search_pro"}`
- 返回准确的搜索结果并生成完整回答

### 测试 2：工具单元测试
```python
tool = ZhipuWebSearchTool(api_key=api_key, max_results=3)
result = await tool.execute(query="Python 编程", count=3)
```

**结果**：✅ 成功
- 返回格式化的搜索结果
- 包含标题、链接、内容摘要、发布日期

## 调试信息

### Debug 文件位置

所有 LLM 请求的调试信息保存在：`/tmp/nanobot_request_debug.json`

### Debug 文件内容示例

```json
{
  "model": "zai/glm-4-flash",
  "messages_count": 52,
  "first_message": {
    "role": "system",
    "content": "# nanobot 🐈\n\nYou are nanobot..."
  },
  "tools_count": 10,
  "first_tool": {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read the contents of a file at the given path.",
      "parameters": {...}
    }
  }
}
```

### 查看调试信息

```bash
# 查看最新的请求调试信息
cat /tmp/nanobot_request_debug.json | jq

# 实时监控调试日志
tail -f ~/.nanobot/logs/nanobot.log | grep -E "DEBUG|Tool:"
```

### 工具调用日志

当 Agent 调用工具时，会在日志中输出：
```
2026-02-09 01:04:27.775 | DEBUG | nanobot.providers.litellm_provider:chat:173 - Received 1 tool calls
2026-02-09 01:04:27.775 | DEBUG | nanobot.providers.litellm_provider:chat:175 - Tool: zhipu_web_search, Args: {"count": 5, "query": "2026年春节假期安排", "search_engine": "search_pro"}
```

## 依赖项

- `zai-sdk==0.2.2`：智谱 AI SDK（已安装）
- 智谱 API 密钥：从 https://open.bigmodel.cn/usercenter/apikeys 获取

## 注意事项

1. **代理问题**：如果系统配置了 socks 代理，需要临时禁用或安装 `httpx[socks]`
2. **API 费用**：
   - search_std: ¥0.01/次
   - search_pro: ¥0.03/次（推荐）
   - search_pro_sogou/quark: ¥0.05/次
3. **结果数量**：最多支持 50 条结果，远超 Brave Search 的 10 条限制
4. **内容截断**：为提高可读性，内容摘要自动截断到 200 字符

## 与 Brave Search 对比

| 特性 | Brave Search | 智谱搜索 |
|------|--------------|----------|
| 工具名称 | web_search | zhipu_web_search |
| API 密钥 | Brave API Key | 智谱 API Key |
| 结果数量 | 1-10 | 1-50 |
| 搜索引擎 | Brave | 多引擎聚合 |
| 费用 | ~$5/月（2000次） | ¥0.01-0.05/次 |
| 中文支持 | 一般 | 优秀 |
| 发布日期 | 无 | 有 |

## 后续优化建议

1. **配置化搜索引擎**：允许在 config.json 中配置默认搜索引擎类型
2. **智能选择**：根据查询语言自动选择 Brave 或智谱搜索
3. **缓存机制**：对相同查询结果进行缓存，减少 API 调用
4. **结果排序**：支持按相关性、时间等维度排序
5. **高级过滤**：支持域名过滤、时间范围过滤等高级功能

## 相关文档

- 智谱 Web Search API：https://docs.bigmodel.cn/cn/guide/tools/web-search.md
- nanobot 工具系统：`/nanobot/agent/tools/`
- 配置文件：`~/.nanobot/config.json`
