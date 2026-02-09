# NEXUS API Gateway 集成文档

## 概述

成功将 NEXUS API 代理网关集成到 nanobot 中，支持通过 NEXUS 使用 Claude 和 GLM 模型。

## 关键发现

### 1. API 格式

NEXUS 使用 **Anthropic 原生 Messages API 格式**（`/v1/messages`），而非 OpenAI 格式。

- 请求格式：Anthropic Messages API
- 认证方式：`x-api-key` header
- API 版本：`anthropic-version: 2023-06-01`

### 2. 模型路由

NEXUS 可能会根据后端配置将请求路由到不同的模型。例如：
- 请求 `claude-opus-4-6` 可能实际使用 `glm-4.7`
- 响应中的 `model` 字段显示实际使用的模型

### 3. 端点结构

正确的 API 端点：
```
https://nexusacc.itssx.com/api/claude_code/cc_glm
```

完整请求路径：
```
https://nexusacc.itssx.com/api/claude_code/cc_glm/v1/messages
```

## 修改文件清单

### 1. `/nanobot/providers/registry.py`

**修改内容**：添加 NEXUS 网关提供商

```python
# NEXUS: API proxy gateway using Anthropic's native API format
# Supports both Claude and GLM models, but uses Anthropic Messages API
# Note: NEXUS may route requests to different models (e.g., GLM) regardless of requested model name
ProviderSpec(
    name="nexus",
    keywords=("nexus",),
    env_key="ANTHROPIC_API_KEY",        # Uses Anthropic format
    display_name="NEXUS",
    litellm_prefix="anthropic",         # Use Anthropic API format
    skip_prefixes=("anthropic/",),      # Don't double-prefix
    env_extras=(),
    is_gateway=True,
    is_local=False,
    detect_by_key_prefix="",
    detect_by_base_keyword="nexus",     # Match "nexus" in api_base URL
    default_api_base="",
    strip_model_prefix=False,
    model_overrides=(),
),
```

**功能特点**：
- 自动检测：当 `api_base` URL 中包含 "nexus" 时自动识别为 NEXUS 网关
- API 格式：使用 Anthropic Messages API 格式
- 模型前缀：添加 `anthropic/` 前缀以告知 LiteLLM 使用 Anthropic API 格式
- 跳过前缀：避免重复添加 `anthropic/` 前缀

**修改 2**：更新智谱提供商的 skip_prefixes

```python
skip_prefixes=("zhipu/", "zai/", "openrouter/", "hosted_vllm/", "nexus/"),
```

**原因**：防止智谱模型名称与 NEXUS 前缀冲突

## 配置方法

### 1. 配置文件

在 `~/.nanobot/config.json` 中配置：

```json
{
  "agents": {
    "defaults": {
      "model": "claude-opus-4-6"
    }
  },
  "providers": {
    "anthropic": {
      "apiKey": "your-nexus-api-key",
      "apiBase": "https://nexusacc.itssx.com/api/claude_code/cc_glm"
    }
  }
}
```

### 2. 支持的模型

#### Claude 模型
- `claude-opus-4-6`
- `claude-opus-4-5`
- `claude-sonnet-4-5`
- `claude-haiku-4-5`
- 其他 Claude 3.x 和 4.x 系列模型

#### GLM 模型
- `glm-4.7`
- `glm-4.6`
- `glm-4.5`
- `glm-4.5-flash`
- `glm-4.5-air`
- `glm-4.5-airx`

### 3. 切换模型

修改配置文件中的 `model` 字段：

```json
{
  "agents": {
    "defaults": {
      "model": "glm-4.7"  // 或其他支持的模型
    }
  }
}
```

## 使用方法

### 1. 基础对话

```bash
python -m nanobot agent -m "你好，请做个简单的自我介绍"
```

### 2. 工具调用

```bash
python -m nanobot agent -m "请帮我查询一下北京今天的天气"
```

### 3. 文件操作

```bash
python -m nanobot agent -m "请在/tmp目录下创建一个测试文件"
```

## 测试结果

### 测试 1：基础对话
```bash
python -m nanobot agent -m "你好，请做个简单的自我介绍"
```

**结果**：✅ 成功
- 成功连接 NEXUS API
- 返回正确的响应
- 模型：实际使用 GLM-4.7（即使请求的是 claude-opus-4-6）

### 测试 2：工具调用
```bash
python -m nanobot agent -m "请帮我查询一下北京今天的天气"
```

**结果**：✅ 成功
- 正确调用 `exec` 工具
- 正确调用 `zhipu_web_search` 工具
- 工具调用参数正确传递
- 生成完整的响应

### 测试 3：文件操作
```bash
python -m nanobot agent -m "请在/tmp目录下创建一个名为test_nanobot.txt的文件"
```

**结果**：✅ 成功
- 正确调用 `write_file` 工具
- 文件创建成功
- 内容写入正确

## 技术细节

### 1. 网关检测机制

nanobot 通过 `find_gateway()` 函数检测 NEXUS 网关：

```python
def find_gateway(api_key: str | None, api_base: str | None) -> ProviderSpec | None:
    """Detect gateway/local by api_key prefix or api_base substring."""
    for spec in PROVIDERS:
        if spec.detect_by_base_keyword and api_base and spec.detect_by_base_keyword in api_base:
            return spec
    # ...
```

当 `api_base` 包含 "nexus" 时，返回 NEXUS 提供商规范。

### 2. 模型名称解析

```python
def _resolve_model(self, model: str) -> str:
    """Resolve model name by applying provider/gateway prefixes."""
    if self._gateway:
        prefix = self._gateway.litellm_prefix
        if prefix and not model.startswith(f"{prefix}/"):
            model = f"{prefix}/{model}"
        return model
    # ...
```

对于 NEXUS 网关：
- 输入：`claude-opus-4-6`
- 输出：`anthropic/claude-opus-4-6`
- LiteLLM 使用 Anthropic API 格式发送请求

### 3. API 请求流程

1. 用户配置 `apiBase` 包含 "nexus"
2. nanobot 检测到 NEXUS 网关
3. 模型名称添加 `anthropic/` 前缀
4. LiteLLM 使用 Anthropic Messages API 格式
5. 请求发送到 NEXUS 端点
6. NEXUS 路由到实际模型（可能是 GLM）
7. 返回 Anthropic 格式的响应

## 调试信息

### 查看请求详情

```bash
# 查看最新的请求调试信息
cat /home/chris/Desktop/my_workspace/nanobot/tmp/nanobot_request_debug.json | jq
```

### 调试文件内容示例

```json
{
  "model": "anthropic/claude-opus-4-6",
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

### 查看工具调用日志

```bash
# 实时监控日志
tail -f ~/.nanobot/logs/nanobot.log | grep -E "DEBUG|Tool:"
```

示例输出：
```
2026-02-10 01:49:43.702 | DEBUG | nanobot.providers.litellm_provider:chat:175 - Received 1 tool calls
2026-02-10 01:49:43.702 | DEBUG | nanobot.providers.litellm_provider:chat:177 - Tool: exec, Args: {"command": "curl -s \"wttr.in/Beijing?format=3\""}
```

## 故障排查

### 问题 1：404 NOT_FOUND

**原因**：端点路径错误或 API 格式不匹配

**解决方案**：
- 确认 `apiBase` 设置为 `https://nexusacc.itssx.com/api/claude_code/cc_glm`
- 确认 NEXUS 网关被正确检测（检查日志）

### 问题 2：模型不支持

**原因**：请求的模型名称 NEXUS 不识别

**解决方案**：
- 使用 NEXUS 支持的模型名称
- 查询可用模型：`curl -H "Authorization: Bearer YOUR_KEY" https://nexusacc.itssx.com/api/claude_code/cc_glm/v1/models`

### 问题 3：hosted_vllm 前缀错误

**原因**：NEXUS 未被识别为网关，被当作 vLLM 本地部署

**解决方案**：
- 确认 `api_base` URL 包含 "nexus" 关键字
- 检查 registry.py 中 NEXUS 提供商配置

## 与其他提供商对比

| 特性 | 官方 Anthropic | NEXUS | OpenRouter |
|------|---------------|-------|------------|
| API 格式 | Anthropic Messages | Anthropic Messages | OpenAI Chat Completions |
| 模型支持 | Claude only | Claude + GLM | 多提供商 |
| 模型路由 | 固定 | 可能重定向 | 固定 |
| 费用 | 官方定价 | 代理定价 | 统一定价 |
| 检测方式 | 默认 | URL 包含 "nexus" | API key 前缀 "sk-or-" |

## 注意事项

1. **模型路由**：NEXUS 可能将请求路由到不同的模型，实际使用的模型可能与请求的不同
2. **API 格式**：必须使用 Anthropic Messages API 格式，不支持 OpenAI 格式
3. **端点配置**：确保 `apiBase` 正确配置，不要包含 `/v1` 后缀
4. **模型名称**：使用 Claude 模型名称（如 `claude-opus-4-6`）以确保正确的 API 格式检测

## 相关文档

- NEXUS 文档：https://cc.yoouu.cn/
- Anthropic Messages API：https://docs.anthropic.com/claude/reference/messages_post
- nanobot 提供商系统：`/nanobot/providers/`
- 配置文件：`~/.nanobot/config.json`

## 提交记录

**Commit**: `bce2ef0`
**Branch**: `dev1`
**Message**: feat: add NEXUS API gateway support

**变更内容**：
- 添加 NEXUS 提供商规范到 registry.py
- 配置 NEXUS 使用 Anthropic API 格式
- 更新智谱提供商的 skip_prefixes

**测试状态**：✅ 所有功能测试通过
