# nanobot 项目理解框架 (Project Understanding Framework)

> 本文档面向初学者，详细讲解 nanobot 的架构、数据流向和代码执行流程

## 目录 (Table of Contents)

1. [整体架构概览](#整体架构概览)
2. [核心组件详解](#核心组件详解)
3. [数据流向图](#数据流向图)
4. [函数调用流程](#函数调用流程)
5. [完整执行流程示例](#完整执行流程示例)
6. [关键代码路径](#关键代码路径)

---

## 整体架构概览

nanobot 是一个轻量级 AI 助手框架，采用 **异步消息总线架构**，核心思想是：

```
用户消息 → 消息总线 → Agent 处理 → LLM 调用 → 工具执行 → 响应返回
```

### 核心设计理念

1. **解耦设计**: 通道(Channels)和智能体(Agent)通过消息总线(Bus)解耦
2. **异步处理**: 所有 I/O 操作使用 async/await
3. **工具系统**: Agent 通过工具(Tools)与外部世界交互
4. **会话管理**: 每个对话维护独立的历史记录

---

## 核心组件详解

### 1. 消息总线 (Message Bus)

**位置**: `nanobot/bus/queue.py`

**作用**: 连接所有组件的中央枢纽

```python
class MessageBus:
    inbound: Queue[InboundMessage]   # 入站队列：通道 → Agent
    outbound: Queue[OutboundMessage] # 出站队列：Agent → 通道
```

**工作原理**:
- 通道将用户消息放入 `inbound` 队列
- Agent 从 `inbound` 队列取消息处理
- Agent 将响应放入 `outbound` 队列
- 通道从 `outbound` 队列取消息发送给用户

### 2. Agent 循环 (Agent Loop)

**位置**: `nanobot/agent/loop.py`

**作用**: 核心处理引擎，协调所有操作

**主要方法**:
```python
async def run():
    while running:
        msg = await bus.consume_inbound()  # 1. 等待消息
        response = await _process_message(msg)  # 2. 处理消息
        await bus.publish_outbound(response)  # 3. 发送响应
```

**处理流程**:
1. 从消息总线获取消息
2. 构建上下文（历史记录 + 系统提示）
3. 调用 LLM
4. 执行工具调用
5. 返回响应

### 3. 上下文构建器 (Context Builder)

**位置**: `nanobot/agent/context.py`

**作用**: 组装发送给 LLM 的完整上下文

**构建内容**:
```
系统提示 =
    核心身份 (nanobot 是谁)
  + 引导文件 (AGENTS.md, SOUL.md, USER.md, TOOLS.md)
  + 记忆上下文 (memory/MEMORY.md)
  + 技能列表 (skills/)
  + 当前时间和环境信息
```

### 4. 工具注册表 (Tool Registry)

**位置**: `nanobot/agent/tools/registry.py`

**作用**: 管理所有可用工具

**内置工具**:
- `read_file`: 读取文件
- `write_file`: 写入文件
- `edit_file`: 编辑文件
- `list_dir`: 列出目录
- `exec`: 执行 shell 命令
- `web_search`: 网络搜索
- `web_fetch`: 获取网页内容
- `message`: 发送消息
- `spawn`: 创建子 Agent
- `cron`: 定时任务

### 5. LLM 提供者 (LLM Provider)

**位置**: `nanobot/providers/base.py`

**作用**: 统一的 LLM 接口抽象

```python
class LLMProvider:
    async def chat(messages, tools, model) -> LLMResponse:
        # 调用 LLM API
        # 返回内容和/或工具调用
```

### 6. 会话管理器 (Session Manager)

**位置**: `nanobot/session/manager.py`

**作用**: 管理对话历史

```python
class Session:
    key: str  # "channel:chat_id"
    messages: list[dict]  # 消息历史

    def add_message(role, content):
        # 添加消息到历史

    def get_history(max_messages=50):
        # 获取最近的消息用于上下文
```

### 7. 通道系统 (Channels)

**位置**: `nanobot/channels/`

**支持的通道**:
- Telegram (`telegram.py`)
- Discord (`discord.py`)
- WhatsApp (`whatsapp.py`)
- Feishu/飞书 (`feishu.py`)

**工作方式**:
```python
class BaseChannel:
    async def start():
        # 1. 连接到平台 API
        # 2. 订阅消息总线的出站消息
        # 3. 开始监听用户消息

    async def _handle_message(message):
        # 1. 解析平台消息
        # 2. 转换为 InboundMessage
        # 3. 发布到消息总线
```

---

## 数据流向图

### 用户发送消息的完整流程

```
┌─────────────┐
│   用户输入   │ (Telegram/Discord/WhatsApp/CLI)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Channel 接收   │ (telegram.py, discord.py, etc.)
│  - 解析消息      │
│  - 提取内容      │
└──────┬──────────┘
       │
       │ InboundMessage
       ▼
┌─────────────────┐
│  Message Bus    │ (bus/queue.py)
│  inbound.put()  │
└──────┬──────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Agent Loop                         │ (agent/loop.py)
│  1. consume_inbound()               │
│  2. 获取/创建 Session                │
│  3. 构建上下文                       │
│     ├─ 系统提示 (Context Builder)   │
│     ├─ 历史记录 (Session Manager)   │
│     └─ 当前消息                      │
└──────┬──────────────────────────────┘
       │
       │ messages + tools
       ▼
┌─────────────────┐
│  LLM Provider   │ (providers/)
│  - 调用 API      │
│  - 返回响应      │
└──────┬──────────┘
       │
       │ LLMResponse
       ▼
┌─────────────────────────────┐
│  工具执行? (如果有工具调用)  │
│  ├─ Tool Registry           │
│  ├─ 执行工具                 │
│  └─ 获取结果                 │
└──────┬──────────────────────┘
       │
       │ (循环直到没有工具调用)
       ▼
┌─────────────────┐
│  保存到 Session │
│  - 用户消息      │
│  - Agent 响应   │
└──────┬──────────┘
       │
       │ OutboundMessage
       ▼
┌─────────────────┐
│  Message Bus    │
│  outbound.put() │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Channel 发送   │
│  - 格式化响应    │
│  - 发送给用户    │
└──────┬──────────┘
       │
       ▼
┌─────────────┐
│  用户收到    │
└─────────────┘
```

---

## 函数调用流程

### 启动流程 (Gateway 模式)

```python
# 入口: nanobot/cli/commands.py
@app.command()
def gateway():
    asyncio.run(_run_gateway())

async def _run_gateway():
    # 1. 加载配置
    config = load_config()

    # 2. 创建消息总线
    bus = MessageBus()

    # 3. 创建 LLM 提供者
    provider = create_provider(config)

    # 4. 创建 Agent Loop
    agent = AgentLoop(bus, provider, workspace)

    # 5. 创建通道管理器
    channels = ChannelManager(config, bus)

    # 6. 启动所有组件
    await asyncio.gather(
        agent.run(),              # Agent 循环
        bus.dispatch_outbound(),  # 消息分发
        channels.start_all()      # 所有通道
    )
```

### 消息处理流程 (核心)

```python
# nanobot/agent/loop.py
async def _process_message(msg: InboundMessage):
    # 1. 获取会话
    session = sessions.get_or_create(msg.session_key)

    # 2. 构建消息列表
    messages = context.build_messages(
        history=session.get_history(),
        current_message=msg.content
    )

    # 3. Agent 循环 (最多 20 次迭代)
    for iteration in range(max_iterations):
        # 3.1 调用 LLM
        response = await provider.chat(
            messages=messages,
            tools=tools.get_definitions()
        )

        # 3.2 检查是否有工具调用
        if response.has_tool_calls:
            # 3.3 执行每个工具
            for tool_call in response.tool_calls:
                result = await tools.execute(
                    tool_call.name,
                    tool_call.arguments
                )
                # 3.4 将结果添加到消息列表
                messages.append({
                    "role": "tool",
                    "content": result
                })
            # 3.5 继续循环，让 LLM 看到工具结果
        else:
            # 3.6 没有工具调用，完成
            final_content = response.content
            break

    # 4. 保存到会话
    session.add_message("user", msg.content)
    session.add_message("assistant", final_content)

    # 5. 返回响应
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=final_content
    )
```

### 工具执行流程

```python
# nanobot/agent/tools/registry.py
async def execute(name: str, arguments: dict):
    # 1. 查找工具
    tool = self._tools.get(name)

    # 2. 验证参数
    # (根据工具的 parameters schema)

    # 3. 执行工具
    result = await tool.execute(**arguments)

    # 4. 返回结果字符串
    return result
```

### 工具示例: read_file

```python
# nanobot/agent/tools/filesystem.py
class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }

    async def execute(self, path: str) -> str:
        # 1. 验证路径 (防止路径遍历攻击)
        validated_path = self._validate_path(path)

        # 2. 读取文件
        content = Path(validated_path).read_text()

        # 3. 返回内容
        return content
```

---

## 完整执行流程示例

### 场景: 用户通过 Telegram 发送 "帮我创建一个 hello.txt 文件"

#### 步骤 1: Telegram 接收消息

```python
# nanobot/channels/telegram.py
async def _handle_message(update, context):
    # 解析 Telegram 消息
    text = update.message.text
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    # 创建 InboundMessage
    msg = InboundMessage(
        channel="telegram",
        sender_id=str(user_id),
        chat_id=str(chat_id),
        content=text
    )

    # 发布到消息总线
    await self.bus.publish_inbound(msg)
```

#### 步骤 2: Agent 处理消息

```python
# Agent Loop 从总线获取消息
msg = await bus.consume_inbound()
# msg.content = "帮我创建一个 hello.txt 文件"

# 获取会话历史
session = sessions.get_or_create("telegram:123456")
history = session.get_history()  # 之前的对话

# 构建完整上下文
messages = [
    {
        "role": "system",
        "content": "你是 nanobot... [系统提示] ...可用工具: write_file..."
    },
    # ... 历史消息 ...
    {
        "role": "user",
        "content": "帮我创建一个 hello.txt 文件"
    }
]
```

#### 步骤 3: 第一次 LLM 调用

```python
# 调用 LLM
response = await provider.chat(messages, tools)

# LLM 返回:
# {
#     "content": "我来帮你创建这个文件",
#     "tool_calls": [
#         {
#             "id": "call_123",
#             "name": "write_file",
#             "arguments": {
#                 "path": "hello.txt",
#                 "content": "Hello, World!"
#             }
#         }
#     ]
# }
```

#### 步骤 4: 执行工具

```python
# 执行 write_file 工具
result = await tools.execute("write_file", {
    "path": "hello.txt",
    "content": "Hello, World!"
})

# 工具返回: "File written successfully: hello.txt"

# 将工具结果添加到消息列表
messages.append({
    "role": "assistant",
    "content": "我来帮你创建这个文件",
    "tool_calls": [...]
})
messages.append({
    "role": "tool",
    "tool_call_id": "call_123",
    "name": "write_file",
    "content": "File written successfully: hello.txt"
})
```

#### 步骤 5: 第二次 LLM 调用

```python
# 再次调用 LLM，让它看到工具执行结果
response = await provider.chat(messages, tools)

# LLM 返回:
# {
#     "content": "已经成功创建了 hello.txt 文件！",
#     "tool_calls": []  # 没有更多工具调用
# }

# 没有工具调用，循环结束
final_content = "已经成功创建了 hello.txt 文件！"
```

#### 步骤 6: 保存并返回

```python
# 保存到会话
session.add_message("user", "帮我创建一个 hello.txt 文件")
session.add_message("assistant", "已经成功创建了 hello.txt 文件！")

# 创建响应消息
response = OutboundMessage(
    channel="telegram",
    chat_id="123456",
    content="已经成功创建了 hello.txt 文件！"
)

# 发布到消息总线
await bus.publish_outbound(response)
```

#### 步骤 7: Telegram 发送响应

```python
# Telegram 通道从总线获取消息
msg = await bus.consume_outbound()

# 发送给用户
await context.bot.send_message(
    chat_id=msg.chat_id,
    text=msg.content
)
```

---

## 关键代码路径

### 路径 1: CLI 单次对话

```
cli/commands.py:agent()
  → agent/loop.py:process_direct()
    → agent/loop.py:_process_message()
      → agent/context.py:build_messages()
      → providers/litellm.py:chat()
      → agent/tools/registry.py:execute()
    → 返回响应字符串
  → 打印到控制台
```

### 路径 2: Gateway 持续运行

```
cli/commands.py:gateway()
  → 创建 MessageBus, AgentLoop, ChannelManager
  → 并发运行:
    ├─ agent/loop.py:run()  # Agent 循环
    ├─ bus/queue.py:dispatch_outbound()  # 消息分发
    └─ channels/manager.py:start_all()  # 所有通道
```

### 路径 3: 子 Agent 执行

```
用户消息: "spawn: 搜索最新的 AI 新闻"
  → agent/loop.py:_process_message()
    → LLM 调用 spawn 工具
      → agent/tools/spawn.py:execute()
        → agent/subagent.py:spawn()
          → 创建后台任务
          → subagent.py:_run_subagent()
            → 独立的 Agent 循环
            → 完成后发送系统消息
              → bus.publish_inbound(InboundMessage(
                    channel="system",
                    content="任务完成: ..."
                  ))
          → 主 Agent 接收系统消息
          → 处理并响应用户
```

### 路径 4: 定时任务

```
cli/commands.py:gateway()
  → cron/service.py:CronService.start()
    → 后台线程检查任务
    → 到时间时:
      → cron/service.py:_execute_job()
        → agent/loop.py:process_direct()
          → 处理定时消息
          → 通过 message 工具发送到指定通道
```

---

## 配置和初始化

### 配置文件结构

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "BOT_TOKEN",
      "allowFrom": ["123456"]
    }
  },
  "tools": {
    "restrictToWorkspace": false,
    "exec": {
      "timeout": 60
    },
    "web": {
      "search": {
        "apiKey": "BRAVE_API_KEY"
      }
    }
  }
}
```

### 工作空间结构

```
~/.nanobot/
├── config.json           # 配置文件
└── workspace/            # 工作空间
    ├── AGENTS.md         # Agent 指令
    ├── SOUL.md           # 个性设定
    ├── USER.md           # 用户偏好
    ├── TOOLS.md          # 工具文档
    ├── HEARTBEAT.md      # 定期任务
    ├── memory/           # 记忆系统
    │   ├── MEMORY.md     # 持久记忆
    │   └── 2026-02-08.md # 每日笔记
    ├── sessions/         # 会话历史
    │   └── telegram_123456.jsonl
    └── skills/           # 自定义技能
        └── my-skill/
            └── SKILL.md
```

---

## 扩展开发指南

### 添加新工具

1. 创建工具类:

```python
# nanobot/agent/tools/my_tool.py
from nanobot.agent.tools.base import Tool

class MyTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "工具描述"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            },
            "required": ["param1"]
        }

    async def execute(self, param1: str) -> str:
        # 实现工具逻辑
        result = do_something(param1)
        return f"结果: {result}"
```

2. 注册工具:

```python
# nanobot/agent/loop.py
def _register_default_tools(self):
    # ... 现有工具 ...
    from nanobot.agent.tools.my_tool import MyTool
    self.tools.register(MyTool())
```

### 添加新通道

1. 创建通道类:

```python
# nanobot/channels/my_channel.py
from nanobot.channels.base import BaseChannel

class MyChannel(BaseChannel):
    async def start(self):
        # 1. 连接到平台
        # 2. 订阅出站消息
        self.bus.subscribe_outbound("my_channel", self._send_message)
        # 3. 开始监听
        await self._listen()

    async def _listen(self):
        # 监听平台消息
        while True:
            message = await platform.get_message()
            await self._handle_message(message)

    async def _handle_message(self, message):
        # 转换为 InboundMessage
        msg = InboundMessage(
            channel="my_channel",
            sender_id=message.user_id,
            chat_id=message.chat_id,
            content=message.text
        )
        await self.bus.publish_inbound(msg)

    async def _send_message(self, msg: OutboundMessage):
        # 发送到平台
        await platform.send(msg.chat_id, msg.content)
```

2. 注册通道:

```python
# nanobot/channels/manager.py
def _create_channel(self, name: str):
    if name == "my_channel":
        from nanobot.channels.my_channel import MyChannel
        return MyChannel(config, bus)
```

---

## 调试技巧

### 1. 启用详细日志

```python
# 在代码中添加
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

### 2. 查看消息流

```python
# 在 MessageBus 中添加日志
async def publish_inbound(self, msg):
    logger.debug(f"Inbound: {msg.channel}:{msg.chat_id} - {msg.content[:50]}")
    await self.inbound.put(msg)
```

### 3. 检查工具调用

```python
# 在 ToolRegistry.execute 中添加
logger.info(f"Executing tool: {name} with args: {arguments}")
```

### 4. 查看 LLM 请求/响应

```python
# 在 Provider.chat 中添加
logger.debug(f"LLM Request: {len(messages)} messages")
logger.debug(f"LLM Response: {response.content[:100]}")
```

---

## 常见问题

### Q: Agent 循环为什么限制 20 次迭代？

A: 防止无限循环。如果 LLM 持续调用工具而不给出最终响应，20 次后会强制结束。

### Q: 为什么使用消息总线而不是直接调用？

A: 解耦设计。通道和 Agent 可以独立开发、测试和部署。也支持多通道同时运行。

### Q: 会话历史如何管理？

A: 每个 `channel:chat_id` 维护独立会话，保留最近 50 条消息。超过会自动截断。

### Q: 子 Agent 和主 Agent 有什么区别？

A: 子 Agent 是简化版，没有完整的上下文和技能，专注于单一任务。完成后通过系统消息通知主 Agent。

### Q: 如何保证安全性？

A:
- 文件操作有路径验证
- Shell 命令有危险模式检测
- 可启用 `restrictToWorkspace` 沙箱模式
- 通道有 `allowFrom` 白名单

---

## 总结

nanobot 的核心架构可以总结为：

1. **消息驱动**: 所有交互通过消息总线
2. **异步处理**: 高效的并发处理
3. **工具系统**: Agent 通过工具与世界交互
4. **上下文管理**: 智能的提示词构建
5. **可扩展**: 易于添加新工具、通道、提供者

理解这个框架的关键是跟踪消息的流动：

```
用户 → 通道 → 总线 → Agent → LLM → 工具 → Agent → 总线 → 通道 → 用户
```

每个组件都有明确的职责，通过标准接口通信，使得系统既简单又强大。
