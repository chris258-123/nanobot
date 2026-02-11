# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nanobot** is an ultra-lightweight personal AI assistant framework (~3,500 lines of core agent code). It provides a complete agentic system with tool execution, multi-channel support (Telegram, Discord, WhatsApp, Feishu), and extensible skills.

## Development Commands

### Setup & Installation
```bash
# Install from source (development)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with Feishu support
pip install -e ".[feishu]"

# Initialize config and workspace
nanobot onboard
```

### Testing
```bash
# Run tests
pytest

# Run specific test file
pytest tests/test_tool_validation.py

# Run with asyncio mode
pytest --asyncio-mode=auto
```

### Code Quality
```bash
# Run linter (Ruff)
ruff check nanobot/

# Format code
ruff format nanobot/

# Check line count (core agent only)
bash core_agent_lines.sh
```

### Running the Agent
```bash
# CLI chat (single message)
nanobot agent -m "Your message here"

# CLI chat with specific session
nanobot agent -m "Your message" --session cli:mysession

# CLI chat (interactive mode)
nanobot agent

# Start gateway (for Telegram/Discord/WhatsApp/Feishu)
nanobot gateway

# Check status
nanobot status

# WhatsApp login (scan QR code)
nanobot channels login
```

### Cron/Scheduled Tasks
```bash
# Add scheduled job
nanobot cron add --name "daily" --message "Good morning!" --cron "0 9 * * *"
nanobot cron add --name "hourly" --message "Check status" --every 3600

# List jobs
nanobot cron list

# Remove job
nanobot cron remove <job_id>
```

### Docker
```bash
# Build image
docker build -t nanobot .

# Initialize config
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot onboard

# Run gateway
docker run -v ~/.nanobot:/root/.nanobot -p 18790:18790 nanobot gateway
```

## Architecture

### Core Components

**Agent Loop** (`nanobot/agent/loop.py`)
- The heart of the system: receives messages, builds context, calls LLM, executes tools
- Implements the agentic loop with max iterations (default: 20)
- Handles both user messages and system messages (from subagents)
- Session management for conversation history

**Tool System** (`nanobot/agent/tools/`)
- `registry.py`: Dynamic tool registration and execution
- `base.py`: Base `Tool` class with JSON schema validation
- Built-in tools: filesystem (read/write/edit/list), shell execution, web search/fetch, browser automation, message sending, spawn (subagents), cron scheduling
- Tools can be restricted to workspace directory via `restrict_to_workspace` config

**Browser Tool** (`nanobot/agent/tools/browser.py`)
- Requires `agent-browser` CLI: `npm install -g agent-browser && agent-browser install`
- Supports: open, snapshot, click, fill, type, screenshot, close, navigation, scroll
- Uses element references (@e1, @e2) for reliable interaction
- Session and profile support for persistent state

**Provider Registry** (`nanobot/providers/registry.py`)
- **Single source of truth** for LLM provider metadata
- Declarative `ProviderSpec` entries define all provider behavior
- Supports: OpenRouter, Anthropic, OpenAI, DeepSeek, Gemini, Zhipu, DashScope (Qwen), Moonshot (Kimi), Groq, vLLM/local
- Auto-detects gateways by API key prefix or base URL
- Handles model name prefixing for LiteLLM routing
- `find_gateway()` function checks `default_api_base` to avoid misidentifying standard providers as vLLM

**Message Bus** (`nanobot/bus/`)
- Event-driven architecture with `InboundMessage` and `OutboundMessage`
- Async queue-based routing between channels and agent loop
- Session keys: `{channel}:{chat_id}`

**Channels** (`nanobot/channels/`)
- Multi-platform support: Telegram, Discord, WhatsApp (via Node.js bridge), Feishu (WebSocket)
- Base class pattern with `ChannelAdapter` interface
- Security: `allowFrom` whitelist for each channel
- Groq provider enables automatic voice transcription for Telegram voice messages

**Skills** (`nanobot/skills/`)
- Markdown-based skill system (YAML frontmatter + instructions)
- Loaded dynamically into agent context
- Built-in skills: github, weather, summarize, tmux, cron, skill-creator, zhipu-search, novel-crawler
- Skills can reference external scripts and assets

**Context Builder** (`nanobot/agent/context.py`)
- Assembles system prompt with workspace files (SOUL.md, TOOLS.md, AGENTS.md, etc.)
- Loads skills and memory
- Formats conversation history for LLM

**Session Manager** (`nanobot/session/manager.py`)
- Persistent conversation history per session
- JSON-based storage in workspace: `~/.nanobot/workspace/sessions/{session_key}.json`
- Clear session history: `rm ~/.nanobot/workspace/sessions/{session_key}.json`

**Subagent Manager** (`nanobot/agent/subagent.py`)
- Background task execution via `spawn` tool
- Subagents announce completion back to main agent via system messages

**Cron Service** (`nanobot/cron/service.py`)
- Scheduled task execution with natural language or cron syntax
- Persistent job storage

**Heartbeat Service** (`nanobot/heartbeat/service.py`)
- Proactive agent wake-up for scheduled tasks

### Configuration

Config file: `~/.nanobot/config.json`

Key sections:
- `providers`: API keys and base URLs for LLM providers
- `agents.defaults`: workspace path, model, temperature, max_tool_iterations
- `channels`: Enable/configure Telegram, Discord, WhatsApp, Feishu
- `tools.restrictToWorkspace`: Security sandbox (default: false)
- `tools.web.search.apiKey`: Brave Search API key (optional)

### Adding a New LLM Provider

**Only 2 steps required:**

1. Add `ProviderSpec` to `PROVIDERS` in `nanobot/providers/registry.py`:
```python
ProviderSpec(
    name="myprovider",
    keywords=("myprovider", "mymodel"),
    env_key="MYPROVIDER_API_KEY",
    display_name="My Provider",
    litellm_prefix="myprovider",
    skip_prefixes=("myprovider/",),
    default_api_base="https://api.myprovider.com",  # Important: prevents vLLM misdetection
)
```

2. Add field to `ProvidersConfig` in `nanobot/config/schema.py`:
```python
class ProvidersConfig(BaseModel):
    ...
    myprovider: ProviderConfig = Field(default_factory=ProviderConfig)
```

Environment variables, model prefixing, config matching, and status display all derive automatically from the registry.

**Important**: Always set `default_api_base` for standard providers to prevent `find_gateway()` from misidentifying them as vLLM local deployments.

### Adding a New Tool

1. Create tool class inheriting from `Tool` in `nanobot/agent/tools/`
2. Implement: `name`, `description`, `parameters` (JSON schema), `execute(**kwargs)`
3. Register in `AgentLoop._register_default_tools()`:
```python
self.tools.register(MyTool())
```

Tools automatically get JSON schema validation via the base class.

### Adding a New Skill

Skills are markdown files with YAML frontmatter:
```markdown
---
name: my-skill
description: Does something useful
---

# Instructions for the agent
...
```

Place in `nanobot/skills/my-skill/SKILL.md`. Skills are auto-loaded by the context builder.

Skills can reference external scripts:
- Place scripts in the same directory as SKILL.md
- Reference them in the skill documentation
- Use relative paths or absolute paths in the workspace

## Key Design Patterns

1. **Declarative Registry Pattern**: Provider metadata lives in a single registry, not scattered across if-elif chains
2. **Tool Validation**: JSON schema-based parameter validation in base `Tool` class
3. **Event-Driven Messaging**: Async message bus decouples channels from agent logic
4. **Session-Based Context**: Each `{channel}:{chat_id}` gets persistent conversation history
5. **Workspace Sandboxing**: Optional `restrict_to_workspace` flag for production security
6. **Subagent Pattern**: Background tasks via `spawn` tool with completion announcements

## Important Notes

- **Line count matters**: This project aims to stay under 4,000 core lines. Check with `bash core_agent_lines.sh`
- **LiteLLM integration**: All LLM calls go through LiteLLM for unified provider interface
- **Security**: Use `restrictToWorkspace: true` in production to sandbox file/shell tools
- **WhatsApp requires Node.js ≥18**: Uses a TypeScript bridge in `bridge/`
- **Feishu uses WebSocket**: No webhook or public IP needed for Feishu integration
- **Python ≥3.11 required**: Uses modern type hints (`str | None`, etc.)
- **Voice transcription**: Configure Groq provider to enable automatic transcription of Telegram voice messages

## Testing Philosophy

- Tests use pytest with async support (`pytest-asyncio`)
- Tool validation tests in `tests/test_tool_validation.py` demonstrate JSON schema validation
- Docker test script: `tests/test_docker.sh`

## Common Pitfalls

- **Don't add provider if-elif chains**: Use the provider registry instead
- **Don't skip tool validation**: Inherit from `Tool` base class for automatic validation
- **Don't hardcode workspace paths**: Use `get_workspace_path()` helper
- **Don't forget to register tools**: New tools must be registered in `AgentLoop._register_default_tools()`
- **Don't forget default_api_base**: Always set it for standard providers to prevent gateway misdetection
- **Session history conflicts**: If switching models causes errors, clear session history with `rm ~/.nanobot/workspace/sessions/{session_key}.json`

## Debugging

- **Request debug logs**: Check `/tmp/nanobot_request_debug.json` or `{workspace}/tmp/nanobot_request_debug.json` for LLM request details
- **Model resolution**: Use `python3 -c "from nanobot.config.loader import load_config; print(load_config().agents.defaults.model)"` to verify current model
- **Gateway detection**: Use `python3 -c "from nanobot.providers.registry import find_gateway; from nanobot.config.loader import load_config; p = load_config().get_provider(); print(find_gateway(p.api_key, p.api_base))"` to check if provider is being detected as gateway
