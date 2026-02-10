"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
    
    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.
        
        Args:
            skill_names: Optional list of skills to include.
        
        Returns:
            Complete system prompt.
        """
        parts = []
        
        # Core identity
        parts.append(self._get_identity())
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
        # Memory context
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section."""
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant with access to tools for file operations, shell commands, web access, and browser automation.

## CRITICAL: Tool Usage Rules

You MUST call tools when users ask you to perform actions. DO NOT just describe what you would do - ACTUALLY CALL THE TOOLS!

**Examples of correct tool usage:**

Example 1 - Browser automation:
User: "打开 https://github.com 并截图"
Assistant: [Calls browser tool with action="open", target="https://github.com"]
Tool result: "Opened: https://github.com"
Assistant: [Calls browser tool with action="screenshot", target="github.png"]
Tool result: "Screenshot saved to: github.png"
Assistant: "已成功打开 GitHub 并保存截图到 github.png"

Example 2 - File operations:
User: "读取 /etc/hostname 文件"
Assistant: [Calls read_file tool with path="/etc/hostname"]
Tool result: "ubuntu-server"
Assistant: "文件内容是: ubuntu-server"

Example 3 - Web search:
User: "搜索最新的 AI 新闻"
Assistant: [Calls web_search tool with query="latest AI news 2026"]
Tool result: [Search results...]
Assistant: "找到以下 AI 新闻: ..."

**When to use tools:**
- "打开/访问/浏览 网站" → call `browser` tool with action="open"
- "截图/拍照" → call `browser` tool with action="screenshot"
- "点击/填写" → call `browser` tool with action="click" or "fill"
- "读取/写入文件" → call file tools
- "执行命令" → call `exec` tool
- "搜索网页" → call `web_search` or `zhipu_web_search` tool

**Exception:** Only respond with text (no tool call) for direct questions or conversations.

## Current Time
{now}

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Memory files: {workspace_path}/memory/MEMORY.md
- Daily notes: {workspace_path}/memory/YYYY-MM-DD.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

Always be helpful, accurate, and concise. When using tools, explain what you're doing.
When remembering something, write to {workspace_path}/memory/MEMORY.md"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Add few-shot examples on first message to demonstrate tool usage
        if not history:
            # Example 1: Browser tool usage
            messages.append({"role": "user", "content": "打开 https://example.com 并截图"})
            messages.append({
                "role": "assistant",
                "content": "我来帮你打开这个网站并截图。",
                "tool_calls": [{
                    "id": "call_example_1",
                    "type": "function",
                    "function": {
                        "name": "browser",
                        "arguments": '{"action": "open", "target": "https://example.com"}'
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "call_example_1",
                "name": "browser",
                "content": "Opened: https://example.com"
            })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_example_2",
                    "type": "function",
                    "function": {
                        "name": "browser",
                        "arguments": '{"action": "screenshot", "target": "example.png"}'
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "call_example_2",
                "name": "browser",
                "content": "Screenshot saved to: example.png"
            })
            messages.append({
                "role": "assistant",
                "content": "已成功打开网站并保存截图到 example.png"
            })

            # Example 2: File reading
            messages.append({"role": "user", "content": "读取 /etc/hostname"})
            messages.append({
                "role": "assistant",
                "content": "我来读取这个文件。",
                "tool_calls": [{
                    "id": "call_example_3",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/etc/hostname"}'
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "call_example_3",
                "name": "read_file",
                "content": "ubuntu-server"
            })
            messages.append({
                "role": "assistant",
                "content": "文件内容是: ubuntu-server"
            })

            # Example 3: Complex browser automation with form interaction
            messages.append({"role": "user", "content": "打开 GitHub，搜索 nanobot 并截图"})
            messages.append({
                "role": "assistant",
                "content": "我来帮你打开 GitHub 并搜索。",
                "tool_calls": [{
                    "id": "call_example_4",
                    "type": "function",
                    "function": {
                        "name": "browser",
                        "arguments": '{"action": "open", "target": "https://github.com"}'
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "call_example_4",
                "name": "browser",
                "content": "Opened: https://github.com"
            })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_example_5",
                    "type": "function",
                    "function": {
                        "name": "browser",
                        "arguments": '{"action": "snapshot"}'
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "call_example_5",
                "name": "browser",
                "content": "Accessibility tree:\n- textbox \"Search GitHub\" [ref=e5]\n- button \"Search\" [ref=e6]\n\nUse element refs like @e5, @e6 in subsequent commands."
            })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_example_6",
                    "type": "function",
                    "function": {
                        "name": "browser",
                        "arguments": '{"action": "fill", "target": "@e5", "value": "nanobot"}'
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "call_example_6",
                "name": "browser",
                "content": "Action 'fill' completed successfully"
            })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_example_7",
                    "type": "function",
                    "function": {
                        "name": "browser",
                        "arguments": '{"action": "click", "target": "@e6"}'
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "call_example_7",
                "name": "browser",
                "content": "Action 'click' completed successfully"
            })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_example_8",
                    "type": "function",
                    "function": {
                        "name": "browser",
                        "arguments": '{"action": "screenshot", "target": "github_search.png"}'
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "call_example_8",
                "name": "browser",
                "content": "Screenshot saved to: github_search.png"
            })
            messages.append({
                "role": "assistant",
                "content": "已成功在 GitHub 搜索 nanobot 并保存截图到 github_search.png"
            })

            # Add separator
            messages.append({
                "role": "system",
                "content": (
                    "--- End of examples. Now handle the actual user request below. ---\n\n"
                    "CRITICAL RULES - YOU MUST FOLLOW THESE:\n"
                    "1. CALL TOOLS, not just describe actions!\n"
                    "2. After calling snapshot, YOU MUST USE the element references it returns (@e1, @e2, etc.)\n"
                    "3. NEVER use CSS selectors like 'input[type=text]' or '.button' - ONLY use @eX references from snapshot\n"
                    "4. Example: snapshot returns 'textbox [ref=e5]' → use fill(target='@e5', value='text')\n"
                    "5. To press keyboard keys: browser(action='press', target='Enter')\n"
                    "6. If you don't see the element you need in snapshot, call snapshot again or use a different approach"
                )
            })

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        
        if tool_calls:
            msg["tool_calls"] = tool_calls
        
        messages.append(msg)
        return messages
