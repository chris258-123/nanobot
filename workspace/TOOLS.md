# Available Tools

This document describes the tools available to nanobot.

## File Operations

### read_file
Read the contents of a file.
```
read_file(path: str) -> str
```

### write_file
Write content to a file (creates parent directories if needed).
```
write_file(path: str, content: str) -> str
```

### edit_file
Edit a file by replacing specific text.
```
edit_file(path: str, old_text: str, new_text: str) -> str
```

### list_dir
List contents of a directory.
```
list_dir(path: str) -> str
```

## Shell Execution

### exec
Execute a shell command and return output.
```
exec(command: str, working_dir: str = None) -> str
```

**Safety Notes:**
- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- Optional `restrictToWorkspace` config to limit paths

## Web Access

### web_search
Search the web using Brave Search API.
```
web_search(query: str, count: int = 5) -> str
```

Returns search results with titles, URLs, and snippets. Requires `tools.web.search.apiKey` in config.

### web_fetch
Fetch and extract main content from a URL.
```
web_fetch(url: str, extractMode: str = "markdown", maxChars: int = 50000) -> str
```

**Notes:**
- Content is extracted using readability
- Supports markdown or plain text extraction
- Output is truncated at 50,000 characters by default

### zhipu_web_search
Search using Zhipu AI web search (optimized for Chinese content).
```
zhipu_web_search(query: str) -> str
```

Requires `tools.web.zhipuApiKey` in config.

## Browser Automation

### browser
**Control a headless browser for web automation and interaction.**

**⚠️ IMPORTANT: Use this tool when the user asks to open, visit, browse, screenshot, or interact with websites!**

```
browser(
    action: str,           # Required: open, snapshot, click, fill, type, screenshot, close, back, forward, reload, scroll
    target: str = None,    # URL for open, element ref (@e1) or selector for click/fill, file path for screenshot
    value: str = None,     # Text value for fill/type actions
    session: str = None,   # Session ID for multiple isolated browser instances
    profile: str = None    # Profile name for persistent state (cookies, localStorage)
) -> str
```

**When to use browser vs web_fetch:**
- Use `browser` when:
  - User asks to "打开" (open), "访问" (visit), "浏览" (browse) a website
  - User asks to "截图" (screenshot), "拍照" (take photo) of a page
  - Page requires JavaScript to render
  - Need to interact with page (click, fill forms, scroll)
  - Need screenshots of rendered page
  - Need to maintain session state
- Use `web_fetch` when:
  - Just need static HTML content
  - Faster and lighter weight

**Typical workflow:**
```python
# 1. Open a website
browser(action="open", target="https://github.com")

# 2. Get page structure with element references
browser(action="snapshot")
# Returns: heading "GitHub" [ref=e1], link "Sign in" [ref=e2], ...

# 3. Interact with elements using refs
browser(action="click", target="@e2")  # Click "Sign in" link

# 4. Fill forms
browser(action="fill", target="@e5", value="username")

# 5. Take screenshot
browser(action="screenshot", target="page.png")

# 6. Close browser
browser(action="close")
```

**Actions:**
- `open`: Open a URL
- `snapshot`: Get accessibility tree with element references (@e1, @e2, ...)
- `click`: Click an element (use @e1 refs from snapshot)
- `fill`: Fill an input field
- `type`: Type text with keyboard
- `screenshot`: Save screenshot to file
- `close`: Close browser session
- `back`, `forward`, `reload`: Navigation
- `scroll`: Scroll the page

**Notes:**
- Requires `agent-browser` CLI to be installed: `npm install -g agent-browser && agent-browser install`
- Element references (@e1, @e2) are easier than CSS selectors
- Use `session` parameter for multiple isolated browser instances
- Use `profile` parameter to persist cookies and login state across sessions
- Browser operations are slower than web_fetch but support full JavaScript and interaction

**Examples of when to use:**
- User: "打开 GitHub" → browser(action="open", target="https://github.com")
- User: "截图这个网页" → browser(action="screenshot", target="screenshot.png")
- User: "访问 example.com" → browser(action="open", target="https://example.com")
- User: "在 Google 搜索" → browser(action="open", target="https://google.com"), then snapshot, fill, click

## Communication

### message
Send a message to the user (used internally).
```
message(content: str, channel: str = None, chat_id: str = None) -> str
```

## Background Tasks

### spawn
Spawn a subagent to handle a task in the background.
```
spawn(task: str, label: str = None) -> str
```

Use for complex or time-consuming tasks that can run independently. The subagent will complete the task and report back when done.

## Scheduled Reminders (Cron)

Use the `exec` tool to create scheduled reminders with `nanobot cron add`:

### Set a recurring reminder
```bash
# Every day at 9am
nanobot cron add --name "morning" --message "Good morning! ☀️" --cron "0 9 * * *"

# Every 2 hours
nanobot cron add --name "water" --message "Drink water! 💧" --every 7200
```

### Set a one-time reminder
```bash
# At a specific time (ISO format)
nanobot cron add --name "meeting" --message "Meeting starts now!" --at "2025-01-31T15:00:00"
```

### Manage reminders
```bash
nanobot cron list              # List all jobs
nanobot cron remove <job_id>   # Remove a job
```

## Heartbeat Task Management

The `HEARTBEAT.md` file in the workspace is checked every 30 minutes.
Use file operations to manage periodic tasks:

### Add a heartbeat task
```python
# Append a new task
edit_file(
    path="HEARTBEAT.md",
    old_text="## Example Tasks",
    new_text="- [ ] New periodic task here\n\n## Example Tasks"
)
```

### Remove a heartbeat task
```python
# Remove a specific task
edit_file(
    path="HEARTBEAT.md",
    old_text="- [ ] Task to remove\n",
    new_text=""
)
```

### Rewrite all tasks
```python
# Replace the entire file
write_file(
    path="HEARTBEAT.md",
    content="# Heartbeat Tasks\n\n- [ ] Task 1\n- [ ] Task 2\n"
)
```

---

## Adding Custom Tools

To add custom tools:
1. Create a class that extends `Tool` in `nanobot/agent/tools/`
2. Implement `name`, `description`, `parameters`, and `execute`
3. Register it in `AgentLoop._register_default_tools()`
