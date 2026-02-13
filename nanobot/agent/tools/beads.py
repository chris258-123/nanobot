"""Beads task management tool."""

import subprocess
from pathlib import Path
from typing import Any
from nanobot.agent.tools.base import Tool


class BeadsTool(Tool):
    """Beads task management integration."""

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path).expanduser()

    @property
    def name(self) -> str:
        return "beads"

    @property
    def description(self) -> str:
        return """Beads task management.

Actions:
- add: Create task (requires: title; optional: description, blocks, parent)
- list: List tasks (optional: doable=true for actionable tasks only)
- update: Update task status (requires: task_id, status)
- query: Query tasks by filter (optional: status, tag)

Status values: todo, doing, done"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "update", "query"]
                },
                "title": {"type": "string"},
                "description": {"type": "string"},
                "blocks": {"type": "string"},  # Comma-separated task IDs
                "parent": {"type": "string"},  # Parent task ID
                "task_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["todo", "doing", "done"]
                },
                "doable": {"type": "boolean"},
                "tag": {"type": "string"}
            },
            "required": ["action"]
        }

    async def execute(self, action: str, **kwargs) -> str:
        """Execute Beads operation."""
        try:
            if action == "add":
                return await self._add(**kwargs)
            elif action == "list":
                return await self._list(**kwargs)
            elif action == "update":
                return await self._update(**kwargs)
            elif action == "query":
                return await self._query(**kwargs)
            else:
                return f"Unknown action: {action}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _add(self, title: str, description: str | None = None,
                   blocks: str | None = None, parent: str | None = None) -> str:
        """Create task with dependencies."""
        cmd = ["bd", "add", title]

        if description:
            cmd.extend(["--description", description])
        if blocks:
            cmd.extend(["--blocks", blocks])
        if parent:
            cmd.extend(["--parent", parent])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.workspace_path))

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return f"Task created: {title}\n{result.stdout}"

    async def _list(self, doable: bool = False) -> str:
        """List tasks."""
        cmd = ["bd", "list"]
        if doable:
            cmd.append("--doable")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.workspace_path))

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return result.stdout if result.stdout else "No tasks found"

    async def _update(self, task_id: str, status: str) -> str:
        """Update task status."""
        cmd = ["bd", "update", task_id, "--status", status]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.workspace_path))

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return f"Task {task_id} updated to {status}\n{result.stdout}"

    async def _query(self, status: str | None = None, tag: str | None = None) -> str:
        """Query tasks by filter."""
        cmd = ["bd", "query"]

        if status:
            cmd.extend(["--status", status])
        if tag:
            cmd.extend(["--tag", tag])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.workspace_path))

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return result.stdout if result.stdout else "No tasks found"
