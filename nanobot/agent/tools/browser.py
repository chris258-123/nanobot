"""Browser automation tool using agent-browser CLI."""

import asyncio
import json
import subprocess
from typing import Any, Dict, Optional

from nanobot.agent.tools.base import Tool


class BrowserTool(Tool):
    """Tool for browser automation using agent-browser CLI.

    Provides AI agents with browser control capabilities including:
    - Opening URLs and navigating
    - Taking snapshots (accessibility tree with element refs)
    - Clicking elements
    - Filling forms
    - Taking screenshots
    - Managing browser sessions
    """

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return (
            "Open websites, take screenshots, and interact with web pages. "
            "Use this when user asks to open/visit/browse a URL or take a screenshot. "
            "Actions: open (URL), screenshot (save to file), snapshot (get page structure), "
            "click/fill/type (interact with elements), press (press keyboard keys like Enter, Tab)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "open",
                        "snapshot",
                        "click",
                        "fill",
                        "type",
                        "press",
                        "screenshot",
                        "close",
                        "back",
                        "forward",
                        "reload",
                        "scroll",
                    ],
                    "description": "Browser action to perform",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Target for the action: URL for 'open', "
                        "element ref (@e1) or selector for 'click'/'fill', "
                        "key name for 'press' (e.g., 'Enter', 'Tab', 'Escape'), "
                        "file path for 'screenshot'"
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "Value for 'fill' or 'type' actions",
                },
                "session": {
                    "type": "string",
                    "description": "Browser session ID (optional, for multiple sessions)",
                },
                "profile": {
                    "type": "string",
                    "description": "Profile name for persistent state (optional)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        target: Optional[str] = None,
        value: Optional[str] = None,
        session: Optional[str] = None,
        profile: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Execute browser action using agent-browser CLI."""
        # Build command
        cmd = ["agent-browser"]

        # Add session/profile flags
        if session:
            cmd.extend(["--session", session])
        if profile:
            cmd.extend(["--profile", profile])

        # Add action and arguments
        cmd.append(action)
        if target:
            cmd.append(target)
        if value:
            cmd.append(value)

        try:
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                return f"Browser command failed: {error_msg}"

            output = stdout.decode().strip()

            # Format output based on action
            if action == "snapshot":
                return f"Accessibility tree:\n{output}\n\nUse element refs like @e1, @e2 in subsequent commands."
            elif action == "screenshot":
                return f"Screenshot saved to: {target}"
            elif action == "open":
                return f"Opened: {target}"
            else:
                return output if output else f"Action '{action}' completed successfully"

        except FileNotFoundError:
            return (
                "agent-browser not found. Install it with:\n"
                "npm install -g agent-browser && agent-browser install"
            )
        except Exception as e:
            return f"Browser error: {str(e)}"
