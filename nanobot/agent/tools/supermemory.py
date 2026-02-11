"""Supermemory integration tool for semantic memory storage and retrieval."""

import json
import os
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


class SupermemoryTool(Tool):
    """
    Tool for interacting with Supermemory API.

    Provides semantic memory storage and retrieval capabilities.
    """

    def __init__(self):
        self.api_key = os.getenv("SUPERMEMORY_API_KEY")
        self.base_url = os.getenv("SUPERMEMORY_API_URL", "https://api.supermemory.ai/v1")
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            timeout=30.0
        )

    @property
    def name(self) -> str:
        return "supermemory"

    @property
    def description(self) -> str:
        return """Store and retrieve memories using semantic search.

Actions:
- store: Store a memory with optional tags
- search: Search memories semantically
- recall: Get recent memories

Examples:
- supermemory(action="store", content="User prefers dark mode", tags=["preferences"])
- supermemory(action="search", query="what are user's preferences?")
- supermemory(action="recall", limit=10)
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "search", "recall"],
                    "description": "Action to perform"
                },
                "content": {
                    "type": "string",
                    "description": "Content to store (for store action)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search action)"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization (optional)"
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of results to return"
                }
            },
            "required": ["action"]
        }

    def execute(self, action: str, content: str = None, query: str = None,
                tags: list[str] = None, limit: int = 5, **kwargs) -> str:
        """Execute supermemory action."""

        if not self.api_key:
            return "Error: SUPERMEMORY_API_KEY not set in environment"

        try:
            if action == "store":
                return self._store_memory(content, tags)
            elif action == "search":
                return self._search_memories(query, limit)
            elif action == "recall":
                return self._recall_memories(limit)
            else:
                return f"Unknown action: {action}"

        except Exception as e:
            return f"Supermemory error: {str(e)}"

    def _store_memory(self, content: str, tags: list[str] = None) -> str:
        """Store a memory."""
        if not content:
            return "Error: content is required for store action"

        payload = {"content": content}
        if tags:
            payload["tags"] = tags

        response = self.client.post("/memories", json=payload)
        response.raise_for_status()

        result = response.json()
        return f"Memory stored successfully (ID: {result.get('id', 'unknown')})"

    def _search_memories(self, query: str, limit: int = 5) -> str:
        """Search memories semantically."""
        if not query:
            return "Error: query is required for search action"

        response = self.client.get("/search", params={"q": query, "limit": limit})
        response.raise_for_status()

        results = response.json()

        if not results or len(results) == 0:
            return "No memories found"

        # Format results
        output = [f"Found {len(results)} memories:\n"]
        for i, memory in enumerate(results, 1):
            output.append(f"{i}. {memory.get('content', '')}")
            if memory.get('tags'):
                output.append(f"   Tags: {', '.join(memory['tags'])}")
            output.append("")

        return "\n".join(output)

    def _recall_memories(self, limit: int = 5) -> str:
        """Get recent memories."""
        response = self.client.get("/memories/recent", params={"limit": limit})
        response.raise_for_status()

        results = response.json()

        if not results or len(results) == 0:
            return "No recent memories"

        output = [f"Recent {len(results)} memories:\n"]
        for i, memory in enumerate(results, 1):
            output.append(f"{i}. {memory.get('content', '')}")
            output.append("")

        return "\n".join(output)
