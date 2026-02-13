"""Letta agent memory management tool."""

import httpx
import json
from typing import Any
from nanobot.agent.tools.base import Tool


class LettaTool(Tool):
    """Letta agent operations for Writer and Archivist agents."""

    def __init__(self, url: str, api_key: str = ""):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    @property
    def name(self) -> str:
        return "letta"

    @property
    def description(self) -> str:
        return """Letta agent memory management.

Actions:
- create_agent: Create Writer or Archivist agent (requires: agent_type)
- send_message: Send message to agent (requires: agent_id, message)
- update_core_memory: Update core memory (requires: agent_id, memory_key, memory_value)
- add_archival: Add to archival memory (requires: agent_id, content)
- search_archival: Search archival memory (requires: agent_id, query)
- list_agents: List all agents

Agent types: writer, archivist"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create_agent", "send_message", "update_core_memory",
                            "add_archival", "search_archival", "list_agents"]
                },
                "agent_id": {"type": "string"},
                "agent_type": {
                    "type": "string",
                    "enum": ["writer", "archivist"]
                },
                "message": {"type": "string"},
                "memory_key": {"type": "string"},
                "memory_value": {"type": "string"},
                "content": {"type": "string"},
                "query": {"type": "string"}
            },
            "required": ["action"]
        }

    async def execute(self, action: str, **kwargs) -> str:
        """Execute Letta operation."""
        try:
            if action == "create_agent":
                return await self._create_agent(**kwargs)
            elif action == "send_message":
                return await self._send_message(**kwargs)
            elif action == "update_core_memory":
                return await self._update_core_memory(**kwargs)
            elif action == "add_archival":
                return await self._add_archival(**kwargs)
            elif action == "search_archival":
                return await self._search_archival(**kwargs)
            elif action == "list_agents":
                return await self._list_agents()
            else:
                return f"Unknown action: {action}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _create_agent(self, agent_type: str) -> str:
        """Create Writer or Archivist agent."""
        personas = {
            "writer": "You are a creative writer. Generate engaging novel chapters based on plot beats and character cards provided in context. Maintain consistency with character states and plot threads.",
            "archivist": "You are an archivist. Extract plot beats and character cards from novel chapters in structured JSON format. Focus on significant story events and character development."
        }

        if agent_type not in personas:
            return f"Unknown agent type: {agent_type}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}/v1/agents",
                headers=self.headers,
                json={
                    "name": f"{agent_type}_agent",
                    "persona": personas[agent_type],
                    "human": "User providing context and instructions",
                    "model": "gpt-4"  # Default model, can be configured
                },
                timeout=30.0
            )
            response.raise_for_status()
            agent = response.json()
            return f"Created {agent_type} agent: {agent['id']}"

    async def _send_message(self, agent_id: str, message: str) -> str:
        """Send message to agent."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}/v1/agents/{agent_id}/messages",
                headers=self.headers,
                json={"message": message, "role": "user"},
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()

            # Extract agent's response
            messages = result.get("messages", [])
            agent_responses = [msg["text"] for msg in messages if msg.get("role") == "assistant"]

            if agent_responses:
                return "\n".join(agent_responses)
            return "No response from agent"

    async def _update_core_memory(self, agent_id: str, memory_key: str, memory_value: str) -> str:
        """Update core memory (persona or human)."""
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{self.url}/v1/agents/{agent_id}/memory/core",
                headers=self.headers,
                json={memory_key: memory_value},
                timeout=30.0
            )
            response.raise_for_status()
            return f"Updated core memory: {memory_key}"

    async def _add_archival(self, agent_id: str, content: str) -> str:
        """Add to archival memory."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}/v1/agents/{agent_id}/archival",
                headers=self.headers,
                json={"content": content},
                timeout=30.0
            )
            response.raise_for_status()
            return f"Added to archival memory (length: {len(content)} chars)"

    async def _search_archival(self, agent_id: str, query: str) -> str:
        """Search archival memory."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.url}/v1/agents/{agent_id}/archival",
                headers=self.headers,
                params={"query": query, "limit": 5},
                timeout=30.0
            )
            response.raise_for_status()
            results = response.json()

            if not results:
                return "No results found in archival memory"

            output = [f"Found {len(results)} results:\n"]
            for i, item in enumerate(results, 1):
                output.append(f"{i}. {item['content'][:100]}...")

            return "\n".join(output)

    async def _list_agents(self) -> str:
        """List all agents."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.url}/v1/agents",
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            agents = response.json()

            if not agents:
                return "No agents found"

            output = ["Agents:\n"]
            for agent in agents:
                output.append(f"- {agent['name']} (ID: {agent['id']})")

            return "\n".join(output)
