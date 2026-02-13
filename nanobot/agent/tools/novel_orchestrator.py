"""Novel workflow orchestrator tool."""

from typing import Any
from nanobot.agent.tools.base import Tool


class NovelOrchestratorTool(Tool):
    """High-level novel workflow orchestration."""

    def __init__(self, qdrant_tool, letta_tool, beads_tool):
        self.qdrant = qdrant_tool
        self.letta = letta_tool
        self.beads = beads_tool

    @property
    def name(self) -> str:
        return "novel_orchestrator"

    @property
    def description(self) -> str:
        return """Orchestrate novel workflows.

Actions:
- init_library: Initialize library for a book (requires: book_id)
- generate_chapter: Generate chapter using template (requires: template_book_id, new_book_id, chapter_num)
- extract_assets: Extract assets from chapter (requires: book_id, chapter_path)

This tool coordinates qdrant, letta, and beads tools for complex workflows."""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["init_library", "generate_chapter", "extract_assets"]
                },
                "book_id": {"type": "string"},
                "template_book_id": {"type": "string"},
                "new_book_id": {"type": "string"},
                "chapter_num": {"type": "integer"},
                "chapter_path": {"type": "string"}
            },
            "required": ["action"]
        }

    async def execute(self, action: str, **kwargs) -> str:
        """Execute orchestrated workflow."""
        try:
            if action == "init_library":
                return await self._init_library(**kwargs)
            elif action == "generate_chapter":
                return await self._generate_chapter(**kwargs)
            elif action == "extract_assets":
                return await self._extract_assets(**kwargs)
            else:
                return f"Unknown action: {action}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _init_library(self, book_id: str) -> str:
        """Initialize library for a book."""
        # Create Beads task
        task_result = await self.beads.execute(
            action="add",
            title=f"Build library for {book_id}",
            description=f"Extract and index assets for {book_id}"
        )

        # Ensure Qdrant collection exists
        collection_result = await self.qdrant.execute(action="info")
        if "Error" in collection_result:
            collection_result = await self.qdrant.execute(action="create_collection")

        return f"Library initialized for {book_id}\n{task_result}\n{collection_result}"

    async def _generate_chapter(self, template_book_id: str, new_book_id: str, chapter_num: int) -> str:
        """Generate chapter using template book."""
        # 1. Search template book for relevant plot beats
        search_result = await self.qdrant.execute(
            action="scroll",
            book_id=template_book_id,
            asset_type="plot_beat",
            limit=20
        )

        # 2. Create Writer agent if not exists
        agents_list = await self.letta.execute(action="list_agents")
        if "writer_agent" not in agents_list:
            await self.letta.execute(action="create_agent", agent_type="writer")

        # 3. Send context to Writer agent
        # Note: This is a simplified version. Full implementation would assemble context pack
        message = f"""Generate chapter {chapter_num} for {new_book_id}.

Template book context:
{search_result}

Write an engaging chapter that follows similar narrative structure."""

        # Extract agent ID from list (simplified)
        # In production, would parse agent ID properly
        writer_response = await self.letta.execute(
            action="send_message",
            agent_id="writer_agent",  # Simplified
            message=message
        )

        return f"Chapter {chapter_num} generated:\n{writer_response}"

    async def _extract_assets(self, book_id: str, chapter_path: str) -> str:
        """Extract assets from chapter."""
        # This would call the asset_extractor.py script
        # For now, return a placeholder
        return f"Asset extraction for {book_id} from {chapter_path} would be triggered here.\nUse the asset_extractor.py script directly for now."
