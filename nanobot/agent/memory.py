"""Memory system for persistent agent memory."""

from pathlib import Path
from datetime import datetime

from nanobot.utils.helpers import ensure_dir, today_date


class MemoryStore:
    """
    Memory system for the agent.
    
    Supports daily notes (memory/YYYY-MM-DD.md) and long-term memory (MEMORY.md).
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
    
    def get_today_file(self) -> Path:
        """Get path to today's memory file."""
        return self.memory_dir / f"{today_date()}.md"
    
    def read_today(self) -> str:
        """Read today's memory notes."""
        today_file = self.get_today_file()
        if today_file.exists():
            return today_file.read_text(encoding="utf-8")
        return ""
    
    def append_today(self, content: str) -> None:
        """Append content to today's memory notes."""
        today_file = self.get_today_file()
        
        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            content = existing + "\n" + content
        else:
            # Add header for new day
            header = f"# {today_date()}\n\n"
            content = header + content
        
        today_file.write_text(content, encoding="utf-8")
    
    def read_long_term(self) -> str:
        """Read long-term memory (MEMORY.md)."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""
    
    def write_long_term(self, content: str) -> None:
        """Write to long-term memory (MEMORY.md)."""
        self.memory_file.write_text(content, encoding="utf-8")
    
    def get_recent_memories(self, days: int = 7) -> str:
        """
        Get memories from the last N days.
        
        Args:
            days: Number of days to look back.
        
        Returns:
            Combined memory content.
        """
        from datetime import timedelta
        
        memories = []
        today = datetime.now().date()
        
        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.memory_dir / f"{date_str}.md"
            
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                memories.append(content)
        
        return "\n\n---\n\n".join(memories)
    
    def list_memory_files(self) -> list[Path]:
        """List all memory files sorted by date (newest first)."""
        if not self.memory_dir.exists():
            return []
        
        files = list(self.memory_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)
    
    def get_history_file(self) -> Path:
        """Get path to HISTORY.md file."""
        return self.memory_dir / "HISTORY.md"

    def append_to_history(self, content: str) -> None:
        """Append event to HISTORY.md with timestamp."""
        history_file = self.get_history_file()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"\n## {timestamp}\n{content}\n"

        if history_file.exists():
            existing = history_file.read_text(encoding="utf-8")
            content = existing + entry
        else:
            content = f"# Conversation History\n{entry}"

        history_file.write_text(content, encoding="utf-8")

    def append_facts(self, facts: str) -> None:
        """Append facts to MEMORY.md."""
        if self.memory_file.exists():
            existing = self.memory_file.read_text(encoding="utf-8")
            content = existing + f"\n\n{facts}"
        else:
            content = f"# Long-term Memory\n\n{facts}"

        self.memory_file.write_text(content, encoding="utf-8")

    async def consolidate_session(
        self,
        messages: list[dict],
        provider,
        model: str
    ) -> tuple[str, str]:
        """
        Consolidate old messages into facts and events.

        Args:
            messages: Messages to consolidate (oldest messages from session)
            provider: LLM provider for summarization
            model: Model to use

        Returns:
            Tuple of (facts, events) extracted from messages
        """
        # Build conversation text
        conversation = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in messages
        ])

        # Consolidation prompt
        prompt = f"""Analyze this conversation and extract:
1. FACTS: Important information to remember long-term (preferences, decisions, context)
2. EVENTS: Key events that happened (actions taken, milestones)

Conversation:
{conversation}

Format your response as:
FACTS:
- [fact 1]
- [fact 2]

EVENTS:
- [event 1]
- [event 2]

Be concise. Only include truly important information."""

        # Call LLM
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model
        )

        # Parse response
        content = response.content
        facts = ""
        events = ""

        if "FACTS:" in content and "EVENTS:" in content:
            parts = content.split("EVENTS:")
            facts = parts[0].replace("FACTS:", "").strip()
            events = parts[1].strip()

        return facts, events

    def get_memory_context(self) -> str:
        """
        Get memory context for the agent.

        Returns:
            Formatted memory context including long-term and recent memories.
        """
        parts = []

        # Long-term memory
        long_term = self.read_long_term()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)

        # History (recent entries only, last 20 lines)
        history_file = self.get_history_file()
        if history_file.exists():
            history = history_file.read_text(encoding="utf-8")
            lines = history.split("\n")
            recent_history = "\n".join(lines[-20:]) if len(lines) > 20 else history
            parts.append("## Recent History\n" + recent_history)

        # Today's notes
        today = self.read_today()
        if today:
            parts.append("## Today's Notes\n" + today)

        return "\n\n".join(parts) if parts else ""
