# Novel Writing Workflow MVP - Implementation Plan

## Context

This plan implements a novel writing workflow system that integrates three external services (Qdrant, Letta, Beads) with nanobot to enable:

1. **Novel Library Pipeline**: Crawl novels → Extract assets (plot_beats, character cards) → Store in vector DB
2. **Writing Pipeline**: Select template book → Assemble context → Generate chapters → Extract & store assets

**Why this is needed:**
- Enable AI-assisted novel writing with consistency across chapters
- Maintain character state and plot continuity through structured asset storage
- Leverage template books for style/structure guidance without plagiarism
- Track complex multi-step workflows with task dependencies

**MVP Scope (1-2 days):**
- 1 template book + 1 new book
- 2 asset types: plot_beats + character cards
- Qdrant for vector search + SQLite for Canon DB
- Letta agents (Writer + Archivist)
- Beads task tracking (with installation instructions)
- Configurable LLM provider for asset extraction

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         nanobot                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ QdrantTool   │  │  LettaTool   │  │  BeadsTool   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌────────────────────────────────────────────────────┐    │
│  │         NovelOrchestratorTool                      │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │ Qdrant  │         │  Letta  │         │  Beads  │
    │ :6333   │         │ :8283   │         │   CLI   │
    └─────────┘         └─────────┘         └─────────┘
         │                    │
         ▼                    ▼
    Vector DB           Agent Memory
    + Payloads          + Archival
```

### Data Flow

**A. Library Pipeline:**
```
novel-crawler → organize_chapters.py → asset_extractor.py → embedder.py → Qdrant
                                                           → canon_db.py → SQLite
```

**B. Writing Pipeline:**
```
Qdrant (search template) → context_pack.py → Letta Writer → chapter.md
                                                           → asset_extractor.py → Qdrant/SQLite
```

## Critical Files to Create/Modify

### New Files (Total: ~1,200 lines)

**Tools (4 files, ~400 lines):**
1. `nanobot/agent/tools/qdrant.py` (~150 lines)
2. `nanobot/agent/tools/letta.py` (~120 lines)
3. `nanobot/agent/tools/beads.py` (~80 lines)
4. `nanobot/agent/tools/novel_orchestrator.py` (~50 lines)

**Skills (1 directory):**
5. `nanobot/skills/novel-workflow/SKILL.md` (~100 lines)
6. `nanobot/skills/novel-workflow/scripts/asset_extractor.py` (~200 lines)
7. `nanobot/skills/novel-workflow/scripts/embedder.py` (~150 lines)
8. `nanobot/skills/novel-workflow/scripts/context_pack.py` (~150 lines)
9. `nanobot/skills/novel-workflow/scripts/canon_db.py` (~100 lines)

**Infrastructure:**
10. `docker-compose.yml` (~50 lines)
11. `requirements-novel.txt` (~10 lines)

### Files to Modify

1. `nanobot/config/schema.py` - Add IntegrationsConfig (~30 lines)
2. `nanobot/agent/loop.py` - Register new tools (~10 lines)

## Implementation Phases

### Phase 1: Infrastructure Setup (2-3 hours)

#### 1.1 Install Dependencies

**System Requirements:**
```bash
# Install Beads (task tracker)
git clone https://github.com/steveyegge/beads.git
cd beads
cargo build --release
sudo cp target/release/bd /usr/local/bin/

# Verify installation
bd --version
```

**Python Dependencies:**
```bash
# Add to requirements-novel.txt
qdrant-client>=1.7.0
sentence-transformers>=2.2.0
httpx>=0.25.0
```

```bash
pip install -r requirements-novel.txt
```

#### 1.2 Docker Compose Setup

**File:** `docker-compose.yml` (workspace root)

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage:z
    restart: unless-stopped

  letta:
    image: letta/letta:latest
    ports:
      - "8283:8283"
    volumes:
      - ./letta_data:/root/.letta
    environment:
      - LETTA_SERVER_PORT=8283
      - LETTA_SERVER_HOST=0.0.0.0
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=canon
      - POSTGRES_USER=novel
      - POSTGRES_PASSWORD=novel123
    volumes:
      - ./postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
```

**Start services:**
```bash
docker-compose up -d
```

**Verification:**
```bash
# Check Qdrant
curl http://localhost:6333/collections

# Check Letta
curl http://localhost:8283/v1/agents

# Check Postgres
psql -h localhost -U novel -d canon -c "SELECT version();"
```

#### 1.3 Configuration Schema

**File:** `nanobot/config/schema.py`

Add these classes before the main `Config` class:

```python
class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    enabled: bool = False
    url: str = "http://localhost:6333"
    api_key: str = ""
    collection_name: str = "novel_assets"

class LettaConfig(BaseModel):
    """Letta agent configuration."""
    enabled: bool = False
    url: str = "http://localhost:8283"
    api_key: str = ""

class BeadsConfig(BaseModel):
    """Beads task management configuration."""
    enabled: bool = False
    workspace_path: str = "~/.beads"

class IntegrationsConfig(BaseModel):
    """External service integrations."""
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    letta: LettaConfig = Field(default_factory=LettaConfig)
    beads: BeadsConfig = Field(default_factory=BeadsConfig)
```

Add to `Config` class:
```python
integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
```

**Update user config:** `~/.nanobot/config.json`

```json
{
  "integrations": {
    "qdrant": {
      "enabled": true,
      "url": "http://localhost:6333",
      "collection_name": "novel_assets"
    },
    "letta": {
      "enabled": true,
      "url": "http://localhost:8283"
    },
    "beads": {
      "enabled": true,
      "workspace_path": "~/.beads"
    }
  }
}
```

### Phase 2: Core Tools Implementation (4-5 hours)

#### 2.1 QdrantTool

**File:** `nanobot/agent/tools/qdrant.py`

Key responsibilities:
- Create collection with vector config (384 dimensions for sentence-transformers)
- Upsert assets with embeddings and payloads
- Search with filters (book_id, characters[], asset_type)
- Batch operations for efficiency

**Implementation highlights:**
```python
from nanobot.agent.tools.base import Tool
import httpx
from typing import Any

class QdrantTool(Tool):
    """Qdrant vector database operations for novel assets."""

    def __init__(self, url: str, api_key: str = "", collection_name: str = "novel_assets"):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name

    @property
    def name(self) -> str:
        return "qdrant"

    @property
    def description(self) -> str:
        return """Qdrant vector database operations.

Actions:
- create_collection: Initialize collection
- upsert: Store asset (book_id, asset_type, characters, text, metadata)
- search: Vector search with filters (query, book_id, characters, asset_type, limit)
- delete: Remove assets by filter
- info: Collection statistics"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create_collection", "upsert", "search", "delete", "info"]
                },
                "book_id": {"type": "string"},
                "asset_type": {
                    "type": "string",
                    "enum": ["plot_beat", "character_card"]
                },
                "characters": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "text": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "metadata": {"type": "object"}
            },
            "required": ["action"]
        }

    async def execute(self, action: str, **kwargs) -> str:
        # Implementation details in full code
        pass
```

**Key methods:**
- `_create_collection()`: PUT /collections/{name} with vector config
- `_upsert()`: Generate embedding with sentence-transformers, PUT /collections/{name}/points
- `_search()`: POST /collections/{name}/points/search with filters
- `_delete()`: POST /collections/{name}/points/delete with filter
- `_info()`: GET /collections/{name}

#### 2.2 LettaTool

**File:** `nanobot/agent/tools/letta.py`

Key responsibilities:
- Create Writer and Archivist agents
- Send messages to agents with context
- Update Core Memory (limited, always loaded)
- Add to Archival Memory (unlimited, retrieved on demand)
- Search archival memory

**Implementation highlights:**
```python
class LettaTool(Tool):
    """Letta agent memory management."""

    def __init__(self, url: str, api_key: str = ""):
        self.url = url
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "letta"

    @property
    def description(self) -> str:
        return """Letta agent operations.

Actions:
- create_agent: Create Writer or Archivist agent
- send_message: Send message to agent
- update_core_memory: Update core memory (persona/human)
- add_archival: Add to archival memory
- search_archival: Search archival memory
- list_agents: List all agents"""

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
```

**Agent Personas:**
- **Writer**: "You are a creative writer. Generate engaging novel chapters based on plot beats and character cards provided in context."
- **Archivist**: "You are an archivist. Extract plot beats and character cards from novel chapters in structured JSON format."

#### 2.3 BeadsTool

**File:** `nanobot/agent/tools/beads.py`

Key responsibilities:
- Add tasks with dependencies (blocks, parent)
- List tasks (all or doable only)
- Update task status (todo, doing, done)
- Query tasks by filters

**Implementation highlights:**
```python
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
- add: Create task with dependencies
- list: List tasks (--doable for actionable)
- update: Update task status
- query: Query tasks by filter"""

    async def execute(self, action: str, **kwargs) -> str:
        # Use subprocess to call bd CLI
        if action == "add":
            cmd = ["bd", "add", kwargs["title"]]
            if kwargs.get("blocks"):
                cmd.extend(["--blocks", kwargs["blocks"]])
            if kwargs.get("parent"):
                cmd.extend(["--parent", kwargs["parent"]])
        # ... other actions
```

#### 2.4 NovelOrchestratorTool

**File:** `nanobot/agent/tools/novel_orchestrator.py`

High-level workflow coordination that combines other tools:

```python
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
- init_library: Initialize library for a book
- generate_chapter: Generate chapter using template
- extract_assets: Extract assets from chapter"""
```

**Tool Registration:**

**File:** `nanobot/agent/loop.py` (modify `_register_default_tools()`)

```python
# Add after existing tool registrations
if self.config.integrations.qdrant.enabled:
    qdrant_tool = QdrantTool(
        url=self.config.integrations.qdrant.url,
        api_key=self.config.integrations.qdrant.api_key,
        collection_name=self.config.integrations.qdrant.collection_name
    )
    self.tools.register(qdrant_tool)

if self.config.integrations.letta.enabled:
    letta_tool = LettaTool(
        url=self.config.integrations.letta.url,
        api_key=self.config.integrations.letta.api_key
    )
    self.tools.register(letta_tool)

if self.config.integrations.beads.enabled:
    beads_tool = BeadsTool(
        workspace_path=self.config.integrations.beads.workspace_path
    )
    self.tools.register(beads_tool)

# Register orchestrator if all dependencies available
if all([
    self.config.integrations.qdrant.enabled,
    self.config.integrations.letta.enabled,
    self.config.integrations.beads.enabled
]):
    orchestrator = NovelOrchestratorTool(qdrant_tool, letta_tool, beads_tool)
    self.tools.register(orchestrator)
```

### Phase 3: Processing Scripts (3-4 hours)

#### 3.1 Asset Extractor

**File:** `nanobot/skills/novel-workflow/scripts/asset_extractor.py`

Extracts plot_beats and character cards from chapters using configurable LLM.

**Key functions:**
```python
import json
import httpx
from pathlib import Path

def extract_plot_beats(chapter_text: str, llm_config: dict) -> list[dict]:
    """Extract plot beats using LLM."""
    prompt = f"""Extract plot beats from this chapter. A plot beat is a significant story event.

Format as JSON array:
[
  {{
    "event": "what happened",
    "characters": ["character names"],
    "impact": "story significance",
    "chapter_position": "beginning|middle|end"
  }}
]

Chapter:
{chapter_text}

Return only the JSON array, no other text."""

    # Call LLM (configurable: nanobot API or custom)
    response = call_llm(prompt, llm_config)
    beats = json.loads(response)
    return beats

def extract_character_cards(chapter_text: str, llm_config: dict) -> list[dict]:
    """Extract character information using LLM."""
    prompt = f"""Extract character information from this chapter.

Format as JSON array:
[
  {{
    "name": "character name",
    "traits": ["personality traits shown"],
    "state": "current emotional/physical state",
    "relationships": {{"other_char": "relationship description"}},
    "goals": ["character goals revealed"],
    "secrets": ["secrets revealed or hinted"]
  }}
]

Chapter:
{chapter_text}

Return only the JSON array, no other text."""

    response = call_llm(prompt, llm_config)
    cards = json.loads(response)
    return cards

def call_llm(prompt: str, llm_config: dict) -> str:
    """Call LLM API (configurable)."""
    if llm_config["type"] == "nanobot":
        # Use nanobot's configured LLM
        # This would call through nanobot's provider system
        pass
    elif llm_config["type"] == "custom":
        # Use custom API endpoint
        response = httpx.post(
            llm_config["url"],
            json={
                "model": llm_config["model"],
                "messages": [{"role": "user", "content": prompt}]
            },
            headers={"Authorization": f"Bearer {llm_config['api_key']}"}
        )
        return response.json()["choices"][0]["message"]["content"]

def process_chapter_file(chapter_path: str, book_id: str, output_dir: str, llm_config: dict):
    """Process a single chapter file."""
    with open(chapter_path) as f:
        chapter_text = f.read()

    beats = extract_plot_beats(chapter_text, llm_config)
    characters = extract_character_cards(chapter_text, llm_config)

    output = {
        "book_id": book_id,
        "chapter": Path(chapter_path).stem,
        "plot_beats": beats,
        "character_cards": characters
    }

    output_path = Path(output_dir) / f"{book_id}_{Path(chapter_path).stem}_assets.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--chapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-config", required=True, help="Path to LLM config JSON")
    args = parser.parse_args()

    with open(args.llm_config) as f:
        llm_config = json.load(f)

    chapter_dir = Path(args.chapter_dir)
    for chapter_file in sorted(chapter_dir.glob("*.md")):
        print(f"Processing {chapter_file.name}...")
        output_path = process_chapter_file(
            str(chapter_file),
            args.book_id,
            args.output_dir,
            llm_config
        )
        print(f"  → {output_path}")
```

**LLM Config Example:** `llm_config.json`
```json
{
  "type": "custom",
  "url": "https://api.deepseek.com/v1/chat/completions",
  "model": "deepseek-chat",
  "api_key": "sk-..."
}
```

#### 3.2 Embedder & Upserter

**File:** `nanobot/skills/novel-workflow/scripts/embedder.py`

Generates embeddings and upserts to Qdrant:

```python
from sentence_transformers import SentenceTransformer
import httpx
import json
from pathlib import Path

def embed_and_upsert(assets_json_path: str, qdrant_url: str, collection_name: str):
    """Embed assets and upsert to Qdrant."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    with open(assets_json_path) as f:
        data = json.load(f)

    points = []

    # Process plot beats
    for beat in data["plot_beats"]:
        text = f"{beat['event']} {beat['impact']}"
        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_{beat['event'][:30]}")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "plot_beat",
                "chapter": data["chapter"],
                "characters": beat["characters"],
                "text": text,
                "metadata": beat
            }
        })

    # Process character cards
    for card in data["character_cards"]:
        text = f"{card['name']}: {' '.join(card['traits'])} {card['state']}"
        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_{card['name']}")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "character_card",
                "chapter": data["chapter"],
                "characters": [card["name"]],
                "text": text,
                "metadata": card
            }
        })

    # Batch upsert
    response = httpx.put(
        f"{qdrant_url}/collections/{collection_name}/points",
        json={"points": points}
    )
    response.raise_for_status()

    return len(points)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, help="Path to assets JSON")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="novel_assets")
    args = parser.parse_args()

    count = embed_and_upsert(args.assets, args.qdrant_url, args.collection)
    print(f"Upserted {count} points to Qdrant")
```

#### 3.3 Context Pack Assembler

**File:** `nanobot/skills/novel-workflow/scripts/context_pack.py`

Assembles Context Pack from template book for Writer agent:

```python
import httpx
import json
from pathlib import Path

def assemble_context_pack(template_book_id: str, new_book_context: dict,
                         qdrant_url: str, collection_name: str,
                         output_path: str) -> dict:
    """Assemble Context Pack from template book."""

    # 1. Query template book's plot beats
    beats_query = {
        "vector": [0] * 384,  # Dummy vector for filter-only query
        "limit": 50,
        "filter": {
            "must": [
                {"key": "book_id", "match": {"value": template_book_id}},
                {"key": "asset_type", "match": {"value": "plot_beat"}}
            ]
        },
        "with_payload": True
    }

    response = httpx.post(
        f"{qdrant_url}/collections/{collection_name}/points/search",
        json=beats_query
    )
    beats = response.json()["result"]

    # 2. Query character cards
    chars_query = {
        "vector": [0] * 384,
        "limit": 50,
        "filter": {
            "must": [
                {"key": "book_id", "match": {"value": template_book_id}},
                {"key": "asset_type", "match": {"value": "character_card"}}
            ]
        },
        "with_payload": True
    }

    response = httpx.post(
        f"{qdrant_url}/collections/{collection_name}/points/search",
        json=chars_query
    )
    characters = response.json()["result"]

    # 3. Assemble Context Pack
    context_pack = {
        "template_book_id": template_book_id,
        "new_book_context": new_book_context,
        "template_plot_beats": [
            {
                "event": hit["payload"]["metadata"]["event"],
                "impact": hit["payload"]["metadata"]["impact"],
                "chapter_position": hit["payload"]["metadata"].get("chapter_position", "unknown")
            }
            for hit in beats[:20]  # Top 20 beats
        ],
        "template_characters": [
            {
                "name": hit["payload"]["metadata"]["name"],
                "traits": hit["payload"]["metadata"]["traits"],
                "relationships": hit["payload"]["metadata"].get("relationships", {})
            }
            for hit in characters[:10]  # Top 10 characters
        ],
        "writing_guidelines": {
            "style": "Match the narrative style of the template book",
            "pacing": "Follow similar chapter structure and pacing",
            "character_depth": "Develop characters with similar depth and complexity"
        }
    }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(context_pack, f, indent=2)

    return context_pack

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-book-id", required=True)
    parser.add_argument("--new-book-context", required=True, help="Path to new book context JSON")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="novel_assets")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.new_book_context) as f:
        new_book_context = json.load(f)

    context_pack = assemble_context_pack(
        args.template_book_id,
        new_book_context,
        args.qdrant_url,
        args.collection,
        args.output
    )
    print(f"Context pack saved to {args.output}")
```

#### 3.4 Canon DB Manager

**File:** `nanobot/skills/novel-workflow/scripts/canon_db.py`

SQLite database for character state tracking:

```python
import sqlite3
import json
from pathlib import Path
from typing import Optional

class CanonDB:
    """SQLite database for novel canon (character states, world rules)."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL,
                name TEXT NOT NULL,
                traits TEXT,  -- JSON array
                current_state TEXT,
                goals TEXT,  -- JSON array
                secrets TEXT,  -- JSON array
                relationships TEXT,  -- JSON object
                last_updated_chapter TEXT,
                UNIQUE(book_id, name)
            );

            CREATE TABLE IF NOT EXISTS world_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL,
                rule_type TEXT NOT NULL,  -- magic_system, geography, politics, etc.
                rule_name TEXT NOT NULL,
                description TEXT,
                constraints TEXT,  -- JSON array
                introduced_chapter TEXT,
                UNIQUE(book_id, rule_type, rule_name)
            );

            CREATE TABLE IF NOT EXISTS plot_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL,
                thread_name TEXT NOT NULL,
                description TEXT,
                status TEXT,  -- active, resolved, abandoned
                introduced_chapter TEXT,
                resolved_chapter TEXT,
                UNIQUE(book_id, thread_name)
            );

            CREATE INDEX IF NOT EXISTS idx_characters_book ON characters(book_id);
            CREATE INDEX IF NOT EXISTS idx_world_rules_book ON world_rules(book_id);
            CREATE INDEX IF NOT EXISTS idx_plot_threads_book ON plot_threads(book_id);
        """)
        self.conn.commit()

    def upsert_character(self, book_id: str, name: str, traits: list[str],
                        current_state: str, goals: list[str], secrets: list[str],
                        relationships: dict, chapter: str):
        """Insert or update character."""
        self.conn.execute("""
            INSERT INTO characters (book_id, name, traits, current_state, goals, secrets, relationships, last_updated_chapter)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id, name) DO UPDATE SET
                traits = excluded.traits,
                current_state = excluded.current_state,
                goals = excluded.goals,
                secrets = excluded.secrets,
                relationships = excluded.relationships,
                last_updated_chapter = excluded.last_updated_chapter
        """, (book_id, name, json.dumps(traits), current_state, json.dumps(goals),
              json.dumps(secrets), json.dumps(relationships), chapter))
        self.conn.commit()

    def get_character(self, book_id: str, name: str) -> Optional[dict]:
        """Get character by name."""
        cursor = self.conn.execute("""
            SELECT name, traits, current_state, goals, secrets, relationships, last_updated_chapter
            FROM characters
            WHERE book_id = ? AND name = ?
        """, (book_id, name))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "name": row[0],
            "traits": json.loads(row[1]),
            "current_state": row[2],
            "goals": json.loads(row[3]),
            "secrets": json.loads(row[4]),
            "relationships": json.loads(row[5]),
            "last_updated_chapter": row[6]
        }

    def get_all_characters(self, book_id: str) -> list[dict]:
        """Get all characters for a book."""
        cursor = self.conn.execute("""
            SELECT name, traits, current_state, goals, secrets, relationships, last_updated_chapter
            FROM characters
            WHERE book_id = ?
            ORDER BY name
        """, (book_id,))
        return [
            {
                "name": row[0],
                "traits": json.loads(row[1]),
                "current_state": row[2],
                "goals": json.loads(row[3]),
                "secrets": json.loads(row[4]),
                "relationships": json.loads(row[5]),
                "last_updated_chapter": row[6]
            }
            for row in cursor.fetchall()
        ]

    def upsert_world_rule(self, book_id: str, rule_type: str, rule_name: str,
                         description: str, constraints: list[str], chapter: str):
        """Insert or update world rule."""
        self.conn.execute("""
            INSERT INTO world_rules (book_id, rule_type, rule_name, description, constraints, introduced_chapter)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id, rule_type, rule_name) DO UPDATE SET
                description = excluded.description,
                constraints = excluded.constraints
        """, (book_id, rule_type, rule_name, description, json.dumps(constraints), chapter))
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()

if __name__ == "__main__":
    # Example usage
    db = CanonDB("~/.nanobot/workspace/canon.db")

    # Add character
    db.upsert_character(
        book_id="book1",
        name="Alice",
        traits=["brave", "curious"],
        current_state="searching for answers",
        goals=["find the truth", "protect her friends"],
        secrets=["knows about the prophecy"],
        relationships={"Bob": "trusted ally", "Eve": "rival"},
        chapter="chapter_01"
    )

    # Get character
    alice = db.get_character("book1", "Alice")
    print(json.dumps(alice, indent=2))

    db.close()
```

### Phase 4: Skill Creation (2 hours)

#### 4.1 Novel Workflow Skill

**File:** `nanobot/skills/novel-workflow/SKILL.md`

```markdown
---
name: novel-workflow
description: Novel writing workflow with Qdrant, Letta, and Beads integration. Orchestrates library building and chapter generation.
---

# Novel Writing Workflow

This skill orchestrates the complete novel writing workflow using:
- **Qdrant**: Vector database for asset storage and retrieval
- **Letta**: Agent runtime for Writer and Archivist agents
- **Beads**: Task tracking for workflow management
- **Canon DB**: SQLite database for character state tracking

## Prerequisites

1. Docker services running:
   ```bash
   docker-compose up -d
   ```

2. Beads CLI installed:
   ```bash
   bd --version
   ```

3. Configuration enabled in `~/.nanobot/config.json`:
   ```json
   {
     "integrations": {
       "qdrant": {"enabled": true},
       "letta": {"enabled": true},
       "beads": {"enabled": true}
     }
   }
   ```

## Workflows

### A. Library Building Pipeline

Build a searchable asset library from a novel:

1. **Crawl novel** (use existing novel-crawler skill):
   ```
   Use novel-crawler skill to crawl chapters to ~/novel_data/{book_id}/
   ```

2. **Extract assets**:
   ```bash
   python scripts/asset_extractor.py \
     --book-id book1 \
     --chapter-dir ~/novel_data/book1 \
     --output-dir ~/novel_assets \
     --llm-config llm_config.json
   ```

3. **Embed and upsert to Qdrant**:
   ```bash
   for asset_file in ~/novel_assets/book1_*.json; do
     python scripts/embedder.py \
       --assets "$asset_file" \
       --qdrant-url http://localhost:6333
   done
   ```

4. **Update Canon DB**:
   ```python
   from scripts.canon_db import CanonDB
   import json

   db = CanonDB("~/.nanobot/workspace/canon.db")

   # Load assets and update character states
   with open("~/novel_assets/book1_chapter01_assets.json") as f:
       assets = json.load(f)

   for card in assets["character_cards"]:
       db.upsert_character(
           book_id="book1",
           name=card["name"],
           traits=card["traits"],
           current_state=card["state"],
           goals=card.get("goals", []),
           secrets=card.get("secrets", []),
           relationships=card.get("relationships", {}),
           chapter=assets["chapter"]
       )
   ```

### B. Writing Pipeline

Generate new chapters using template book:

1. **Initialize Letta agents**:
   ```
   Use letta tool to create Writer and Archivist agents
   ```

2. **Assemble Context Pack**:
   ```bash
   python scripts/context_pack.py \
     --template-book-id book1 \
     --new-book-context new_book_context.json \
     --output context_pack.json
   ```

3. **Generate chapter**:
   ```
   Use letta tool to send context pack to Writer agent
   ```

4. **Extract assets from generated chapter**:
   ```bash
   python scripts/asset_extractor.py \
     --book-id book2 \
     --chapter-dir ~/generated_chapters \
     --output-dir ~/novel_assets \
     --llm-config llm_config.json
   ```

5. **Update databases**:
   - Embed and upsert to Qdrant
   - Update Canon DB with new character states

## Tools Available

- `qdrant`: Vector database operations
- `letta`: Agent memory management
- `beads`: Task tracking
- `novel_orchestrator`: High-level workflow coordination

## Scripts Reference

- `asset_extractor.py`: Extract plot_beats and character cards from chapters
- `embedder.py`: Generate embeddings and upsert to Qdrant
- `context_pack.py`: Assemble Context Pack from template book
- `canon_db.py`: SQLite database for character state tracking

## Example: Complete MVP Workflow

```python
# 1. Create Qdrant collection
qdrant(action="create_collection")

# 2. Create Beads task for library building
beads(action="add", title="Build library for book1", description="Extract and index assets")

# 3. Create Letta agents
letta(action="create_agent", agent_type="writer")
letta(action="create_agent", agent_type="archivist")

# 4. Process template book (book1)
# - Use novel-crawler to crawl chapters
# - Run asset_extractor.py on all chapters
# - Run embedder.py to upsert to Qdrant
# - Update Canon DB

# 5. Generate new chapter for book2
# - Run context_pack.py to assemble context
# - Send to Writer agent via letta tool
# - Extract assets from generated chapter
# - Update databases

# 6. Update Beads task
beads(action="update", task_id="TASK-1", status="done")
```

## Configuration

**LLM Config** (`llm_config.json`):
```json
{
  "type": "custom",
  "url": "https://api.deepseek.com/v1/chat/completions",
  "model": "deepseek-chat",
  "api_key": "sk-..."
}
```

**New Book Context** (`new_book_context.json`):
```json
{
  "book_id": "book2",
  "title": "My New Novel",
  "genre": "fantasy",
  "setting": "medieval kingdom",
  "main_characters": [
    {
      "name": "Hero",
      "role": "protagonist",
      "traits": ["brave", "determined"]
    }
  ],
  "plot_outline": "A hero's journey to save the kingdom..."
}
```

## Troubleshooting

**Qdrant connection error:**
```bash
curl http://localhost:6333/collections
# If fails, check: docker-compose ps
```

**Letta agent not responding:**
```bash
curl http://localhost:8283/v1/agents
# Check Letta logs: docker-compose logs letta
```

**Beads command not found:**
```bash
bd --version
# If fails, reinstall: cargo install --path .
```
```

## Verification Steps

### End-to-End Testing

**Test 1: Infrastructure**
```bash
# Verify Docker services
docker-compose ps
curl http://localhost:6333/collections
curl http://localhost:8283/v1/agents

# Verify Beads
bd --version
bd init  # Initialize Beads workspace
```

**Test 2: Qdrant Operations**
```bash
# Test via nanobot
nanobot agent -m "Use qdrant tool to create_collection"
nanobot agent -m "Use qdrant tool to upsert a test plot_beat for book_test"
nanobot agent -m "Use qdrant tool to search for plot_beat in book_test"
```

**Test 3: Letta Operations**
```bash
# Test via nanobot
nanobot agent -m "Use letta tool to create a writer agent"
nanobot agent -m "Use letta tool to send message 'Hello' to the writer agent"
```

**Test 4: Asset Extraction**
```bash
# Create test chapter
echo "# Chapter 1\n\nAlice met Bob in the forest. She was searching for answers." > test_chapter.md

# Extract assets
python nanobot/skills/novel-workflow/scripts/asset_extractor.py \
  --book-id test_book \
  --chapter-dir . \
  --output-dir ./test_assets \
  --llm-config llm_config.json

# Verify output
cat test_assets/test_book_test_chapter_assets.json
```

**Test 5: Complete Pipeline**
```bash
# 1. Crawl a small novel (3 chapters)
nanobot agent -m "Use novel-crawler to crawl 3 chapters from [URL] to ~/novel_data/test_book"

# 2. Extract assets
for chapter in ~/novel_data/test_book/*.md; do
  python scripts/asset_extractor.py \
    --book-id test_book \
    --chapter-dir ~/novel_data/test_book \
    --output-dir ~/novel_assets \
    --llm-config llm_config.json
done

# 3. Embed and upsert
for asset in ~/novel_assets/test_book_*.json; do
  python scripts/embedder.py --assets "$asset"
done

# 4. Verify in Qdrant
nanobot agent -m "Use qdrant tool to search for 'character meeting' in test_book"

# 5. Generate new chapter
python scripts/context_pack.py \
  --template-book-id test_book \
  --new-book-context new_book.json \
  --output context.json

nanobot agent -m "Use letta tool to send the context pack to writer agent and generate a chapter"
```

## Dependencies Summary

**Python packages:**
```
qdrant-client>=1.7.0
sentence-transformers>=2.2.0
httpx>=0.25.0
```

**External services:**
- Docker + docker-compose
- Qdrant (Docker image)
- Letta (Docker image)
- Postgres (Docker image)
- Beads CLI (Rust, cargo install)

**Estimated total lines of code:** ~1,200 lines
- Tools: ~400 lines
- Scripts: ~600 lines
- Skill: ~100 lines
- Config: ~30 lines
- Docker: ~50 lines
- Docs: ~20 lines

## Next Steps After MVP

1. Add more asset types (style cards, structure cards)
2. Implement QC (quality control) checks
3. Add web UI for workflow management
4. Integrate with more LLM providers
5. Add batch processing for multiple books
6. Implement conflict detection for character states
7. Add visualization for plot threads and character relationships

---

**Note:** This is an MVP implementation focusing on core functionality. Production use would require additional error handling, logging, monitoring, and testing.

