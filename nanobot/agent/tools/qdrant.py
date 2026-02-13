"""Qdrant vector database tool for novel asset storage and retrieval."""

import httpx
from typing import Any
from nanobot.agent.tools.base import Tool


class QdrantTool(Tool):
    """Qdrant vector database operations for novel assets."""

    def __init__(self, url: str, api_key: str = "", collection_name: str = "novel_assets"):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.collection_name = collection_name
        self.headers = {"api-key": api_key} if api_key else {}

    @property
    def name(self) -> str:
        return "qdrant"

    @property
    def description(self) -> str:
        return """Qdrant vector database operations for novel assets.

Actions:
- create_collection: Initialize collection with vector config
- upsert: Store asset (requires: book_id, asset_type, text; optional: characters, metadata)
- search: Vector search (requires: query; optional: book_id, characters, asset_type, limit)
- scroll: Filter-only search without vectors (requires: book_id; optional: asset_type, limit)
- delete: Remove assets by filter (requires: book_id; optional: asset_type)
- info: Collection statistics

Asset types: plot_beat, character_card"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create_collection", "upsert", "search", "scroll", "delete", "info"]
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
        """Execute Qdrant operation."""
        try:
            if action == "create_collection":
                return await self._create_collection()
            elif action == "upsert":
                return await self._upsert(**kwargs)
            elif action == "search":
                return await self._search(**kwargs)
            elif action == "scroll":
                return await self._scroll(**kwargs)
            elif action == "delete":
                return await self._delete(**kwargs)
            elif action == "info":
                return await self._info()
            else:
                return f"Unknown action: {action}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _create_collection(self) -> str:
        """Create collection with vector config (384 dimensions for sentence-transformers)."""
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.url}/collections/{self.collection_name}",
                headers=self.headers,
                json={
                    "vectors": {
                        "size": 384,  # all-MiniLM-L6-v2 embedding size
                        "distance": "Cosine"
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            return f"Collection '{self.collection_name}' created successfully"

    async def _upsert(self, book_id: str, asset_type: str, text: str,
                     characters: list[str] | None = None, metadata: dict | None = None) -> str:
        """Upsert asset with embedding (requires sentence-transformers)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return "Error: sentence-transformers not installed. Run: pip install sentence-transformers"

        # Generate embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text).tolist()

        # Create point
        point_id = abs(hash(f"{book_id}_{asset_type}_{text[:50]}"))
        point = {
            "id": point_id,
            "vector": embedding,
            "payload": {
                "book_id": book_id,
                "asset_type": asset_type,
                "text": text,
                "characters": characters or [],
                "metadata": metadata or {}
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.url}/collections/{self.collection_name}/points",
                headers=self.headers,
                json={"points": [point]},
                timeout=30.0
            )
            response.raise_for_status()
            return f"Asset upserted: {asset_type} for {book_id} (ID: {point_id})"

    async def _search(self, query: str, book_id: str | None = None,
                     characters: list[str] | None = None, asset_type: str | None = None,
                     limit: int = 10) -> str:
        """Vector search with filters."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return "Error: sentence-transformers not installed"

        # Generate query embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_vector = model.encode(query).tolist()

        # Build filter
        filter_conditions = []
        if book_id:
            filter_conditions.append({"key": "book_id", "match": {"value": book_id}})
        if asset_type:
            filter_conditions.append({"key": "asset_type", "match": {"value": asset_type}})
        if characters:
            for char in characters:
                filter_conditions.append({"key": "characters", "match": {"value": char}})

        search_params = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True
        }
        if filter_conditions:
            search_params["filter"] = {"must": filter_conditions}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}/collections/{self.collection_name}/points/search",
                headers=self.headers,
                json=search_params,
                timeout=30.0
            )
            response.raise_for_status()
            results = response.json()["result"]

            if not results:
                return "No results found"

            output = [f"Found {len(results)} results:\n"]
            for i, hit in enumerate(results, 1):
                payload = hit["payload"]
                output.append(f"{i}. [{payload['asset_type']}] {payload['book_id']}")
                output.append(f"   Text: {payload['text'][:100]}...")
                output.append(f"   Score: {hit['score']:.3f}\n")

            return "\n".join(output)

    async def _scroll(self, book_id: str, asset_type: str | None = None, limit: int = 50) -> str:
        """Filter-only search without vectors (for retrieving all assets of a book)."""
        filter_conditions = [{"key": "book_id", "match": {"value": book_id}}]
        if asset_type:
            filter_conditions.append({"key": "asset_type", "match": {"value": asset_type}})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}/collections/{self.collection_name}/points/scroll",
                headers=self.headers,
                json={
                    "filter": {"must": filter_conditions},
                    "limit": limit,
                    "with_payload": True,
                    "with_vector": False
                },
                timeout=30.0
            )
            response.raise_for_status()
            results = response.json()["result"]["points"]

            if not results:
                return f"No assets found for {book_id}"

            output = [f"Found {len(results)} assets:\n"]
            for i, point in enumerate(results, 1):
                payload = point["payload"]
                output.append(f"{i}. [{payload['asset_type']}] {payload['text'][:80]}...")

            return "\n".join(output)

    async def _delete(self, book_id: str, asset_type: str | None = None) -> str:
        """Delete assets by filter."""
        filter_conditions = [{"key": "book_id", "match": {"value": book_id}}]
        if asset_type:
            filter_conditions.append({"key": "asset_type", "match": {"value": asset_type}})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}/collections/{self.collection_name}/points/delete",
                headers=self.headers,
                json={"filter": {"must": filter_conditions}},
                timeout=30.0
            )
            response.raise_for_status()
            return f"Deleted assets for {book_id}" + (f" ({asset_type})" if asset_type else "")

    async def _info(self) -> str:
        """Get collection statistics."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.url}/collections/{self.collection_name}",
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            info = response.json()["result"]
            return f"""Collection: {self.collection_name}
Points: {info['points_count']}
Vectors: {info['vectors_count']}
Status: {info['status']}"""

