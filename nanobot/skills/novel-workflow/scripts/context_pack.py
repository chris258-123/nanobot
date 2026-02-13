"""Context pack assembler for novel writing.

Assembles Context Pack from template book for Writer agent.
"""

import httpx
import json
from pathlib import Path
import argparse


def assemble_context_pack(template_book_id: str, new_book_context: dict,
                         qdrant_url: str, collection_name: str,
                         output_path: str) -> dict:
    """Assemble Context Pack from template book."""

    # 1. Query template book's plot beats using scroll API (filter-only, no vector search)
    beats_response = httpx.post(
        f"{qdrant_url}/collections/{collection_name}/points/scroll",
        json={
            "filter": {
                "must": [
                    {"key": "book_id", "match": {"value": template_book_id}},
                    {"key": "asset_type", "match": {"value": "plot_beat"}}
                ]
            },
            "limit": 50,
            "with_payload": True,
            "with_vector": False
        },
        timeout=30.0
    )
    beats_response.raise_for_status()
    beats = beats_response.json()["result"]["points"]

    # 2. Query character cards using scroll API
    chars_response = httpx.post(
        f"{qdrant_url}/collections/{collection_name}/points/scroll",
        json={
            "filter": {
                "must": [
                    {"key": "book_id", "match": {"value": template_book_id}},
                    {"key": "asset_type", "match": {"value": "character_card"}}
                ]
            },
            "limit": 50,
            "with_payload": True,
            "with_vector": False
        },
        timeout=30.0
    )
    chars_response.raise_for_status()
    characters = chars_response.json()["result"]["points"]

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
                "traits": hit["payload"]["metadata"].get("traits", []),
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
        json.dump(context_pack, f, indent=2, ensure_ascii=False)

    return context_pack


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble context pack from template book")
    parser.add_argument("--template-book-id", required=True, help="Template book ID")
    parser.add_argument("--new-book-context", required=True, help="Path to new book context JSON")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--collection", default="novel_assets", help="Collection name")
    parser.add_argument("--output", required=True, help="Output path for context pack")
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
    print(f"  - {len(context_pack['template_plot_beats'])} plot beats")
    print(f"  - {len(context_pack['template_characters'])} characters")

