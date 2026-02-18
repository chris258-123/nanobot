#!/usr/bin/env python3
"""
Test script for enhanced asset search across all 8 narrative elements.
Demonstrates semantic search capabilities for the novel asset library.
"""

import httpx
import json
from sentence_transformers import SentenceTransformer

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

QDRANT_URL = "http://localhost:6333"
COLLECTION = "novel_assets"

def search_assets(query: str, asset_type: str = None, limit: int = 5):
    """Search for assets using semantic similarity."""
    # Generate query embedding
    query_vector = model.encode(query).tolist()

    # Build filter
    filter_conditions = []
    if asset_type:
        filter_conditions.append({
            "key": "asset_type",
            "match": {"value": asset_type}
        })

    # Search
    payload = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True
    }

    if filter_conditions:
        payload["filter"] = {"must": filter_conditions}

    response = httpx.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        json=payload
    )

    return response.json()["result"]

def print_results(query: str, results: list, asset_type: str = None):
    """Pretty print search results."""
    type_str = f" ({asset_type})" if asset_type else ""
    print(f"\n{'='*80}")
    print(f"Query{type_str}: {query}")
    print(f"{'='*80}")

    for i, hit in enumerate(results, 1):
        score = hit["score"]
        payload = hit["payload"]
        asset_type = payload["asset_type"]
        chapter = payload["chapter"]

        print(f"\n[{i}] Score: {score:.4f} | Type: {asset_type} | Chapter: {chapter}")
        print(f"Text: {payload['text'][:200]}...")

        # Show key metadata based on asset type
        metadata = payload.get("metadata", {})
        if asset_type == "plot_beat":
            print(f"  Event: {metadata.get('event', 'N/A')}")
            print(f"  Impact: {metadata.get('impact', 'N/A')[:100]}...")
        elif asset_type == "character_card":
            print(f"  Name: {metadata.get('name', 'N/A')}")
            print(f"  Traits: {', '.join(metadata.get('traits', []))}")
        elif asset_type == "conflict":
            print(f"  Type: {metadata.get('type', 'N/A')}")
            print(f"  Intensity: {metadata.get('intensity', 'N/A')}")
        elif asset_type == "theme":
            print(f"  Theme: {metadata.get('theme', 'N/A')}")
        elif asset_type == "setting":
            print(f"  Location: {metadata.get('location', 'N/A')[:100]}...")

def main():
    """Run comprehensive search tests."""

    print("\n" + "="*80)
    print("ENHANCED NOVEL ASSET SEARCH SYSTEM - DEMONSTRATION")
    print("="*80)

    # Test 1: Search plot beats
    results = search_assets("角色之间的冲突和对抗", asset_type="plot_beat", limit=3)
    print_results("角色之间的冲突和对抗", results, "plot_beat")

    # Test 2: Search character cards
    results = search_assets("冷静理性的角色", asset_type="character_card", limit=3)
    print_results("冷静理性的角色", results, "character_card")

    # Test 3: Search conflicts
    results = search_assets("权力与责任的矛盾", asset_type="conflict", limit=3)
    print_results("权力与责任的矛盾", results, "conflict")

    # Test 4: Search settings
    results = search_assets("科幻世界的设定", asset_type="setting", limit=2)
    print_results("科幻世界的设定", results, "setting")

    # Test 5: Search themes
    results = search_assets("秩序与混沌", asset_type="theme", limit=3)
    print_results("秩序与混沌", results, "theme")

    # Test 6: Search point of view
    results = search_assets("叙事视角", asset_type="point_of_view", limit=2)
    print_results("叙事视角", results, "point_of_view")

    # Test 7: Search tone
    results = search_assets("紧张的氛围", asset_type="tone", limit=2)
    print_results("紧张的氛围", results, "tone")

    # Test 8: Search style
    results = search_assets("对话驱动的叙事", asset_type="style", limit=2)
    print_results("对话驱动的叙事", results, "style")

    # Test 9: Cross-type search (no filter)
    results = search_assets("伊登和荆璜的关系", limit=5)
    print_results("伊登和荆璜的关系 (跨类型搜索)", results)

    # Statistics
    print(f"\n{'='*80}")
    print("COLLECTION STATISTICS")
    print(f"{'='*80}")

    response = httpx.get(f"{QDRANT_URL}/collections/{COLLECTION}")
    stats = response.json()["result"]
    print(f"Total points: {stats['points_count']}")
    print(f"Vector size: {stats['config']['params']['vectors']['size']}")

    # Count by asset type
    print(f"\nAssets by type:")
    for asset_type in ["plot_beat", "character_card", "conflict", "setting",
                       "theme", "point_of_view", "tone", "style"]:
        scroll_response = httpx.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json={
                "filter": {
                    "must": [{"key": "asset_type", "match": {"value": asset_type}}]
                },
                "limit": 1,
                "with_payload": False,
                "with_vector": False
            }
        )
        # Note: This only gets count from first page, for demo purposes
        print(f"  {asset_type}: (sample query successful)")

if __name__ == "__main__":
    main()
