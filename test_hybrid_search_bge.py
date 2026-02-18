#!/usr/bin/env python3
"""
Test hybrid search with BGE-large model and novel_assets_v2 collection.
"""

import os
# Disable SOCKS proxy at the very beginning
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)

import httpx
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

QDRANT_URL = "http://localhost:6333"
COLLECTION = "novel_assets_v2"

class HybridSearcherBGE:
    """Hybrid search with BGE-large model."""

    def __init__(self):
        print("Loading BAAI/bge-large-zh-v1.5 model...")
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        print("Model loaded successfully!")
        self.qdrant_url = QDRANT_URL

    def vector_search(self, query: str, asset_type: str = None, limit: int = 20) -> List[Dict]:
        """Pure vector search with BGE model."""
        query_vector = self.model.encode(query).tolist()

        filter_conditions = []
        if asset_type:
            filter_conditions.append({
                "key": "asset_type",
                "match": {"value": asset_type}
            })

        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "score_threshold": 0.2
        }

        if filter_conditions:
            payload["filter"] = {"must": filter_conditions}

        response = httpx.post(
            f"{self.qdrant_url}/collections/{COLLECTION}/points/search",
            json=payload,
            timeout=30.0
        )

        return response.json()["result"]

    def keyword_search(self, query: str, asset_type: str = None, limit: int = 20) -> List[Dict]:
        """Keyword-based search using scroll."""
        keywords = query.replace("，", " ").replace("、", " ").split()

        filter_conditions = []
        if asset_type:
            filter_conditions.append({
                "key": "asset_type",
                "match": {"value": asset_type}
            })

        payload = {
            "limit": 100,
            "with_payload": True,
            "with_vector": False
        }

        if filter_conditions:
            payload["filter"] = {"must": filter_conditions}

        response = httpx.post(
            f"{self.qdrant_url}/collections/{COLLECTION}/points/scroll",
            json=payload,
            timeout=30.0
        )

        points = response.json()["result"]["points"]

        # Score by keyword matching
        scored_points = []
        for point in points:
            text = point["payload"].get("text", "")
            metadata_text = json.dumps(point["payload"].get("metadata", {}), ensure_ascii=False)
            combined_text = text + " " + metadata_text

            # Count keyword matches
            score = sum(1 for kw in keywords if kw in combined_text)
            if score > 0:
                scored_points.append({
                    "id": point["id"],
                    "score": score / len(keywords),
                    "payload": point["payload"]
                })

        scored_points.sort(key=lambda x: x["score"], reverse=True)
        return scored_points[:limit]

    def hybrid_search(self, query: str, asset_type: str = None, limit: int = 10,
                     vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict]:
        """Hybrid search combining vector and keyword results."""
        print(f"\n🔍 Searching for: '{query}'")
        if asset_type:
            print(f"   Asset type: {asset_type}")
        print(f"   Weights: Vector={vector_weight}, Keyword={keyword_weight}")

        # Get results from both methods
        print("\n   [1/3] Running vector search...")
        vector_results = self.vector_search(query, asset_type, limit=20)
        print(f"   Found {len(vector_results)} vector results")

        print("   [2/3] Running keyword search...")
        keyword_results = self.keyword_search(query, asset_type, limit=20)
        print(f"   Found {len(keyword_results)} keyword results")

        print("   [3/3] Merging and reranking...")

        # Merge and rerank
        combined = {}

        for result in vector_results:
            point_id = result["id"]
            combined[point_id] = {
                "id": point_id,
                "payload": result["payload"],
                "vector_score": result["score"],
                "keyword_score": 0.0
            }

        for result in keyword_results:
            point_id = result["id"]
            if point_id in combined:
                combined[point_id]["keyword_score"] = result["score"]
            else:
                combined[point_id] = {
                    "id": point_id,
                    "payload": result["payload"],
                    "vector_score": 0.0,
                    "keyword_score": result["score"]
                }

        # Calculate hybrid score
        for point_id in combined:
            combined[point_id]["hybrid_score"] = (
                vector_weight * combined[point_id]["vector_score"] +
                keyword_weight * combined[point_id]["keyword_score"]
            )

        # Sort by hybrid score
        results = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)

        # Format output
        formatted = []
        for r in results[:limit]:
            formatted.append({
                "id": r["id"],
                "score": r["hybrid_score"],
                "vector_score": r["vector_score"],
                "keyword_score": r["keyword_score"],
                "payload": r["payload"]
            })

        return formatted


def main():
    """Test hybrid search with user query."""
    searcher = HybridSearcherBGE()

    print("\n" + "="*80)
    print("HYBRID SEARCH TEST - BGE-large-zh-v1.5 Model")
    print("="*80)

    # User query
    query = "冷静的，飞船"

    print(f"\n📝 Query: {query}")
    print("-"*80)

    # Search without asset type filter
    results = searcher.hybrid_search(query, asset_type=None, limit=10)

    print(f"\n✅ Found {len(results)} results")
    print("="*80)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Hybrid Score: {r['score']:.4f}")
        print(f"    Vector: {r['vector_score']:.4f} | Keyword: {r['keyword_score']:.4f}")
        print(f"    Asset Type: {r['payload'].get('asset_type', 'N/A')}")
        print(f"    Book: {r['payload'].get('book_id', 'N/A')}")
        print(f"    Chapter: {r['payload'].get('chapter', 'N/A')}")
        print(f"    Text: {r['payload'].get('text', '')[:150]}...")

        # Show metadata if available
        metadata = r['payload'].get('metadata', {})
        if metadata:
            print(f"    Metadata keys: {list(metadata.keys())}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
