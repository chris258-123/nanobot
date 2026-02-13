#!/usr/bin/env python3
"""
Hybrid search combining vector similarity and keyword matching for better recall.
"""

import httpx
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

QDRANT_URL = "http://localhost:6333"
COLLECTION = "novel_assets"

class HybridSearcher:
    """Hybrid search with vector + keyword + reranking."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.qdrant_url = QDRANT_URL

    def expand_query(self, query: str, asset_type: str = None) -> List[str]:
        """Expand query with synonyms and related terms."""
        expansions = {
            "冷静": ["理性", "沉着", "镇定", "冷静"],
            "理性": ["冷静", "逻辑", "分析"],
            "冲突": ["矛盾", "对抗", "争执", "纠纷"],
            "权力": ["权威", "统治", "控制"],
            "责任": ["义务", "职责", "担当"],
            "秩序": ["规则", "体系", "制度"],
            "混沌": ["无序", "混乱", "失控"],
        }

        # Extract key terms and expand
        expanded = [query]
        for key, synonyms in expansions.items():
            if key in query:
                for syn in synonyms:
                    expanded.append(query.replace(key, syn))

        return list(set(expanded))[:3]  # Top 3 variations

    def vector_search(self, query: str, asset_type: str = None, limit: int = 20) -> List[Dict]:
        """Pure vector search."""
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
            "score_threshold": 0.2  # Lower threshold for better recall
        }

        if filter_conditions:
            payload["filter"] = {"must": filter_conditions}

        response = httpx.post(
            f"{self.qdrant_url}/collections/{COLLECTION}/points/search",
            json=payload
        )

        return response.json()["result"]

    def keyword_search(self, query: str, asset_type: str = None, limit: int = 20) -> List[Dict]:
        """Keyword-based search using scroll with text matching."""
        # Extract keywords from query
        keywords = query.split()

        # Use scroll API to get candidates
        filter_conditions = []
        if asset_type:
            filter_conditions.append({
                "key": "asset_type",
                "match": {"value": asset_type}
            })

        payload = {
            "limit": 100,  # Get more candidates
            "with_payload": True,
            "with_vector": False
        }

        if filter_conditions:
            payload["filter"] = {"must": filter_conditions}

        response = httpx.post(
            f"{self.qdrant_url}/collections/{COLLECTION}/points/scroll",
            json=payload
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
                    "score": score / len(keywords),  # Normalize
                    "payload": point["payload"]
                })

        # Sort by score
        scored_points.sort(key=lambda x: x["score"], reverse=True)
        return scored_points[:limit]

    def hybrid_search(self, query: str, asset_type: str = None, limit: int = 10,
                     vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict]:
        """Hybrid search combining vector and keyword results."""

        # Get results from both methods
        vector_results = self.vector_search(query, asset_type, limit=20)
        keyword_results = self.keyword_search(query, asset_type, limit=20)

        # Merge and rerank
        combined = {}

        # Add vector results
        for result in vector_results:
            point_id = result["id"]
            combined[point_id] = {
                "id": point_id,
                "payload": result["payload"],
                "vector_score": result["score"],
                "keyword_score": 0.0
            }

        # Add keyword results
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

    def multi_query_search(self, query: str, asset_type: str = None, limit: int = 10) -> List[Dict]:
        """Search with query expansion for better recall."""
        expanded_queries = self.expand_query(query, asset_type)

        all_results = {}
        for q in expanded_queries:
            results = self.hybrid_search(q, asset_type, limit=20)
            for r in results:
                point_id = r["id"]
                if point_id not in all_results:
                    all_results[point_id] = r
                else:
                    # Boost score if found in multiple queries
                    all_results[point_id]["score"] = max(
                        all_results[point_id]["score"],
                        r["score"]
                    ) * 1.1  # 10% boost

        # Sort and return top results
        results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return results[:limit]


def main():
    """Test hybrid search."""
    searcher = HybridSearcher()

    print("\n" + "="*80)
    print("HYBRID SEARCH DEMONSTRATION")
    print("="*80)

    # Test 1: Character search
    print("\n[Test 1] Character search: 冷静理性的角色")
    results = searcher.hybrid_search("冷静理性的角色", asset_type="character_card", limit=5)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Hybrid: {r['score']:.4f} (V:{r['vector_score']:.4f}, K:{r['keyword_score']:.4f})")
        print(f"    {r['payload']['text'][:100]}...")

    # Test 2: Conflict search with query expansion
    print("\n[Test 2] Conflict search with expansion: 权力与责任的矛盾")
    results = searcher.multi_query_search("权力与责任的矛盾", asset_type="conflict", limit=5)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']:.4f}")
        print(f"    {r['payload']['text'][:100]}...")

    # Test 3: Theme search
    print("\n[Test 3] Theme search: 秩序与混沌")
    results = searcher.hybrid_search("秩序与混沌", asset_type="theme", limit=5)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Hybrid: {r['score']:.4f} (V:{r['vector_score']:.4f}, K:{r['keyword_score']:.4f})")
        print(f"    {r['payload']['text'][:100]}...")


if __name__ == "__main__":
    main()
