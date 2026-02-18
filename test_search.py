#!/usr/bin/env python3
"""Test Qdrant search functionality."""

import httpx
from sentence_transformers import SentenceTransformer

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test query
query = "伊登和荆璜的对话"
query_vector = model.encode(query).tolist()

# Search
response = httpx.post(
    "http://localhost:6333/collections/novel_assets/points/search",
    json={
        "vector": query_vector,
        "limit": 5,
        "with_payload": True
    },
    timeout=30.0
)

results = response.json()["result"]

print(f"搜索: '{query}'")
print(f"找到 {len(results)} 个结果:\n")

for i, hit in enumerate(results, 1):
    payload = hit["payload"]
    print(f"{i}. [{payload['asset_type']}] 章节: {payload['chapter']}")
    print(f"   文本: {payload['text'][:80]}...")
    print(f"   相关度: {hit['score']:.3f}\n")
