#!/usr/bin/env python3
"""
Enhanced embedder with Chinese-optimized models for better recall.
"""

from sentence_transformers import SentenceTransformer
import httpx
import json
from pathlib import Path
import argparse

# Chinese-optimized models (choose one):
# 1. paraphrase-multilingual-MiniLM-L12-v2 (384 dim, multilingual)
# 2. distiluse-base-multilingual-cased-v2 (512 dim, better quality)
# 3. m3e-base (768 dim, Chinese-specific, best for Chinese)

MODEL_OPTIONS = {
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 384 dim
    "multilingual-large": "distiluse-base-multilingual-cased-v2",  # 512 dim
    "chinese": "moka-ai/m3e-base",  # 768 dim, Chinese-specific
}

def embed_and_upsert(assets_json_path: str, qdrant_url: str, collection_name: str,
                     model_name: str = "multilingual"):
    """Embed assets and upsert to Qdrant with Chinese-optimized model."""

    # Load model
    model_key = MODEL_OPTIONS.get(model_name, MODEL_OPTIONS["multilingual"])
    print(f"Loading model: {model_key}")
    model = SentenceTransformer(model_key)

    # Get vector dimension
    vector_dim = model.get_sentence_embedding_dimension()
    print(f"Vector dimension: {vector_dim}")

    with open(assets_json_path) as f:
        data = json.load(f)

    points = []

    # Process plot beats with enhanced text representation
    for beat in data.get("plot_beats", []):
        # Combine multiple fields for richer semantic representation
        text_parts = [
            beat.get('event', ''),
            beat.get('impact', ''),
            beat.get('causality', ''),
            f"角色: {', '.join(beat.get('characters', []))}"
        ]
        text = ' '.join(filter(None, text_parts))

        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_beat_{beat.get('event', '')[:30]}")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "plot_beat",
                "chapter": data["chapter"],
                "characters": beat.get("characters", []),
                "text": text,
                "metadata": beat
            }
        })

    # Process character cards with enhanced text
    for card in data.get("character_cards", []):
        text_parts = [
            f"角色: {card.get('name', '')}",
            f"特质: {' '.join(card.get('traits', []))}",
            f"状态: {card.get('state', '')}",
            f"目标: {' '.join(card.get('goals', []))}",
            f"缺陷: {' '.join(card.get('flaws', []))}",
            f"成长: {card.get('growth', '')}"
        ]
        text = ' '.join(filter(None, text_parts))

        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_char_{card.get('name', '')}")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "character_card",
                "chapter": data["chapter"],
                "characters": [card.get("name", "")],
                "text": text,
                "metadata": card
            }
        })

    # Process conflicts with enhanced text
    for conflict in data.get("conflicts", []):
        text_parts = [
            f"冲突类型: {conflict.get('type', '')}",
            f"描述: {conflict.get('description', '')}",
            f"涉及方: {', '.join(conflict.get('parties', []))}",
            f"强度: {conflict.get('intensity', '')}",
            f"利害关系: {conflict.get('stakes', '')}"
        ]
        text = ' '.join(filter(None, text_parts))

        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_conflict_{conflict.get('type', '')}_{len(points)}")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "conflict",
                "chapter": data["chapter"],
                "characters": conflict.get("parties", []),
                "text": text,
                "metadata": conflict
            }
        })

    # Process settings with enhanced text
    for setting in data.get("settings", []):
        text_parts = [
            f"地点: {setting.get('location', '')}",
            f"时期: {setting.get('time_period', '')}",
            f"氛围: {setting.get('atmosphere', '')}",
            f"世界规则: {' '.join(setting.get('world_rules', [])[:3])}",  # Top 3 rules
            f"社会背景: {setting.get('social_context', '')[:200]}"  # First 200 chars
        ]
        text = ' '.join(filter(None, text_parts))

        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_setting_{len(points)}")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "setting",
                "chapter": data["chapter"],
                "characters": [],
                "text": text,
                "metadata": setting
            }
        })

    # Process themes with enhanced text
    for theme in data.get("themes", []):
        text_parts = [
            f"主题: {theme.get('theme', '')}",
            f"表现: {theme.get('manifestation', '')}",
            f"象征: {' '.join(theme.get('symbols', []))}",
            f"提出的问题: {' '.join(theme.get('questions_raised', [])[:2])}"  # Top 2 questions
        ]
        text = ' '.join(filter(None, text_parts))

        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_theme_{theme.get('theme', '')}")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "theme",
                "chapter": data["chapter"],
                "characters": [],
                "text": text,
                "metadata": theme
            }
        })

    # Process point of view
    pov = data.get("point_of_view", {})
    if pov:
        text_parts = [
            f"视角: {pov.get('person', '')} {pov.get('knowledge', '')}",
            f"聚焦: {pov.get('focalization', '')}",
            f"距离: {pov.get('distance', '')}",
            f"可靠性: {pov.get('reliability', '')}",
            f"转换: {' '.join(pov.get('shifts', []))[:200]}"
        ]
        text = ' '.join(filter(None, text_parts))

        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_pov")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "point_of_view",
                "chapter": data["chapter"],
                "characters": [pov.get('focalization', '')] if pov.get('focalization') else [],
                "text": text,
                "metadata": pov
            }
        })

    # Process tone
    tone = data.get("tone", {})
    if tone:
        text_parts = [
            f"整体语气: {tone.get('overall_tone', '')}",
            f"情感弧线: {tone.get('emotional_arc', '')}",
            f"氛围关键词: {' '.join(tone.get('mood_keywords', []))}",
            f"紧张度: {tone.get('tension_level', '')}",
            f"节奏: {tone.get('pacing', '')}"
        ]
        text = ' '.join(filter(None, text_parts))

        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_tone")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "tone",
                "chapter": data["chapter"],
                "characters": [],
                "text": text,
                "metadata": tone
            }
        })

    # Process style
    style = data.get("style", {})
    if style:
        text_parts = [
            f"句式结构: {style.get('sentence_structure', '')}",
            f"词汇水平: {style.get('vocabulary_level', '')}",
            f"修辞手法: {' '.join(style.get('rhetoric_devices', []))}",
            f"对话比例: {style.get('dialogue_ratio', '')}",
            f"描写风格: {style.get('description_style', '')}",
            f"特色: {' '.join(style.get('distinctive_features', []))}"
        ]
        text = ' '.join(filter(None, text_parts))

        embedding = model.encode(text).tolist()
        points.append({
            "id": abs(hash(f"{data['book_id']}_{data['chapter']}_style")),
            "vector": embedding,
            "payload": {
                "book_id": data["book_id"],
                "asset_type": "style",
                "chapter": data["chapter"],
                "characters": [],
                "text": text,
                "metadata": style
            }
        })

    # Batch upsert
    if points:
        response = httpx.put(
            f"{qdrant_url}/collections/{collection_name}/points",
            json={"points": points},
            timeout=30.0
        )
        response.raise_for_status()
        print(f"Upserted {len(points)} points to Qdrant")
    else:
        print("No points to upsert")

    return len(points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, help="Path to assets JSON")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="novel_assets_chinese")
    parser.add_argument("--model", default="multilingual",
                       choices=["multilingual", "multilingual-large", "chinese"],
                       help="Embedding model to use")
    args = parser.parse_args()

    count = embed_and_upsert(args.assets, args.qdrant_url, args.collection, args.model)
    print(f"Total: {count} points upserted")
