#!/usr/bin/env python3
"""
测试 Phase 1：处理前 100 章到三层记忆系统

使用真实的小说数据测试：
- Qdrant (召回记忆) - 已完成嵌入
- Canon DB (权威记忆) - 本脚本处理
- Neo4j (结构记忆) - 本脚本处理
"""

import sys
import json
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from chapter_processor import ChapterProcessor


def load_asset_file(asset_path: Path) -> dict:
    """Load asset file and extract delta information."""
    with open(asset_path, 'r', encoding='utf-8') as f:
        assets = json.load(f)

    # Convert assets to delta format for chapter_processor
    delta = {
        "entities_new": [],
        "events": [],
        "relations_delta": [],
        "fact_changes": [],
        "hooks": []
    }

    # Extract characters from character_cards
    for idx, char in enumerate(assets.get("character_cards", [])):
        entity = {
            "name": char.get("name", f"unknown_{idx}"),
            "type": "character",
            "aliases": [],  # Could extract from metadata
            "traits": {k: v for k, v in char.items() if k not in ["name", "relationships"]},
            "status": char.get("state", "active")
        }
        delta["entities_new"].append(entity)

        # Extract relationships
        relationships = char.get("relationships", {})
        for target, rel_type in relationships.items():
            delta["relations_delta"].append({
                "from_id": char.get("name"),
                "to_id": target,
                "kind": rel_type,
                "op": "INSERT"
            })

    # Extract events from plot_beats
    for idx, beat in enumerate(assets.get("plot_beats", [])):
        event = {
            "event_id": f"{assets.get('chapter', 'unknown')}_evt_{idx}",  # Include chapter in event_id
            "type": "plot_beat",
            "summary": beat.get("event", ""),
            "participants": beat.get("characters", [])
        }
        delta["events"].append(event)

    # Extract hooks from plot_beats (if they mention foreshadowing)
    for idx, beat in enumerate(assets.get("plot_beats", [])):
        impact = beat.get("impact", "").lower()
        if any(keyword in impact for keyword in ["伏笔", "暗示", "铺垫", "预示"]):
            delta["hooks"].append({
                "name": f"Hook from {assets.get('chapter', 'unknown')}",
                "summary": beat.get("event", ""),
                "priority": 1
            })

    return delta


def main():
    print("=" * 70)
    print("Phase 1 测试：处理前 100 章到三层记忆系统")
    print("=" * 70)

    # Initialize processor
    processor = ChapterProcessor(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pass="novel123",
        canon_db_path="~/.nanobot/workspace/canon_novel04_test.db",
        qdrant_url="http://localhost:6333"
    )

    # Get chapter files and asset files
    chapter_dir = Path("/home/chris/Desktop/my_workspace/novel_data/04")
    asset_dir = Path("/home/chris/novel_assets_test100")

    chapter_files = sorted(chapter_dir.glob("0*.md"))[:100]
    asset_files = sorted(asset_dir.glob("novel_04_0*.json"))

    print(f"\n找到 {len(chapter_files)} 个章节文件")
    print(f"找到 {len(asset_files)} 个资产文件")

    if len(chapter_files) != len(asset_files):
        print(f"⚠️  警告：章节文件和资产文件数量不匹配")

    # Process all 100 chapters
    test_count = 100
    print(f"\n开始处理全部 {test_count} 章...")
    print()

    results = []
    for i in range(min(test_count, len(chapter_files))):
        chapter_file = chapter_files[i]
        asset_file = asset_files[i]

        # Extract chapter info from filename
        # Format: 0001_第1章 xxx.md
        filename = chapter_file.stem
        chapter_no = filename.split("_")[0]  # "0001"
        title = "_".join(filename.split("_")[1:])  # "第1章 xxx"

        print(f"[{i+1}/{test_count}] 处理 {chapter_no}: {title[:40]}...")

        # Read chapter text
        with open(chapter_file, 'r', encoding='utf-8') as f:
            chapter_text = f.read()

        # Load assets and convert to delta
        delta = load_asset_file(asset_file)

        # Process chapter
        try:
            result = processor.process_chapter(
                book_id="novel_04",
                chapter_no=chapter_no,
                chapter_text=chapter_text,
                title=title,
                pov=None,  # Could extract from assets
                delta=delta
            )

            results.append(result)

            if result["status"] == "success":
                print(f"  ✓ 成功 (commit: {result['commit_id'][:8]}...)")
                if result.get("warnings"):
                    print(f"  ⚠️  {len(result['warnings'])} 个警告")
            elif result["status"] == "blocked":
                print(f"  ❌ 阻塞: {len(result['conflicts']['blocking'])} 个冲突")
            else:
                print(f"  ❌ 失败: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"  ❌ 异常: {e}")
            results.append({"status": "error", "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)

    successful = sum(1 for r in results if r.get("status") == "success")
    blocked = sum(1 for r in results if r.get("status") == "blocked")
    failed = sum(1 for r in results if r.get("status") in ["failed", "error"])

    print(f"成功: {successful}")
    print(f"阻塞: {blocked}")
    print(f"失败: {failed}")

    # Verify data
    print("\n" + "=" * 70)
    print("验证数据")
    print("=" * 70)

    # Canon DB
    entities = processor.canon_db.get_all_entities("character")
    print(f"\nCanon DB - 角色数量: {len(entities)}")
    if entities:
        print(f"  示例: {entities[0]['canonical_name']}")

    # Neo4j
    print(f"\nNeo4j - 查询测试...")
    # Could add more verification here

    processor.close()
    print("\n测试完成！")


if __name__ == "__main__":
    main()
