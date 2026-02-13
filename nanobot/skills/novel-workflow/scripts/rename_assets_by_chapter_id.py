#!/usr/bin/env python3
"""
重命名资产文件以对齐整理后的章节 ID

将 novel_04_第1章 xxx_assets.json
重命名为 novel_04_0001_第1章 xxx_assets.json
"""

import os
import re
import sys
from pathlib import Path


def extract_chapter_number(title):
    """从标题中提取章节号"""
    # 提取阿拉伯数字 "第123章"
    match = re.search(r'第(\d+)章', title)
    if match:
        return int(match.group(1))
    return None


def main():
    if len(sys.argv) < 3:
        print("用法: python rename_assets_by_chapter_id.py <章节目录> <资产目录>")
        print("示例: python rename_assets_by_chapter_id.py ~/novel_data/04 ~/novel_assets_enhanced")
        sys.exit(1)

    chapter_dir = Path(sys.argv[1]).expanduser()
    asset_dir = Path(sys.argv[2]).expanduser()

    if not chapter_dir.exists():
        print(f"错误: 章节目录不存在 {chapter_dir}")
        sys.exit(1)

    if not asset_dir.exists():
        print(f"错误: 资产目录不存在 {asset_dir}")
        sys.exit(1)

    print("=" * 60)
    print("重命名资产文件以对齐章节 ID")
    print("=" * 60)
    print(f"章节目录: {chapter_dir}")
    print(f"资产目录: {asset_dir}")
    print()

    # 构建章节标题到 ID 的映射
    # 格式: "第1章 xxx" -> "0001"
    title_to_id = {}

    for chapter_file in sorted(chapter_dir.glob("*.md")):
        # 文件名格式: 0001_第1章 xxx.md
        filename = chapter_file.stem  # 去掉 .md

        # 提取前缀 ID (0001)
        match = re.match(r'^(\d{4})_(.+)$', filename)
        if match:
            chapter_id = match.group(1)
            title_part = match.group(2)  # "第1章 xxx"

            # 提取章节号用于匹配
            chapter_num = extract_chapter_number(title_part)
            if chapter_num:
                # 使用章节号作为key，因为资产文件名中只有章节号
                title_to_id[chapter_num] = (chapter_id, title_part)

    print(f"找到 {len(title_to_id)} 个章节映射")
    print()

    # 重命名资产文件
    renamed_count = 0
    skipped_count = 0
    error_count = 0

    for asset_file in sorted(asset_dir.glob("novel_04_第*章*_assets.json")):
        old_name = asset_file.name

        # 提取章节号
        # 格式: novel_04_第123章 xxx_assets.json
        match = re.search(r'novel_04_第(\d+)章', old_name)
        if not match:
            print(f"⚠️  无法解析: {old_name}")
            skipped_count += 1
            continue

        chapter_num = int(match.group(1))

        # 查找对应的章节 ID
        if chapter_num not in title_to_id:
            print(f"⚠️  找不到映射: 第{chapter_num}章")
            skipped_count += 1
            continue

        chapter_id, title_part = title_to_id[chapter_num]

        # 构建新文件名
        # novel_04_第123章 xxx_assets.json -> novel_04_0123_第123章 xxx_assets.json
        new_name = old_name.replace(
            f"novel_04_第{chapter_num}章",
            f"novel_04_{chapter_id}_第{chapter_num}章"
        )

        new_path = asset_dir / new_name

        # 检查新文件是否已存在
        if new_path.exists():
            print(f"⚠️  目标文件已存在: {new_name}")
            skipped_count += 1
            continue

        # 重命名
        try:
            asset_file.rename(new_path)
            renamed_count += 1

            # 只显示前10个和每100个
            if renamed_count <= 10 or renamed_count % 100 == 0:
                print(f"✓ {chapter_id}: {old_name[:60]}...")
        except Exception as e:
            print(f"❌ 重命名失败 {old_name}: {e}")
            error_count += 1

    print()
    print("=" * 60)
    print("重命名完成")
    print("=" * 60)
    print(f"成功重命名: {renamed_count} 个文件")
    print(f"跳过: {skipped_count} 个文件")
    print(f"错误: {error_count} 个文件")
    print()


if __name__ == "__main__":
    main()
