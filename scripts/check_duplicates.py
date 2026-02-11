#!/usr/bin/env python3
"""
检查并删除重复章节
"""

import os
import re
from pathlib import Path
from collections import defaultdict

NOVEL_DIR = "/home/chris/Desktop/my_workspace/novel_data"

def get_chapter_title(filepath):
    """获取章节标题（文件第一行）"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            title = first_line.lstrip('#').strip()
            return title
    except Exception as e:
        return None

def extract_chapter_number(title):
    """从标题中提取章节号"""
    # 尝试提取"第X章"
    match = re.search(r'第(\d+)章', title)
    if match:
        return int(match.group(1))
    return None

def main():
    print("=" * 50)
    print("  检查重复章节")
    print("=" * 50)

    # 获取所有MD文件
    md_files = [f for f in Path(NOVEL_DIR).glob("*.md")
                if not f.name.startswith('temp_')]

    print(f"总文件数: {len(md_files)}")
    print()

    # 按章节号分组
    chapter_groups = defaultdict(list)
    no_chapter_num = []

    for filepath in md_files:
        title = get_chapter_title(filepath)
        if title:
            chapter_num = extract_chapter_number(title)
            if chapter_num:
                chapter_groups[chapter_num].append((filepath, title))
            else:
                no_chapter_num.append((filepath, title))

    # 找出重复的章节
    duplicates = []
    for chapter_num, files in sorted(chapter_groups.items()):
        if len(files) > 1:
            duplicates.append((chapter_num, files))

    if duplicates:
        print(f"发现 {len(duplicates)} 个重复的章节号：")
        print()

        for chapter_num, files in duplicates[:20]:  # 只显示前20个
            print(f"第 {chapter_num} 章 - {len(files)} 个文件:")
            for filepath, title in files:
                print(f"  - {filepath.name}")
                print(f"    标题: {title}")
            print()

        # 询问是否删除
        print(f"\n总共发现 {len(duplicates)} 组重复章节")
        print("准备删除重复文件（保留文件名最短的）...")

        deleted_count = 0
        for chapter_num, files in duplicates:
            # 按文件名长度排序，保留最短的
            files_sorted = sorted(files, key=lambda x: len(x[0].name))
            keep_file = files_sorted[0]

            for filepath, title in files_sorted[1:]:
                print(f"删除: {filepath.name}")
                try:
                    filepath.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  错误: {e}")

        print(f"\n删除了 {deleted_count} 个重复文件")
    else:
        print("没有发现重复的章节号")

    # 检查没有章节号的文件
    if no_chapter_num:
        print(f"\n没有章节号的文件 ({len(no_chapter_num)} 个):")
        for filepath, title in no_chapter_num[:10]:
            print(f"  - {filepath.name}: {title}")

    print("\n" + "=" * 50)
    print("  检查完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
