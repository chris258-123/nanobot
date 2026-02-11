#!/usr/bin/env python3
"""
整理小说章节文件
- 删除重复章节（基于标题）
- 按章节号排序
- 番外章节放在最后
- 重命名为 0001_, 0002_ 格式
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# 支持环境变量或命令行参数
NOVEL_DIR = os.environ.get('NOVEL_DIR', os.path.expanduser("~/novel_data"))
if len(sys.argv) > 1:
    NOVEL_DIR = sys.argv[1]

def get_chapter_title(filepath):
    """获取章节标题（文件第一行）"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            title = first_line.lstrip('#').strip()
            return title
    except Exception as e:
        print(f"警告: 无法读取 {filepath.name}: {e}")
        return None

def chinese_to_arabic(chinese_num):
    """将中文数字转换为阿拉伯数字"""
    chinese_numerals = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
    }

    units = {
        '十': 10, '百': 100, '千': 1000, '万': 10000
    }

    result = 0
    temp_num = 0

    i = 0
    while i < len(chinese_num):
        char = chinese_num[i]

        if char in chinese_numerals:
            temp_num = chinese_numerals[char]
        elif char in units:
            unit = units[char]
            if temp_num == 0:
                temp_num = 1
            result += temp_num * unit
            temp_num = 0

        i += 1

    result += temp_num
    return result if result > 0 else None

def extract_chapter_number(title):
    """从标题中提取章节号"""
    # 先尝试提取阿拉伯数字 "第123章"
    match = re.search(r'第(\d+)章', title)
    if match:
        return int(match.group(1))

    # 尝试提取中文数字 "第一百二十三章"
    match = re.search(r'第([零一二三四五六七八九十百千万]+)章', title)
    if match:
        chinese_num = match.group(1)
        arabic_num = chinese_to_arabic(chinese_num)
        if arabic_num:
            return arabic_num

    # 番外章节
    if '番外' in title:
        match = re.search(r'第(\d+)章', title)
        if match:
            return 10000 + int(match.group(1))
        match = re.search(r'第([零一二三四五六七八九十百千万]+)章', title)
        if match:
            chinese_num = match.group(1)
            arabic_num = chinese_to_arabic(chinese_num)
            if arabic_num:
                return 10000 + arabic_num
        return 10000

    return 99999

def clean_title(title):
    """清理标题，用于比较"""
    title = re.sub(r'^\d+_', '', title)
    title = re.sub(r'笔趣阁\s*', '', title)
    title = re.sub(r'^\d+\.', '', title)
    return title.strip()

def main():
    print("=" * 50)
    print("  整理小说章节文件")
    print("=" * 50)

    novel_path = Path(NOVEL_DIR)
    if not novel_path.exists():
        print(f"错误: 目录不存在 {NOVEL_DIR}")
        return

    md_files = [f for f in novel_path.glob("*.md")
                if not f.name.startswith('temp_')]
    print(f"找到 {len(md_files)} 个文件")
    print()

    # 按章节标题分组（去重）
    title_groups = defaultdict(list)
    chapter_number_groups = defaultdict(list)

    for filepath in md_files:
        title = get_chapter_title(filepath)
        if title:
            clean_t = clean_title(title)
            title_groups[clean_t].append(filepath)

            # 同时按章节号分组
            chapter_num = extract_chapter_number(title)
            chapter_number_groups[chapter_num].append((filepath, title))

    # 删除重复文件（基于标题）
    duplicates_removed = 0
    unique_files = []

    for clean_t, files in title_groups.items():
        if len(files) > 1:
            files_sorted = sorted(files, key=lambda f: (
                '笔趣阁' in f.name,
                len(f.name),
            ))
            keep_file = files_sorted[0]
            unique_files.append(keep_file)

            for f in files_sorted[1:]:
                print(f"删除重复标题: {f.name}")
                try:
                    f.unlink()
                    duplicates_removed += 1
                except Exception as e:
                    print(f"  错误: {e}")
        else:
            unique_files.append(files[0])

    # 删除重复章节号的文件
    for chapter_num, files_with_titles in chapter_number_groups.items():
        if len(files_with_titles) > 1 and chapter_num != 99999:
            print(f"\n发现重复章节号 {chapter_num}:")
            for filepath, title in files_with_titles:
                print(f"  - {filepath.name}: {title}")

            # 保留文件大小最大的（内容最多的）
            files_sorted = sorted(files_with_titles, key=lambda x: x[0].stat().st_size, reverse=True)
            keep_file = files_sorted[0][0]
            print(f"  保留: {keep_file.name}")

            for filepath, title in files_sorted[1:]:
                if filepath in unique_files:
                    unique_files.remove(filepath)
                print(f"  删除: {filepath.name}")
                try:
                    filepath.unlink()
                    duplicates_removed += 1
                except Exception as e:
                    print(f"    错误: {e}")

    print(f"\n删除了 {duplicates_removed} 个重复文件")
    print(f"剩余 {len(unique_files)} 个唯一文件")

    # 重新获取文件信息并排序
    files_to_rename = []
    for filepath in unique_files:
        title = get_chapter_title(filepath)
        if title:
            is_extra = '番外' in title
            chapter_num = extract_chapter_number(title)
            files_to_rename.append((is_extra, chapter_num, title, filepath))

    files_to_rename.sort(key=lambda x: (x[0], x[1]))

    # 统计
    main_chapters = sum(1 for x in files_to_rename if not x[0])
    extra_chapters = sum(1 for x in files_to_rename if x[0])

    print(f"\n正文章节: {main_chapters}")
    print(f"番外章节: {extra_chapters}")
    print()

    # 重命名文件
    print("开始重命名...")
    renamed_count = 0

    for index, (is_extra, chapter_num, title, old_path) in enumerate(files_to_rename, 1):
        clean_name = clean_title(title)
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', clean_name)
        clean_name = clean_name[:100]

        new_name = f"{chapter_num:04d}_{clean_name}.md"
        new_path = old_path.parent / new_name

        if old_path != new_path:
            try:
                if new_path.exists():
                    new_path.unlink()

                old_path.rename(new_path)
                renamed_count += 1

                if renamed_count <= 10 or renamed_count == main_chapters or renamed_count == main_chapters + 1:
                    print(f"  {chapter_num:04d}: {clean_name[:50]}")
                elif renamed_count % 100 == 0:
                    print(f"  {chapter_num:04d}: {clean_name[:50]}")
            except Exception as e:
                print(f"  错误: 无法重命名 {old_path.name}: {e}")

    print(f"\n重命名了 {renamed_count} 个文件")
    print("\n" + "=" * 50)
    print("  整理完成")
    print("=" * 50)
    print(f"正文章节: 0001 - {main_chapters:04d}")
    print(f"番外章节: {main_chapters+1:04d} - {len(files_to_rename):04d}")
    print(f"保存位置: {NOVEL_DIR}")

if __name__ == "__main__":
    main()
