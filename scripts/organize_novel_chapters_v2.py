#!/usr/bin/env python3
"""
整理和排序小说章节文件 - 改进版
- 基于章节标题删除重复文件
- 按章节号排序
- 重命名为 0001_, 0002_ 格式
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
            # 移除 markdown 标题符号
            title = first_line.lstrip('#').strip()
            return title
    except Exception as e:
        print(f"警告: 无法读取 {filepath.name}: {e}")
        return None

def extract_chapter_number(filename):
    """从文件名中提取章节号"""
    name = filename.replace('.md', '')

    # 尝试多种模式提取章节号
    patterns = [
        r'第(\d+)章',  # 第123章
        r'(\d+)\.第\d+章',  # 1610.第1606章
        r'^(\d+)_',  # 0001_
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))

    # 番外章节
    if '番外' in name:
        # 尝试提取番外中的章节号
        match = re.search(r'第(\d+)章', name)
        if match:
            return 10000 + int(match.group(1))
        return 10000

    return 99999  # 无法识别的放到最后

def clean_title(title):
    """清理标题，用于比较"""
    # 移除常见的前缀
    title = re.sub(r'^\d+_', '', title)
    title = re.sub(r'笔趣阁\s*', '', title)
    title = re.sub(r'^\d+\.', '', title)
    return title.strip()

def main():
    print("=" * 50)
    print("  整理小说章节文件")
    print("=" * 50)

    # 获取所有MD文件
    md_files = [f for f in Path(NOVEL_DIR).glob("*.md")
                if not f.name.startswith('temp_')]
    print(f"找到 {len(md_files)} 个文件")

    # 按章节标题分组
    title_groups = defaultdict(list)
    files_with_info = []

    for filepath in md_files:
        title = get_chapter_title(filepath)
        if title:
            clean_t = clean_title(title)
            title_groups[clean_t].append(filepath)
            chapter_num = extract_chapter_number(filepath.name)
            files_with_info.append((chapter_num, title, filepath))

    # 删除重复文件
    duplicates_removed = 0
    unique_files = []

    for clean_t, files in title_groups.items():
        if len(files) > 1:
            # 优先保留文件名最短、不含"笔趣阁"的
            files_sorted = sorted(files, key=lambda f: (
                '笔趣阁' in f.name,
                len(f.name),
            ))
            keep_file = files_sorted[0]
            unique_files.append(keep_file)

            # 删除其他重复文件
            for f in files_sorted[1:]:
                print(f"删除重复: {f.name}")
                try:
                    f.unlink()
                    duplicates_removed += 1
                except Exception as e:
                    print(f"  错误: {e}")
        else:
            unique_files.append(files[0])

    print(f"\n删除了 {duplicates_removed} 个重复文件")
    print(f"剩余 {len(unique_files)} 个唯一文件")

    # 重新获取文件信息并排序
    files_to_rename = []
    for filepath in unique_files:
        title = get_chapter_title(filepath)
        chapter_num = extract_chapter_number(filepath.name)
        if title:
            files_to_rename.append((chapter_num, title, filepath))

    files_to_rename.sort(key=lambda x: x[0])

    # 重命名文件
    print("\n开始重命名...")
    renamed_count = 0

    for index, (chapter_num, title, old_path) in enumerate(files_to_rename, 1):
        # 清理标题作为文件名
        clean_name = clean_title(title)
        # 移除文件名中的非法字符
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', clean_name)
        clean_name = clean_name[:100]  # 限制长度

        new_name = f"{index:04d}_{clean_name}.md"
        new_path = old_path.parent / new_name

        if old_path != new_path:
            try:
                # 如果目标文件已存在，添加后缀
                if new_path.exists():
                    new_path = old_path.parent / f"{index:04d}_{clean_name}_dup.md"

                old_path.rename(new_path)
                renamed_count += 1

                if renamed_count <= 10 or renamed_count % 100 == 0:
                    print(f"  {index:04d}: {clean_name[:50]}")
            except Exception as e:
                print(f"  错误: 无法重命名 {old_path.name}: {e}")

    print(f"\n重命名了 {renamed_count} 个文件")
    print("\n" + "=" * 50)
    print("  整理完成")
    print("=" * 50)
    print(f"最终文件数: {len(files_to_rename)}")
    print(f"保存位置: {NOVEL_DIR}")

if __name__ == "__main__":
    main()
