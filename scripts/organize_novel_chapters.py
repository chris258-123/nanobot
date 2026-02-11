#!/usr/bin/env python3
"""
整理和排序小说章节文件
- 删除重复文件
- 按章节号排序
- 重命名为 0001_, 0002_ 格式
"""

import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict

NOVEL_DIR = "/home/chris/Desktop/my_workspace/novel_data"

def get_file_hash(filepath):
    """获取文件内容的MD5哈希值"""
    with open(filepath, 'rb') as f:
        # 只读取前1000字节来判断是否重复（标题+部分内��）
        content = f.read(1000)
        return hashlib.md5(content).hexdigest()

def extract_chapter_number(filename):
    """从文件名中提取章节号"""
    # 移除前缀和后缀
    name = filename.replace('.md', '')

    # 尝试多种模式提取章节号
    patterns = [
        r'第(\d+)章',  # 第123章
        r'(\d+)\.第\d+章',  # 1610.第1606章
        r'^(\d+)_',  # 0001_
        r'番外.*第(\d+)章',  # 番外：第5章
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))

    # 如果是番外章节但没有数字，返回一个很大的数
    if '番外' in name:
        return 10000

    return 99999  # 无法识别的放到最后

def clean_filename(filename):
    """清理文件名，移除不需要的前缀"""
    name = filename.replace('.md', '')

    # 移除 "笔趣阁" 前缀
    name = re.sub(r'笔趣阁\s*', '', name)

    # 移除已有的数字前缀 (0001_, 0002_, etc.)
    name = re.sub(r'^\d+_', '', name)

    return name + '.md'

def main():
    print("=" * 50)
    print("  整理小说章节文件")
    print("=" * 50)

    # 获取所有MD文件
    md_files = list(Path(NOVEL_DIR).glob("*.md"))
    print(f"找到 {len(md_files)} 个文件")

    # 按内容哈希分组，找出重复文件
    hash_groups = defaultdict(list)
    for filepath in md_files:
        if filepath.name.startswith('temp_'):
            continue
        try:
            file_hash = get_file_hash(filepath)
            hash_groups[file_hash].append(filepath)
        except Exception as e:
            print(f"警告: 无法读取 {filepath.name}: {e}")

    # 删除重复文件，保留最干净的版本
    unique_files = []
    duplicates_removed = 0

    for file_hash, files in hash_groups.items():
        if len(files) > 1:
            # 优先保留不含"笔趣阁"的文件
            files_sorted = sorted(files, key=lambda f: (
                '笔趣阁' in f.name,  # 不含笔趣阁的优先
                len(f.name),  # 文件名短的优先
            ))
            keep_file = files_sorted[0]
            unique_files.append(keep_file)

            # 删除其他重复文件
            for f in files_sorted[1:]:
                print(f"删除重复: {f.name}")
                f.unlink()
                duplicates_removed += 1
        else:
            unique_files.append(files[0])

    print(f"\n删除了 {duplicates_removed} 个重复文件")
    print(f"剩余 {len(unique_files)} 个唯一文件")

    # 按章节号排序
    files_with_numbers = []
    for filepath in unique_files:
        chapter_num = extract_chapter_number(filepath.name)
        clean_name = clean_filename(filepath.name)
        files_with_numbers.append((chapter_num, clean_name, filepath))

    files_with_numbers.sort(key=lambda x: x[0])

    # 重命名文件
    print("\n开始重命名...")
    renamed_count = 0

    for index, (chapter_num, clean_name, old_path) in enumerate(files_with_numbers, 1):
        new_name = f"{index:04d}_{clean_name}"
        new_path = old_path.parent / new_name

        if old_path != new_path:
            # 如果目标文件已存在，先删除
            if new_path.exists():
                new_path.unlink()

            old_path.rename(new_path)
            renamed_count += 1

            if renamed_count <= 10 or renamed_count % 100 == 0:
                print(f"  {index:04d}: {clean_name}")

    print(f"\n重命名了 {renamed_count} 个文件")
    print("\n" + "=" * 50)
    print("  整理完成")
    print("=" * 50)
    print(f"最终文件数: {len(files_with_numbers)}")
    print(f"保存位置: {NOVEL_DIR}")

if __name__ == "__main__":
    main()
