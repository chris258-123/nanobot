#!/usr/bin/env python3
"""
按章节名称排序，番外放在最后
"""

import os
import re
from pathlib import Path

NOVEL_DIR = "/home/chris/Desktop/my_workspace/novel_data"

def extract_chapter_info(filename):
    """提取章节信息，区分正文和番外"""
    name = filename.replace('.md', '')

    # 移除已有的数字前缀
    name = re.sub(r'^\d+_', '', name)

    # 判断是否为番外
    is_extra = '番外' in name

    # 提取章节号
    chapter_num = 99999

    if not is_extra:
        # 正文章节：提取"第X章"
        match = re.search(r'第(\d+)章', name)
        if match:
            chapter_num = int(match.group(1))
    else:
        # 番外章节：提取番外中的章节号（如果有）
        match = re.search(r'第(\d+)章', name)
        if match:
            chapter_num = int(match.group(1))
        else:
            # 没有章节号的番外，按文件名排序
            chapter_num = 0

    return (is_extra, chapter_num, name)

def main():
    print("=" * 50)
    print("  重新排序：番外放在最后")
    print("=" * 50)

    # 获取所有MD文件
    md_files = [f for f in Path(NOVEL_DIR).glob("*.md")
                if not f.name.startswith('temp_')]

    print(f"找到 {len(md_files)} 个文件")

    # 提取章节信息并排序
    files_with_info = []
    for filepath in md_files:
        is_extra, chapter_num, clean_name = extract_chapter_info(filepath.name)
        files_with_info.append((is_extra, chapter_num, clean_name, filepath))

    # 排序：先按是否番外（False在前），再按章节号
    files_with_info.sort(key=lambda x: (x[0], x[1]))

    # 统计
    main_chapters = sum(1 for x in files_with_info if not x[0])
    extra_chapters = sum(1 for x in files_with_info if x[0])

    print(f"正文章节: {main_chapters}")
    print(f"番外章节: {extra_chapters}")
    print()

    # 重命名文件
    print("开始重命名...")
    renamed_count = 0

    for index, (is_extra, chapter_num, clean_name, old_path) in enumerate(files_with_info, 1):
        new_name = f"{index:04d}_{clean_name}.md"
        new_path = old_path.parent / new_name

        if old_path != new_path:
            try:
                # 如果目标文件已存在，先删除
                if new_path.exists():
                    new_path.unlink()

                old_path.rename(new_path)
                renamed_count += 1

                # 显示进度
                if renamed_count <= 10:
                    print(f"  {index:04d}: {clean_name[:60]}")
                elif renamed_count == main_chapters:
                    print(f"  {index:04d}: {clean_name[:60]} (最后一个正文章节)")
                elif renamed_count == main_chapters + 1:
                    print(f"  {index:04d}: {clean_name[:60]} (第一个番外章节)")
                elif renamed_count % 100 == 0:
                    print(f"  {index:04d}: {clean_name[:60]}")
            except Exception as e:
                print(f"  错误: 无法重命名 {old_path.name}: {e}")

    print(f"\n重命名了 {renamed_count} 个文件")
    print("\n" + "=" * 50)
    print("  排序完成")
    print("=" * 50)
    print(f"正文章节: 0001 - {main_chapters:04d}")
    print(f"番外章节: {main_chapters+1:04d} - {len(files_with_info):04d}")
    print(f"保存位置: {NOVEL_DIR}")

if __name__ == "__main__":
    main()
