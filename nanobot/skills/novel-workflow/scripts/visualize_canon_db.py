#!/usr/bin/env python3
"""
Canon DB 可视化工具
可视化实体注册表、角色状态、事实历史、关系变化等
"""

import sys
from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# 配置中文字体 - 自动检测可用的 CJK 字体
import matplotlib.font_manager as fm
cjk_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name]
if cjk_fonts:
    matplotlib.rcParams['font.sans-serif'] = [cjk_fonts[0], 'DejaVu Sans']
else:
    # 备用方案：尝试常见中文字体
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import pandas as pd
from collections import defaultdict
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from nanobot.config.loader import load_config


class CanonDBVisualizer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def get_statistics(self):
        """获取数据库统计信息"""
        cursor = self.conn.cursor()
        stats = {}

        # 实体统计
        cursor.execute("SELECT type, COUNT(*) as count FROM entity_registry GROUP BY type")
        entity_counts = {row['type']: row['count'] for row in cursor.fetchall()}
        stats['entities'] = entity_counts

        # 总实体数
        cursor.execute("SELECT COUNT(*) as count FROM entity_registry")
        stats['total_entities'] = cursor.fetchone()['count']

        # 事实历史记录数
        cursor.execute("SELECT COUNT(*) as count FROM fact_history")
        stats['fact_history_count'] = cursor.fetchone()['count']

        # 关系历史记录数
        cursor.execute("SELECT COUNT(*) as count FROM relationship_history")
        stats['relationship_history_count'] = cursor.fetchone()['count']

        # 提交数
        cursor.execute("SELECT COUNT(*) as count FROM commit_log")
        stats['commit_count'] = cursor.fetchone()['count']

        # 角色数
        cursor.execute("SELECT COUNT(*) as count FROM character_current")
        stats['character_count'] = cursor.fetchone()['count']

        return stats

    def visualize_entity_distribution(self, output_path='canon_entity_distribution.png'):
        """可视化实体类型分布"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT type, COUNT(*) as count FROM entity_registry GROUP BY type")
        data = cursor.fetchall()

        if not data:
            print("没有实体数据")
            return

        types = [row['type'] for row in data]
        counts = [row['count'] for row in data]

        plt.figure(figsize=(10, 6))
        plt.bar(types, counts, color='skyblue', alpha=0.8)
        plt.xlabel('实体类型', fontsize=12)
        plt.ylabel('数量', fontsize=12)
        plt.title('实体类型分布', fontsize=16)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 实体分布图已保存到: {output_path}")
        plt.close()

    def visualize_fact_history_timeline(self, output_path='canon_fact_timeline.png'):
        """可视化事实历史时间线"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT chapter_no, predicate, COUNT(*) as count
            FROM fact_history
            GROUP BY chapter_no, predicate
            ORDER BY chapter_no
            LIMIT 200
        """)
        data = cursor.fetchall()

        if not data:
            print("没有事实历史数据")
            return

        # 按章节和谓词组织数据
        chapters = sorted(set(row['chapter_no'] for row in data))
        predicates = sorted(set(row['predicate'] for row in data))

        # 创建数据矩阵
        matrix = defaultdict(lambda: defaultdict(int))
        for row in data:
            matrix[row['chapter_no']][row['predicate']] = row['count']

        # 绘图
        fig, ax = plt.subplots(figsize=(16, 8))

        x_pos = range(len(chapters))
        width = 0.8 / len(predicates) if predicates else 0.8

        for i, predicate in enumerate(predicates):
            values = [matrix[ch][predicate] for ch in chapters]
            ax.bar([x + i * width for x in x_pos], values,
                  width=width, label=predicate, alpha=0.8)

        ax.set_xlabel('章节', fontsize=12)
        ax.set_ylabel('事实变更数量', fontsize=12)
        ax.set_title('事实历史时间线', fontsize=16)
        ax.set_xticks([x + width * len(predicates) / 2 for x in x_pos])
        ax.set_xticklabels(chapters, rotation=90, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 事实时间线图已保存到: {output_path}")
        plt.close()

    def visualize_relationship_changes(self, output_path='canon_relationship_changes.png'):
        """可视化关系变化"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT chapter_no, kind, op, COUNT(*) as count
            FROM relationship_history
            GROUP BY chapter_no, kind, op
            ORDER BY chapter_no
            LIMIT 200
        """)
        data = cursor.fetchall()

        if not data:
            print("没有关系历史数据")
            return

        # 按章节统计
        chapters = sorted(set(row['chapter_no'] for row in data))
        ops = ['INSERT', 'UPDATE', 'DELETE']

        matrix = defaultdict(lambda: defaultdict(int))
        for row in data:
            matrix[row['chapter_no']][row['op']] += row['count']

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 6))

        x_pos = range(len(chapters))
        width = 0.25

        for i, op in enumerate(ops):
            values = [matrix[ch][op] for ch in chapters]
            ax.bar([x + i * width for x in x_pos], values,
                  width=width, label=op, alpha=0.8)

        ax.set_xlabel('章节', fontsize=12)
        ax.set_ylabel('关系变更数量', fontsize=12)
        ax.set_title('关系变化统计', fontsize=16)
        ax.set_xticks([x + width for x in x_pos])
        ax.set_xticklabels(chapters, rotation=90, fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 关系变化图已保存到: {output_path}")
        plt.close()

    def visualize_commit_status(self, output_path='canon_commit_status.png'):
        """可视化提交状态"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM commit_log
            GROUP BY status
        """)
        data = cursor.fetchall()

        if not data:
            print("没有提交日志数据")
            return

        statuses = [row['status'] for row in data]
        counts = [row['count'] for row in data]

        plt.figure(figsize=(10, 6))
        colors = ['green' if 'DONE' in s else 'red' if 'FAILED' in s else 'orange'
                 for s in statuses]
        plt.bar(statuses, counts, color=colors, alpha=0.8)
        plt.xlabel('提交状态', fontsize=12)
        plt.ylabel('数量', fontsize=12)
        plt.title('提交状态分布', fontsize=16)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 提交状态图已保存到: {output_path}")
        plt.close()

    def visualize_top_characters(self, output_path='canon_top_characters.png', top_n=20):
        """可视化出现最多的角色"""
        cursor = self.conn.cursor()
        cursor.execute("""
            WITH chars AS (
                SELECT entity_id, canonical_name
                FROM entity_registry
                WHERE type = 'character'
                  AND canonical_name NOT LIKE 'character_%'
                  AND canonical_name NOT IN ('母亲', '妹妹', '姐姐', '出租车师傅', '未具名仇人', '蓝发女孩', '∈')
            ),
            fact_counts AS (
                SELECT subject_id AS entity_id, COUNT(*) AS fact_count
                FROM fact_history
                GROUP BY subject_id
            ),
            rel_counts AS (
                SELECT entity_id, COUNT(*) AS rel_count
                FROM (
                    SELECT from_id AS entity_id FROM relationship_history
                    UNION ALL
                    SELECT to_id AS entity_id FROM relationship_history
                )
                GROUP BY entity_id
            )
            SELECT
                c.canonical_name,
                COALESCE(f.fact_count, 0) AS fact_count,
                COALESCE(r.rel_count, 0) AS rel_count,
                (COALESCE(f.fact_count, 0) + COALESCE(r.rel_count, 0)) AS activity_score
            FROM chars c
            LEFT JOIN fact_counts f ON c.entity_id = f.entity_id
            LEFT JOIN rel_counts r ON c.entity_id = r.entity_id
            WHERE (COALESCE(f.fact_count, 0) + COALESCE(r.rel_count, 0)) > 0
            ORDER BY activity_score DESC, fact_count DESC, rel_count DESC, c.canonical_name
            LIMIT ?
        """, (top_n,))
        data = cursor.fetchall()

        if not data:
            print("没有角色数据")
            return

        names = [row['canonical_name'] for row in data]
        counts = [row['activity_score'] for row in data]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(names)), counts, color='coral', alpha=0.8)
        plt.yticks(range(len(names)), names, fontsize=10)
        plt.xlabel('活跃度分数（事实+关系）', fontsize=12)
        plt.ylabel('角色', fontsize=12)
        plt.title(f'Top {len(names)} 活跃角色', fontsize=16)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Top角色图已保存到: {output_path}")
        plt.close()

    def export_summary_report(self, output_path='canon_summary.txt'):
        """导出摘要报告"""
        stats = self.get_statistics()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Canon DB 数据摘要报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"总实体数: {stats['total_entities']}\n")
            f.write(f"角色数: {stats['character_count']}\n")
            f.write(f"事实历史记录: {stats['fact_history_count']}\n")
            f.write(f"关系历史记录: {stats['relationship_history_count']}\n")
            f.write(f"提交数: {stats['commit_count']}\n\n")

            f.write("实体类型分布:\n")
            for entity_type, count in stats['entities'].items():
                f.write(f"  {entity_type}: {count}\n")

            # 获取示例角色
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT canonical_name, first_seen_chapter
                FROM entity_registry
                WHERE type = 'character'
                ORDER BY first_seen_chapter
                LIMIT 10
            """)
            f.write("\n前10个出现的角色:\n")
            for row in cursor.fetchall():
                f.write(f"  {row['canonical_name']} (首次出现: {row['first_seen_chapter']})\n")

        print(f"✓ 摘要报告已保存到: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Canon DB 数据可视化')
    parser.add_argument('--db-path', type=str, help='Canon DB 路径（可选）')
    args = parser.parse_args()

    print("=" * 70)
    print("Canon DB 数据可视化")
    print("=" * 70)

    # 确定数据库路径
    if args.db_path:
        canon_db_path = Path(args.db_path).expanduser()
    else:
        # 加载配置
        config = load_config()
        canon_db_path = Path(config.integrations.canon_db.db_path).expanduser()

    if not canon_db_path.exists():
        print(f"错误: Canon DB 不存在: {canon_db_path}")
        return

    print(f"数据库路径: {canon_db_path}\n")

    # 创建可视化器
    visualizer = CanonDBVisualizer(str(canon_db_path))

    try:
        # 获取统计信息
        print("\n统计信息:")
        stats = visualizer.get_statistics()
        print(f"  总实体数: {stats['total_entities']}")
        print(f"  角色数: {stats['character_count']}")
        print(f"  事实历史: {stats['fact_history_count']}")
        print(f"  关系历史: {stats['relationship_history_count']}")
        print(f"  提交数: {stats['commit_count']}")

        # 生成可视化
        print("\n生成可视化图表...")
        visualizer.visualize_entity_distribution('canon_entity_distribution.png')
        visualizer.visualize_fact_history_timeline('canon_fact_timeline.png')
        visualizer.visualize_relationship_changes('canon_relationship_changes.png')
        visualizer.visualize_commit_status('canon_commit_status.png')
        visualizer.visualize_top_characters('canon_top_characters.png')
        visualizer.export_summary_report('canon_summary.txt')

        print("\n✓ 所有可视化完成！")

    finally:
        visualizer.close()


if __name__ == '__main__':
    main()
