#!/usr/bin/env python3
"""
Neo4j 图数据可视化工具
可视化角色关系网络、事件时间线、实体连接等
"""

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase

# 配置中文字体 - 自动检测可用的 CJK 字体
cjk_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name]
if cjk_fonts:
    matplotlib.rcParams['font.sans-serif'] = [cjk_fonts[0], 'DejaVu Sans']
else:
    # 备用方案：尝试常见中文字体
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class Neo4jVisualizer:
    def __init__(self, uri, username, password, *, book_id: str = "", commit_ids: list[str] | None = None):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.book_id = (book_id or "").strip()
        self.commit_ids = commit_ids or []

    def close(self):
        self.driver.close()

    def get_character_network(
        self,
        limit=160,
        include_associate=False,
        include_implied=True,
        min_weight=1,
    ):
        """获取角色关系网络"""
        with self.driver.session() as session:
            if self.commit_ids:
                # Book-scoped mode: only relations written by this book's commits.
                result = session.run(
                    """
                    MATCH (c1:Character)-[r:RELATES]-(c2:Character)
                    WHERE c1.entity_id < c2.entity_id
                      AND r.commit_id IN $commit_ids
                      AND ($include_associate OR r.kind <> 'ASSOCIATE')
                      AND (
                        ($include_implied AND r.status IN ['confirmed', 'implied'])
                        OR (NOT $include_implied AND r.status = 'confirmed')
                      )
                    WITH c1, c2, collect(DISTINCT r.kind) as rel_kinds, count(r) as rel_weight
                    WITH c1, c2, rel_kinds, rel_weight, 0 as co_event_weight, rel_weight as total_weight
                    WHERE total_weight >= $min_weight
                    RETURN c1.canonical_name as from, c2.canonical_name as to,
                           rel_kinds as relations, rel_weight, co_event_weight, total_weight as weight
                    ORDER BY total_weight DESC
                    LIMIT $limit
                    """,
                    limit=limit,
                    include_associate=include_associate,
                    include_implied=include_implied,
                    min_weight=min_weight,
                    commit_ids=self.commit_ids,
                )
            else:
                result = session.run(
                    """
                    MATCH (c1:Character)-[r:RELATES]-(c2:Character)
                    WHERE c1.entity_id < c2.entity_id
                      AND ($include_associate OR r.kind <> 'ASSOCIATE')
                      AND (
                        ($include_implied AND r.status IN ['confirmed', 'implied'])
                        OR (NOT $include_implied AND r.status = 'confirmed')
                      )
                    WITH c1, c2, collect(DISTINCT r.kind) as rel_kinds, count(r) as rel_weight
                    OPTIONAL MATCH (c1)-[:PARTICIPATES_IN]->(ev:Event)<-[:PARTICIPATES_IN]-(c2)
                    WITH c1, c2, rel_kinds, rel_weight, count(DISTINCT ev) as co_event_weight
                    WITH c1, c2, rel_kinds, rel_weight, co_event_weight, (rel_weight + co_event_weight) as total_weight
                    WHERE total_weight >= $min_weight
                    RETURN c1.canonical_name as from, c2.canonical_name as to,
                           rel_kinds as relations, rel_weight, co_event_weight, total_weight as weight
                    ORDER BY total_weight DESC
                    LIMIT $limit
                    """,
                    limit=limit,
                    include_associate=include_associate,
                    include_implied=include_implied,
                    min_weight=min_weight,
                )

            edges = []
            for record in result:
                labels = list(record["relations"] or [])
                if record["co_event_weight"] and record["co_event_weight"] > 0:
                    labels.append("CO_EVENT")
                edges.append({
                    'from': record['from'],
                    'to': record['to'],
                    'relation': '/'.join(labels[:2]) if labels else 'CO_EVENT',
                    'weight': record['weight'],
                    'rel_weight': record['rel_weight'],
                    'co_event_weight': record['co_event_weight'],
                })
            return edges

    def get_all_characters(self, limit=100):
        """获取所有角色节点"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Character)
                RETURN c.entity_id as id, c.canonical_name as name
                LIMIT $limit
            """, limit=limit)

            nodes = []
            for record in result:
                nodes.append({
                    'id': record['id'],
                    'name': record['name']
                })
            return nodes

    def get_events_timeline(self, limit=100, chapter_from: int | None = None, chapter_to: int | None = None):
        """获取事件时间线"""
        chapter_expr = "toInteger(replace(ch.chapter_no, 'ch', ''))"
        with self.driver.session() as session:
            if self.book_id:
                if limit and limit > 0:
                    result = session.run(
                        """
                        MATCH (e:Event)-[:OCCURS_IN]->(ch:Chapter {book_id: $book_id})
                        WHERE ($chapter_from IS NULL OR """ + chapter_expr + """ >= $chapter_from)
                          AND ($chapter_to IS NULL OR """ + chapter_expr + """ <= $chapter_to)
                        RETURN e.event_id as id, e.summary as summary,
                               e.type as type, ch.chapter_no as chapter
                        ORDER BY """ + chapter_expr + """
                        LIMIT $limit
                        """,
                        limit=limit,
                        book_id=self.book_id,
                        chapter_from=chapter_from,
                        chapter_to=chapter_to,
                    )
                else:
                    result = session.run(
                        """
                        MATCH (e:Event)-[:OCCURS_IN]->(ch:Chapter {book_id: $book_id})
                        WHERE ($chapter_from IS NULL OR """ + chapter_expr + """ >= $chapter_from)
                          AND ($chapter_to IS NULL OR """ + chapter_expr + """ <= $chapter_to)
                        RETURN e.event_id as id, e.summary as summary,
                               e.type as type, ch.chapter_no as chapter
                        ORDER BY """ + chapter_expr + """
                        """,
                        book_id=self.book_id,
                        chapter_from=chapter_from,
                        chapter_to=chapter_to,
                    )
            else:
                if limit and limit > 0:
                    result = session.run(
                        """
                        MATCH (e:Event)-[:OCCURS_IN]->(ch:Chapter)
                        WHERE ($chapter_from IS NULL OR """ + chapter_expr + """ >= $chapter_from)
                          AND ($chapter_to IS NULL OR """ + chapter_expr + """ <= $chapter_to)
                        RETURN e.event_id as id, e.summary as summary,
                               e.type as type, ch.chapter_no as chapter
                        ORDER BY """ + chapter_expr + """
                        LIMIT $limit
                        """,
                        limit=limit,
                        chapter_from=chapter_from,
                        chapter_to=chapter_to,
                    )
                else:
                    result = session.run(
                        """
                        MATCH (e:Event)-[:OCCURS_IN]->(ch:Chapter)
                        WHERE ($chapter_from IS NULL OR """ + chapter_expr + """ >= $chapter_from)
                          AND ($chapter_to IS NULL OR """ + chapter_expr + """ <= $chapter_to)
                        RETURN e.event_id as id, e.summary as summary,
                               e.type as type, ch.chapter_no as chapter
                        ORDER BY """ + chapter_expr + """
                        """,
                        chapter_from=chapter_from,
                        chapter_to=chapter_to,
                    )

            events = []
            for record in result:
                events.append({
                    'id': record['id'],
                    'summary': record['summary'],
                    'type': record['type'],
                    'chapter': record['chapter']
                })
            return events

    def visualize_character_network(
        self,
        output_path='neo4j_character_network.png',
        title='角色关系网络图',
        exclude_character_name=None,
    ):
        """可视化角色关系网络"""
        print("正在获取角色关系数据...")
        edges = self.get_character_network(
            limit=220,
            include_associate=False,
            include_implied=True,
            min_weight=1,
        )

        if exclude_character_name:
            edges = [
                edge
                for edge in edges
                if edge['from'] != exclude_character_name and edge['to'] != exclude_character_name
            ]

        if not edges:
            print("没有找到角色关系数据")
            return

        # 创建网络图
        graph = nx.Graph()

        # 添加边（聚合图）
        edge_labels = {}
        for edge in edges:
            graph.add_edge(edge['from'], edge['to'], weight=edge.get('weight', 1))
            edge_labels[(edge['from'], edge['to'])] = edge['relation']

        if graph.number_of_nodes() == 0:
            print("没有可视化节点")
            return

        # 只画最大连通子图，突出主关系网络
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        graph = graph.subgraph(components[0]).copy()
        edge_labels = {
            (u, v): label
            for (u, v), label in edge_labels.items()
            if graph.has_edge(u, v)
        }

        # 绘图
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(graph, k=2, iterations=50)

        # 绘制节点
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue',
                              node_size=3000, alpha=0.9)

        # 绘制边
        widths = [1 + 0.4 * graph[u][v].get('weight', 1) for u, v in graph.edges()]
        nx.draw_networkx_edges(graph, pos, edge_color='gray', width=widths, alpha=0.55)

        # 绘制标签
        nx.draw_networkx_labels(graph, pos, font_size=10)

        # 绘制边标签（关系类型）
        nx.draw_networkx_edge_labels(graph, pos, edge_labels,
                                    font_size=8, font_color='red')

        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 角色关系网络图已保存到: {output_path}")
        plt.close()

    def visualize_events_timeline(
        self,
        output_path='neo4j_events_timeline.png',
        limit=100,
        chapter_from: int | None = None,
        chapter_to: int | None = None,
    ):
        """可视化事件时间线"""
        print("正在获取事件数据...")
        events = self.get_events_timeline(limit=limit, chapter_from=chapter_from, chapter_to=chapter_to)

        if not events:
            print("没有找到事件数据")
            return

        # 按章节分组事件
        events_by_chapter = defaultdict(list)
        for event in events:
            chapter = event['chapter']
            events_by_chapter[chapter].append(event)

        # 绘图
        fig, ax = plt.subplots(figsize=(16, 10))

        chapters = sorted(events_by_chapter.keys())
        y_pos = 0

        for chapter in chapters:
            chapter_events = events_by_chapter[chapter]
            for event in chapter_events:
                ax.barh(y_pos, 1, left=int(chapter.replace('ch', '')),
                       height=0.8, alpha=0.7)
                ax.text(int(chapter.replace('ch', '')) + 0.5, y_pos,
                       f"{event['type']}: {event['summary'][:30]}...",
                       va='center', fontsize=8)
                y_pos += 1

        ax.set_xlabel('章节', fontsize=12)
        ax.set_ylabel('事件', fontsize=12)
        ax.set_title('事件时间线', fontsize=16)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 事件时间线图已保存到: {output_path}")
        plt.close()

    def get_statistics(self):
        """获取图数据库统计信息"""
        with self.driver.session() as session:
            stats = {}

            if self.commit_ids and self.book_id:
                result = session.run(
                    """
                    MATCH (c:Character)
                    WHERE EXISTS {
                        MATCH (c)-[:PARTICIPATES_IN]->(:Event)-[:OCCURS_IN]->(:Chapter {book_id: $book_id})
                    } OR EXISTS {
                        MATCH (c)-[r:RELATES]-()
                        WHERE r.commit_id IN $commit_ids
                    }
                    RETURN count(DISTINCT c) as count
                    """,
                    book_id=self.book_id,
                    commit_ids=self.commit_ids,
                )
                stats['characters'] = result.single()['count']

                result = session.run(
                    "MATCH ()-[r:RELATES]->() WHERE r.commit_id IN $commit_ids RETURN count(r) as count",
                    commit_ids=self.commit_ids,
                )
                stats['relations'] = result.single()['count']

                result = session.run(
                    "MATCH (e:Event)-[:OCCURS_IN]->(ch:Chapter {book_id: $book_id}) RETURN count(DISTINCT e) as count",
                    book_id=self.book_id,
                )
                stats['events'] = result.single()['count']

                result = session.run(
                    "MATCH (ch:Chapter {book_id: $book_id}) RETURN count(ch) as count",
                    book_id=self.book_id,
                )
                stats['chapters'] = result.single()['count']

                result = session.run(
                    """
                    MATCH (:Chapter {book_id: $book_id})-[:HAS_CHUNK]->(ck:Chunk)
                    RETURN count(DISTINCT ck) as count
                    """,
                    book_id=self.book_id,
                )
                stats['chunks'] = result.single()['count']
            else:
                # 角色数量
                result = session.run("MATCH (c:Character) RETURN count(c) as count")
                stats['characters'] = result.single()['count']

                # 关系数量
                result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as count")
                stats['relations'] = result.single()['count']

                # 事件数量
                result = session.run("MATCH (e:Event) RETURN count(e) as count")
                stats['events'] = result.single()['count']

                # 章节数量
                result = session.run("MATCH (ch:Chapter) RETURN count(ch) as count")
                stats['chapters'] = result.single()['count']

                # 文本块数量
                result = session.run("MATCH (ck:Chunk) RETURN count(ck) as count")
                stats['chunks'] = result.single()['count']

            return stats


def _load_commit_ids_from_canon(canon_db_path: str, book_id: str) -> list[str]:
    path = Path(canon_db_path).expanduser()
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            """
            SELECT commit_id
            FROM commit_log
            WHERE book_id = ? AND status = 'ALL_DONE'
            ORDER BY chapter_no ASC
            """,
            (book_id,),
        ).fetchall()
        return [str(row[0]) for row in rows if row and row[0]]
    finally:
        conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Neo4j 图数据可视化')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--username', type=str, default='neo4j', help='Neo4j username')
    parser.add_argument('--password', type=str, default='novel123', help='Neo4j password')
    parser.add_argument('--protagonist-name', type=str, default='罗彬瀚', help='主角名（用于去主角网络图）')
    parser.add_argument('--book-id', type=str, default='', help='仅可视化指定 book_id（需配合 canon commit 过滤关系）')
    parser.add_argument('--canon-db-path', type=str, default='', help='Canon DB 路径（用于按 book_id 过滤 commit_id）')
    parser.add_argument('--events-limit', type=int, default=100, help='事件时间线最大条数，0表示全部')
    parser.add_argument('--chapter-from', type=int, default=0, help='事件时间线起始章节（含），0表示不限制')
    parser.add_argument('--chapter-to', type=int, default=0, help='事件时间线结束章节（含），0表示不限制')
    args = parser.parse_args()

    print("=" * 70)
    print("Neo4j 图数据可视化")
    print("=" * 70)

    # 使用命令行参数或配置
    try:
        from nanobot.config.loader import load_config

        config = load_config()
        neo4j_config = config.integrations.neo4j
        if neo4j_config.enabled:
            uri = neo4j_config.uri
            username = neo4j_config.username
            password = neo4j_config.password
        else:
            # 使用命令行参数
            uri = args.uri
            username = args.username
            password = args.password
    except Exception:
        # 使用命令行参数
        uri = args.uri
        username = args.username
        password = args.password

    book_id = (args.book_id or "").strip()
    commit_ids: list[str] = []
    if book_id and args.canon_db_path:
        commit_ids = _load_commit_ids_from_canon(args.canon_db_path, book_id)
    elif book_id:
        print("警告: 指定了 --book-id 但未提供 --canon-db-path，关系图仍可能包含其他书。")

    print(f"连接到: {uri}")
    if book_id:
        print(f"Book过滤: {book_id} (commit_ids={len(commit_ids)})")
    print("")

    # 连接 Neo4j
    visualizer = Neo4jVisualizer(uri, username, password, book_id=book_id, commit_ids=commit_ids)

    try:
        # 获取统计信息
        print("\n统计信息:")
        stats = visualizer.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 生成可视化
        print("\n生成可视化图表...")
        visualizer.visualize_character_network(
            'neo4j_character_network.png',
            title='角色关系网络图（全网）',
        )
        visualizer.visualize_character_network(
            'neo4j_character_network_no_protagonist.png',
            title=f'角色关系网络图（去主角：{args.protagonist_name}）',
            exclude_character_name=args.protagonist_name,
        )
        chapter_from = args.chapter_from if args.chapter_from > 0 else None
        chapter_to = args.chapter_to if args.chapter_to > 0 else None
        visualizer.visualize_events_timeline(
            'neo4j_events_timeline.png',
            limit=args.events_limit,
            chapter_from=chapter_from,
            chapter_to=chapter_to,
        )

        print("\n✓ 所有可视化完成！")

    finally:
        visualizer.close()


if __name__ == '__main__':
    main()
