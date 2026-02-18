#!/bin/bash
# 三层记忆系统可视化汇总脚本

echo "======================================================================"
echo "三层记忆系统数据可视化"
echo "======================================================================"
echo ""

# 1. Canon DB 可视化
echo "1. 生成 Canon DB 可视化..."
python3 nanobot/skills/novel-workflow/scripts/visualize_canon_db.py \
    --db-path ~/.nanobot/workspace/canon_novel04_test.db 2>&1 | \
    grep -E "^(统计|✓|错误)" || echo "Canon DB 可视化完成"

echo ""

# 2. Neo4j 可视化
echo "2. 生成 Neo4j 图数据可视化..."
python3 nanobot/skills/novel-workflow/scripts/visualize_neo4j.py 2>&1 | \
    grep -E "^(统计|✓|错误)" || echo "Neo4j 可视化完成"

echo ""
echo "======================================================================"
echo "可视化文件生成完成"
echo "======================================================================"
echo ""
echo "Canon DB 可视化:"
ls -lh canon_*.png canon_*.txt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Neo4j 可视化:"
ls -lh neo4j_*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "查看图片: eog *.png  # 或使用你喜欢的图片查看器"
echo ""
