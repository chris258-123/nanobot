#!/bin/bash
# 监控资产提取进度

OUTPUT_DIR="/home/chris/novel_assets_enhanced"
TOTAL=985

while true; do
    current=$(ls "$OUTPUT_DIR" 2>/dev/null | grep -c "\.json$")
    percent=$(echo "scale=2; $current * 100 / $TOTAL" | bc)

    clear
    echo "========================================"
    echo "资产提取进度监控"
    echo "========================================"
    echo "总章节数: $TOTAL"
    echo "已完成: $current"
    echo "进度: $percent%"
    echo "========================================"
    echo ""
    echo "最近处理的文件:"
    ls -lt "$OUTPUT_DIR" | head -6 | tail -5
    echo ""
    echo "按 Ctrl+C 退出监控"

    sleep 5
done
