#!/bin/bash
# 批量资产提取脚本 - 支持断点续传

BOOK_ID="novel_04"
CHAPTER_DIR="/home/chris/Desktop/my_workspace/novel_data/04"
OUTPUT_DIR="/home/chris/novel_assets_enhanced"
LLM_CONFIG="/home/chris/Desktop/my_workspace/nanobot/llm_config.json"
SCRIPT_PATH="/home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/scripts/asset_extractor_enhanced.py"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 统计信息
total_chapters=$(ls "$CHAPTER_DIR" | wc -l)
processed=0
skipped=0
failed=0

echo "========================================="
echo "批量资产提取"
echo "========================================="
echo "源目录: $CHAPTER_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "总章节数: $total_chapters"
echo "========================================="
echo ""

# 遍历所有章节文件
for chapter_file in "$CHAPTER_DIR"/*.md; do
    chapter_name=$(basename "$chapter_file" .md)
    output_file="$OUTPUT_DIR/${BOOK_ID}_${chapter_name}_assets.json"

    # 检查是否已处理
    if [ -f "$output_file" ]; then
        ((skipped++))
        echo "[跳过] $chapter_name (已存在)"
        continue
    fi

    echo "[处理] $chapter_name ..."

    # 执行提取
    python "$SCRIPT_PATH" \
        --book-id "$BOOK_ID" \
        --chapter-file "$chapter_file" \
        --output-dir "$OUTPUT_DIR" \
        --llm-config "$LLM_CONFIG" \
        --extract-all \
        2>&1 | grep -E "(Processing|Extracted|Error)" || true

    # 检查是否成功
    if [ -f "$output_file" ]; then
        ((processed++))
        echo "  ✓ 成功 ($processed/$total_chapters)"
    else
        ((failed++))
        echo "  ✗ 失败"
        echo "$chapter_name" >> "$OUTPUT_DIR/failed_chapters.txt"
    fi

    # 每10个章节显示进度
    if [ $((($processed + $skipped + $failed) % 10)) -eq 0 ]; then
        echo ""
        echo "--- 进度报告 ---"
        echo "已处理: $processed"
        echo "已跳过: $skipped"
        echo "失败: $failed"
        echo "总计: $(($processed + $skipped + $failed))/$total_chapters"
        echo "----------------"
        echo ""
    fi

    # 短暂延迟避免API限流
    sleep 1
done

echo ""
echo "========================================="
echo "提取完成！"
echo "========================================="
echo "成功处理: $processed"
echo "跳过(已存在): $skipped"
echo "失败: $failed"
echo "总计: $(($processed + $skipped + $failed))/$total_chapters"
echo "========================================="

if [ $failed -gt 0 ]; then
    echo ""
    echo "失败的章节已记录到: $OUTPUT_DIR/failed_chapters.txt"
fi
