#!/bin/bash
# 使用browser工具爬取22biqu小说
# 先获取章节列表，再批量爬取

set -e

OUTPUT_DIR="/home/chris/Desktop/my_workspace/novel_data"
CRAWLEE_SCRIPT="/home/chris/Desktop/my_workspace/nanobot/scripts/crawlee/crawl-novel-chapter.js"
CHAPTER_LIST="/tmp/22biqu_chapters.txt"
TMUX_SESSION="novel_crawler_22biqu"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

mkdir -p "$OUTPUT_DIR"

log "=========================================="
log "  获取章节列表"
log "=========================================="

# 使用browser获取章节列表
agent-browser open "https://www.22biqu.com/biqu7034/"
sleep 2

# 提取所有章节链接
agent-browser snapshot | \
    grep -A 2 "link.*第.*章" | \
    grep "url:" | \
    sed 's/.*url: //' | \
    sort -u > "$CHAPTER_LIST"

TOTAL_CHAPTERS=$(wc -l < "$CHAPTER_LIST")
log "找到 $TOTAL_CHAPTERS 个章节"

if [ $TOTAL_CHAPTERS -eq 0 ]; then
    log "错误：未找到章节链接"
    exit 1
fi

# 创建爬取脚本
CRAWLER_SCRIPT="/tmp/crawl_22biqu_task.sh"
cat > "$CRAWLER_SCRIPT" << 'SCRIPT_EOF'
#!/bin/bash

OUTPUT_DIR="/home/chris/Desktop/my_workspace/novel_data"
CRAWLEE_SCRIPT="/home/chris/Desktop/my_workspace/nanobot/scripts/crawlee/crawl-novel-chapter.js"
CHAPTER_LIST="/tmp/22biqu_chapters.txt"

SUCCESS=0
FAILED=0
INDEX=1

while IFS= read -r chapter_url; do
    # 构建完整URL
    if [[ ! "$chapter_url" =~ ^http ]]; then
        FULL_URL="https://www.22biqu.com${chapter_url}"
    else
        FULL_URL="$chapter_url"
    fi

    # 先用临时文件名爬取
    TEMP_FILE="${OUTPUT_DIR}/temp_${INDEX}.md"

    echo "[$(date '+%H:%M:%S')] [$INDEX] 爬取: $FULL_URL"

    if node "$CRAWLEE_SCRIPT" "$FULL_URL" "$TEMP_FILE" --format=markdown 2>&1 | grep -q "✓"; then
        # 从文件第一行提取章节标题
        CHAPTER_TITLE=$(head -1 "$TEMP_FILE" | sed 's/^# //' | sed 's/[<>:"/\\|?*]/_/g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        # 如果标题为空或是"未找到标题"，使用序号
        if [ -z "$CHAPTER_TITLE" ] || [ "$CHAPTER_TITLE" = "未找到标题" ]; then
            CHAPTER_TITLE="第${INDEX}章"
        fi

        # 重命名文件
        FINAL_FILE="${OUTPUT_DIR}/${CHAPTER_TITLE}.md"
        mv "$TEMP_FILE" "$FINAL_FILE"

        echo "✓ 已保存: ${CHAPTER_TITLE}.md"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "✗ 第 $INDEX 章失败"
        FAILED=$((FAILED + 1))
        rm -f "$TEMP_FILE"
    fi

    INDEX=$((INDEX + 1))
    sleep 1
done < "$CHAPTER_LIST"

echo ""
echo "=========================================="
echo "  爬取完成"
echo "=========================================="
echo "成功: $SUCCESS 章"
echo "失败: $FAILED 章"
echo "=========================================="
SCRIPT_EOF

chmod +x "$CRAWLER_SCRIPT"

# 检查tmux会话
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "关闭已存在的会话..."
    tmux kill-session -t "$TMUX_SESSION"
fi

# 在tmux中启动
log "在tmux会话中启动爬取: $TMUX_SESSION"
tmux new-session -d -s "$TMUX_SESSION" "bash $CRAWLER_SCRIPT; echo '任务完成，5秒后关闭...'; sleep 5"

log "✓ 爬取任务已启动"
log ""
log "查看进度："
log "  tmux attach -t $TMUX_SESSION"
log ""
log "监控中..."

# 监控
while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do
    sleep 5
    COUNT=$(find "$OUTPUT_DIR" -name "*.md" -type f ! -name "temp_*.md" 2>/dev/null | wc -l)
    log "已爬取: $COUNT 章"
done

log ""
log "✓ 任务完成"

# 清理临时文件
rm -f "$OUTPUT_DIR"/temp_*.md

# 统计
FINAL_COUNT=$(find "$OUTPUT_DIR" -name "*.md" -type f 2>/dev/null | wc -l)
log ""
log "=========================================="
log "  最终统计"
log "=========================================="
log "总章节数: $FINAL_COUNT"
log "保存位置: $OUTPUT_DIR"
log ""
log "部分章节列表:"
find "$OUTPUT_DIR" -name "*.md" -type f | sort | head -10 | while read file; do
    log "  - $(basename "$file")"
done
log "=========================================="

# 关闭browser
agent-browser close
