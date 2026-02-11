#!/bin/bash
# 批量爬取小说章节

set -e

# 允许通过环境变量覆盖输出目录
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/novel_data}"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAYWRIGHT_SCRIPT="${SKILL_DIR}/crawl-chapter-playwright.js"
CHAPTER_LIST="/tmp/all_chapters.txt"
TMUX_SESSION="novel_crawler"

GREEN='\033[0;32m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

mkdir -p "$OUTPUT_DIR"

log "=========================================="
log "  准备爬取小说"
log "=========================================="
log "总章节数: $(wc -l < $CHAPTER_LIST)"
log ""

# 创建爬取任务脚本
CRAWLER_SCRIPT="/tmp/crawl_task.sh"
cat > "$CRAWLER_SCRIPT" << SCRIPT_EOF
#!/bin/bash

OUTPUT_DIR="${OUTPUT_DIR}"
PLAYWRIGHT_SCRIPT="${PLAYWRIGHT_SCRIPT}"
CHAPTER_LIST="${CHAPTER_LIST}"

SUCCESS=0
FAILED=0
INDEX=1
TOTAL=\$(wc -l < "\$CHAPTER_LIST")

echo "开始爬取 \$TOTAL 章..."
echo ""

while IFS= read -r chapter_url; do
    # 构建完整 URL
    if [[ ! "\$chapter_url" =~ ^http ]]; then
        FULL_URL="https://www.22biqu.com\${chapter_url}"
    else
        FULL_URL="\$chapter_url"
    fi

    # 使用临时文件名
    TEMP_FILE="\${OUTPUT_DIR}/temp_\${INDEX}.md"

    echo "[\$(date '+%H:%M:%S')] [\$INDEX/\$TOTAL] 爬取: \$FULL_URL"

    # 调用 Playwright 脚本
    if node "\$PLAYWRIGHT_SCRIPT" "\$FULL_URL" "\$TEMP_FILE" 2>&1 | grep -q "✓"; then
        # 从文件第一行提取章节标题
        CHAPTER_TITLE=\$(head -1 "\$TEMP_FILE" | sed 's/^# //' | sed 's/[<>:"/\\\\|?*]/_/g' | sed 's/^[[:space:]]*//;s/[[:space:]]*\$//')

        # 如果标题为空或是"未找到标题"，使用序号
        if [ -z "\$CHAPTER_TITLE" ] || [ "\$CHAPTER_TITLE" = "未找到标题" ]; then
            CHAPTER_TITLE="第\${INDEX}章"
        fi

        # 重命名文件
        FINAL_FILE="\${OUTPUT_DIR}/\${CHAPTER_TITLE}.md"
        mv "\$TEMP_FILE" "\$FINAL_FILE"

        echo "  ✓ 已保存: \${CHAPTER_TITLE}.md"
        SUCCESS=\$((SUCCESS + 1))
    else
        echo "  ✗ 失败"
        FAILED=\$((FAILED + 1))
        rm -f "\$TEMP_FILE"
    fi

    INDEX=\$((INDEX + 1))
    sleep 1  # 延迟避免被封
done < "\$CHAPTER_LIST"

echo ""
echo "=========================================="
echo "  爬取完成"
echo "=========================================="
echo "成功: \$SUCCESS 章"
echo "失败: \$FAILED 章"
echo "=========================================="
SCRIPT_EOF

chmod +x "$CRAWLER_SCRIPT"

# 检查 tmux 会话
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "关闭已存在的会话..."
    tmux kill-session -t "$TMUX_SESSION"
fi

# 在 tmux 中启动
log "在 tmux 会话中启动爬取: $TMUX_SESSION"
tmux new-session -d -s "$TMUX_SESSION" "bash $CRAWLER_SCRIPT; echo '任务完成，10秒后关闭...'; sleep 10"

log "✓ 爬取任务已启动"
log ""
log "查看进度："
log "  tmux attach -t $TMUX_SESSION"
log ""
log "监控中..."

# 监控进度
while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do
    sleep 10
    COUNT=$(find "$OUTPUT_DIR" -name "*.md" -type f ! -name "temp_*.md" | wc -l)
    log "已爬取: $COUNT 章"
done

log ""
log "✓ 任务完成"

# 清理临时文件
rm -f "$OUTPUT_DIR"/temp_*.md

# 统计
FINAL_COUNT=$(find "$OUTPUT_DIR" -name "*.md" -type f | wc -l)
log ""
log "=========================================="
log "  最终统计"
log "=========================================="
log "总章节数: $FINAL_COUNT"
log "保存位置: $OUTPUT_DIR"
log "=========================================="
