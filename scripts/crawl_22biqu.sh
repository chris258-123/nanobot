#!/bin/bash
# 批量爬取22biqu小说 - 使用Crawlee工具
# 用法: bash crawl_22biqu.sh

set -e

# 配置
BASE_URL="https://www.22biqu.com/biqu7034"
OUTPUT_DIR="/home/chris/Desktop/my_workspace/novel_data"
CRAWLEE_SCRIPT="/home/chris/Desktop/my_workspace/nanobot/scripts/crawlee/crawl-novel-chapter.js"
TMUX_SESSION="novel_crawler_22biqu"
DELAY=1

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查tmux
if ! command -v tmux &> /dev/null; then
    error "tmux未安装，请先安装: sudo apt install tmux"
    exit 1
fi

log "=========================================="
log "  批量爬取22biqu小说"
log "=========================================="
log "书籍URL: $BASE_URL"
log "输出目录: $OUTPUT_DIR"
log "=========================================="

# 创建爬取脚本
CRAWLER_SCRIPT="/tmp/crawl_22biqu_task.sh"
cat > "$CRAWLER_SCRIPT" << 'SCRIPT_EOF'
#!/bin/bash

BASE_URL="https://www.22biqu.com/biqu7034"
OUTPUT_DIR="/home/chris/Desktop/my_workspace/novel_data"
CRAWLEE_SCRIPT="/home/chris/Desktop/my_workspace/nanobot/scripts/crawlee/crawl-novel-chapter.js"
DELAY=1

SUCCESS=0
FAILED=0
CHAPTER_INDEX=1  # 全局章节计数器

# 爬取9批章节 (2-9目录，每个200章)
for batch in $(seq 2 9); do
    echo ""
    echo "=========================================="
    echo "开始爬取第 $((batch-1)) 批章节 (目录: ${batch}/)..."
    echo "=========================================="

    for i in $(seq 1 200); do
        CHAPTER_URL="${BASE_URL}/${batch}/${i}.html"
        OUTPUT_FILE="${OUTPUT_DIR}/$(printf "%04d" $CHAPTER_INDEX).md"

        echo "[$(date '+%H:%M:%S')] [$CHAPTER_INDEX] 爬取: ${batch}/${i}.html"

        if node "$CRAWLEE_SCRIPT" "$CHAPTER_URL" "$OUTPUT_FILE" --format=markdown 2>&1 | grep -q "✓"; then
            echo "✓ 第 $CHAPTER_INDEX 章成功"
            SUCCESS=$((SUCCESS + 1))
            CHAPTER_INDEX=$((CHAPTER_INDEX + 1))
            FAILED=0  # 重置失败计数
        else
            echo "✗ 失败"
            FAILED=$((FAILED + 1))

            # 连续失败10次则跳过当前批次
            if [ $FAILED -ge 10 ]; then
                echo "连续失败过多，跳过当前批次"
                break
            fi
        fi

        sleep $DELAY
    done
done

echo ""
echo "=========================================="
echo "  爬取完成"
echo "=========================================="
echo "成功: $SUCCESS 章"
echo "失败: $FAILED 章"
echo "=========================================="
SCRIPT_EOF

chmod +x "$CRAWLER_SCRIPT"

# 检查是否已有同名会话
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "发现已存在的会话，正在关闭..."
    tmux kill-session -t "$TMUX_SESSION"
fi

# 在tmux中启动爬取任务
log "在tmux会话中启动爬取: $TMUX_SESSION"
tmux new-session -d -s "$TMUX_SESSION" "bash $CRAWLER_SCRIPT; echo '任务完成，5秒后自动关闭...'; sleep 5"

log "✓ 爬取任务已启动"
log ""
log "查看进度："
log "  tmux attach -t $TMUX_SESSION"
log ""
log "手动关闭："
log "  tmux kill-session -t $TMUX_SESSION"
log ""
log "监控中..."

# 监控任务
while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do
    sleep 5
    if [ -d "$OUTPUT_DIR" ]; then
        COUNT=$(find "$OUTPUT_DIR" -name "[0-9]*.md" -type f 2>/dev/null | wc -l)
        log "已爬取: $COUNT 章"
    fi
done

log ""
log "✓ 任务已完成，tmux会话已自动关闭"

# 显示最终统计
if [ -d "$OUTPUT_DIR" ]; then
    FINAL_COUNT=$(find "$OUTPUT_DIR" -name "[0-9]*.md" -type f 2>/dev/null | wc -l)
    log ""
    log "=========================================="
    log "  最终统计"
    log "=========================================="
    log "总章节数: $FINAL_COUNT"
    log "保存位置: $OUTPUT_DIR"
    log ""
    log "文件列表示例:"
    find "$OUTPUT_DIR" -name "[0-9]*.md" -type f | sort | head -10 | while read file; do
        log "  - $(basename "$file")"
    done
    log "=========================================="
fi
