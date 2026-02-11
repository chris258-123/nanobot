#!/bin/bash
# 批量爬取小说章节 - 使用现有的Crawlee工具
# 用法: bash batch_crawl_novel.sh [起始章节] [结束章节]

set -e

# 配置
BOOK_URL="https://www.bqg518.cc/#/book/341"
OUTPUT_DIR="/home/chris/Desktop/my_workspace/novel_data"
CRAWLEE_SCRIPT="/home/chris/Desktop/my_workspace/nanobot/scripts/crawlee/crawl-novel-chapter.js"
START_CHAPTER=${1:-1}
END_CHAPTER=${2:-100}
DELAY=1  # 每章之间延迟（秒）

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

log "=========================================="
log "  批量爬取小说章节"
log "=========================================="
log "书籍URL: $BOOK_URL"
log "输出目录: $OUTPUT_DIR"
log "章节范围: $START_CHAPTER - $END_CHAPTER"
log "=========================================="

# 统计
SUCCESS=0
FAILED=0

# 循环爬取
for i in $(seq $START_CHAPTER $END_CHAPTER); do
    log "正在爬取第 $i 章..."

    CHAPTER_URL="${BOOK_URL}/${i}"
    OUTPUT_FILE="${OUTPUT_DIR}/chapter_${i}.md"

    if node "$CRAWLEE_SCRIPT" "$CHAPTER_URL" "$OUTPUT_FILE" --format=markdown 2>&1 | grep -q "✓"; then
        log "✓ 第 $i 章爬取成功"
        SUCCESS=$((SUCCESS + 1))
    else
        error "✗ 第 $i 章爬取失败"
        FAILED=$((FAILED + 1))

        # 连续失败3次则停止
        if [ $FAILED -ge 3 ]; then
            error "连续失败过多，停止爬取"
            break
        fi
    fi

    # 延迟
    if [ $i -lt $END_CHAPTER ]; then
        sleep $DELAY
    fi
done

log ""
log "=========================================="
log "  爬取完成"
log "=========================================="
log "成功: $SUCCESS 章"
log "失败: $FAILED 章"
log "保存位置: $OUTPUT_DIR"
log "=========================================="

# 列出文件
log ""
log "已爬取的章节:"
ls -lh "$OUTPUT_DIR"/*.md 2>/dev/null | tail -10 || echo "无文件"
