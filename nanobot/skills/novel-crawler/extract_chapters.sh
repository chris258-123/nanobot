#!/bin/bash
# 提取多个目录页的章节列表

set -e

CHAPTER_LIST="/tmp/all_chapters.txt"
BASE_URL="${1:-https://www.example.com/book/12345}"
START_PAGE="${2:-2}"
END_PAGE="${3:-9}"

GREEN='\033[0;32m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

log "=========================================="
log "  提取章节列表"
log "=========================================="
log "基础 URL: $BASE_URL"
log "目录页范围: $START_PAGE - $END_PAGE"
log ""

# 清空章节列表
> "$CHAPTER_LIST"

# 遍历目录页
for page in $(seq $START_PAGE $END_PAGE); do
    log "正在提取第 $page 页..."

    agent-browser open "${BASE_URL}/${page}/"
    sleep 2

    agent-browser snapshot | \
        grep -A 2 "link.*第.*章" | \
        grep "url:" | \
        sed 's/.*url: //' >> "$CHAPTER_LIST"
done

# 关闭浏览器
agent-browser close

# 去重并排序
sort -u "$CHAPTER_LIST" -o "$CHAPTER_LIST"

TOTAL=$(wc -l < "$CHAPTER_LIST")

log ""
log "=========================================="
log "  提取完成"
log "=========================================="
log "总章节数: $TOTAL"
log "保存位置: $CHAPTER_LIST"
log "=========================================="
