#!/bin/bash
# 提取22biqu所有目录页的章节链接

set -e

CHAPTER_LIST="/tmp/22biqu_all_chapters.txt"
TEMP_LIST="/tmp/22biqu_temp_chapters.txt"

GREEN='\033[0;32m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

# 清空章节列表
> "$CHAPTER_LIST"

log "=========================================="
log "  提取所有目录页的章节链接"
log "=========================================="

# 遍历目录页 2-9
for page in {2..9}; do
    log "正在提取第 $page 页..."

    # 打开目录页
    agent-browser open "https://www.22biqu.com/biqu7034/${page}/"
    sleep 2

    # 提取章节链接
    agent-browser snapshot | \
        grep -A 2 "link.*第.*章" | \
        grep "url:" | \
        sed 's/.*url: //' | \
        sort -u > "$TEMP_LIST"

    CHAPTER_COUNT=$(wc -l < "$TEMP_LIST")
    log "  找到 $CHAPTER_COUNT 个章节"

    # 追加到总列表
    cat "$TEMP_LIST" >> "$CHAPTER_LIST"
done

# 关闭浏览器
agent-browser close

# 去重并排序
sort -u "$CHAPTER_LIST" -o "$CHAPTER_LIST"

TOTAL_CHAPTERS=$(wc -l < "$CHAPTER_LIST")

log ""
log "=========================================="
log "  提取完成"
log "=========================================="
log "总章节数: $TOTAL_CHAPTERS"
log "保存位置: $CHAPTER_LIST"
log "=========================================="
