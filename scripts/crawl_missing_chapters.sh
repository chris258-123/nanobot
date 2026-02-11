#!/bin/bash
# 补充爬取第9页缺失的章节

set -e

OUTPUT_DIR="/home/chris/Desktop/my_workspace/novel_data"
PLAYWRIGHT_SCRIPT="/home/chris/Desktop/my_workspace/nanobot/scripts/crawl-chapter-playwright.js"

GREEN='\033[0;32m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

# 缺失的章节列表
MISSING_CHAPTERS=(
    "/biqu7034/10935372.html"
    "/biqu7034/26957715.html"
    "/biqu7034/26957720.html"
    "/biqu7034/37494030.html"
    "/biqu7034/37494031.html"
    "/biqu7034/37494032.html"
    "/biqu7034/37494033.html"
    "/biqu7034/37494034.html"
    "/biqu7034/6850606.html"
    "/biqu7034/6850613.html"
    "/biqu7034/6850620.html"
    "/biqu7034/6850626.html"
)

log "=========================================="
log "  补充爬取缺失章节"
log "=========================================="
log "总计: ${#MISSING_CHAPTERS[@]} 章"
log ""

SUCCESS=0
FAILED=0
INDEX=1

for chapter_url in "${MISSING_CHAPTERS[@]}"; do
    FULL_URL="https://www.22biqu.com${chapter_url}"
    TEMP_FILE="${OUTPUT_DIR}/temp_missing_${INDEX}.md"

    log "[$INDEX/${#MISSING_CHAPTERS[@]}] 爬取: $FULL_URL"

    if node "$PLAYWRIGHT_SCRIPT" "$FULL_URL" "$TEMP_FILE" 2>&1 | grep -q "✓"; then
        CHAPTER_TITLE=$(head -1 "$TEMP_FILE" | sed 's/^# //' | sed 's/[<>:"/\\|?*]/_/g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        if [ -z "$CHAPTER_TITLE" ] || [ "$CHAPTER_TITLE" = "未找到标题" ]; then
            CHAPTER_TITLE="补充章节_${INDEX}"
        fi

        FINAL_FILE="${OUTPUT_DIR}/${CHAPTER_TITLE}.md"
        mv "$TEMP_FILE" "$FINAL_FILE"

        log "  ✓ 已保存: ${CHAPTER_TITLE}.md"
        SUCCESS=$((SUCCESS + 1))
    else
        log "  ✗ 失败"
        FAILED=$((FAILED + 1))
        rm -f "$TEMP_FILE"
    fi

    INDEX=$((INDEX + 1))
    sleep 1
done

log ""
log "=========================================="
log "  补充爬取完成"
log "=========================================="
log "成功: $SUCCESS 章"
log "失败: $FAILED 章"
log "=========================================="
