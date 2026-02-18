#!/bin/bash
# Monitor Book A reprocessing progress

LOG_FILE=$(ls -t /home/chris/Desktop/my_workspace/novel_data/04/log/reprocess_full_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No log file found"
    exit 1
fi

echo "Monitoring: $LOG_FILE"
echo "============================================================"
echo ""

# Show configuration
echo "Configuration:"
grep -A 20 "Configuration:" "$LOG_FILE" | head -20
echo ""

# Show progress
echo "Progress:"
TOTAL=$(grep "Asset Files Found:" "$LOG_FILE" | awk '{print $NF}')
PROCESSED=$(grep -c "Qdrant write complete" "$LOG_FILE")
echo "  Processed: $PROCESSED / $TOTAL chapters"
if [ "$TOTAL" -gt 0 ]; then
    PERCENT=$(echo "scale=2; $PROCESSED * 100 / $TOTAL" | bc)
    echo "  Progress: ${PERCENT}%"
fi
echo ""

# Show recent chapters
echo "Recent chapters:"
tail -20 "$LOG_FILE" | grep "Qdrant write complete" | tail -5
echo ""

# Show errors
ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE")
echo "Errors: $ERROR_COUNT"
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "Recent errors:"
    grep "ERROR" "$LOG_FILE" | tail -3
fi
echo ""

# Show last 10 lines
echo "Last 10 lines:"
tail -10 "$LOG_FILE"
