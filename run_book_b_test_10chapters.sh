#!/bin/bash

# Book B Generation Test: 10 Chapters
# Based on Book A's 10-chapter memory database

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOVEL_DATA_DIR="/home/chris/Desktop/my_workspace/novel_data/04"
LOG_DIR="${NOVEL_DATA_DIR}/log"
OUTPUT_DIR="${NOVEL_DATA_DIR}/book_b_output"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/book_b_generation_10chapters_${TIMESTAMP}.log"

echo "=== Book B Generation Test: 10 Chapters ===" | tee "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Configuration:" | tee -a "${LOG_FILE}"
echo "  Template (Book A): novel_04_test" | tee -a "${LOG_FILE}"
echo "  Target (Book B): novel_04_book_b_test" | tee -a "${LOG_FILE}"
echo "  Chapters: 10" | tee -a "${LOG_FILE}"
echo "  Output: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "  Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run Book B generation
cd "${SCRIPT_DIR}/nanobot/skills/novel-workflow/scripts"

python3 generate_book_ab.py \
  --world-config "${NOVEL_DATA_DIR}/world_spec_novel_04.json" \
  --template-book-id "novel_04_test" \
  --template-canon-db-path "${NOVEL_DATA_DIR}/novel_DB/canon_novel_04_v2_test.db" \
  --template-neo4j-uri "bolt://localhost:7687" \
  --template-neo4j-user "neo4j" \
  --template-neo4j-pass "novel123" \
  --template-neo4j-database "neo4j" \
  --template-qdrant-url "http://localhost:6333" \
  --template-qdrant-collection "novel_04_assets_v2_test" \
  --template-semantic-search \
  --target-book-id "novel_04_book_b_test" \
  --target-canon-db-path "${NOVEL_DATA_DIR}/novel_DB/canon_novel_04_book_b_test.db" \
  --target-neo4j-uri "bolt://localhost:7689" \
  --target-neo4j-user "neo4j" \
  --target-neo4j-pass "novel123" \
  --target-neo4j-database "neo4j" \
  --target-qdrant-url "http://localhost:6333" \
  --target-qdrant-collection "novel_04_book_b_test" \
  --chapter-count 10 \
  --output-dir "${OUTPUT_DIR}" \
  --llm-config-path "${SCRIPT_DIR}/nanobot/skills/novel-workflow/llm_config_claude.json" \
  --log-dir "${LOG_DIR}" \
  --log-injections \
  --reference-top-k 20 \
  --consistency-policy warn_only \
  2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "=== Generation Complete ===" | tee -a "${LOG_FILE}"
echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Book B generation succeeded!" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    echo "Generated chapters:" | tee -a "${LOG_FILE}"
    ls -lh "${OUTPUT_DIR}"/*.md 2>/dev/null | tee -a "${LOG_FILE}" || echo "  No chapters found" | tee -a "${LOG_FILE}"
else
    echo "✗ Book B generation failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
fi

exit ${EXIT_CODE}
