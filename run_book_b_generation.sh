#!/usr/bin/env bash
set -euo pipefail

# Book B Generation Script - A/B Separation Architecture
# Reads from Book A (novel_04) and generates Book B with its own three-tier memory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOVEL_DATA_DIR="/home/chris/Desktop/my_workspace/novel_data/04"
LOG_DIR="${NOVEL_DATA_DIR}/log"
OUTPUT_DIR="${NOVEL_DATA_DIR}/new_book"
WORLD_SPEC="${SCRIPT_DIR}/nanobot/skills/novel-workflow/templates/world_spec.example.json"
LLM_CONFIG="${SCRIPT_DIR}/nanobot/skills/novel-workflow/llm_config_claude.json"

# Book A (template) - read-only
TEMPLATE_BOOK_ID="novel_04"
TEMPLATE_CANON_DB="${NOVEL_DATA_DIR}/novel_DB/canon_novel_04_v2.db"
TEMPLATE_NEO4J_URI="bolt://localhost:7687"
TEMPLATE_NEO4J_USER="neo4j"
TEMPLATE_NEO4J_PASS="novel123"
TEMPLATE_NEO4J_DATABASE="neo4j"
TEMPLATE_QDRANT_URL="http://localhost:6333"
TEMPLATE_QDRANT_COLLECTION="novel_04_assets_v2"

# Book B (target) - write
TARGET_BOOK_ID="novel_04_b"
TARGET_CANON_DB="${OUTPUT_DIR}/canon_novel_04_b.db"
TARGET_NEO4J_URI="bolt://localhost:7689"  # Dedicated Neo4j container for Book B
TARGET_NEO4J_USER="neo4j"
TARGET_NEO4J_PASS="novel123"
TARGET_NEO4J_DATABASE="neo4j"
TARGET_QDRANT_URL="http://localhost:6333"
TARGET_QDRANT_COLLECTION="novel_04_b_assets"

# Generation parameters
CHAPTER_COUNT=5  # Start with 5 chapters for testing
START_CHAPTER=1
TEMPERATURE=0.8
CHAPTER_MIN_CHARS=2600
CHAPTER_MAX_CHARS=3800
REFERENCE_TOP_K=8

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/book_b_generation_${TIMESTAMP}.log"

echo "=== Book B Generation - A/B Separation Architecture ===" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Book A (Template - Read Only):" | tee -a "${LOG_FILE}"
echo "  Book ID: ${TEMPLATE_BOOK_ID}" | tee -a "${LOG_FILE}"
echo "  Canon DB: ${TEMPLATE_CANON_DB}" | tee -a "${LOG_FILE}"
echo "  Neo4j: ${TEMPLATE_NEO4J_URI} / ${TEMPLATE_NEO4J_DATABASE}" | tee -a "${LOG_FILE}"
echo "  Qdrant: ${TEMPLATE_QDRANT_URL} / ${TEMPLATE_QDRANT_COLLECTION}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Book B (Target - Write):" | tee -a "${LOG_FILE}"
echo "  Book ID: ${TARGET_BOOK_ID}" | tee -a "${LOG_FILE}"
echo "  Canon DB: ${TARGET_CANON_DB}" | tee -a "${LOG_FILE}"
echo "  Neo4j: ${TARGET_NEO4J_URI} / ${TARGET_NEO4J_DATABASE}" | tee -a "${LOG_FILE}"
echo "  Qdrant: ${TARGET_QDRANT_URL} / ${TARGET_QDRANT_COLLECTION}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Generation Parameters:" | tee -a "${LOG_FILE}"
echo "  World Spec: ${WORLD_SPEC}" | tee -a "${LOG_FILE}"
echo "  Output Dir: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "  Chapter Count: ${CHAPTER_COUNT}" | tee -a "${LOG_FILE}"
echo "  Temperature: ${TEMPERATURE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Verify Book A data exists
if [ ! -f "${TEMPLATE_CANON_DB}" ]; then
    echo "ERROR: Book A Canon DB not found: ${TEMPLATE_CANON_DB}" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${WORLD_SPEC}" ]; then
    echo "ERROR: World spec not found: ${WORLD_SPEC}" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${LLM_CONFIG}" ]; then
    echo "ERROR: LLM config not found: ${LLM_CONFIG}" | tee -a "${LOG_FILE}"
    exit 1
fi

# Create Neo4j database for Book B (if it doesn't exist)
echo "Creating Neo4j database for Book B..." | tee -a "${LOG_FILE}"
cypher-shell -a "${TARGET_NEO4J_URI}" -u "${TARGET_NEO4J_USER}" -p "${TARGET_NEO4J_PASS}" \
    "CREATE DATABASE ${TARGET_NEO4J_DATABASE} IF NOT EXISTS" 2>&1 | tee -a "${LOG_FILE}" || true

# Create Qdrant collection for Book B
echo "Creating Qdrant collection for Book B..." | tee -a "${LOG_FILE}"
curl -X DELETE "${TARGET_QDRANT_URL}/collections/${TARGET_QDRANT_COLLECTION}" 2>&1 | tee -a "${LOG_FILE}" || true
curl -X PUT "${TARGET_QDRANT_URL}/collections/${TARGET_QDRANT_COLLECTION}" \
    -H "Content-Type: application/json" \
    -d '{
        "vectors": {
            "size": 1024,
            "distance": "Cosine"
        }
    }' 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Starting Book B generation..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Disable proxy to avoid network issues
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

# Run generation
cd "${SCRIPT_DIR}/nanobot/skills/novel-workflow/scripts"

python3 generate_book_ab.py \
    --target-book-id "${TARGET_BOOK_ID}" \
    --template-book-id "${TEMPLATE_BOOK_ID}" \
    --world-config "${WORLD_SPEC}" \
    --chapter-count ${CHAPTER_COUNT} \
    --start-chapter ${START_CHAPTER} \
    --output-dir "${OUTPUT_DIR}" \
    --llm-config "${LLM_CONFIG}" \
    --temperature ${TEMPERATURE} \
    --chapter-min-chars ${CHAPTER_MIN_CHARS} \
    --chapter-max-chars ${CHAPTER_MAX_CHARS} \
    --reference-top-k ${REFERENCE_TOP_K} \
    --template-canon-db-path "${TEMPLATE_CANON_DB}" \
    --template-neo4j-uri "${TEMPLATE_NEO4J_URI}" \
    --template-neo4j-user "${TEMPLATE_NEO4J_USER}" \
    --template-neo4j-pass "${TEMPLATE_NEO4J_PASS}" \
    --template-neo4j-database "${TEMPLATE_NEO4J_DATABASE}" \
    --template-qdrant-url "${TEMPLATE_QDRANT_URL}" \
    --template-qdrant-collection "${TEMPLATE_QDRANT_COLLECTION}" \
    --target-canon-db-path "${TARGET_CANON_DB}" \
    --target-neo4j-uri "${TARGET_NEO4J_URI}" \
    --target-neo4j-user "${TARGET_NEO4J_USER}" \
    --target-neo4j-pass "${TARGET_NEO4J_PASS}" \
    --target-neo4j-database "${TARGET_NEO4J_DATABASE}" \
    --target-qdrant-url "${TARGET_QDRANT_URL}" \
    --target-qdrant-collection "${TARGET_QDRANT_COLLECTION}" \
    --commit-memory \
    --enforce-isolation \
    --log-dir "${LOG_DIR}" \
    --consistency-policy strict_blocking \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "${LOG_FILE}"
echo "=== Generation Complete ===" | tee -a "${LOG_FILE}"
echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Book B generation succeeded!" | tee -a "${LOG_FILE}"
else
    echo "✗ Book B generation failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
fi

exit ${EXIT_CODE}
