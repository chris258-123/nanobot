#!/bin/bash
# Full reprocessing script for Book A (novel_04)
# Generated: 2026-02-18

set -e  # Exit on error

# Disable proxy to avoid interference with model downloads
unset ALL_PROXY
unset all_proxy
unset HTTP_PROXY
unset http_proxy
unset HTTPS_PROXY
unset https_proxy

# Enable offline mode for HuggingFace (use cached models)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Configuration
BOOK_ID="novel_04"
ASSET_DIR="/home/chris/Desktop/my_workspace/novel_data/04/novel_assets"
DB_DIR="/home/chris/Desktop/my_workspace/novel_data/04/novel_DB"
LOG_DIR="/home/chris/Desktop/my_workspace/novel_data/04/log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/reprocess_full_${TIMESTAMP}.log"

# Service configuration
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION="novel_04_assets_v2"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASS="novel123"
NEO4J_DATABASE="neo4j"
CANON_DB_PATH="${DB_DIR}/canon_novel_04_v2.db"

# LLM configuration
LLM_CONFIG="/home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config_claude.json"

# Embedding configuration
EMBEDDING_MODEL="chinese-large"

# Processing configuration
FROM_CHAPTER="0001"
LLM_MAX_TOKENS=4096
CONTEXT_STATE_LIMIT=30
CONTEXT_RELATION_LIMIT=30
CONTEXT_THREAD_LIMIT=20

# Create directories if needed
mkdir -p "${LOG_DIR}"
mkdir -p "${DB_DIR}"

# Log header
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Book A Full Reprocessing - $(date)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Log configuration
echo "Configuration:" | tee -a "${LOG_FILE}"
echo "  Book ID: ${BOOK_ID}" | tee -a "${LOG_FILE}"
echo "  Asset Directory: ${ASSET_DIR}" | tee -a "${LOG_FILE}"
echo "  Database Directory: ${DB_DIR}" | tee -a "${LOG_FILE}"
echo "  Log File: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Service Endpoints:" | tee -a "${LOG_FILE}"
echo "  Qdrant URL: ${QDRANT_URL}" | tee -a "${LOG_FILE}"
echo "  Qdrant Collection: ${QDRANT_COLLECTION}" | tee -a "${LOG_FILE}"
echo "  Neo4j URI: ${NEO4J_URI}" | tee -a "${LOG_FILE}"
echo "  Neo4j User: ${NEO4J_USER}" | tee -a "${LOG_FILE}"
echo "  Neo4j Database: ${NEO4J_DATABASE}" | tee -a "${LOG_FILE}"
echo "  Canon DB Path: ${CANON_DB_PATH}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Embedding Configuration:" | tee -a "${LOG_FILE}"
echo "  Model: ${EMBEDDING_MODEL}" | tee -a "${LOG_FILE}"
echo "  Integrated: Yes (embeddings generated during processing)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Processing Configuration:" | tee -a "${LOG_FILE}"
echo "  From Chapter: ${FROM_CHAPTER}" | tee -a "${LOG_FILE}"
echo "  LLM Max Tokens: ${LLM_MAX_TOKENS}" | tee -a "${LOG_FILE}"
echo "  Context Limits: state=${CONTEXT_STATE_LIMIT}, relation=${CONTEXT_RELATION_LIMIT}, thread=${CONTEXT_THREAD_LIMIT}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Count asset files
ASSET_COUNT=$(ls "${ASSET_DIR}"/*.json 2>/dev/null | wc -l)
echo "Asset Files Found: ${ASSET_COUNT}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Starting reprocessing at $(date)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run reprocessing
cd /home/chris/Desktop/my_workspace/nanobot

python nanobot/skills/novel-workflow/scripts/reprocess_all.py \
  --mode llm \
  --book-id "${BOOK_ID}" \
  --asset-dir "${ASSET_DIR}" \
  --from-chapter "${FROM_CHAPTER}" \
  --llm-config "${LLM_CONFIG}" \
  --canon-db-path "${CANON_DB_PATH}" \
  --neo4j-uri "${NEO4J_URI}" \
  --neo4j-user "${NEO4J_USER}" \
  --neo4j-pass "${NEO4J_PASS}" \
  --neo4j-database "${NEO4J_DATABASE}" \
  --qdrant-url "${QDRANT_URL}" \
  --qdrant-collection "${QDRANT_COLLECTION}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --llm-max-tokens "${LLM_MAX_TOKENS}" \
  --context-state-limit "${CONTEXT_STATE_LIMIT}" \
  --context-relation-limit "${CONTEXT_RELATION_LIMIT}" \
  --context-thread-limit "${CONTEXT_THREAD_LIMIT}" \
  --reset-canon \
  --reset-neo4j \
  --reset-qdrant \
  --log-file "${LOG_FILE}"

EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Reprocessing completed at $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Processing completed successfully" | tee -a "${LOG_FILE}"
else
    echo "✗ Processing failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
fi

exit ${EXIT_CODE}
