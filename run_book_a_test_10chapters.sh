#!/usr/bin/env bash
set -euo pipefail

# Test Script: Reprocess first 10 chapters of Book A with 8-asset extraction

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOVEL_DATA_DIR="/home/chris/Desktop/my_workspace/novel_data/04"
CHAPTER_DIR="${NOVEL_DATA_DIR}/novel_assets"  # Asset files location
LOG_DIR="${NOVEL_DATA_DIR}/log"
DB_DIR="${NOVEL_DATA_DIR}/novel_DB"

# Database paths
CANON_DB_PATH="${DB_DIR}/canon_novel_04_v2_test.db"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASS="novel123"
NEO4J_DATABASE="neo4j"
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION="novel_04_assets_v2_test"

# LLM config
LLM_CONFIG="${SCRIPT_DIR}/nanobot/skills/novel-workflow/llm_config_claude.json"

# Test parameters
BOOK_ID="novel_04_test"
TEST_CHAPTERS=10

# Create log directory
mkdir -p "${LOG_DIR}"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/test_10chapters_${TIMESTAMP}.log"

echo "=== Book A Test: First 10 Chapters with 8-Asset Extraction ===" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Configuration:" | tee -a "${LOG_FILE}"
echo "  Book ID: ${BOOK_ID}" | tee -a "${LOG_FILE}"
echo "  Chapters: First ${TEST_CHAPTERS}" | tee -a "${LOG_FILE}"
echo "  Canon DB: ${CANON_DB_PATH}" | tee -a "${LOG_FILE}"
echo "  Neo4j: ${NEO4J_URI} / ${NEO4J_DATABASE}" | tee -a "${LOG_FILE}"
echo "  Qdrant: ${QDRANT_URL} / ${QDRANT_COLLECTION}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Remove old test Canon DB
if [ -f "${CANON_DB_PATH}" ]; then
    echo "Removing old test Canon DB..." | tee -a "${LOG_FILE}"
    rm "${CANON_DB_PATH}"
fi

# Reset Neo4j (delete all nodes and relationships)
echo "Resetting Neo4j database..." | tee -a "${LOG_FILE}"
docker exec nanobot-neo4j-1 cypher-shell -u "${NEO4J_USER}" -p "${NEO4J_PASS}" \
    "MATCH (n) DETACH DELETE n" 2>&1 | tee -a "${LOG_FILE}" || true

# Reset Qdrant collection
echo "Resetting Qdrant collection..." | tee -a "${LOG_FILE}"
curl -X DELETE "${QDRANT_URL}/collections/${QDRANT_COLLECTION}" 2>&1 | tee -a "${LOG_FILE}" || true
curl -X PUT "${QDRANT_URL}/collections/${QDRANT_COLLECTION}" \
    -H "Content-Type: application/json" \
    -d '{
        "vectors": {
            "size": 1024,
            "distance": "Cosine"
        }
    }' 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Starting test processing..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Disable proxy
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

# Enable offline mode for embedding model
export HF_HUB_OFFLINE=1

# Run reprocessing for first 10 chapters
cd "${SCRIPT_DIR}/nanobot/skills/novel-workflow/scripts"

python3 reprocess_all.py \
    --mode delta \
    --book-id "${BOOK_ID}" \
    --asset-dir "${CHAPTER_DIR}" \
    --llm-config "${LLM_CONFIG}" \
    --canon-db-path "${CANON_DB_PATH}" \
    --neo4j-uri "${NEO4J_URI}" \
    --neo4j-user "${NEO4J_USER}" \
    --neo4j-pass "${NEO4J_PASS}" \
    --neo4j-database "${NEO4J_DATABASE}" \
    --qdrant-url "${QDRANT_URL}" \
    --qdrant-collection "${QDRANT_COLLECTION}" \
    --embedding-model chinese-large \
    --reset-canon \
    --reset-neo4j \
    --max-chapters ${TEST_CHAPTERS} \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "${LOG_FILE}"
echo "=== Test Complete ===" | tee -a "${LOG_FILE}"
echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Test processing succeeded!" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    echo "Verifying asset types in Qdrant..." | tee -a "${LOG_FILE}"

    # Check what asset types were created
    python3 << 'EOF' | tee -a "${LOG_FILE}"
import httpx

qdrant_url = "http://localhost:6333"
collection = "novel_04_assets_v2_test"

try:
    response = httpx.post(
        f"{qdrant_url}/collections/{collection}/points/scroll",
        json={"limit": 100, "with_payload": ["asset_type"], "with_vector": False},
        timeout=20.0,
        trust_env=False
    )

    points = response.json()["result"]["points"]
    asset_types = {}
    for point in points:
        asset_type = point.get("payload", {}).get("asset_type", "unknown")
        asset_types[asset_type] = asset_types.get(asset_type, 0) + 1

    print("\n✓ Asset types found in Qdrant:")
    for asset_type, count in sorted(asset_types.items()):
        print(f"  - {asset_type}: {count}")

    # Check if we have the 8 asset types
    expected = ["plot_beat", "character_card", "conflict", "setting", "theme", "pov", "tone", "style"]
    found = [at for at in expected if at in asset_types]
    missing = [at for at in expected if at not in asset_types]

    print(f"\n✓ Found {len(found)}/8 expected asset types")
    if missing:
        print(f"⚠ Missing: {', '.join(missing)}")

    # Check for Chinese names in fact_digest
    response = httpx.post(
        f"{qdrant_url}/collections/{collection}/points/scroll",
        json={
            "filter": {"must": [{"key": "asset_type", "match": {"value": "fact_digest"}}]},
            "limit": 3,
            "with_payload": True,
            "with_vector": False
        },
        timeout=20.0,
        trust_env=False
    )

    points = response.json()["result"]["points"]
    print("\n✓ Sample fact_digest (checking for Chinese names):")
    for point in points:
        text = point.get("payload", {}).get("text", "")
        print(f"  {text[:80]}...")

except Exception as e:
    print(f"✗ Error checking Qdrant: {e}")
EOF

else
    echo "✗ Test processing failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
fi

exit ${EXIT_CODE}
