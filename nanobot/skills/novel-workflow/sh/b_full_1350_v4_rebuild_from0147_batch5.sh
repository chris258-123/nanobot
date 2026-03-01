#!/usr/bin/env bash
set -euo pipefail

unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG=/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/b_full_1350_v4_rebuild_from0147_batch5.log
FAILED_BATCHES=/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/b_full_1350_v4_failed_batches_from0147.txt
FAILED_CHAPTERS=/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/b_full_1350_v4_failed_chapters_from0147.txt
RUN_REPORT=/home/chris/Desktop/my_workspace/novel_data/04/new_book/full_1350_v4/novel_04_b_full_1350_v4_run_report.json

mkdir -p "$(dirname "$LOG")"
: > "$FAILED_BATCHES"
: > "$FAILED_CHAPTERS"

append_unique() {
  local file="$1"
  local line="$2"
  if ! grep -Fqx "$line" "$file" 2>/dev/null; then
    echo "$line" >> "$file"
  fi
}

record_failed_chapters_for_batch() {
  local s="$1"
  local e="$2"
  local rows=""
  rows=$(python - "$RUN_REPORT" "$s" "$e" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
s = int(sys.argv[2])
e = int(sys.argv[3])
if not path.exists():
    sys.exit(0)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(0)
items = data.get("items")
if not isinstance(items, list):
    sys.exit(0)
for item in items:
    try:
        chapter_no = int(str(item.get("chapter_no")))
    except Exception:
        continue
    if chapter_no < s or chapter_no > e:
        continue
    if str(item.get("status") or "") != "failed":
        continue
    error = str(item.get("error") or "").replace("\n", " ").strip()
    print(f"{chapter_no:04d}\t{error}")
PY
)
  if [ -z "$rows" ]; then
    echo "[$(date +"%F %T")] WARN no failed chapter rows parsed for batch $(printf "%04d" "$s")-$(printf "%04d" "$e")" | tee -a "$LOG"
    return
  fi
  while IFS=$'\t' read -r chapter_no error_msg; do
    [ -z "${chapter_no:-}" ] && continue
    append_unique "$FAILED_CHAPTERS" "$chapter_no"
    if [ -n "${error_msg:-}" ]; then
      echo "[$(date +"%F %T")] FAIL chapter ${chapter_no}: ${error_msg}" | tee -a "$LOG"
    else
      echo "[$(date +"%F %T")] FAIL chapter ${chapter_no}: <empty error>" | tee -a "$LOG"
    fi
  done <<< "$rows"
}

total_batches=0
success_batches=0
failed_batches=0

echo "[$(date +"%F %T")] ==== START rebuild book=novel_04_b_full_1350_v4 range=0147-1350 batch=5 ====" | tee -a "$LOG"

for s in $(seq 147 5 1350); do
  total_batches=$((total_batches+1))
  e=$((s+4))
  [ $e -gt 1350 ] && e=1350
  ok=0
  overwrite_range="$(printf "%04d-%04d" "$s" "$e")"

  for a in 1 2 3; do
    echo "[$(date +"%F %T")] BATCH $(printf "%04d" "$s")-$(printf "%04d" "$e") attempt $a/3" | tee -a "$LOG"

    /home/chris/miniforge3/envs/nanobot/bin/python /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/scripts/generate_book_ab.py \
      --target-book-id novel_04_b_full_1350_v4 \
      --template-book-id novel_04_a_full \
      --world-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/templates/world_spec.updated.json \
      --chapter-count $((e-s+1)) \
      --start-chapter $s \
      --output-dir /home/chris/Desktop/my_workspace/novel_data/04/new_book/full_1350_v4 \
      --llm-config /home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/llm_config_claude.json \
      --temperature 0.62 \
      --plan-max-tokens 20000 \
      --chapter-max-tokens 10000 \
      --chapter-min-chars 2800 \
      --chapter-max-chars 4200 \
      --resume \
      --resume-overwrite-range "$overwrite_range" \
      --commit-memory \
      --consistency-policy warn_only \
      --continuity-mode strict_gate \
      --continuity-retry 4 \
      --continuity-window 12 \
      --continuity-min-entities 3 \
      --continuity-min-open-threads 0 \
      --continuity-min-chars 2600 \
      --continuity-max-opening-overlap 0.72 \
      --opening-rewrite-by-llm \
      --opening-rewrite-max-attempts 3 \
      --opening-rewrite-max-chars 900 \
      --batch-boundary-gate \
      --chapter-summary-style structured \
      --ending-style closure \
      --enforce-isolation \
      --template-semantic-search \
      --template-semantic-model chinese-large \
      --reference-top-k 12 \
      --llm-max-retries 3 \
      --llm-retry-backoff 3.0 \
      --llm-backoff-factor 2.0 \
      --llm-backoff-max 60.0 \
      --llm-retry-jitter 0.5 \
      --template-canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/novel_DB/canon_novel_04_a_full.db \
      --template-neo4j-uri bolt://localhost:7689 \
      --template-neo4j-user neo4j \
      --template-neo4j-pass novel123 \
      --template-neo4j-database neo4j \
      --template-qdrant-url http://localhost:6333 \
      --template-qdrant-collection novel_04_a_full_assets \
      --target-canon-db-path /home/chris/Desktop/my_workspace/novel_data/04/new_book/canon_novel_04_b_full_1350_v4.db \
      --target-neo4j-uri bolt://localhost:7695 \
      --target-neo4j-user neo4j \
      --target-neo4j-pass novel123 \
      --target-neo4j-database neo4j \
      --target-qdrant-url http://localhost:6333 \
      --target-qdrant-collection novel_04_b_full_1350_v4_assets \
      --blueprint-mode hierarchical \
      --stage-size 100 \
      --batch-size 5 \
      --book-total-chapters 1350 \
      --no-freeze-published-blueprint \
      --blueprint-template-source none \
      --enforce-chinese-on-injection \
      --enforce-chinese-on-commit \
      --enforce-chinese-fields rule,status,trait,goal,secret,state \
      --log-dir /home/chris/Desktop/my_workspace/novel_data/04/new_book/log \
      --log-injections 2>&1 | tee -a "$LOG" && ok=1 && break

    echo "[$(date +"%F %T")] Retry after 20s..." | tee -a "$LOG"
    sleep 20
  done

  if [ $ok -eq 1 ]; then
    success_batches=$((success_batches+1))
    continue
  fi

  failed_batches=$((failed_batches+1))
  batch_range="$(printf "%04d" "$s")-$(printf "%04d" "$e")"
  append_unique "$FAILED_BATCHES" "$batch_range"
  echo "[$(date +"%F %T")] SKIP failed batch $batch_range after 3 attempts; continue next batch" | tee -a "$LOG"
  record_failed_chapters_for_batch "$s" "$e"
done

echo "[$(date +"%F %T")] ==== FINISH rebuild (batch=5) ====" | tee -a "$LOG"
echo "[$(date +"%F %T")] SUMMARY total_batches=$total_batches success_batches=$success_batches failed_batches=$failed_batches" | tee -a "$LOG"
echo "[$(date +"%F %T")] failed_batch_list=$FAILED_BATCHES" | tee -a "$LOG"
echo "[$(date +"%F %T")] failed_chapter_list=$FAILED_CHAPTERS" | tee -a "$LOG"
