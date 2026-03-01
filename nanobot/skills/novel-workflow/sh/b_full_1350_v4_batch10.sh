#!/usr/bin/env bash
set -euo pipefail

unset ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG=/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/b_full_1350_v4_batch10.log
mkdir -p "$(dirname "$LOG")"

echo "[$(date +"%F %T")] ==== START full rerun book=novel_04_b_full_1350_v4 range=1-1350 batch=10 ====" | tee -a "$LOG"

for s in $(seq 1 10 1350); do
  e=$((s+9)); [ $e -gt 1350 ] && e=1350
  ok=0

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
      --temperature 0.63 \
      --plan-max-tokens 10000 \
      --chapter-max-tokens 10000 \
      --resume \
      --commit-memory \
      --consistency-policy warn_only \
      --continuity-mode strict_gate \
      --continuity-retry 4 \
      --continuity-window 12 \
      --continuity-min-entities 3 \
      --continuity-min-open-threads 0 \
      --continuity-min-chars 2600 \
      --continuity-max-opening-overlap 0.72 \
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
      --batch-size 10 \
      --book-total-chapters 1350 \
      --freeze-published-blueprint \
      --blueprint-template-source none \
      --enforce-chinese-on-injection \
      --enforce-chinese-on-commit \
      --enforce-chinese-fields rule,status,trait,goal,secret,state \
      --log-dir /home/chris/Desktop/my_workspace/novel_data/04/new_book/log \
      --log-injections 2>&1 | tee -a "$LOG" && ok=1 && break

    echo "[$(date +"%F %T")] Retry after 20s..." | tee -a "$LOG"
    sleep 20
  done

  if [ $ok -ne 1 ]; then
    echo "[$(date +"%F %T")] STOP on batch $(printf "%04d" "$s")-$(printf "%04d" "$e")" | tee -a "$LOG"
    exit 1
  fi
done

echo "[$(date +"%F %T")] ==== FINISH full rerun ====" | tee -a "$LOG"
