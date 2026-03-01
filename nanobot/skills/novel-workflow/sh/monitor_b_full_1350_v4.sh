#!/usr/bin/env bash
set -euo pipefail

TARGET_BOOK_ID="novel_04_b_full_1350_v4"
SESSION_NAME="b_full_v4_full0157"
BATCH_SCRIPT="/home/chris/Desktop/my_workspace/nanobot/nanobot/skills/novel-workflow/sh/b_full_1350_v4_full_from0157_batch5.sh"
MAIN_LOG="/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/b_full_1350_v4_full_from0157_batch5.log"
MONITOR_LOG="/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/b_full_1350_v4_monitor_full0157.log"
GEN_LOG_ROOT="/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/generate_book_ab"
RUN_REPORT="/home/chris/Desktop/my_workspace/novel_data/04/new_book/full_1350_v4/novel_04_b_full_1350_v4_run_report.json"
OUTPUT_DIR="/home/chris/Desktop/my_workspace/novel_data/04/new_book/full_1350_v4"
SQLITE_DB="/home/chris/Desktop/my_workspace/novel_data/04/new_book/canon_novel_04_b_full_1350_v4.db"
NEO4J_CONTAINER="neo4j-b-full-1350-v4"
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION="novel_04_b_full_1350_v4_assets"
LOCK_DIR="/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/.b_full_1350_v4_monitor_full0157.lockdir"
DEFAULT_START=157
TOTAL_CHAPTERS=1350
BATCH_STEP=5

mkdir -p "$(dirname "$MAIN_LOG")" "$(dirname "$MONITOR_LOG")"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  stale_pid=""
  if [[ -f "$LOCK_DIR/pid" ]]; then
    stale_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  fi
  if [[ -n "$stale_pid" ]] && ! kill -0 "$stale_pid" 2>/dev/null; then
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null || true
  fi
fi

if [[ ! -d "$LOCK_DIR" ]]; then
  echo "[$(date +"%F %T")] SKIP monitor already running" >> "$MONITOR_LOG"
  exit 0
fi

echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

log() {
  echo "[$(date +"%F %T")] $*" | tee -a "$MONITOR_LOG"
}

latest_progress() {
  if [[ -f "$MAIN_LOG" ]]; then
    local line
    line=$(grep -E "\[[0-9]+/[0-9]+[[:space:]]+[0-9]+\.[0-9]+%.*last=[0-9]+" "$MAIN_LOG" | tail -n 1 || true)
    [[ -n "$line" ]] && log "PROGRESS $line"
  fi
}

compute_resume_start() {
  python - "$OUTPUT_DIR" "$TARGET_BOOK_ID" "$DEFAULT_START" "$TOTAL_CHAPTERS" <<'PYIN'
import glob
import os
import re
import sys

output_dir, target_book_id, default_start, total_chapters = sys.argv[1:5]
default_start = int(default_start)
total_chapters = int(total_chapters)
pattern = os.path.join(output_dir, f"{target_book_id}_chapter_*.md")
chapter_files = glob.glob(pattern)
if not chapter_files:
    print(f"{default_start}\tfallback_default")
    raise SystemExit(0)

latest = 0
for path in chapter_files:
    m = re.search(r"_chapter_(\d{4})\.md$", os.path.basename(path))
    if not m:
        continue
    latest = max(latest, int(m.group(1)))

if latest <= 0:
    print(f"{default_start}\tfallback_default")
elif latest >= total_chapters:
    print(f"{total_chapters + 1}\tcompleted_from_md_{latest:04d}")
else:
    print(f"{latest + 1}\tlatest_md_{latest:04d}")
PYIN
}

continuity_probe() {
  local latest_run
  latest_run=$(ls -td "$GEN_LOG_ROOT/${TARGET_BOOK_ID}_"* 2>/dev/null | head -n 1 || true)
  if [[ -z "$latest_run" ]]; then
    log "CONTINUITY no_run_dir"
    return
  fi

  while IFS= read -r line; do
    [[ -n "$line" ]] && log "$line"
  done < <(python - "$latest_run" <<'PYIN'
import glob
import json
import os
import sys

run_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(run_dir, "chapters", "*_pre_generation_injection.json")))
if not files:
    print("CONTINUITY no_injection_files")
    raise SystemExit(0)

def chapter_no(path, data):
    if isinstance(data, dict) and data.get("chapter_no"):
        try:
            return int(data["chapter_no"])
        except Exception:
            pass
    base = os.path.basename(path)
    return int(base.split("_", 1)[0])

latest_file = files[-1]
with open(latest_file, "r", encoding="utf-8") as f:
    latest = json.load(f)

latest_ch = chapter_no(latest_file, latest)
open_with = latest.get("current_chapter_open_with")
carry = latest.get("carry_over_required_for_gate") or latest.get("previous_chapter_carry_over")
open_count = len(open_with) if isinstance(open_with, list) else (1 if open_with else 0)
carry_count = len(carry) if isinstance(carry, list) else (1 if carry else 0)
source = latest.get("resolved_open_with_source")
cross_source = latest.get("cross_batch_anchor_source")

print(
    "CONTINUITY latest_chapter={:04d} open_with_items={} carry_items={} resolved_open_with_source={} cross_batch_anchor_source={}"
    .format(latest_ch, open_count, carry_count, source, cross_source)
)

boundary_rows = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not bool(data.get("is_batch_first")):
        continue
    ch = chapter_no(path, data)
    ow = data.get("current_chapter_open_with")
    co = data.get("carry_over_required_for_gate") or data.get("previous_chapter_carry_over")
    owc = len(ow) if isinstance(ow, list) else (1 if ow else 0)
    coc = len(co) if isinstance(co, list) else (1 if co else 0)
    src = data.get("resolved_open_with_source")
    ok = 1 if (owc > 0 and coc > 0) else 0
    boundary_rows.append((ch, ok, owc, coc, src))

if not boundary_rows:
    print("CONTINUITY boundary_check none_yet")
else:
    for ch, ok, owc, coc, src in boundary_rows[-5:]:
        print(
            "CONTINUITY boundary chapter={:04d} ok={} open_with_items={} carry_items={} source={}"
            .format(ch, ok, owc, coc, src)
        )
PYIN
)
}

memory_probe() {
  while IFS= read -r line; do
    [[ -n "$line" ]] && log "$line"
  done < <(python - "$SQLITE_DB" "$RUN_REPORT" "$QDRANT_URL" "$QDRANT_COLLECTION" <<'PYIN'
import json
import sqlite3
import sys
import urllib.request

sqlite_db, run_report, qdrant_url, collection = sys.argv[1:5]

latest_done = None
all_done = None
failed_commits = None
try:
    conn = sqlite3.connect(sqlite_db)
    cur = conn.cursor()
    all_done = cur.execute("SELECT COUNT(*) FROM commit_log WHERE status='ALL_DONE'").fetchone()[0]
    failed_commits = cur.execute("SELECT COUNT(*) FROM commit_log WHERE status='FAILED'").fetchone()[0]
    latest_done = cur.execute("SELECT MAX(chapter_no) FROM commit_log WHERE status='ALL_DONE'").fetchone()[0]
    conn.close()
    print(f"MEMORY sqlite all_done={all_done} failed={failed_commits} latest_done={latest_done}")
except Exception as exc:
    print(f"MEMORY sqlite error={exc}")

try:
    with urllib.request.urlopen(f"{qdrant_url}/collections/{collection}", timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    points = data.get("result", {}).get("points_count")
    print(f"MEMORY qdrant collection={collection} points={points}")
except Exception as exc:
    print(f"MEMORY qdrant error={exc}")

try:
    with open(run_report, "r", encoding="utf-8") as f:
        report = json.load(f)
    print(
        "MEMORY report ok={} failed={} memory_failed={} generated_at={}".format(
            report.get("ok"),
            report.get("failed"),
            report.get("memory_failed"),
            report.get("generated_at"),
        )
    )
except Exception as exc:
    print(f"MEMORY report error={exc}")
PYIN
)

  local neo4j_raw
  neo4j_raw=$(docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p novel123 "MATCH (c:Chapter {book_id:'${TARGET_BOOK_ID}'}) RETURN count(c) AS chapters, max(c.chapter_no) AS latest;" 2>/dev/null | tail -n +2 | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' || true)
  if [[ -n "$neo4j_raw" ]]; then
    log "MEMORY neo4j ${neo4j_raw}"
  else
    log "MEMORY neo4j unavailable"
  fi
}

log "MONITOR tick start"

resume_info="$(compute_resume_start)"
resume_start="$(echo "$resume_info" | awk '{print $1}')"
resume_source="$(echo "$resume_info" | awk '{print $2}')"
if [[ -z "${resume_start:-}" ]]; then
  resume_start="$DEFAULT_START"
  resume_source="resume_parse_error"
fi
log "RESUME target_start=$(printf "%04d" "$resume_start") source=${resume_source}"

if (( resume_start > TOTAL_CHAPTERS )); then
  log "STATUS finished_by_output latest>=${TOTAL_CHAPTERS}"
  latest_progress
  continuity_probe
  memory_probe
  exit 0
fi

if [[ -f "$MAIN_LOG" ]] && grep -q "==== FINISH full rerun" "$MAIN_LOG"; then
  log "STATUS finished"
  latest_progress
  continuity_probe
  memory_probe
  exit 0
fi

run_match=$(pgrep -af "b_full_1350_v4_full_from0157_batch5.sh|generate_book_ab.py --target-book-id ${TARGET_BOOK_ID}" || true)
if [[ -n "$run_match" ]]; then
  log "STATUS running"
  log "PIDS $(echo "$run_match" | tr '\n' ';' | sed 's/;$/ /')"
  running_start="$(echo "$run_match" | sed -n 's/.*--start-chapter \([0-9]\+\).*/\1/p' | sort -n | tail -n 1 || true)"
  if [[ -n "$running_start" ]] && (( running_start + BATCH_STEP < resume_start )); then
    log "WARN stale_restart_detected running_start=$(printf "%04d" "$running_start") expected_start>=$(printf "%04d" "$resume_start")"
  fi
else
  log "STATUS stopped_unfinished"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    window_name="auto_resume_$(date +"%H%M%S")"
    tmux new-window -d -t "$SESSION_NAME" -n "$window_name" "START=$resume_start END=$TOTAL_CHAPTERS STEP=$BATCH_STEP bash $BATCH_SCRIPT"
    log "RESTART_TRIGGERED existing_session window=${window_name} start=$(printf "%04d" "$resume_start")"
  else
    tmux new-session -d -s "$SESSION_NAME" "START=$resume_start END=$TOTAL_CHAPTERS STEP=$BATCH_STEP bash $BATCH_SCRIPT"
    log "RESTART_TRIGGERED new_session=${SESSION_NAME} start=$(printf "%04d" "$resume_start")"
  fi
fi

latest_progress
continuity_probe
memory_probe

log "MONITOR tick end"
