---
name: vimax-bridge
description: Bridge markdown novel chapters to per-chapter anime shorts using an external ViMax repository.
metadata: {"nanobot":{"emoji":"🎬","os":["darwin","linux"],"requires":{"bins":["python3"]}}}
---

# ViMax Bridge Skill

Use this skill when you want to convert local chapter markdown files into per-chapter anime short videos.

## What this skill does

1. Reads chapter `.md` files from a directory.
2. Builds per-chapter scene plans (`scene_plan.json`) via:
   - optional API LLM planner, or
   - local heuristic fallback.
3. Converts scene plans to screenplay text and feeds ViMax `Script2VideoPipeline`.
4. Saves one video per chapter plus full run metadata/logs.

## Prerequisites

- Clone ViMax locally (example path): `/tmp/vimax_repo_1588772`
- Prepare ViMax dependencies in its own environment (ViMax requires Python 3.12+ and its deps)
- Prepare ViMax config yaml (usually `configs/script2video.yaml`)
- Optional: prepare API planner config (`llm_config.json`) in OpenAI-compatible format

Example `llm_config.json`:

```json
{
  "type": "custom",
  "url": "https://api.deepseek.com/v1/chat/completions",
  "model": "deepseek-chat",
  "api_key": "YOUR_API_KEY"
}
```

## Run command

```bash
python nanobot/skills/vimax-bridge/scripts/novel_md_to_anime.py \
  --chapter-dir /path/to/chapters_md \
  --output-dir /path/to/anime_output \
  --vimax-repo /tmp/vimax_repo_1588772 \
  --vimax-config /tmp/vimax_repo_1588772/configs/script2video.yaml \
  --llm-config /path/to/llm_config.json \
  --style "Anime Style" \
  --resume
```

If ViMax uses a separate venv Python:

```bash
python nanobot/skills/vimax-bridge/scripts/novel_md_to_anime.py \
  ... \
  --vimax-python /path/to/vimax_venv/bin/python
```

## Useful flags

- `--dry-run`: only generate chapter manifest / scene plans / screenplay, skip ViMax rendering
- `--max-chapters N`: run small batch first
- `--max-retries N`: retry failed chapters
- `--glob "*.md"`: customize chapter filename pattern

## Output layout

- `<output_dir>/chapter_manifest.json`
- `<output_dir>/run_report.json`
- `<output_dir>/failed_chapters.json` (only when failures exist)
- `<output_dir>/videos/chapter_0001.mp4`
- `<output_dir>/runs/0001/scene_plan.json`
- `<output_dir>/runs/0001/vimax_input.json`
- `<output_dir>/runs/0001/run.log`

## Notes

- Bridge mode is intentionally lightweight: ViMax remains an external repo/process.
- Script currently supports `--mode script2video` only.
- If LLM planning fails, the script automatically falls back to heuristic scene planning.
