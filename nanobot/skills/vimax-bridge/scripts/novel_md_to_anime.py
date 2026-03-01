#!/usr/bin/env python3
"""Bridge script: convert chapter markdown files into per-chapter ViMax videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

_VIMAX_RUNNER_CODE = """
import asyncio
import json
import sys

from pipelines.script2video_pipeline import Script2VideoPipeline


async def _run(config_path: str, payload_path: str) -> str:
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    pipeline = Script2VideoPipeline.init_from_config(config_path=config_path)
    output_path = await pipeline(
        script=payload["script"],
        user_requirement=payload["user_requirement"],
        style=payload["style"],
    )
    return str(output_path)


if __name__ == "__main__":
    config_path = sys.argv[1]
    payload_path = sys.argv[2]
    video_path = asyncio.run(_run(config_path, payload_path))
    print(f"FINAL_VIDEO={video_path}")
""".strip()


@dataclass(slots=True)
class ChapterRecord:
    """One chapter parsed from markdown."""

    index: int
    chapter_no: str
    title: str
    body: str
    source_path: Path
    sha256: str


class LLMError(RuntimeError):
    """Raised when LLM scene planning fails."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_chapter_number(name: str) -> int | None:
    patterns = [
        r"第\s*(\d{1,6})\s*章",
        r"chapter[_\-\s]*(\d{1,6})",
        r"chap[_\-\s]*(\d{1,6})",
        r"(\d{1,6})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def chapter_sort_key(path: Path) -> tuple[int, int, str]:
    chapter_num = extract_chapter_number(path.stem)
    if chapter_num is None:
        return (1, 0, path.name)
    return (0, chapter_num, path.name)


def parse_markdown_chapter(path: Path, index: int) -> ChapterRecord:
    raw = read_text(path)
    title = ""
    body_lines: list[str] = []
    heading_seen = False

    for line in raw.splitlines():
        stripped = line.strip()
        if not heading_seen and stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            heading_seen = True
            continue
        body_lines.append(line)

    if not title:
        title = path.stem

    filtered_lines = []
    for line in body_lines:
        s = line.strip()
        if s in {"", "***", "---"}:
            filtered_lines.append("")
            continue
        if "上一章目录下一章" in s or "请收藏本站" in s:
            continue
        filtered_lines.append(line)

    body = "\n".join(filtered_lines)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    chapter_num = extract_chapter_number(path.name)
    chapter_no = f"{chapter_num:04d}" if chapter_num is not None else f"{index:04d}"

    return ChapterRecord(
        index=index,
        chapter_no=chapter_no,
        title=title,
        body=body,
        source_path=path,
        sha256=sha256_text(raw),
    )


def discover_chapters(chapter_dir: Path, glob_pattern: str, max_chapters: int | None) -> list[ChapterRecord]:
    files = sorted(chapter_dir.glob(glob_pattern), key=chapter_sort_key)
    chapters: list[ChapterRecord] = []
    for idx, path in enumerate(files, start=1):
        if not path.is_file():
            continue
        chapters.append(parse_markdown_chapter(path, index=idx))
        if max_chapters and len(chapters) >= max_chapters:
            break
    return chapters


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n..."


def heuristic_scene_plan(
    chapter: ChapterRecord,
    style: str,
    min_scenes: int,
    max_scenes: int,
) -> dict[str, Any]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chapter.body) if p.strip()]
    if not paragraphs:
        paragraphs = [chapter.title]

    if len(paragraphs) <= max_scenes:
        selected = paragraphs
    else:
        step = max(1, len(paragraphs) // max_scenes)
        selected = [paragraphs[i] for i in range(0, len(paragraphs), step)][:max_scenes]

    if len(selected) < min_scenes and len(paragraphs) > len(selected):
        for para in paragraphs:
            if para not in selected:
                selected.append(para)
            if len(selected) >= min_scenes:
                break

    selected = selected[:max_scenes]
    scene_count = max(1, len(selected))
    duration = max(3, min(8, round(45 / scene_count)))

    cameras = [
        "wide establishing shot",
        "medium tracking shot",
        "over-the-shoulder shot",
        "close-up reaction shot",
        "dynamic low-angle shot",
    ]

    scenes = []
    for idx, paragraph in enumerate(selected, start=1):
        prompt = truncate_text(paragraph.replace("\n", " "), 220)
        scenes.append(
            {
                "scene_id": f"S{idx:02d}",
                "prompt": prompt,
                "duration_s": duration,
                "camera": cameras[(idx - 1) % len(cameras)],
                "style": style,
            }
        )

    return {
        "chapter_no": chapter.chapter_no,
        "title": chapter.title,
        "source": "heuristic",
        "scenes": scenes,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])

    raise ValueError("Unterminated JSON object in response")


def load_llm_config(path: Path) -> dict[str, Any]:
    data = json.loads(read_text(path))

    if data.get("type") == "custom":
        url = data.get("url", "").strip()
        if not url:
            raise LLMError("llm_config custom mode requires non-empty 'url'")
        return {
            "url": url,
            "model": data.get("model", ""),
            "api_key": data.get("api_key", ""),
            "headers": {},
        }

    providers = data.get("providers")
    if isinstance(providers, dict):
        for provider_cfg in providers.values():
            if not isinstance(provider_cfg, dict):
                continue
            api_key = provider_cfg.get("apiKey") or provider_cfg.get("api_key")
            api_base = provider_cfg.get("apiBase") or provider_cfg.get("api_base")
            if not api_key or not api_base:
                continue
            url = api_base.rstrip("/")
            if not url.endswith("/chat/completions"):
                url = f"{url}/chat/completions"
            extra_headers = provider_cfg.get("extraHeaders") or provider_cfg.get("extra_headers") or {}
            headers = {k: str(v) for k, v in extra_headers.items() if v is not None}
            return {
                "url": url,
                "model": data.get("model", ""),
                "api_key": api_key,
                "headers": headers,
            }

    raise LLMError("Unsupported llm_config format")


def llm_scene_plan(
    chapter: ChapterRecord,
    llm_cfg: dict[str, Any],
    style: str,
    min_scenes: int,
    max_scenes: int,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    if not llm_cfg.get("url") or not llm_cfg.get("model") or not llm_cfg.get("api_key"):
        raise LLMError("llm_config must provide url/model/api_key")

    system_prompt = (
        "You are a screenplay planner. Output strict JSON only. "
        "Create coherent anime scenes from chapter text."
    )
    user_prompt = (
        f"Chapter title: {chapter.title}\n"
        f"Scene count range: {min_scenes}-{max_scenes}\n"
        "Return JSON with key 'scenes', where each scene has: "
        "scene_id, prompt, duration_s, camera.\n"
        "Keep duration_s between 3 and 8 seconds.\n"
        f"Preferred style: {style}\n\n"
        "Chapter content:\n"
        f"{truncate_text(chapter.body, 9000)}"
    )

    headers = {
        "Authorization": f"Bearer {llm_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    headers.update(llm_cfg.get("headers", {}))

    payload = {
        "model": llm_cfg["model"],
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(llm_cfg["url"], headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise LLMError("LLM response does not contain choices")

    content = choices[0].get("message", {}).get("content")
    if isinstance(content, list):
        text_parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
        content = "\n".join(part for part in text_parts if part)
    if not isinstance(content, str) or not content.strip():
        raise LLMError("LLM response content is empty")

    parsed = extract_json_object(content)
    scenes = parsed.get("scenes") or parsed.get("scene_plan")
    if not isinstance(scenes, list) or not scenes:
        raise LLMError("LLM response JSON missing non-empty scenes list")

    normalized = []
    for idx, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        prompt = str(scene.get("prompt") or scene.get("description") or "").strip()
        if not prompt:
            continue
        duration_raw = scene.get("duration_s", 5)
        try:
            duration_val = int(duration_raw)
        except (TypeError, ValueError):
            duration_val = 5
        duration_val = max(3, min(8, duration_val))
        camera = str(scene.get("camera") or "medium cinematic shot").strip()
        normalized.append(
            {
                "scene_id": f"S{idx:02d}",
                "prompt": truncate_text(prompt.replace("\n", " "), 220),
                "duration_s": duration_val,
                "camera": camera,
                "style": style,
            }
        )

    if not normalized:
        raise LLMError("LLM scene normalization produced empty scene list")

    return {
        "chapter_no": chapter.chapter_no,
        "title": chapter.title,
        "source": "llm",
        "scenes": normalized[:max_scenes],
    }


def scene_plan_to_script(chapter: ChapterRecord, scene_plan: dict[str, Any]) -> str:
    lines = [f"# {chapter.title}", ""]
    scenes = scene_plan.get("scenes") or []
    for idx, scene in enumerate(scenes, start=1):
        prompt = str(scene.get("prompt", "")).strip()
        camera = str(scene.get("camera", "")).strip()
        duration = int(scene.get("duration_s", 5))
        lines.append(f"INT. SCENE {idx} - DAY")
        lines.append(prompt or "Character-driven anime action sequence.")
        lines.append(f"CAMERA: {camera or 'medium cinematic shot'}")
        lines.append(f"SHOT_DURATION: {duration}s")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def override_working_dir_in_yaml(config_text: str, working_dir: Path) -> str:
    escaped = str(working_dir).replace("'", "''")
    replacement = f"working_dir: '{escaped}'"
    if re.search(r"(?m)^working_dir\s*:\s*.*$", config_text):
        return re.sub(r"(?m)^working_dir\s*:\s*.*$", replacement, config_text)
    return config_text.rstrip() + "\n" + replacement + "\n"


def parse_working_dir_from_yaml(config_text: str) -> Path:
    match = re.search(r"(?m)^working_dir\s*:\s*(.+?)\s*$", config_text)
    if not match:
        raise RuntimeError("working_dir is missing in ViMax config")
    raw = match.group(1).strip().strip("\"'")
    return Path(raw)


def run_vimax_script2video(
    vimax_repo: Path,
    python_bin: str,
    config_path: Path,
    payload_path: Path,
    log_path: Path,
    timeout_seconds: int,
) -> Path:
    command = [python_bin, "-c", _VIMAX_RUNNER_CODE, str(config_path), str(payload_path)]
    result = subprocess.run(
        command,
        cwd=str(vimax_repo),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n=== {utc_now_iso()} ===\n")
        f.write("$ " + " ".join(command) + "\n")
        f.write("\n--- stdout ---\n")
        f.write(result.stdout or "")
        f.write("\n--- stderr ---\n")
        f.write(result.stderr or "")

    if result.returncode != 0:
        raise RuntimeError(f"ViMax command failed with exit code {result.returncode}")

    final_video_rel = None
    for line in (result.stdout or "").splitlines():
        if line.startswith("FINAL_VIDEO="):
            final_video_rel = line.split("=", 1)[1].strip()
            break

    if final_video_rel:
        final_video_path = Path(final_video_rel)
        if not final_video_path.is_absolute():
            final_video_path = vimax_repo / final_video_path
        return final_video_path

    config_text = read_text(config_path)
    working_dir = parse_working_dir_from_yaml(config_text)
    if not working_dir.is_absolute():
        working_dir = vimax_repo / working_dir
    return working_dir / "final_video.mp4"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert markdown novel chapters into per-chapter anime videos")
    parser.add_argument("--chapter-dir", required=True, help="Directory containing chapter markdown files")
    parser.add_argument("--output-dir", required=True, help="Output directory for videos and metadata")
    parser.add_argument("--vimax-repo", required=True, help="Path to cloned ViMax repository")
    parser.add_argument("--vimax-config", required=True, help="Path to ViMax script2video YAML config")
    parser.add_argument("--mode", default="script2video", choices=["script2video"])
    parser.add_argument("--llm-config", help="Optional JSON config for scene planning API")
    parser.add_argument("--resume", action="store_true", help="Skip chapters that already have final videos")
    parser.add_argument("--max-chapters", type=int, help="Process at most N chapters")
    parser.add_argument("--glob", default="*.md", help="Glob pattern for chapter files")
    parser.add_argument("--style", default="Anime Style", help="Style passed to ViMax")
    parser.add_argument(
        "--user-requirement",
        default="Fast-paced anime short with coherent visual continuity.",
        help="Creative requirements passed to ViMax",
    )
    parser.add_argument("--min-scenes", type=int, default=6, help="Minimum scene count")
    parser.add_argument("--max-scenes", type=int, default=10, help="Maximum scene count")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retries per chapter after the first failure",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate scene plans only, do not run ViMax")
    parser.add_argument(
        "--vimax-python",
        default=sys.executable,
        help="Python executable used to run ViMax (recommend ViMax venv python)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7200,
        help="Per-attempt timeout when calling ViMax",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    chapter_dir = Path(args.chapter_dir)
    if not chapter_dir.is_dir():
        raise SystemExit(f"chapter-dir does not exist: {chapter_dir}")

    vimax_repo = Path(args.vimax_repo)
    if not (vimax_repo / "main_script2video.py").exists():
        raise SystemExit(f"vimax-repo does not look like ViMax root: {vimax_repo}")

    vimax_config = Path(args.vimax_config)
    if not vimax_config.is_file():
        raise SystemExit(f"vimax-config file not found: {vimax_config}")

    if args.min_scenes <= 0 or args.max_scenes <= 0 or args.min_scenes > args.max_scenes:
        raise SystemExit("scene count constraints are invalid")


def run_pipeline(args: argparse.Namespace) -> int:
    validate_args(args)

    chapter_dir = Path(args.chapter_dir)
    output_dir = Path(args.output_dir)
    vimax_repo = Path(args.vimax_repo)
    vimax_config = Path(args.vimax_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    runs_dir = output_dir / "runs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    chapters = discover_chapters(chapter_dir, args.glob, args.max_chapters)
    if not chapters:
        raise SystemExit("No chapter markdown files were found")

    llm_cfg = None
    if args.llm_config:
        llm_cfg = load_llm_config(Path(args.llm_config))

    manifest = [
        {
            "index": c.index,
            "chapter_no": c.chapter_no,
            "title": c.title,
            "source_path": str(c.source_path),
            "sha256": c.sha256,
        }
        for c in chapters
    ]
    dump_json(output_dir / "chapter_manifest.json", manifest)

    report: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "chapters_total": len(chapters),
        "chapters": [],
    }
    failures: list[dict[str, str]] = []

    base_config_text = read_text(vimax_config)

    for chapter in chapters:
        chapter_run_dir = runs_dir / chapter.chapter_no
        chapter_run_dir.mkdir(parents=True, exist_ok=True)
        chapter_video = videos_dir / f"chapter_{chapter.chapter_no}.mp4"

        chapter_report: dict[str, Any] = {
            "chapter_no": chapter.chapter_no,
            "title": chapter.title,
            "source_path": str(chapter.source_path),
            "video_path": str(chapter_video),
            "status": "pending",
            "started_at": utc_now_iso(),
        }

        if args.resume and chapter_video.exists():
            chapter_report["status"] = "skipped"
            chapter_report["reason"] = "existing_video"
            chapter_report["ended_at"] = utc_now_iso()
            report["chapters"].append(chapter_report)
            continue

        if llm_cfg:
            try:
                scene_plan = llm_scene_plan(
                    chapter=chapter,
                    llm_cfg=llm_cfg,
                    style=args.style,
                    min_scenes=args.min_scenes,
                    max_scenes=args.max_scenes,
                )
            except Exception as exc:  # noqa: BLE001
                scene_plan = heuristic_scene_plan(chapter, args.style, args.min_scenes, args.max_scenes)
                chapter_report["scene_plan_fallback"] = str(exc)
        else:
            scene_plan = heuristic_scene_plan(chapter, args.style, args.min_scenes, args.max_scenes)

        screenplay = scene_plan_to_script(chapter, scene_plan)
        vimax_input = {
            "chapter_no": chapter.chapter_no,
            "title": chapter.title,
            "script": screenplay,
            "user_requirement": args.user_requirement,
            "style": args.style,
        }

        dump_json(chapter_run_dir / "scene_plan.json", scene_plan)
        (chapter_run_dir / "screenplay.txt").write_text(screenplay, encoding="utf-8")
        dump_json(chapter_run_dir / "vimax_input.json", vimax_input)

        if args.dry_run:
            chapter_report["status"] = "dry_run"
            chapter_report["ended_at"] = utc_now_iso()
            report["chapters"].append(chapter_report)
            continue

        working_dir = chapter_run_dir / "vimax_workdir"
        config_text = override_working_dir_in_yaml(base_config_text, working_dir)
        chapter_config_path = chapter_run_dir / "vimax_config.yaml"
        chapter_config_path.write_text(config_text, encoding="utf-8")

        payload_path = chapter_run_dir / "vimax_input.json"
        log_path = chapter_run_dir / "run.log"

        max_attempts = args.max_retries + 1
        error_msg = ""

        for attempt in range(1, max_attempts + 1):
            try:
                final_video_path = run_vimax_script2video(
                    vimax_repo=vimax_repo,
                    python_bin=args.vimax_python,
                    config_path=chapter_config_path,
                    payload_path=payload_path,
                    log_path=log_path,
                    timeout_seconds=args.timeout_seconds,
                )
                if not final_video_path.exists():
                    raise RuntimeError(f"ViMax finished but final video was not found: {final_video_path}")
                shutil.copy2(final_video_path, chapter_video)
                chapter_report["status"] = "success"
                chapter_report["attempts"] = attempt
                chapter_report["vimax_final_video"] = str(final_video_path)
                break
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"\n[attempt {attempt}/{max_attempts}] failed: {error_msg}\n")

        if chapter_report["status"] != "success":
            chapter_report["status"] = "failed"
            chapter_report["error"] = error_msg or "unknown_error"
            failures.append(
                {
                    "chapter_no": chapter.chapter_no,
                    "title": chapter.title,
                    "error": chapter_report["error"],
                    "run_log": str(log_path),
                }
            )

        chapter_report["ended_at"] = utc_now_iso()
        report["chapters"].append(chapter_report)

    report["ended_at"] = utc_now_iso()
    report["success_count"] = sum(1 for row in report["chapters"] if row["status"] == "success")
    report["failed_count"] = sum(1 for row in report["chapters"] if row["status"] == "failed")
    report["skipped_count"] = sum(1 for row in report["chapters"] if row["status"] == "skipped")
    report["dry_run_count"] = sum(1 for row in report["chapters"] if row["status"] == "dry_run")

    dump_json(output_dir / "run_report.json", report)
    if failures:
        dump_json(output_dir / "failed_chapters.json", failures)

    print(
        "Done: "
        f"success={report['success_count']} "
        f"failed={report['failed_count']} "
        f"skipped={report['skipped_count']} "
        f"dry_run={report['dry_run_count']}"
    )
    print(f"Run report: {output_dir / 'run_report.json'}")

    return 0 if report["failed_count"] == 0 else 1


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
