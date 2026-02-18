"""Parallel asset extractor - Multi-threaded for maximum speed.

Uses ThreadPoolExecutor to process multiple chapters simultaneously.
Recommended: 3-5 workers to balance speed and API rate limits.
"""

import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx
from loguru import logger

# Thread-safe counters
stats_lock = Lock()
stats = {"processed": 0, "skipped": 0, "failed": 0}


def configure_logger(log_file: str | None):
    """Configure loguru sinks for console and optional file output."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
            rotation="50 MB",
            encoding="utf-8",
        )


def strip_markdown_code_blocks(response: str) -> str:
    """Strip markdown code blocks from LLM response."""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    return response.strip()


def extract_all_assets(chapter_text: str, llm_config: dict) -> dict:
    """Extract all 8 asset types in a single API call."""
    prompt = f"""从这一章中提取所有叙事元素。返回完整的JSON对象，包含以下8种元素：

返回格式：
{{
  "plot_beats": [
    {{
      "event": "发生了什么事件",
      "characters": ["涉及的角色名"],
      "impact": "对故事的影响",
      "chapter_position": "beginning|middle|end",
      "causality": "因果关系说明"
    }}
  ],
  "character_cards": [
    {{
      "name": "角色名",
      "traits": ["性格特征"],
      "state": "当前状态",
      "relationships": {{"其他角色": "关系描述"}},
      "goals": ["角色目标"],
      "secrets": ["秘密或隐藏信息"]
    }}
  ],
  "conflicts": [
    {{
      "type": "internal|external|interpersonal",
      "description": "冲突描述",
      "parties": ["涉及方"],
      "stakes": "冲突的赌注/后果",
      "resolution_status": "unresolved|partially_resolved|resolved"
    }}
  ],
  "settings": [
    {{
      "location": "地点名称",
      "time": "时间（时代/季节/时刻）",
      "atmosphere": "氛围描述",
      "significance": "对情节的重要性",
      "sensory_details": ["感官细节"]
    }}
  ],
  "themes": [
    {{
      "theme": "主题名称",
      "manifestation": "如何体现",
      "symbols": ["相关象征"],
      "development": "主题发展"
    }}
  ],
  "point_of_view": {{
    "narrator_type": "first_person|third_person_limited|third_person_omniscient",
    "perspective_character": "视角角色",
    "reliability": "reliable|unreliable",
    "narrative_distance": "close|medium|distant"
  }},
  "tone": {{
    "overall_tone": "整体语气",
    "emotional_range": ["情感范围"],
    "mood": "氛围",
    "shifts": ["语气转变点"]
  }},
  "style": {{
    "sentence_structure": "句式特点",
    "vocabulary_level": "词汇水平",
    "rhetoric_devices": ["修辞手法"],
    "dialogue_ratio": "high|medium|low",
    "description_style": "描写风格",
    "distinctive_features": ["独特风格特征"]
  }}
}}

章节内容：
{chapter_text}

只返回JSON对象，不要其他文字。"""

    response = call_llm(prompt, llm_config)
    response = strip_markdown_code_blocks(response)

    try:
        assets = json.loads(response)
        return assets
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse response JSON: {}", e)
        # Return empty structure
        return {
            "plot_beats": [],
            "character_cards": [],
            "conflicts": [],
            "settings": [],
            "themes": [],
            "point_of_view": {},
            "tone": {},
            "style": {}
        }


def call_llm(prompt: str, llm_config: dict) -> str:
    """Call LLM API (configurable)."""
    if llm_config.get("type") == "custom":
        # Temporarily unset ALL_PROXY to avoid SOCKS proxy issues
        old_all_proxy = os.environ.pop('ALL_PROXY', None)
        old_all_proxy_lower = os.environ.pop('all_proxy', None)

        try:
            response = httpx.post(
                llm_config["url"],
                json={
                    "model": llm_config["model"],
                    "messages": [{"role": "user", "content": prompt}]
                },
                headers={"Authorization": f"Bearer {llm_config['api_key']}"},
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        finally:
            # Restore ALL_PROXY
            if old_all_proxy:
                os.environ['ALL_PROXY'] = old_all_proxy
            if old_all_proxy_lower:
                os.environ['all_proxy'] = old_all_proxy_lower

    if llm_config.get("providers") and llm_config.get("model"):
        provider_cfg = llm_config["providers"].get("anthropic")
        if not provider_cfg:
            raise ValueError("providers.anthropic is required in providers mode")
        api_key = provider_cfg.get("apiKey") or provider_cfg.get("api_key")
        api_base = provider_cfg.get("apiBase") or provider_cfg.get("api_base")
        extra_headers = provider_cfg.get("extraHeaders") or provider_cfg.get("extra_headers")

        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(
            api_key=api_key,
            api_base=api_base,
            default_model=llm_config["model"],
            extra_headers=extra_headers,
        )

        async def _chat_with_timeout():
            return await asyncio.wait_for(
                provider.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=llm_config["model"],
                    max_tokens=4096,
                    temperature=0.2,
                ),
                timeout=120.0,
            )

        old_all_proxy = os.environ.pop('ALL_PROXY', None)
        old_all_proxy_lower = os.environ.pop('all_proxy', None)
        try:
            try:
                asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    response = pool.submit(lambda: asyncio.run(_chat_with_timeout())).result()
            except RuntimeError:
                response = asyncio.run(_chat_with_timeout())
            content = response.content or ""
            if not content.strip():
                raise RuntimeError("empty LLM response")
            return content
        finally:
            if old_all_proxy:
                os.environ['ALL_PROXY'] = old_all_proxy
            if old_all_proxy_lower:
                os.environ['all_proxy'] = old_all_proxy_lower

    raise ValueError("Unsupported LLM config: expected {type: custom} or {providers, model}")


def process_chapter_file(chapter_file: Path, book_id: str, output_dir: Path,
                        llm_config: dict, worker_id: int) -> tuple[bool, str]:
    """Process a single chapter file. Returns (success, message)."""
    try:
        # Check if already processed
        expected_output = output_dir / f"{book_id}_{chapter_file.stem}_assets.json"
        if expected_output.exists():
            with stats_lock:
                stats["skipped"] += 1
            return True, f"[Worker {worker_id}] Skipped {chapter_file.name}"

        # Read chapter
        with open(chapter_file) as f:
            chapter_text = f.read()

        # Extract assets
        assets = extract_all_assets(chapter_text, llm_config)

        # Add metadata
        output = {
            "book_id": book_id,
            "chapter": chapter_file.stem,
            **assets
        }

        # Write output
        with open(expected_output, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        with stats_lock:
            stats["processed"] += 1

        return True, f"[Worker {worker_id}] ✓ {chapter_file.name}"

    except Exception as e:
        with stats_lock:
            stats["failed"] += 1
        # Log failed chapter
        with open(output_dir / "failed_chapters.txt", "a") as f:
            f.write(f"{chapter_file.name}\n")
        return False, f"[Worker {worker_id}] ✗ {chapter_file.name}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel asset extraction")
    parser.add_argument("--book-id", required=True, help="Book ID")
    parser.add_argument("--chapter-dir", required=True, help="Directory containing chapter files")
    parser.add_argument("--output-dir", required=True, help="Output directory for assets")
    parser.add_argument("--llm-config", required=True, help="Path to LLM config JSON")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers (default: 3)")
    parser.add_argument("--log-file", default="", help="Optional log file path")
    args = parser.parse_args()

    configure_logger(args.log_file or None)

    with open(args.llm_config) as f:
        llm_config = json.load(f)

    chapter_dir = Path(args.chapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all chapter files
    chapter_files = sorted(chapter_dir.glob("*.md"))
    total = len(chapter_files)

    logger.info("Found {} chapter files", total)
    logger.info("Output directory: {}", output_dir)
    logger.info("Workers: {}", args.workers)
    logger.info("Mode: Parallel (multi-threaded)")
    logger.info("=" * 60)

    start_time = time.time()
    completed = 0

    # Process chapters in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_chapter = {
            executor.submit(
                process_chapter_file,
                chapter_file,
                args.book_id,
                output_dir,
                llm_config,
                i % args.workers + 1
            ): chapter_file
            for i, chapter_file in enumerate(chapter_files)
        }

        # Process results as they complete
        for future in as_completed(future_to_chapter):
            completed += 1
            success, message = future.result()

            # Print progress every 10 completions or on failure
            if completed % 10 == 0 or not success:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0

                logger.info("[{}/{}] {}", completed, total, message)
                logger.info(
                    "Progress: {} processed, {} skipped, {} failed",
                    stats["processed"],
                    stats["skipped"],
                    stats["failed"],
                )
                logger.info("Speed: {:.2f} chapters/sec | ETA: {:.1f} min", rate, eta / 60)
            elif completed % 50 == 0:
                # Brief update every 50
                logger.info(
                    "[{}/{}] Progress: {} processed, {} skipped",
                    completed,
                    total,
                    stats["processed"],
                    stats["skipped"],
                )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Extraction complete!")
    logger.info("Processed: {}", stats["processed"])
    logger.info("Skipped: {}", stats["skipped"])
    logger.info("Failed: {}", stats["failed"])
    logger.info("Total time: {:.1f} minutes", elapsed / 60)
    logger.info("Average speed: {:.2f} chapters/sec", total / elapsed if elapsed > 0 else 0)
    logger.info("=" * 60)
