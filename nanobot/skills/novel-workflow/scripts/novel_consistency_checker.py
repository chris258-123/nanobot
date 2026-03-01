#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# 同目录导入 audit_utils
sys.path.insert(0, str(Path(__file__).parent))
from audit_utils import (
    chapter_body,
    chapter_title,
    write_json,
)

_PROXY_KEYS = ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")
_VALID_SEVERITIES = {"CRITICAL", "WARNING", "INFO"}
SUMMARY_SCHEMA_VER = "1"
SUMMARY_PROMPT_VER = "2"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="小说长文连载一致性审查工具")
    p.add_argument("--chapter-dir", type=Path, required=True, help="章节 .md 文件目录")
    p.add_argument("--book-id", type=str, required=True, help="书籍 ID，如 novel_04_b_full_1350_v4")
    p.add_argument("--from-chapter", type=int, default=1, help="起始章节号（默认 1）")
    p.add_argument("--to-chapter", type=int, default=9999, help="结束章节号（默认自动检测最大值）")
    p.add_argument("--canon-db", type=Path, default=None, help="Canon DB sqlite 路径（可选）")
    p.add_argument("--gen-log-root", type=Path, default=None, help="生成日志根目录（可选）")
    p.add_argument("--llm-config", type=Path, default=None, help="llm_config.json 路径（可选）")
    p.add_argument("--llm-mode", choices=["raw", "summary"], default="summary", help="LLM 审查模式：raw=原文拼接，summary=摘要压缩（默认）")
    p.add_argument("--llm-batch-size", type=int, default=20, help="LLM 每批章节数（仅 --llm-mode=raw 时生效，默认 20）")
    p.add_argument("--summary-extract-batch", type=int, default=5, help="摘要提取每批章节数（默认 5）")
    p.add_argument("--summary-cache", type=Path, default=None, help="摘要缓存 JSON 路径（可选，避免重复提取）")
    p.add_argument("--context-window", type=int, default=20, help="审查时携带的历史章节摘要数（0=禁用，默认 20）")
    p.add_argument("--output-dir", type=Path, default=Path("./audit_output"), help="输出目录")
    p.add_argument("--skip-llm", action="store_true", help="跳过 LLM 审查")
    p.add_argument("--gender-config", type=Path, default=None, help="角色性别配置 JSON 路径（可选）")
    return p.parse_args()


# ── 章节加载 ──────────────────────────────────────────────────────────────────

def _load_chapters(
    chapter_dir: Path, book_id: str, start: int, to_chapter: int
) -> tuple[dict[int, dict[str, Any]], int]:
    pat = re.compile(rf"^{re.escape(book_id)}_chapter_(\d{{4}})\.md$")
    files: dict[int, Path] = {}
    for p in chapter_dir.iterdir():
        if p.is_file():
            m = pat.match(p.name)
            if m:
                files[int(m.group(1))] = p
    if not files:
        raise ValueError(f"未找到匹配文件：{book_id}_chapter_XXXX.md in {chapter_dir}")

    # end 使用用户指定上限（不截断），让 detect_text_integrity 报告尾部缺失章节
    end = to_chapter if to_chapter < 9999 else max(files)
    if start > end:
        raise ValueError(f"无可审查章节：from={start}, end={end}")

    chapters: dict[int, dict[str, Any]] = {}
    for no in sorted(files):
        if start <= no <= end:
            text = files[no].read_text(encoding="utf-8")
            chapters[no] = {"text": text, "body": chapter_body(text), "title": chapter_title(text)}
    return chapters, end


# ── LLM 调用 ──────────────────────────────────────────────────────────────────

def _pop_proxy() -> dict[str, str]:
    backup: dict[str, str] = {}
    for key in _PROXY_KEYS:
        val = os.environ.pop(key, None)
        if val is not None:
            backup[key] = val
    return backup


def _restore_proxy(backup: dict[str, str]) -> None:
    os.environ.update(backup)


def _extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload or "")
    # OpenAI / DeepSeek 格式
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(x.get("text") or "") for x in content if isinstance(x, dict)).strip()
    # Anthropic 格式
    blocks = payload.get("content")
    if isinstance(blocks, list):
        return "\n".join(str(x.get("text") or "") for x in blocks if isinstance(x, dict)).strip()
    return json.dumps(payload, ensure_ascii=False)


def _call_llm(prompt: str, llm_config: dict[str, Any]) -> str:
    backup = _pop_proxy()
    try:
        if llm_config.get("type") == "custom":
            url = str(llm_config["url"]).strip()
            model = str(llm_config["model"]).strip()
            api_key = str(llm_config.get("api_key") or "").strip()
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            if llm_config.get("max_tokens"):
                body["max_tokens"] = int(llm_config["max_tokens"])
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = httpx.post(url, json=body, headers=headers, timeout=180.0)
            resp.raise_for_status()
            return _extract_text(resp.json())

        providers = llm_config.get("providers")
        model = llm_config.get("model")
        if isinstance(providers, dict) and model:
            cfg = providers.get("anthropic") or {}
            api_key = cfg.get("apiKey") or cfg.get("api_key") or ""
            api_base = (cfg.get("apiBase") or cfg.get("api_base") or "https://api.anthropic.com").rstrip("/")
            extra = cfg.get("extraHeaders") or {}
            headers = {
                "x-api-key": str(api_key),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                **({k: str(v) for k, v in extra.items()} if isinstance(extra, dict) else {}),
            }
            body = {
                "model": str(model),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": int(llm_config.get("max_tokens") or 16384),
            }
            resp = httpx.post(f"{api_base}/v1/messages", json=body, headers=headers, timeout=180.0)
            resp.raise_for_status()
            return _extract_text(resp.json())

        raise ValueError("不支持的 llm_config 格式：需要 {type:custom} 或 {providers, model}")
    finally:
        _restore_proxy(backup)


# ── LLM Prompt & 解析 ─────────────────────────────────────────────────────────

def _build_prompt(book_id: str, batch_nos: list[int], chapters: dict[int, dict[str, Any]]) -> str:
    parts = [
        f"【章节 {no:04d} | {chapters[no].get('title') or f'第{no}章'}】\n{chapters[no].get('body') or ''}"
        for no in batch_nos
    ]
    return (
        "你是一位专业的小说编辑，负责审查长篇连载小说的一致性问题。\n"
        "请仔细阅读以下章节内容，找出所有一致性问题。\n\n"
        "审查维度：\n"
        "1. 情节逻辑矛盾（前后章节事件冲突）\n"
        "2. 人物行为不一致（性格突变、无动机行为）\n"
        "3. 世界观设定冲突（规则/能力/地理前后矛盾）\n"
        "4. 伏笔未回收或突然出现的无根据情节\n"
        "5. 时间线混乱\n"
        "6. 人物关系矛盾\n\n"
        f"书籍 ID：{book_id}\n"
        f"批次范围：{batch_nos[0]:04d}-{batch_nos[-1]:04d}\n\n"
        "章节内容：\n"
        + "\n\n".join(parts)
        + '\n\n请以 JSON 格式返回问题列表：\n'
        '{"issues": [{"severity": "CRITICAL|WARNING|INFO", "chapter_range": "0001-0005", '
        '"category": "plot_contradiction|character_inconsistency|worldbuilding|foreshadowing|timeline|relationship", '
        '"description": "具体问题描述（中文）", "evidence": "引用原文片段（不超过100字）"}]}\n\n'
        "只返回 JSON，不要 markdown 代码块，不要其他内容。"
    )


def _parse_llm_response(raw: str, batch_range: str) -> list[dict[str, str]]:
    text = raw.strip()
    # 去除 markdown 代码块（兼容 ```json 和 ``` 两种形式）
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # 提取 rows：兼容三种格式
    # 1. {"issues": [...]}  2. [{...}, ...]  3. {...}（单条）
    rows: list[Any] | None = None

    # 优先尝试完整文本，再尝试截取 {...} 子串
    candidates: list[str] = [text]
    if "{" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    # 也尝试截取 [...] 子串（LLM 直接返回数组时）
    if "[" in text:
        candidates.append(text[text.find("[") : text.rfind("]") + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            rows = obj
            break
        if isinstance(obj, dict):
            issues = obj.get("issues")
            if isinstance(issues, list):
                rows = issues
                break
            # 单条 issue 对象（无 issues 包装）
            if obj.get("description") or obj.get("severity"):
                rows = [obj]
                break

    if rows is None:
        print(f"[WARN] LLM 返回非 JSON，批次 {batch_range}，原始内容前200字：{raw[:200]}", file=sys.stderr)
        return []

    result: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        desc = str(row.get("description") or "").strip()
        if not desc:
            continue
        sev = str(row.get("severity") or "WARNING").upper()
        result.append({
            "severity": sev if sev in _VALID_SEVERITIES else "WARNING",
            "chapter_range": str(row.get("chapter_range") or batch_range),
            "category": str(row.get("category") or "llm_consistency"),
            "description": desc,
            "evidence": str(row.get("evidence") or "")[:200],
        })
    return result


# ── 摘要压缩法：工具函数 ──────────────────────────────────────────────────────

def _parse_json_robust(raw: str) -> dict | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = re.sub(r",\s*([}\]])", r"\1", text[start : end + 1])
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _chapter_source_hash(chapter: dict) -> str:
    source = f"{chapter.get('title') or ''}{chapter.get('body') or ''}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _load_summary_cache(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_summary_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _resolve_cached_summaries(
    cache: dict, chapters: dict, nos: list[int], schema_ver: str, prompt_ver: str
) -> tuple[dict[int, dict], list[int]]:
    entries = cache.get("entries") if isinstance(cache, dict) else {}
    if not isinstance(entries, dict):
        entries = {}
    hit_map: dict[int, dict] = {}
    miss_nos: list[int] = []
    for no in nos:
        entry = entries.get(str(no))
        chapter = chapters.get(no) or {}
        if not isinstance(entry, dict):
            miss_nos.append(no)
            continue
        if str(entry.get("schema_ver") or "") != schema_ver:
            miss_nos.append(no)
            continue
        if str(entry.get("prompt_ver") or "") != prompt_ver:
            miss_nos.append(no)
            continue
        if str(entry.get("source_hash") or "") != _chapter_source_hash(chapter):
            miss_nos.append(no)
            continue
        summary = entry.get("summary")
        if not isinstance(summary, dict):
            miss_nos.append(no)
            continue
        hit_map[no] = dict(summary)
    return hit_map, miss_nos


def _make_fallback_summary(no: int, chapters: dict) -> dict:
    chapter = chapters.get(no) or {}
    return {
        "chapter_no": no,
        "title": str(chapter.get("title") or f"第{no}章"),
        "characters": [],
        "events": ["摘要提取失败，待人工复核"],
        "state_changes": [],
        "hooks": [],
        "ending_state": "摘要提取失败，待人工复核",
        "quality": "fallback",
    }


def _build_summary_extract_prompt(book_id: str, batch_nos: list[int], chapters: dict) -> str:
    parts = [
        f"【章节 {no:04d} | {chapters.get(no, {}).get('title') or f'第{no}章'}】\n"
        + str(chapters.get(no, {}).get("body") or "")[:4000]
        for no in batch_nos
    ]
    example = json.dumps({
        "summaries": [{
            "chapter_no": batch_nos[0] if batch_nos else 1,
            "title": "章节标题",
            "characters": ["角色A", "角色B"],
            "events": ["A得知秘密", "A与B争执", "A决定离开"],
            "state_changes": ["A从怀疑转为确信", "B对A的信任下降"],
            "hooks": ["神秘来信来源未揭晓"],
            "ending_state": "A带着线索离开城镇，B留在原地观望",
            "gender_refs": {"角色A": "他", "角色B": "她"},
        }]
    }, ensure_ascii=False)
    return (
        "你是小说一致性审查前处理助手，任务是提取章节结构化摘要。\n"
        "请只基于给定章节内容输出 JSON，不要补充额外解释。\n\n"
        f"书籍 ID：{book_id}\n"
        f"输入章节数：{len(batch_nos)}（summaries 长度必须严格等于该值）\n\n"
        "每章输出字段：\n"
        "- chapter_no: 章节号（整数）\n"
        "- title: 章节标题\n"
        "- characters: 主要人物列表\n"
        "- events: 2-4 条关键事件，每条不超过 35 字\n"
        "- state_changes: 1-3 条状态变化，每条不超过 35 字\n"
        "- hooks: 0-2 条悬念，每条不超过 30 字\n"
        "- ending_state: 章节结尾状态，不超过 60 字\n"
        "- gender_refs: 该章中每个主要角色实际使用的性别代词，格式 {角色名: 他|她|混用}，"
        "仅记录章节中明确出现性别代词的角色\n\n"
        f"输出格式示例：\n{example}\n\n"
        "待处理章节：\n"
        + "\n\n".join(parts)
        + '\n\n仅返回 JSON：{"summaries": [...]}'
    )


def _parse_summary_extract_response(raw: str, expected_nos: list[int], chapters: dict) -> list[dict]:
    def _clean_list(value: Any, max_items: int, max_len: int) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text[:max_len])
            if len(out) >= max_items:
                break
        return out

    obj = _parse_json_robust(raw)
    rows = obj.get("summaries") if isinstance(obj, dict) else None
    expected_set = set(expected_nos)
    parsed_map: dict[int, dict] = {}

    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                no = int(row.get("chapter_no"))
            except (TypeError, ValueError):
                continue
            if no not in expected_set or no in parsed_map:
                continue
            chapter = chapters.get(no) or {}
            parsed_map[no] = {
                "chapter_no": no,
                "title": str(row.get("title") or chapter.get("title") or f"第{no}章")[:120],
                "characters": _clean_list(row.get("characters"), 20, 30),
                "events": _clean_list(row.get("events"), 4, 35),
                "state_changes": _clean_list(row.get("state_changes"), 3, 35),
                "hooks": _clean_list(row.get("hooks"), 2, 30),
                "ending_state": str(row.get("ending_state") or "")[:60],
                "quality": "llm",
            }

    return [parsed_map.get(no) or _make_fallback_summary(no, chapters) for no in expected_nos]


def _extract_summaries_with_cache(
    chapters: dict, nos: list[int], llm_config: dict,
    extract_batch: int, cache_path: Path | None, book_id: str,
) -> list[dict]:
    cache = _load_summary_cache(cache_path) if cache_path else {}
    hit_map, miss_nos = _resolve_cached_summaries(cache, chapters, nos, SUMMARY_SCHEMA_VER, SUMMARY_PROMPT_VER)
    resolved: dict[int, dict] = dict(hit_map)
    entries: dict = cache.get("entries") if isinstance(cache.get("entries"), dict) else {}
    cache_dirty = False
    batch_size = max(int(extract_batch or 1), 1)
    batches = _chunks(miss_nos, batch_size)
    for idx, batch in enumerate(batches, 1):
        rng = f"{batch[0]:04d}-{batch[-1]:04d}"
        print(f"[INFO] 摘要提取批次 {idx}/{len(batches)}：{rng}", file=sys.stderr)
        try:
            raw = _call_llm(_build_summary_extract_prompt(book_id, batch, chapters), llm_config)
            parsed = _parse_summary_extract_response(raw, batch, chapters)
        except Exception as exc:
            print(f"[WARN] 摘要提取失败，批次 {rng}：{exc}", file=sys.stderr)
            parsed = [_make_fallback_summary(no, chapters) for no in batch]
        for summary in parsed:
            try:
                no = int(summary.get("chapter_no"))
            except (TypeError, ValueError):
                continue
            resolved[no] = summary
            entries[str(no)] = {
                "source_hash": _chapter_source_hash(chapters.get(no) or {}),
                "schema_ver": SUMMARY_SCHEMA_VER,
                "prompt_ver": SUMMARY_PROMPT_VER,
                "updated_at": datetime.now().isoformat(),
                "summary": summary,
            }
            cache_dirty = True
    if cache_path and cache_dirty:
        cache["schema_ver"] = SUMMARY_SCHEMA_VER
        cache["prompt_ver"] = SUMMARY_PROMPT_VER
        cache["entries"] = entries
        _save_summary_cache(cache_path, cache)
    return [resolved.get(no) or _make_fallback_summary(no, chapters) for no in nos]


def _load_context_summaries(cache_path: Path | None, ctx_nos: list[int]) -> list[dict]:
    # 只从缓存读取历史摘要，未命中直接跳过，不触发额外 LLM 调用
    if not cache_path or not ctx_nos:
        return []
    cache = _load_summary_cache(cache_path)
    entries = cache.get("entries") if isinstance(cache, dict) else {}
    if not isinstance(entries, dict):
        return []
    result: list[dict] = []
    for no in ctx_nos:
        entry = entries.get(str(no))
        if not isinstance(entry, dict):
            continue
        if str(entry.get("schema_ver") or "") != SUMMARY_SCHEMA_VER:
            continue
        if str(entry.get("prompt_ver") or "") != SUMMARY_PROMPT_VER:
            continue
        summary = entry.get("summary")
        if isinstance(summary, dict):
            result.append(dict(summary))
    return result


def _build_summary_review_prompt(
    book_id: str, start: int, end: int, summaries: list[dict],
    context_summaries: list[dict] | None = None,
    gender_map: dict[str, str] | None = None,
) -> str:
    compact = json.dumps({"summaries": summaries}, ensure_ascii=False, separators=(",", ":"))
    # 构建性别设定段落
    gender_section = ""
    if gender_map:
        labels = {"male": "男性(他)", "female": "女性(她)", "unknown": "性别不明"}
        gender_lines = ", ".join(f"{name}={labels.get(g, g)}" for name, g in gender_map.items())
        gender_section = (
            f"【角色性别设定（以此为准）】\n{gender_lines}\n\n"
            "请特别检查：\n"
            "- 上述角色的性别代词（他/她）是否与设定一致\n"
            "- 同一角色在不同章节的性别指代是否前后一致\n\n"
        )
    suffix = (
        "请输出 JSON：\n"
        '{"issues": [{"severity": "CRITICAL|WARNING|INFO", "chapter_range": "0001-0005", '
        '"category": "plot_contradiction|character_inconsistency|worldbuilding|foreshadowing|timeline|relationship|gender_inconsistency", '
        '"description": "具体问题描述（中文）", "evidence": "摘要中的证据片段（不超过100字）"}]}\n\n'
        "只返回 JSON，不要 markdown 代码块，不要其他内容。"
    )
    header = (
        "你是一位专业的小说一致性审查员。\n"
        "请注意：你只能基于提供的章节摘要判断，不要假设原文存在未给出的细节。\n"
        f"如果摘要中没有证据，请不要编造问题。\n\n书籍 ID：{book_id}\n\n"
        + gender_section
    )
    if context_summaries:
        ctx_compact = json.dumps({"summaries": context_summaries}, ensure_ascii=False, separators=(",", ":"))
        return (
            header
            + "【历史上下文摘要（仅供参考，不对此报告问题）】\n"
            + f"{ctx_compact}\n\n"
            + f"【当前审查批次：{start:04d}-{end:04d}】\n"
            + f"{compact}\n\n"
            + "请只针对\"当前审查批次\"报告问题，但可以引用历史上下文中的信息作为证据。\n\n"
            + suffix
        )
    return (
        header
        + f"审查范围：{start:04d}-{end:04d}\n\n"
        + f"摘要数据（紧凑 JSON）：\n{compact}\n\n"
        + suffix
    )


def _run_llm_raw_mode(
    chapters: dict, nos: list[int], llm_config: dict, batch_size: int, book_id: str
) -> list[dict]:
    llm_issues: list[dict] = []
    batches = _chunks(nos, batch_size)
    for i, batch in enumerate(batches, 1):
        rng = f"{batch[0]:04d}-{batch[-1]:04d}"
        print(f"[INFO] LLM 批次 {i}/{len(batches)}：{rng}", file=sys.stderr)
        try:
            raw = _call_llm(_build_prompt(book_id, batch, chapters), llm_config)
            llm_issues.extend(_parse_llm_response(raw, rng))
        except Exception as exc:
            print(f"[WARN] LLM 调用失败，批次 {rng}：{exc}", file=sys.stderr)
    return llm_issues


def _run_llm_summary_mode(
    chapters: dict, nos: list[int], llm_config: dict, args: argparse.Namespace,
    gender_map: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    cache_path = args.summary_cache.expanduser() if args.summary_cache else None
    # 预查缓存（用于统计 cache_hits）
    probe_cache = _load_summary_cache(cache_path) if cache_path else {}
    hit_map_pre, miss_nos_pre = _resolve_cached_summaries(
        probe_cache, chapters, nos, SUMMARY_SCHEMA_VER, SUMMARY_PROMPT_VER
    )
    summaries = _extract_summaries_with_cache(
        chapters=chapters, nos=nos, llm_config=llm_config,
        extract_batch=args.summary_extract_batch,
        cache_path=cache_path, book_id=args.book_id,
    )
    # 滚动窗口：从缓存加载历史摘要作为上下文
    context_summaries: list[dict] = []
    context_window = max(int(getattr(args, "context_window", 0) or 0), 0)
    if context_window > 0 and nos and cache_path:
        # 从缓存 entries 中找出所有章节号，取 nos[0] 之前的最多 context_window 章
        cache_for_ctx = _load_summary_cache(cache_path)
        entries_for_ctx = cache_for_ctx.get("entries") if isinstance(cache_for_ctx, dict) else {}
        if isinstance(entries_for_ctx, dict):
            cached_nos = sorted(int(k) for k in entries_for_ctx if k.isdigit())
            before_nos = [no for no in cached_nos if no < nos[0]]
            ctx_nos = before_nos[-context_window:]
            if ctx_nos:
                context_summaries = _load_context_summaries(cache_path, ctx_nos)
                if context_summaries:
                    span = sorted(int(s.get("chapter_no", 0)) for s in context_summaries if s.get("chapter_no"))
                    print(
                        f"[INFO] 携带历史摘要 {len(context_summaries)} 章"
                        + (f"（{span[0]:04d}-{span[-1]:04d}）" if span else ""),
                        file=sys.stderr,
                    )
    issues: list[dict] = []
    if nos:
        rng = f"{nos[0]:04d}-{nos[-1]:04d}"
        print(f"[INFO] LLM 摘要审查：{rng}（共 {len(summaries)} 章摘要）", file=sys.stderr)
        try:
            raw = _call_llm(
                _build_summary_review_prompt(
                    args.book_id, nos[0], nos[-1], summaries,
                    context_summaries=context_summaries or None,
                    gender_map=gender_map or None,
                ),
                llm_config,
            )
            issues = _parse_llm_response(raw, rng)
        except Exception as exc:
            print(f"[WARN] LLM 摘要审查失败，范围 {rng}：{exc}", file=sys.stderr)
    extract_batch = max(int(args.summary_extract_batch or 1), 1)
    summary_stats = {
        "chapters": len(nos),
        "summary_count": len(summaries),
        "cache_hits": len(hit_map_pre),
        "cache_misses": len(miss_nos_pre),
        "extract_batches": len(_chunks(miss_nos_pre, extract_batch)),
        "fallback_count": sum(1 for s in summaries if str(s.get("quality") or "") == "fallback"),
        "cache_path": str(cache_path) if cache_path else None,
        "context_chapters": len(context_summaries),
    }
    return issues, summary_stats


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def _chunks(lst: list[int], size: int) -> list[list[int]]:
    size = max(size, 1)
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _load_llm_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] llm_config 不存在：{path}", file=sys.stderr)
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception as exc:
        print(f"[WARN] 读取 llm_config 失败：{exc}", file=sys.stderr)
        return None


def _load_gender_config(path: Path | None) -> dict[str, str]:
    # 返回 {角色名: "male"/"female"/"unknown"}
    if path is None:
        return {}
    if not path.exists():
        print(f"[WARN] gender_config 不存在：{path}", file=sys.stderr)
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        chars = obj.get("characters") if isinstance(obj, dict) else None
        if not isinstance(chars, dict):
            return {}
        return {str(k): str(v) for k, v in chars.items() if v in ("male", "female", "unknown")}
    except Exception as exc:
        print(f"[WARN] 读取 gender_config 失败：{exc}", file=sys.stderr)
        return {}


# ── 报告合并 ──────────────────────────────────────────────────────────────────

def _merge_reports(report_dir: Path, book_id: str, output_dir: Path) -> int:
    if not report_dir.is_dir():
        print(f"[ERROR] report_dir 不存在：{report_dir}", file=sys.stderr)
        return 1

    pat = re.compile(r"^consistency_report_(\d{4})_(\d{4})\.json$")
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for p in report_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] 跳过无效报告 {p.name}：{exc}", file=sys.stderr)
            continue
        if not isinstance(obj, dict) or str(obj.get("book_id") or "") != book_id:
            continue
        rows.append((int(m.group(1)), int(m.group(2)), obj))

    if not rows:
        print(f"[ERROR] 未找到可合并报告：book_id={book_id} in {report_dir}", file=sys.stderr)
        return 1

    rows.sort(key=lambda x: (x[0], x[1]))

    def _as_int(v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    severity_rank = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
    llm_issue_map: dict[tuple[str, str], dict[str, Any]] = {}
    llm_issue_order: list[tuple[str, str]] = []
    llm_batches = 0

    for _, _, report in rows:
        llm = report.get("llm_analysis")
        if isinstance(llm, dict):
            llm_batches += _as_int(llm.get("batches_processed"))
            for item in (llm.get("issues") or [] if isinstance(llm.get("issues"), list) else []):
                if not isinstance(item, dict):
                    continue
                sev = str(item.get("severity") or "WARNING").upper()
                sev = sev if sev in _VALID_SEVERITIES else "WARNING"
                cr = str(item.get("chapter_range") or "").strip()
                desc = str(item.get("description") or "").strip()
                if not desc:
                    continue
                key = (cr, desc[:50])
                issue = {**item, "severity": sev, "chapter_range": cr, "description": desc}
                if key not in llm_issue_map:
                    llm_issue_map[key] = issue
                    llm_issue_order.append(key)
                elif severity_rank.get(sev, 0) > severity_rank.get(str(llm_issue_map[key].get("severity")), 0):
                    llm_issue_map[key] = issue

    llm_issues = [llm_issue_map[k] for k in llm_issue_order]
    min_start, max_end = rows[0][0], max(end for _, end, _ in rows)
    summary = {
        "total_issues": len(llm_issues),
        "llm_issues": len(llm_issues),
        "llm_critical": sum(1 for x in llm_issues if x.get("severity") == "CRITICAL"),
        "llm_warning": sum(1 for x in llm_issues if x.get("severity") == "WARNING"),
        "llm_info": sum(1 for x in llm_issues if x.get("severity") == "INFO"),
    }
    merged: dict[str, Any] = {
        "book_id": book_id,
        "range": f"{min_start:04d}-{max_end:04d}",
        "generated_at": datetime.now().isoformat(),
        "merged_reports": len(rows),
        "summary": summary,
        "llm_analysis": {"batches_processed": llm_batches, "issue_count": len(llm_issues), "issues": llm_issues, "llm_mode": "merged"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    range_tag = f"{min_start:04d}_{max_end:04d}"
    json_path = output_dir / f"consistency_report_global_{range_tag}.json"
    md_path = output_dir / f"consistency_report_global_{range_tag}.md"
    write_json(json_path, merged)

    md_lines = [
        f"# 一致性审查合并报告 — {book_id}", "",
        f"- 合并范围：{merged['range']}", f"- 合并报告数：{len(rows)}", f"- 生成时间：{merged['generated_at']}", "",
        "## 摘要", "", "| 指标 | 数量 |", "| --- | ---: |",
        f"| 总问题数 | {summary['total_issues']} |",
        f"| LLM 审查问题 | {summary['llm_issues']} |",
        f"| CRITICAL | {summary['llm_critical']} |",
        f"| WARNING | {summary['llm_warning']} |",
        f"| INFO | {summary['llm_info']} |", "",
    ]
    if llm_issues:
        md_lines += ["## LLM 审查问题", ""]
        for issue in llm_issues:
            sev = issue.get("severity", "WARNING")
            md_lines.append(f"**[{sev}]** 章节 {issue.get('chapter_range')} — {issue.get('description')}")
            if issue.get("evidence"):
                md_lines.append(f"> {issue['evidence']}")
            md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path), "summary": summary}, ensure_ascii=False))
    return 0


def _cmd_merge() -> int:
    p = argparse.ArgumentParser(description="合并一致性审查报告")
    p.add_argument("merge")  # 消耗子命令占位符
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--book-id", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    return _merge_reports(
        report_dir=args.report_dir.expanduser(),
        book_id=args.book_id,
        output_dir=args.output_dir.expanduser(),
    )


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        return _cmd_merge()

    args = _parse_args()
    chapter_dir = args.chapter_dir.expanduser()
    if not chapter_dir.is_dir():
        print(f"[ERROR] chapter_dir 不存在：{chapter_dir}", file=sys.stderr)
        return 1

    try:
        chapters, end_chapter = _load_chapters(chapter_dir, args.book_id, args.from_chapter, args.to_chapter)
    except Exception as exc:
        print(f"[ERROR] 加载章节失败：{exc}", file=sys.stderr)
        return 1

    start = args.from_chapter
    end = end_chapter
    print(f"[INFO] 审查范围：{start:04d}-{end:04d}，共 {len(chapters)} 章", file=sys.stderr)

    # 加载性别配置
    gender_map = _load_gender_config(args.gender_config.expanduser() if args.gender_config else None)
    if gender_map:
        print(f"[INFO] 已加载性别配置：{len(gender_map)} 个角色", file=sys.stderr)

    # LLM 审查
    llm_issues: list[dict[str, str]] = []
    batches_processed = 0
    summary_stats: dict | None = None
    llm_config = _load_llm_config(args.llm_config.expanduser() if args.llm_config else None)
    if not llm_config or args.skip_llm:
        reason = "已传 --skip-llm" if args.skip_llm else "未提供 --llm-config 或加载失败"
        print(f"[INFO] 跳过 LLM 审查（{reason}）", file=sys.stderr)
    if llm_config and not args.skip_llm:
        nos = sorted(no for no in chapters if start <= no <= end)
        if args.llm_mode == "raw":
            llm_issues = _run_llm_raw_mode(chapters, nos, llm_config, args.llm_batch_size, args.book_id)
            batches_processed = len(_chunks(nos, args.llm_batch_size))
        else:
            llm_issues, summary_stats = _run_llm_summary_mode(
                chapters, nos, llm_config, args, gender_map=gender_map or None
            )
            batches_processed = summary_stats["extract_batches"] + 1

    llm_analysis: dict[str, Any] = {
        "batches_processed": batches_processed,
        "issue_count": len(llm_issues),
        "issues": llm_issues,
        "llm_mode": args.llm_mode if (llm_config and not args.skip_llm) else "skipped",
    }
    if summary_stats is not None:
        llm_analysis["summary_stats"] = summary_stats

    # 汇总
    summary = {
        "total_issues": len(llm_issues),
        "llm_issues": len(llm_issues),
        "llm_critical": sum(1 for x in llm_issues if x.get("severity") == "CRITICAL"),
        "llm_warning": sum(1 for x in llm_issues if x.get("severity") == "WARNING"),
        "llm_info": sum(1 for x in llm_issues if x.get("severity") == "INFO"),
    }

    report = {
        "book_id": args.book_id,
        "range": f"{start:04d}-{end:04d}",
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "llm_analysis": llm_analysis,
    }

    # 写出报告
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    range_tag = f"{start:04d}_{end:04d}"
    json_path = output_dir / f"consistency_report_{range_tag}.json"
    md_path = output_dir / f"consistency_report_{range_tag}.md"

    write_json(json_path, report)

    md_lines = [
        f"# 一致性审查报告 — {args.book_id}",
        "",
        f"- 审查范围：{report['range']}",
        f"- 生成时间：{report['generated_at']}",
        "",
        "## 摘要",
        "",
        "| 指标 | 数量 |",
        "| --- | ---: |",
        f"| 总问题数 | {summary['total_issues']} |",
        f"| LLM 审查问题 | {summary['llm_issues']} |",
        f"| CRITICAL | {summary['llm_critical']} |",
        f"| WARNING | {summary['llm_warning']} |",
        f"| INFO | {summary['llm_info']} |",
        "",
    ]
    if llm_issues:
        md_lines += ["## LLM 审查问题", ""]
        for issue in llm_issues:
            sev = issue.get("severity", "WARNING")
            md_lines.append(f"**[{sev}]** 章节 {issue.get('chapter_range')} — {issue.get('description')}")
            if issue.get("evidence"):
                md_lines.append(f"> {issue['evidence']}")
            md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({
        "json_report": str(json_path),
        "markdown_report": str(md_path),
        "summary": summary,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
