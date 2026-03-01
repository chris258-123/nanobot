#!/usr/bin/env python3
"""Generate Book B from world settings while reading Book A template memory."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None

try:
    from FlagEmbedding import FlagModel
    FLAG_MODEL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FlagModel = None
    FLAG_MODEL_AVAILABLE = False

ENTITY_ID_KEY_TO_NAME_KEY: dict[str, str] = {
    "entity_id": "entity_name",
    "subject_id": "subject_name",
    "from_id": "from_name",
    "to_id": "to_name",
    "character_id": "character_name",
    "actor_id": "actor_name",
    "target_id": "target_name",
}
PROMPT_DROP_KEYS: set[str] = {
    "memory_type",
    "chapter_sort_key",
    "evidence_chunk_id",
}
PROMPT_METADATA_LABELS: dict[str, str] = {
    "characters": "角色",
    "event": "事件",
    "impact": "影响",
    "causality": "因果",
    "chapter_position": "章节定位",
    "subject_name": "主体",
    "predicate": "谓词",
    "value": "事实值",
    "from_name": "关系起点",
    "to_name": "关系终点",
    "kind": "关系类型",
    "status": "关系状态",
    "relation": "关系描述",
    "fact": "事实描述",
    "summary": "摘要",
    "goal": "目标",
    "conflict": "冲突",
    "scene": "场景",
    "location": "地点",
    "time": "时间",
    "emotion": "情绪",
    "action": "动作",
    "result": "结果",
    "notes": "备注",
}
DEFAULT_ENFORCE_CHINESE_FIELDS: tuple[str, ...] = ("rule", "status", "trait", "goal", "secret", "state")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
BLUEPRINT_TEMPLATE_SOURCES: set[str] = {"none", "booka"}
BLUEPRINT_MODES: set[str] = {"flat", "hierarchical"}
ENDING_STYLES: set[str] = {"closure", "soft", "hook"}
DEFAULT_CONTINUITY_MAX_OPENING_OVERLAP = 0.72
OPENING_REFERENCE_REWRITE_MAX_ATTEMPTS = 2
OPENING_REFERENCE_MIN_ANCHOR_HITS = 1


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _is_probable_internal_entity_id(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    return bool(re.fullmatch(r"[a-z]+_[0-9a-f]{6,}", text))


def _replace_inline_entity_ids(text: str, entity_name_index: dict[str, str]) -> str:
    if not text or not entity_name_index:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return entity_name_index.get(token, token)

    return re.sub(r"[a-z]+_[0-9a-f]{6,}", _replace, text)


def _coerce_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or _is_probable_internal_entity_id(text):
        return None
    return text


def _collect_entity_name_index(payload: Any, index: dict[str, str]) -> None:
    if isinstance(payload, dict):
        for id_key, name_key in ENTITY_ID_KEY_TO_NAME_KEY.items():
            raw_id = payload.get(id_key)
            if not isinstance(raw_id, str):
                continue
            entity_id = raw_id.strip()
            if not entity_id:
                continue
            name = _coerce_name(
                payload.get(name_key)
                or payload.get("canonical_name")
                or payload.get("name")
                or payload.get("subject")
            )
            if name:
                index.setdefault(entity_id, name)

        generic_id = payload.get("id")
        generic_name = _coerce_name(payload.get("name") or payload.get("canonical_name"))
        if isinstance(generic_id, str) and _is_probable_internal_entity_id(generic_id) and generic_name:
            index.setdefault(generic_id, generic_name)

        for value in payload.values():
            _collect_entity_name_index(value, index)
        return

    if isinstance(payload, list):
        for value in payload:
            _collect_entity_name_index(value, index)


def _replace_entity_ids_with_names(payload: Any, entity_name_index: dict[str, str]) -> Any:
    if isinstance(payload, dict):
        replaced: dict[str, Any] = {}
        for key, value in payload.items():
            if key in PROMPT_DROP_KEYS:
                continue

            if key in ENTITY_ID_KEY_TO_NAME_KEY:
                if isinstance(value, str):
                    entity_id = value.strip()
                    if entity_id:
                        replaced[ENTITY_ID_KEY_TO_NAME_KEY[key]] = entity_name_index.get(entity_id, entity_id)
                continue

            if key.endswith("_id") and isinstance(value, str) and _is_probable_internal_entity_id(value):
                name_key = f"{key[:-3]}_name"
                entity_id = value.strip()
                if entity_id:
                    replaced[name_key] = entity_name_index.get(entity_id, entity_id)
                continue

            child = _replace_entity_ids_with_names(value, entity_name_index)
            if _is_empty_value(child):
                continue
            replaced[key] = child
        return replaced

    if isinstance(payload, list):
        cleaned_list = []
        for value in payload:
            child = _replace_entity_ids_with_names(value, entity_name_index)
            if _is_empty_value(child):
                continue
            cleaned_list.append(child)
        return cleaned_list

    if isinstance(payload, str):
        text = payload.strip()
        if _is_probable_internal_entity_id(text):
            return entity_name_index.get(text, text)
        return _replace_inline_entity_ids(payload, entity_name_index)

    return payload


def _metadata_to_prompt_elements(metadata: dict[str, Any]) -> dict[str, Any]:
    elements: dict[str, Any] = {}
    for key, value in metadata.items():
        if _is_empty_value(value):
            continue
        label = PROMPT_METADATA_LABELS.get(key)
        if label is None and re.search(r"[\u4e00-\u9fff]", key):
            label = key
        if label is None:
            continue
        existing = elements.get(label)
        if existing is None:
            elements[label] = value
            continue
        if isinstance(existing, list):
            existing.append(value)
            continue
        elements[label] = [existing, value]
    return elements


def _sanitize_recent_summaries_for_prompt(recent_summaries: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in recent_summaries:
        text = str(item or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_chapter_title_for_compare(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^[#*\s>]+", "", value)
    value = re.sub(r"[*_`]+", "", value)
    value = re.sub(r"^第[0-9零一二三四五六七八九十百千万两〇]+章[：:\-\s]*", "", value).strip()
    return value.strip("：:，,。；;、-—_ ")


def _sanitize_generated_chapter_text(chapter_text: str, chapter_title: str) -> str:
    lines = [line.rstrip() for line in str(chapter_text or "").splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    chapter_title_norm = _normalize_chapter_title_for_compare(chapter_title)
    removed = 0
    while lines and removed < 3:
        first = lines[0].strip()
        if not first:
            lines.pop(0)
            removed += 1
            continue
        first_norm = _normalize_chapter_title_for_compare(first)
        if first.startswith("#") and first_norm == chapter_title_norm:
            lines.pop(0)
            removed += 1
            continue
        if first_norm and first_norm == chapter_title_norm:
            lines.pop(0)
            removed += 1
            continue
        break
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def _sanitize_text_list(value: Any, *, max_items: int, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        text = text.replace("\n", " ").strip()
        if len(text) > max_len:
            text = text[:max_len].rstrip()
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


ANCHOR_PAIR_CHARS: tuple[tuple[str, str], ...] = (
    ("“", "”"),
    ("‘", "’"),
    ("（", "）"),
    ("【", "】"),
    ("《", "》"),
    ("「", "」"),
    ("『", "』"),
)


def _close_unbalanced_pairs(text: str) -> str:
    value = str(text or "")
    for left, right in ANCHOR_PAIR_CHARS:
        diff = value.count(left) - value.count(right)
        if diff > 0:
            value += right * min(diff, 3)
    return value


def _truncate_text_on_sentence_boundary(text: str, *, max_len: int) -> tuple[str, bool]:
    value = str(text or "").strip()
    if not value:
        return ("", False)
    if len(value) <= max_len:
        return (value, False)
    cut = value[: max(int(max_len), 1)].rstrip()
    min_boundary = max(24, int(max_len * 0.45))
    pivot = -1
    for punct in "。！？!?；;":
        idx = cut.rfind(punct)
        if idx >= min_boundary:
            pivot = max(pivot, idx)
    if pivot < 0:
        for punct in "，,、":
            idx = cut.rfind(punct)
            if idx >= min_boundary:
                pivot = max(pivot, idx)
    if pivot >= 0:
        return (cut[: pivot + 1].rstrip(), True)
    return (cut, True)


def _build_tail_anchor_excerpt(
    chapter_text: str,
    *,
    min_chars: int = 80,
    max_chars: int = 260,
) -> tuple[str, bool]:
    compact = re.sub(r"\s+", " ", str(chapter_text or "").strip())
    if not compact:
        return ("", False)
    segments = [seg.strip() for seg in re.split(r"(?<=[。！？!?])\s+", compact) if seg.strip()]
    merged = ""
    if segments:
        picked: list[str] = []
        total = 0
        for seg in reversed(segments):
            picked.insert(0, seg)
            total += len(seg)
            if total >= min_chars and len(picked) >= 2:
                break
        merged = "".join(picked).strip()
    if not merged:
        merged = compact
    merged, was_truncated = _truncate_text_on_sentence_boundary(merged, max_len=max_chars)
    merged = _close_unbalanced_pairs(merged.strip())
    return (merged, was_truncated)


def _sanitize_cross_batch_anchor_items(
    values: list[str],
    *,
    max_items: int = 3,
    max_len: int = 320,
) -> tuple[list[str], bool]:
    cleaned: list[str] = []
    truncated = False
    seen: set[str] = set()
    for item in values:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if not text:
            continue
        text, was_truncated = _truncate_text_on_sentence_boundary(text, max_len=max_len)
        text = _close_unbalanced_pairs(text.strip())
        if was_truncated:
            truncated = True
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return (cleaned, truncated)


GENERIC_HOOK_PHRASES: tuple[str, ...] = (
    "留下推动下一章的悬念",
    "留下悬念",
    "推动下一章",
    "未完待续",
    "待续",
    "请看下章",
    "请看下一章",
    "to be continued",
)

LINK_WORD_STOPWORDS: set[str] = {
    "下一章",
    "本章",
    "线索",
    "事件",
    "角色",
    "他们",
    "我们",
    "这个",
    "那个",
    "继续",
    "需要",
    "问题",
}

REFERENCE_ENTITY_STOPWORDS: set[str] = {
    *LINK_WORD_STOPWORDS,
    "章节",
    "开头",
    "结尾",
    "开篇",
    "承接",
    "剧情",
    "镜头",
    "场景",
    "状态",
    "目标",
    "冲突",
    "线索",
    "广播",
    "旋律",
    "计划",
}


def _normalize_link_items(raw: Any, *, max_items: int = 3, max_len: int = 120) -> list[str]:
    if isinstance(raw, list):
        return _sanitize_text_list(raw, max_items=max_items, max_len=max_len)
    if isinstance(raw, str):
        parts = [part.strip() for part in re.split(r"[；;。\n]+", raw) if part.strip()]
        return _sanitize_text_list(parts, max_items=max_items, max_len=max_len)
    return []


def _is_generic_ending_hook(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    if "待修复" in text:
        return True
    text_compact = re.sub(r"\s+", "", text)
    for phrase in GENERIC_HOOK_PHRASES:
        if phrase in text_compact:
            return True
    if len(text_compact) <= 10:
        return True
    return False


def _normalize_for_link_match(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower())


def _char_ngrams(text: str, size: int = 2) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", str(text or ""))
    if len(chars) < size:
        return set()
    return {"".join(chars[idx : idx + size]) for idx in range(len(chars) - size + 1)}


def _opening_overlap_ratio(left: str, right: str) -> float:
    left_bigrams = _char_ngrams(left, size=2)
    right_bigrams = _char_ngrams(right, size=2)
    if not left_bigrams or not right_bigrams:
        return 0.0
    union = left_bigrams | right_bigrams
    if not union:
        return 0.0
    return len(left_bigrams & right_bigrams) / len(union)


def _extract_opening_excerpt(text: str, *, max_chars: int = 420) -> str:
    compact = str(text or "").strip()
    if not compact:
        return ""
    return compact[: max(int(max_chars), 80)]


def _split_chapter_opening_window(
    text: str,
    *,
    max_chars: int = 820,
    min_chars: int = 180,
) -> tuple[str, str]:
    compact = str(text or "").strip()
    if not compact:
        return "", ""
    min_chars = max(int(min_chars or 0), 80)
    limit = min(max(int(max_chars or 0), min_chars), len(compact))
    split_idx = limit
    best = -1
    for marker in ("\n\n", "\n", "。", "！", "？", ".", "!", "?"):
        idx = compact.rfind(marker, min_chars, limit)
        if idx < 0:
            continue
        candidate = idx + len(marker)
        if candidate > best:
            best = candidate
    if best >= min_chars:
        split_idx = best
    opening = compact[:split_idx].strip()
    remainder = compact[split_idx:].lstrip()
    return opening, remainder


def _link_items_match(left: str, right: str) -> bool:
    l_norm = _normalize_for_link_match(left)
    r_norm = _normalize_for_link_match(right)
    if not l_norm or not r_norm:
        return False
    if l_norm in r_norm or r_norm in l_norm:
        return True
    left_tokens = {
        token for token in _extract_keywords(left) if len(token) >= 2 and token not in LINK_WORD_STOPWORDS
    }
    right_tokens = {
        token for token in _extract_keywords(right) if len(token) >= 2 and token not in LINK_WORD_STOPWORDS
    }
    if left_tokens & right_tokens:
        return True
    left_bigrams = _char_ngrams(left, size=2)
    right_bigrams = _char_ngrams(right, size=2)
    return len(left_bigrams & right_bigrams) >= 2


def _validate_blueprint_continuity(
    chapters: list[dict[str, Any]],
    *,
    start_chapter: int = 1,
    ending_style: str = "closure",
) -> dict[str, Any]:
    issues: list[str] = []
    pairs: list[dict[str, Any]] = []
    generic_hook_count = 0
    broken_links = 0
    normalized: list[dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        normalized.append(
            {
                "chapter_no": str(chapter.get("chapter_no") or "").zfill(4),
                "title": str(chapter.get("title") or "").strip(),
                "beat_outline": _sanitize_text_list(chapter.get("beat_outline"), max_items=8, max_len=120),
                "ending_hook": str(chapter.get("ending_hook") or "").strip(),
                "carry_over_to_next": _normalize_link_items(chapter.get("carry_over_to_next")),
                "open_with": _normalize_link_items(chapter.get("open_with")),
            }
        )

    style = str(ending_style or "closure").strip().lower()
    if style not in ENDING_STYLES:
        style = "closure"
    hook_required = style == "hook"

    for idx, chapter in enumerate(normalized):
        chapter_no = chapter["chapter_no"] or f"{idx + 1:04d}"
        beats = chapter["beat_outline"]
        ending_hook = chapter["ending_hook"]
        if _is_generic_ending_hook(ending_hook):
            generic_hook_count += 1
            if hook_required:
                issues.append(f"章节 {chapter_no} ending_hook 过于通用或为空")
        requires_open_with = idx > 0 or int(start_chapter or 1) > 1
        if idx < len(normalized) - 1 and not chapter["carry_over_to_next"]:
            issues.append(f"章节 {chapter_no} 缺少 carry_over_to_next")
        if requires_open_with and not chapter["open_with"]:
            issues.append(f"章节 {chapter_no} 缺少 open_with")
        if requires_open_with and chapter["open_with"] and beats:
            first_beat = str(beats[0]).strip()
            if first_beat and not any(_link_items_match(first_beat, item) for item in chapter["open_with"]):
                issues.append(f"章节 {chapter_no} 第一拍未与 open_with 对齐")

    for idx in range(len(normalized) - 1):
        current = normalized[idx]
        nxt = normalized[idx + 1]
        matches: list[dict[str, str]] = []
        for carry in current["carry_over_to_next"]:
            matched_open = next((item for item in nxt["open_with"] if _link_items_match(carry, item)), "")
            if matched_open:
                matches.append({"carry_over": carry, "open_with": matched_open})

        pair_issues: list[str] = []
        if not current["carry_over_to_next"] or not nxt["open_with"]:
            pair_issues.append("缺少 carry_over_to_next 或 open_with")
        elif not matches:
            pair_issues.append("carry_over_to_next 未在下一章 open_with 中承接")

        if pair_issues:
            broken_links += 1
            issues.append(
                f"{current['chapter_no']}->{nxt['chapter_no']} 承接失败: {'; '.join(pair_issues)}"
            )

        pairs.append(
            {
                "from_chapter": current["chapter_no"],
                "to_chapter": nxt["chapter_no"],
                "from_hook": current["ending_hook"],
                "from_carry_over_to_next": current["carry_over_to_next"],
                "to_open_with": nxt["open_with"],
                "matched_links": matches,
                "ok": not pair_issues,
                "issues": pair_issues,
            }
        )

    return {
        "ok": not issues,
        "issues": issues,
        "pairs": pairs,
        "summary": {
            "chapter_count": len(normalized),
            "generic_hook_count": generic_hook_count,
            "broken_link_count": broken_links,
        },
    }


def _auto_patch_blueprint_links(chapters: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Inject one exact carry-over item into next chapter open_with when link is missing."""
    patched: list[dict[str, Any]] = copy.deepcopy(chapters)
    patched_count = 0
    for idx in range(len(patched) - 1):
        current = patched[idx]
        nxt = patched[idx + 1]
        carries = _normalize_link_items(current.get("carry_over_to_next"))
        if not carries:
            continue
        next_open = _normalize_link_items(nxt.get("open_with"))
        if any(_link_items_match(carry, open_item) for carry in carries for open_item in next_open):
            continue
        anchor = carries[0]
        next_open = [anchor] + [item for item in next_open if not _link_items_match(anchor, item)]
        nxt["open_with"] = _sanitize_text_list(next_open, max_items=3, max_len=120)
        patched_count += 1
    return patched, patched_count


def _build_opening_anchor_sentence(open_with_item: str) -> str:
    text = str(open_with_item or "").strip()
    if not text:
        return "承接上章，局势未止。"
    text = text.strip("。！？!?,，；;：:")
    if not text:
        return "承接上章，局势未止。"
    if text.startswith("承接上章"):
        return text if text.endswith(("。", "！", "？")) else f"{text}。"
    return f"承接上章：{text}。"


def _auto_patch_opening_beats(
    chapters: list[dict[str, Any]],
    *,
    start_chapter: int,
) -> tuple[list[dict[str, Any]], int]:
    """Inject a soft opening anchor beat when beat[0] does not reflect open_with."""
    patched: list[dict[str, Any]] = copy.deepcopy(chapters)
    patched_count = 0
    for idx, chapter in enumerate(patched):
        requires_open_with = idx > 0 or int(start_chapter or 1) > 1
        open_with = _normalize_link_items(chapter.get("open_with"))
        if not requires_open_with or not open_with:
            continue
        beats = _sanitize_text_list(chapter.get("beat_outline"), max_items=8, max_len=120)
        first_beat = str(beats[0]).strip() if beats else ""
        if first_beat and any(_link_items_match(first_beat, item) for item in open_with):
            chapter["beat_outline"] = beats
            continue
        anchor = _build_opening_anchor_sentence(open_with[0])
        updated = [anchor] + beats if beats else [anchor]
        chapter["beat_outline"] = _sanitize_text_list(updated, max_items=8, max_len=120)
        patched_count += 1
    return patched, patched_count


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", str(text or ""))
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _extract_reference_entities(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}", str(text or ""))
    entities: list[str] = []
    seen: set[str] = set()
    weak_chars = set("的一了在是和就都而及与着将把被让给对向从到于并其该这那每各先后上中下里外内前后再已仍")
    for block in tokens:
        block = block.strip()
        for size in (2, 3):
            if len(block) < size:
                continue
            for idx in range(len(block) - size + 1):
                token = block[idx : idx + size]
                if token in seen:
                    continue
                seen.add(token)
                if token in REFERENCE_ENTITY_STOPWORDS:
                    continue
                if token.endswith(("章节", "问题", "状态", "线索", "剧情", "事件", "计划", "镜头")):
                    continue
                if token[0] in weak_chars or token[-1] in weak_chars:
                    continue
                entities.append(token)
    return entities


def _count_entity_pronouns(text: str, entity: str) -> dict[str, int]:
    if not entity:
        return {"他": 0, "她": 0, "TA": 0}
    counters = {"他": 0, "她": 0, "TA": 0}
    pattern = re.compile(
        rf"{re.escape(entity)}[^。！？\n]{{0,12}}([他她])|([他她])[^。！？\n]{{0,12}}{re.escape(entity)}|"
        rf"{re.escape(entity)}[^。！？\n]{{0,12}}(?:TA|ta)|(?:TA|ta)[^。！？\n]{{0,12}}{re.escape(entity)}"
    )
    for match in pattern.finditer(str(text or "")):
        token = ""
        if match.group(1):
            token = match.group(1)
        elif match.group(2):
            token = match.group(2)
        else:
            token = "TA"
        if token in {"他", "她", "TA"}:
            counters[token] += 1
    return counters


def _detect_narrative_pov(text: str) -> str:
    content = str(text or "")
    first_hits = len(re.findall(r"(?:我|我们|咱们)", content))
    third_hits = len(re.findall(r"(?:他|她|他们|她们|TA|ta)", content))
    if first_hits >= 3 and first_hits >= int(third_hits * 1.2):
        return "first_person"
    if third_hits >= 3 and third_hits >= int(first_hits * 1.2):
        return "third_person"
    return "mixed"


def _build_opening_reference_profile(
    *,
    previous_chapter_carry_over: list[str],
    current_chapter_open_with: list[str],
    previous_chapter_opening_text: str,
    chapter_opening_text: str,
    recent_continuity_capsules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_chunks: list[str] = []
    source_chunks.extend(_normalize_link_items(previous_chapter_carry_over, max_items=4, max_len=180))
    source_chunks.extend(_normalize_link_items(current_chapter_open_with, max_items=4, max_len=180))
    if previous_chapter_opening_text:
        source_chunks.append(_extract_opening_excerpt(previous_chapter_opening_text, max_chars=560))
    if chapter_opening_text:
        source_chunks.append(_extract_opening_excerpt(chapter_opening_text, max_chars=560))

    entity_scores: dict[str, int] = {}
    entity_first_pos: dict[str, int] = {}
    for idx, chunk in enumerate(source_chunks):
        for token in _extract_reference_entities(chunk):
            entity_scores[token] = entity_scores.get(token, 0) + 1
            entity_first_pos.setdefault(token, idx)

    if recent_continuity_capsules:
        for capsule in recent_continuity_capsules[-3:]:
            for name in _sanitize_text_list(capsule.get("carry_entities"), max_items=8, max_len=32):
                if len(name) < 2:
                    continue
                entity_scores[name] = entity_scores.get(name, 0) + 1
                entity_first_pos.setdefault(name, len(source_chunks))

    anchor_entities = sorted(
        entity_scores.keys(),
        key=lambda item: (-entity_scores.get(item, 0), entity_first_pos.get(item, 10**6), len(item)),
    )[:6]

    context_text = "\n".join(source_chunks)
    entity_pronoun_hint: dict[str, str] = {}
    for entity in anchor_entities:
        counters = _count_entity_pronouns(context_text, entity)
        if counters["他"] >= 2 and counters["他"] > counters["她"]:
            entity_pronoun_hint[entity] = "他"
        elif counters["她"] >= 2 and counters["她"] > counters["他"]:
            entity_pronoun_hint[entity] = "她"
        elif counters["TA"] >= 2:
            entity_pronoun_hint[entity] = "TA"

    pov_hint = _detect_narrative_pov(context_text)
    return {
        "anchor_entities": anchor_entities,
        "entity_pronoun_hint": entity_pronoun_hint,
        "pov_hint": pov_hint,
        "min_anchor_hits": OPENING_REFERENCE_MIN_ANCHOR_HITS,
    }


def _validate_opening_reference_consistency(
    *,
    chapter_text: str,
    reference_profile: dict[str, Any] | None,
    opening_max_chars: int = 900,
    min_anchor_hits: int = OPENING_REFERENCE_MIN_ANCHOR_HITS,
) -> dict[str, Any]:
    profile = reference_profile if isinstance(reference_profile, dict) else {}
    opening_text = _extract_opening_excerpt(chapter_text, max_chars=max(260, int(opening_max_chars or 0)))
    anchor_entities = [
        str(item).strip()
        for item in (profile.get("anchor_entities") or [])
        if str(item).strip()
    ]
    pronoun_hints = {
        str(name).strip(): str(hint).strip()
        for name, hint in (profile.get("entity_pronoun_hint") or {}).items()
        if str(name).strip() and str(hint).strip()
    }
    expected_pov = str(profile.get("pov_hint") or "").strip()

    issues: list[str] = []
    hit_entities = [name for name in anchor_entities if name in opening_text]
    required_hits = max(int(min_anchor_hits or 0), 0)
    if anchor_entities and len(hit_entities) < required_hits:
        issues.append(
            f"章节开头主语锚点不足：要求至少 {required_hits} 个，实际 {len(hit_entities)} 个"
        )

    pronoun_drifts: list[str] = []
    for entity, expected in pronoun_hints.items():
        if entity not in opening_text or expected not in {"他", "她"}:
            continue
        counters = _count_entity_pronouns(opening_text, entity)
        if expected == "他" and counters["她"] > counters["他"] and counters["她"] >= 1:
            pronoun_drifts.append(f"{entity}(期望他, 检测她={counters['她']})")
        if expected == "她" and counters["他"] > counters["她"] and counters["他"] >= 1:
            pronoun_drifts.append(f"{entity}(期望她, 检测他={counters['他']})")
    if pronoun_drifts:
        issues.append("章节开头指代漂移：" + "；".join(pronoun_drifts[:4]))

    opening_pov = _detect_narrative_pov(opening_text)
    if expected_pov in {"first_person", "third_person"} and opening_pov in {"first_person", "third_person"}:
        if expected_pov != opening_pov:
            issues.append(
                f"章节开头叙述视角偏移：期望 {expected_pov}，实际 {opening_pov}"
            )

    return {
        "ok": not issues,
        "issues": issues,
        "metrics": {
            "anchor_expected": len(anchor_entities),
            "anchor_hits": len(hit_entities),
            "anchor_hit_entities": hit_entities[:6],
            "pov_expected": expected_pov,
            "pov_detected": opening_pov,
        },
    }


def _collect_prev_entities_from_hard_pack(hard_pack: dict[str, Any]) -> list[str]:
    prev_context = hard_pack.get("prev_context") if isinstance(hard_pack, dict) else None
    if not isinstance(prev_context, dict):
        return []

    entities: list[str] = []
    for row in prev_context.get("character_state", []) or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("character_name") or row.get("entity_name") or "").strip()
        if name and len(name) >= 2:
            entities.append(name)
    for row in prev_context.get("recent_relations", []) or []:
        if not isinstance(row, dict):
            continue
        for key in ("from_name", "to_name"):
            name = str(row.get(key) or "").strip()
            if name and len(name) >= 2:
                entities.append(name)
    deduped = list(dict.fromkeys(entities))
    return deduped[:20]


def _collect_prev_threads_from_hard_pack(hard_pack: dict[str, Any]) -> list[str]:
    prev_context = hard_pack.get("prev_context") if isinstance(hard_pack, dict) else None
    if not isinstance(prev_context, dict):
        return []
    threads: list[str] = []
    for row in prev_context.get("open_threads", []) or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("notes") or row.get("name") or row.get("summary") or row.get("title") or "").strip()
        if text:
            threads.append(text)
    return list(dict.fromkeys(threads))[:12]


def _build_fallback_continuity_capsule(
    *,
    chapter_no: str,
    chapter_title: str,
    chapter_plan: dict[str, Any],
    chapter_text: str,
    chapter_summary: str,
    hard_pack_b: dict[str, Any],
) -> dict[str, Any]:
    summary = str(chapter_summary or "").strip()
    if not summary:
        compact = str(chapter_text or "").replace("\n", " ").strip()
        summary = compact[:220] if compact else chapter_title

    touched_rules: list[str] = []
    hard_rules = hard_pack_b.get("hard_rules") if isinstance(hard_pack_b, dict) else None
    if isinstance(hard_rules, list):
        for row in hard_rules:
            if not isinstance(row, dict):
                continue
            predicate = str(row.get("predicate") or "").strip()
            value = str(row.get("value") or "").strip()
            if predicate:
                touched_rules.append(f"{predicate}: {value}" if value else predicate)
            if len(touched_rules) >= 6:
                break

    return {
        "chapter_no": str(chapter_no or "").zfill(4),
        "chapter_title": str(chapter_title or "").strip(),
        "chapter_summary": summary[:300],
        "carry_entities": _collect_prev_entities_from_hard_pack(hard_pack_b)[:10],
        "state_deltas": _sanitize_text_list(
            [chapter_plan.get("goal", ""), chapter_plan.get("conflict", "")],
            max_items=4,
            max_len=80,
        ),
        "open_threads": _collect_prev_threads_from_hard_pack(hard_pack_b)[:8],
        "hard_rules_touched": _sanitize_text_list(touched_rules, max_items=6, max_len=100),
        "next_chapter_must_answer": _sanitize_text_list([chapter_plan.get("ending_hook", "")], max_items=3, max_len=120),
    }


def _sanitize_continuity_capsules_for_prompt(capsules: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    if not isinstance(capsules, list):
        return []
    size = max(int(window or 0), 1)
    cleaned: list[dict[str, Any]] = []
    for raw in capsules[-size:]:
        if not isinstance(raw, dict):
            continue
        chapter_no = str(raw.get("chapter_no") or "").strip()
        chapter_title = str(raw.get("chapter_title") or "").strip()
        summary = str(raw.get("chapter_summary") or "").strip()
        carry_entities = _sanitize_text_list(raw.get("carry_entities"), max_items=10, max_len=32)
        state_deltas = _sanitize_text_list(raw.get("state_deltas"), max_items=8, max_len=120)
        open_threads = _sanitize_text_list(raw.get("open_threads"), max_items=8, max_len=120)
        hard_rules_touched = _sanitize_text_list(raw.get("hard_rules_touched"), max_items=8, max_len=120)
        next_chapter_must_answer = _sanitize_text_list(raw.get("next_chapter_must_answer"), max_items=6, max_len=120)
        if not (summary or carry_entities or open_threads):
            continue
        cleaned.append(
            {
                "chapter_no": chapter_no.zfill(4) if chapter_no else "",
                "chapter_title": chapter_title,
                "chapter_summary": summary[:300],
                "carry_entities": carry_entities,
                "state_deltas": state_deltas,
                "open_threads": open_threads,
                "hard_rules_touched": hard_rules_touched,
                "next_chapter_must_answer": next_chapter_must_answer,
            }
        )
    return cleaned


def _validate_chapter_continuity(
    *,
    chapter_text: str,
    recent_capsules: list[dict[str, Any]],
    hard_pack_b: dict[str, Any],
    open_with_expected: list[str] | None,
    required_carry_over: list[str] | None = None,
    previous_chapter_opening_text: str | None = None,
    max_opening_overlap: float = DEFAULT_CONTINUITY_MAX_OPENING_OVERLAP,
    first_beat_expected: str | None = None,
    continuity_window: int,
    min_entities: int,
    min_open_threads: int,
    min_chapter_chars: int = 2600,
) -> dict[str, Any]:
    text = str(chapter_text or "")
    issues: list[str] = []
    metrics: dict[str, Any] = {}

    body_chars = len(text)
    required_chars = max(int(min_chapter_chars or 0), 0)
    metrics["chapter_body_chars"] = body_chars
    metrics["chapter_chars_min_required"] = required_chars
    if required_chars > 0 and body_chars < required_chars:
        issues.append(f"章节正文过短：要求至少 {required_chars} 字，实际 {body_chars} 字")

    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    metrics["zh_chars"] = zh_chars
    metrics["latin_chars"] = latin_chars
    if zh_chars < max(120, latin_chars * 3):
        issues.append("中文占比过低，疑似未稳定中文叙事")

    prompt_capsules = _sanitize_continuity_capsules_for_prompt(recent_capsules, continuity_window)
    candidate_entities: list[str] = []
    for capsule in prompt_capsules:
        candidate_entities.extend(_sanitize_text_list(capsule.get("carry_entities"), max_items=10, max_len=32))
    candidate_entities.extend(_collect_prev_entities_from_hard_pack(hard_pack_b))
    deduped_entities = [name for name in dict.fromkeys(candidate_entities) if len(name) >= 2]
    matched_entities = [name for name in deduped_entities if name in text]
    metrics["entity_candidates"] = len(deduped_entities)
    metrics["entity_matched"] = len(matched_entities)
    required_entities = min(max(int(min_entities), 0), len(deduped_entities))
    if required_entities > 0 and len(matched_entities) < required_entities:
        issues.append(
            f"承接实体不足：要求至少 {required_entities} 个，实际 {len(matched_entities)} 个（候选 {len(deduped_entities)}）"
        )

    candidate_threads: list[str] = []
    for capsule in prompt_capsules:
        candidate_threads.extend(_sanitize_text_list(capsule.get("open_threads"), max_items=8, max_len=120))
    candidate_threads.extend(_collect_prev_threads_from_hard_pack(hard_pack_b))
    deduped_threads = [item for item in dict.fromkeys(candidate_threads) if item]
    thread_hits = 0
    for thread in deduped_threads:
        keywords = _extract_keywords(thread)
        if any(keyword in text for keyword in keywords):
            thread_hits += 1
    metrics["thread_candidates"] = len(deduped_threads)
    metrics["thread_hits"] = thread_hits
    required_threads = min(max(int(min_open_threads), 0), len(deduped_threads))
    if required_threads > 0 and thread_hits < required_threads:
        issues.append(
            f"开放线索推进不足：要求至少 {required_threads} 条，实际 {thread_hits} 条（候选 {len(deduped_threads)}）"
        )

    expected_open_with = _normalize_link_items(open_with_expected)
    opening_text = text[: max(400, int(len(text) * 0.15))]
    opening_hits = 0
    for item in expected_open_with:
        if _link_items_match(item, opening_text):
            opening_hits += 1
    metrics["open_with_expected"] = len(expected_open_with)
    metrics["open_with_opening_hits"] = opening_hits
    if expected_open_with and opening_hits < 1:
        issues.append("章节开头未承接 open_with 要点")

    carry_over_items = _normalize_link_items(required_carry_over)
    carry_hits = 0
    for item in carry_over_items:
        if _link_items_match(item, opening_text):
            carry_hits += 1
    metrics["carry_over_expected"] = len(carry_over_items)
    metrics["carry_over_opening_hits"] = carry_hits
    if carry_over_items and carry_hits < 1:
        issues.append("章节开头未承接上一章 carry_over 要点")

    previous_opening = _extract_opening_excerpt(previous_chapter_opening_text or "", max_chars=len(opening_text))
    opening_overlap = _opening_overlap_ratio(previous_opening, opening_text) if previous_opening else 0.0
    overlap_threshold = min(max(float(max_opening_overlap or 0.0), 0.0), 0.99)
    metrics["opening_overlap_ratio"] = round(opening_overlap, 6)
    metrics["opening_overlap_threshold"] = overlap_threshold
    if previous_opening:
        prev_chars = len(re.findall(r"[\u4e00-\u9fff]", previous_opening))
        curr_chars = len(re.findall(r"[\u4e00-\u9fff]", opening_text))
        metrics["previous_opening_zh_chars"] = prev_chars
        metrics["current_opening_zh_chars"] = curr_chars
        if prev_chars >= 60 and curr_chars >= 60 and opening_overlap >= overlap_threshold:
            issues.append(f"章节开篇疑似复写上一章（开头重叠度 {opening_overlap:.2f}）")

    first_beat = str(first_beat_expected or "").strip()
    first_beat_hit = 0
    if first_beat:
        if _link_items_match(first_beat, opening_text):
            first_beat_hit = 1
        else:
            beat_keywords = [token for token in _extract_keywords(first_beat) if token not in LINK_WORD_STOPWORDS]
            if any(keyword in opening_text for keyword in beat_keywords):
                first_beat_hit = 1
        if first_beat_hit < 1:
            issues.append("章节开篇未落地 beat_outline[0]（承接拍点）")
    metrics["first_beat_expected"] = 1 if first_beat else 0
    metrics["first_beat_opening_hit"] = first_beat_hit

    return {
        "ok": not issues,
        "issues": issues,
        "metrics": metrics,
    }


def _issues_need_opening_rewrite(issues: list[str] | None) -> bool:
    if not issues:
        return False
    markers = (
        "章节开头未承接",
        "章节开篇未落地",
        "章节开篇疑似复写",
        "章节开头主语锚点不足",
        "章节开头指代漂移",
        "章节开头叙述视角偏移",
        "开头未承接",
        "开篇未落地",
        "开篇疑似复写",
        "主语锚点",
        "指代漂移",
        "叙述视角",
        "open_with",
        "carry_over",
        "beat_outline[0]",
    )
    for issue in issues:
        text = str(issue or "")
        if any(marker in text for marker in markers):
            return True
    return False


def _ensure_hard_context_shape(hard_pack: Any) -> dict[str, Any]:
    default_prev = {
        "character_state": [],
        "recent_relations": [],
        "open_threads": [],
    }
    if not isinstance(hard_pack, dict):
        return {
            "hard_rules": [],
            "prev_context": default_prev,
        }

    hard_rules = hard_pack.get("hard_rules")
    if not isinstance(hard_rules, list):
        hard_rules = []

    prev_context_raw = hard_pack.get("prev_context")
    prev_context = prev_context_raw if isinstance(prev_context_raw, dict) else {}
    character_state = prev_context.get("character_state")
    recent_relations = prev_context.get("recent_relations")
    open_threads = prev_context.get("open_threads")
    return {
        "hard_rules": hard_rules,
        "prev_context": {
            "character_state": character_state if isinstance(character_state, list) else [],
            "recent_relations": recent_relations if isinstance(recent_relations, list) else [],
            "open_threads": open_threads if isinstance(open_threads, list) else [],
        },
    }


def _strip_markdown_code_blocks(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = _strip_markdown_code_blocks(content)
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    sliced = cleaned[start : end + 1] if start != -1 and end > start else cleaned
    repaired = re.sub(r",\s*([}\]])", r"\1", sliced)
    parsed = json.loads(repaired)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _is_delta_json_parse_error(exc: BaseException) -> bool:
    for item in _iter_exception_chain(exc):
        if isinstance(item, json.JSONDecodeError):
            return True
        if item.__class__.__name__ == "DeltaParseError":
            return True
        message = str(item)
        if "Expecting ',' delimiter" in message:
            return True
        if "Failed to parse delta JSON" in message:
            return True
    return False


def _classify_runtime_error(exc: BaseException) -> str:
    if _is_delta_json_parse_error(exc):
        return "DELTA_JSON_PARSE_FAILED"
    text = " | ".join(str(item) for item in _iter_exception_chain(exc))
    if "Server disconnected" in text:
        return "LLM_SERVER_DISCONNECTED"
    return "RUNTIME_ERROR"


def _format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _load_world_spec(world_text: str, world_config_path: str) -> dict[str, Any]:
    world_text = (world_text or "").strip()
    if world_config_path:
        loaded = json.loads(Path(world_config_path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--world-config JSON must be an object")
        if world_text:
            loaded["world_appendix"] = world_text
        return loaded
    if world_text:
        return {"world": world_text}
    raise ValueError("Provide --world or --world-config")


def _int_to_chinese(num: int) -> str:
    if num <= 0:
        return "零"
    digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千"]
    chars: list[str] = []
    zero_pending = False
    text = str(num)
    for idx, ch in enumerate(text):
        digit = int(ch)
        unit_pos = len(text) - idx - 1
        if digit == 0:
            zero_pending = True
            continue
        if zero_pending and chars:
            chars.append("零")
        zero_pending = False
        if not (digit == 1 and unit_pos == 1 and not chars):
            chars.append(digits[digit])
        chars.append(units[unit_pos])
    result = "".join(chars).rstrip("零")
    return result or "零"


def _format_chapter_title(chapter_no: int, raw_title: str, raw_goal: str) -> str:
    title = str(raw_title or "").strip()
    title = re.sub(r"^第[0-9零一二三四五六七八九十百千万两〇]+章[：:\-\s]*", "", title).strip()
    title = title.strip("：:，,。；;、-—_ ")
    if not title:
        fallback = str(raw_goal or "").strip().strip("：:，,。；;、-—_ ")
        title = fallback[:18] if fallback else "回声未熄"
    return f"第{_int_to_chinese(chapter_no)}章：{title}"


def _strip_chapter_title_prefix(title: str) -> str:
    text = str(title or "").strip()
    text = re.sub(r"^第[0-9零一二三四五六七八九十百千万两〇]+章[：:\-\s]*", "", text).strip()
    return text.strip("：:，,。；;、-—_ ")


TITLE_TEMPLATE_STOPWORDS: set[str] = {
    "必须",
    "需要",
    "继续",
    "开始",
    "结束",
    "本章",
    "章节",
    "线索",
    "冲突",
    "推进",
    "计划",
    "任务",
    "问题",
}
TITLE_ACTION_VERBS: tuple[str, ...] = (
    "追索",
    "对峙",
    "回收",
    "切断",
    "潜入",
    "审计",
    "破译",
    "夺回",
    "逼近",
    "清算",
)
TITLE_OPENING_FRAMES: tuple[str, ...] = (
    "关于",
    "在第",
    "在",
    "当",
)


def _title_pattern_key(title: str) -> str:
    core = _strip_chapter_title_prefix(title)
    core = re.sub(r"\s+", "", core)
    if not core:
        return "EMPTY"
    core = re.sub(r"[0-9零一二三四五六七八九十百千万两〇]+", "#", core)
    core = re.sub(r"[A-Za-z]+", "A", core)
    core = re.sub(r"[\u4e00-\u9fff]+", "中", core)
    core = re.sub(r"[\W_]+", "|", core).strip("|")
    return core[:120] if core else "EMPTY"


def _title_char_ngrams(text: str, size: int = 2) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text)
    if len(chars) < size:
        return set()
    return {"".join(chars[idx : idx + size]) for idx in range(len(chars) - size + 1)}


def _titles_too_similar(prev_title: str, cur_title: str) -> bool:
    prev_core = _strip_chapter_title_prefix(prev_title)
    cur_core = _strip_chapter_title_prefix(cur_title)
    if not prev_core or not cur_core:
        return False
    if prev_core == cur_core:
        return True
    if any(prev_core.startswith(frame) and cur_core.startswith(frame) for frame in TITLE_OPENING_FRAMES):
        return True
    prev_flags = (
        "：" in prev_core or ":" in prev_core,
        "的" in prev_core,
        "与" in prev_core or "及" in prev_core,
    )
    cur_flags = (
        "：" in cur_core or ":" in cur_core,
        "的" in cur_core,
        "与" in cur_core or "及" in cur_core,
    )
    frame_overlap = sum(1 for left, right in zip(prev_flags, cur_flags) if left == right)
    prev_head = re.split(r"[：:]", prev_core, maxsplit=1)[0].strip()
    cur_head = re.split(r"[：:]", cur_core, maxsplit=1)[0].strip()
    if prev_head and cur_head:
        common_head = os.path.commonprefix([prev_head, cur_head])
        if len(common_head) >= 5:
            return True
    prev_grams = _title_char_ngrams(prev_core, size=2)
    cur_grams = _title_char_ngrams(cur_core, size=2)
    if not prev_grams or not cur_grams:
        return frame_overlap >= 2 and (prev_core in cur_core or cur_core in prev_core)
    overlap = len(prev_grams & cur_grams)
    union = len(prev_grams | cur_grams)
    score = overlap / max(union, 1)
    return frame_overlap >= 2 and score >= 0.55


def _build_diverse_title_seed(chapter: dict[str, Any], chapter_index: int, prev_title: str) -> str:
    raw_prev = _strip_chapter_title_prefix(prev_title)
    source_fragments: list[str] = [
        str(chapter.get("goal") or ""),
        str(chapter.get("conflict") or ""),
        str(chapter.get("ending_hook") or ""),
    ]
    source_fragments.extend(_normalize_link_items(chapter.get("carry_over_to_next")))
    keywords: list[str] = []
    for fragment in source_fragments:
        for token in _extract_keywords(fragment):
            if len(token) < 2:
                continue
            if token in TITLE_TEMPLATE_STOPWORDS:
                continue
            if token in raw_prev:
                continue
            keywords.append(token)
    deduped = list(dict.fromkeys(keywords))
    verb = TITLE_ACTION_VERBS[chapter_index % len(TITLE_ACTION_VERBS)]
    if len(deduped) >= 2:
        options = (
            f"{verb}{deduped[0]}，{deduped[1]}失序",
            f"{deduped[0]}失衡，{verb}{deduped[1]}",
            f"{verb}{deduped[0]}与{deduped[1]}",
        )
        seed = options[chapter_index % len(options)]
    elif deduped:
        seed = f"{verb}{deduped[0]}，边界生变"
    else:
        base = _strip_chapter_title_prefix(chapter.get("title", ""))
        seed = base or "回声未熄"
    return seed.strip("：:，,。；;、-—_ ")[:22]


def _diversify_adjacent_titles(
    chapters: list[dict[str, Any]],
    *,
    previous_title: str = "",
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    prev_title = str(previous_title or "")
    for idx, current in enumerate(chapters):
        current_title = str(current.get("title") or "")
        if idx == 0 and not prev_title:
            prev_title = current_title
            continue
        if not _titles_too_similar(prev_title, current_title):
            prev_title = current_title
            continue
        chapter_no = str(current.get("chapter_no") or f"{idx + 1:04d}")
        chapter_int = int(chapter_no) if chapter_no.isdigit() else idx + 1
        replacement_seed = _build_diverse_title_seed(current, idx, prev_title)
        replacement = _format_chapter_title(chapter_int, replacement_seed, current.get("goal", ""))
        alerts.append(
            {
                "chapter_no": chapter_no,
                "before": current_title,
                "after": replacement,
            }
        )
        current["title"] = replacement
        prev_title = replacement
    return alerts


def _compute_title_style_metrics(
    chapters: list[dict[str, Any]],
    *,
    previous_title: str = "",
) -> dict[str, Any]:
    pattern_distribution: dict[str, int] = {}
    adjacent_dup_count = 0
    titles = [str(chapter.get("title") or "") for chapter in chapters if isinstance(chapter, dict)]
    for title in titles:
        key = _title_pattern_key(title)
        pattern_distribution[key] = int(pattern_distribution.get(key, 0)) + 1
    if previous_title and titles and _titles_too_similar(previous_title, titles[0]):
        adjacent_dup_count += 1
    for idx in range(1, len(titles)):
        if _titles_too_similar(titles[idx - 1], titles[idx]):
            adjacent_dup_count += 1
    return {
        "adjacent_title_dup_count": adjacent_dup_count,
        "title_pattern_distribution": pattern_distribution,
    }


def _normalize_chapter_plan(
    raw_chapters: list[dict[str, Any]],
    chapter_count: int,
    start_chapter: int,
    *,
    ending_style: str = "closure",
) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    style = str(ending_style or "closure").strip().lower()
    if style not in ENDING_STYLES:
        style = "closure"
    hook_required = style == "hook"
    for idx in range(chapter_count):
        chapter_no = start_chapter + idx
        source = raw_chapters[idx] if idx < len(raw_chapters) and isinstance(raw_chapters[idx], dict) else {}
        beats = source.get("beat_outline")
        if not isinstance(beats, list):
            beats = []
        goal = str(source.get("goal") or source.get("core_goal") or "推进主线并制造新冲突")
        conflict = str(source.get("conflict") or source.get("key_conflict") or "角色目标与外部阻力发生碰撞")
        title = _format_chapter_title(chapter_no, str(source.get("title") or ""), goal)
        ending_hook_raw = str(source.get("ending_hook") or "").strip()
        ending_hook = ending_hook_raw
        if hook_required and not ending_hook:
            ending_hook = "【待修复】请给出具体跨章钩子"
        carry_over_to_next = _normalize_link_items(source.get("carry_over_to_next"))
        open_with = _normalize_link_items(source.get("open_with"))
        needs_repair = False
        if hook_required and _is_generic_ending_hook(ending_hook):
            needs_repair = True
        if idx < chapter_count - 1 and not carry_over_to_next:
            needs_repair = True
        if (idx > 0 or int(start_chapter or 1) > 1) and not open_with:
            needs_repair = True
        chapters.append(
            {
                "chapter_no": f"{chapter_no:04d}",
                "title": title,
                "goal": goal,
                "conflict": conflict,
                "beat_outline": [str(item) for item in beats if str(item).strip()],
                "ending_hook": ending_hook,
                "carry_over_to_next": carry_over_to_next,
                "open_with": open_with,
                "needs_repair": needs_repair,
            }
        )
    return chapters


def _default_log_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "logs"


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


@dataclass
class InjectionLogger:
    root_dir: Path
    target_book_id: str

    def __post_init__(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.root_dir / "generate_book_ab" / f"{self.target_book_id}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rel_path: str, payload: dict[str, Any] | list[Any]) -> Path:
        path = self.run_dir / rel_path
        _write_json(path, payload)
        return path

    @staticmethod
    def raw_rel_path(rel_path: str) -> str:
        if rel_path.endswith(".json"):
            return rel_path[:-5] + ".raw.json"
        return rel_path + ".raw"

    def write_with_raw(
        self,
        rel_path: str,
        payload: dict[str, Any] | list[Any],
        *,
        raw_payload: dict[str, Any] | list[Any] | None = None,
    ) -> tuple[Path, Path | None]:
        clean_path = self.write(rel_path, payload)
        raw_path: Path | None = None
        if raw_payload is not None:
            raw_path = self.write(self.raw_rel_path(rel_path), raw_payload)
        return clean_path, raw_path


def _stage_bounds_for_chapter(chapter_no: int, stage_size: int) -> tuple[int, int]:
    safe_stage_size = max(int(stage_size or 1), 1)
    chapter_int = max(int(chapter_no or 1), 1)
    stage_start = ((chapter_int - 1) // safe_stage_size) * safe_stage_size + 1
    stage_end = stage_start + safe_stage_size - 1
    return stage_start, stage_end


def _stage_id(stage_start: int, stage_end: int) -> str:
    return f"{int(stage_start):04d}_{int(stage_end):04d}"


@dataclass
class BlueprintStore:
    root_dir: Path
    target_book_id: str
    stage_size: int
    freeze_published: bool

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.master_path = self.root_dir / "master_arc.json"
        self.index_path = self.root_dir / "index.json"
        self.stage_dir = self.root_dir / "stages"
        self.batch_dir = self.root_dir / "batches"
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_index(self) -> dict[str, Any]:
        data = self._read_json(self.index_path) or {}
        if not data:
            data = {
                "target_book_id": self.target_book_id,
                "blueprint_mode": "hierarchical",
                "stage_size": int(self.stage_size),
                "freeze_published_blueprint": bool(self.freeze_published),
                "locked_until": 0,
                "book_total_chapters": 0,
                "updated_at": datetime.now().isoformat(),
                "stages": [],
                "batches": [],
            }
        data.setdefault("target_book_id", self.target_book_id)
        data.setdefault("blueprint_mode", "hierarchical")
        data.setdefault("stage_size", int(self.stage_size))
        data.setdefault("freeze_published_blueprint", bool(self.freeze_published))
        data.setdefault("locked_until", 0)
        data.setdefault("book_total_chapters", 0)
        data.setdefault("updated_at", datetime.now().isoformat())
        data.setdefault("stages", [])
        data.setdefault("batches", [])
        return data

    def save_index(self, index_payload: dict[str, Any]) -> None:
        payload = dict(index_payload)
        payload["updated_at"] = datetime.now().isoformat()
        self._write_json(self.index_path, payload)

    def load_master_arc(self) -> dict[str, Any] | None:
        return self._read_json(self.master_path)

    def save_master_arc(self, payload: dict[str, Any]) -> None:
        self._write_json(self.master_path, payload)

    def stage_path(self, stage_start: int, stage_end: int) -> Path:
        return self.stage_dir / f"stage_{_stage_id(stage_start, stage_end)}.json"

    def batch_path(self, batch_start: int, batch_end: int) -> Path:
        return self.batch_dir / f"batch_{int(batch_start):04d}_{int(batch_end):04d}.json"

    def load_stage(self, stage_start: int, stage_end: int) -> dict[str, Any] | None:
        return self._read_json(self.stage_path(stage_start, stage_end))

    def save_stage(self, stage_start: int, stage_end: int, payload: dict[str, Any]) -> Path:
        path = self.stage_path(stage_start, stage_end)
        self._write_json(path, payload)
        return path

    def load_batch(self, batch_start: int, batch_end: int) -> dict[str, Any] | None:
        return self._read_json(self.batch_path(batch_start, batch_end))

    def save_batch(self, batch_start: int, batch_end: int, payload: dict[str, Any]) -> Path:
        path = self.batch_path(batch_start, batch_end)
        self._write_json(path, payload)
        return path


@dataclass
class MemoryStore:
    canon_db_path: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_pass: str
    neo4j_database: str
    qdrant_url: str
    qdrant_collection: str
    qdrant_api_key: str = ""

    def canon_norm(self) -> str:
        return str(Path(self.canon_db_path).expanduser().resolve())

    def neo4j_norm(self) -> tuple[str, str]:
        return (self.neo4j_uri.strip(), self.neo4j_database.strip())

    def qdrant_norm(self) -> tuple[str, str]:
        return (self.qdrant_url.rstrip("/"), self.qdrant_collection.strip())


def _assert_isolation(template_store: MemoryStore, target_store: MemoryStore) -> None:
    clashes = []
    if template_store.canon_norm() == target_store.canon_norm():
        clashes.append("Canon DB path is identical")
    if template_store.neo4j_norm() == target_store.neo4j_norm():
        clashes.append("Neo4j target (uri+database) is identical")
    if template_store.qdrant_norm() == target_store.qdrant_norm():
        clashes.append("Qdrant target (url+collection) is identical")
    if clashes:
        raise ValueError("Isolation check failed: " + "; ".join(clashes))


@dataclass
class LLMClient:
    llm_config: dict[str, Any]
    timeout: float = 180.0

    @staticmethod
    def _pop_proxy_env() -> dict[str, str]:
        backup: dict[str, str] = {}
        for key in (
            "ALL_PROXY",
            "all_proxy",
        ):
            value = os.environ.pop(key, None)
            if value is not None:
                backup[key] = value
        return backup

    @staticmethod
    def _restore_proxy_env(backup: dict[str, str]) -> None:
        for key, value in backup.items():
            os.environ[key] = value

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        if self.llm_config.get("type") == "custom":
            proxy_backup = self._pop_proxy_env()
            try:
                response = httpx.post(
                    self.llm_config["url"],
                    json={
                        "model": self.llm_config["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    headers={"Authorization": f"Bearer {self.llm_config['api_key']}"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"] or ""
                if not content.strip():
                    raise RuntimeError("empty LLM response")
                return content
            finally:
                self._restore_proxy_env(proxy_backup)

        if self.llm_config.get("providers") and self.llm_config.get("model"):
            provider_cfg = self.llm_config["providers"].get("anthropic")
            if not provider_cfg:
                raise ValueError("providers.anthropic is required in providers mode")
            api_key = provider_cfg.get("apiKey") or provider_cfg.get("api_key")
            api_base = provider_cfg.get("apiBase") or provider_cfg.get("api_base")
            extra_headers = provider_cfg.get("extraHeaders") or provider_cfg.get("extra_headers")
            from nanobot.providers.litellm_provider import LiteLLMProvider

            provider = LiteLLMProvider(
                api_key=api_key,
                api_base=api_base,
                default_model=self.llm_config["model"],
                extra_headers=extra_headers,
            )
            async def _chat_with_timeout():
                return await asyncio.wait_for(
                    provider.chat(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.llm_config["model"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=self.timeout,
                )

            proxy_backup = self._pop_proxy_env()
            try:
                try:
                    asyncio.get_running_loop()
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        response = pool.submit(lambda: asyncio.run(_chat_with_timeout())).result()
                except RuntimeError:
                    response = asyncio.run(_chat_with_timeout())
            finally:
                self._restore_proxy_env(proxy_backup)
            content = (response.content or "").strip()
            if not content or content.lower().startswith("error calling llm:"):
                raise RuntimeError(content or "empty LLM response")
            return content

        raise ValueError("Unsupported llm_config: expected {type: custom} or {providers, model}")

    def complete_with_retry(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        max_retries: int,
        retry_backoff: float,
        backoff_factor: float,
        backoff_max: float,
        retry_jitter: float,
    ) -> str:
        attempts = max(max_retries, 0) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.complete(prompt, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:  # pragma: no cover - operational retry
                last_error = exc
                if attempt >= attempts:
                    break
                base = retry_backoff * (backoff_factor ** max(0, attempt - 1))
                delay = min(base, backoff_max)
                if retry_jitter > 0:
                    delay += random.uniform(0, retry_jitter)
                time.sleep(delay)
        raise RuntimeError(f"LLM request failed after {attempts} attempts: {last_error}")


def _parse_enforce_chinese_fields(raw_fields: str | list[str] | tuple[str, ...] | None) -> set[str]:
    if raw_fields is None:
        return set(DEFAULT_ENFORCE_CHINESE_FIELDS)
    if isinstance(raw_fields, str):
        tokens = [token.strip().lower() for token in re.split(r"[,\s]+", raw_fields) if token.strip()]
        return set(tokens) if tokens else set(DEFAULT_ENFORCE_CHINESE_FIELDS)
    tokens = [str(token).strip().lower() for token in raw_fields if str(token).strip()]
    return set(tokens) if tokens else set(DEFAULT_ENFORCE_CHINESE_FIELDS)


@dataclass
class ChineseContextNormalizer:
    llm: LLMClient
    fields: set[str]
    enabled: bool = True
    max_tokens: int = 768
    cache_limit: int = 1024

    def __post_init__(self) -> None:
        self.fields = {str(field).strip().lower() for field in self.fields if str(field).strip()}
        self._cache: dict[str, str] = {}

    @staticmethod
    def _needs_translation(text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        if not LATIN_CHAR_RE.search(cleaned):
            return False
        if re.fullmatch(r"[A-Z0-9_:-]{2,}", cleaned):
            return False
        if _is_probable_internal_entity_id(cleaned):
            return False
        return True

    def _translate_text(
        self,
        text: str,
        *,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> str:
        if not self.enabled:
            return text
        original = str(text or "")
        if not self._needs_translation(original):
            return original
        cached = self._cache.get(original)
        if cached is not None:
            return cached

        prompt = (
            "把下面文本翻译为简体中文，只输出翻译后的文本。\n"
            "要求：\n"
            "1) 保留实体ID、关系枚举、技术名词缩写原样（例如 character_ab12cd、CO_PARTICIPANT、Neo4j、Qdrant）。\n"
            "2) 不要补充解释，不要加引号，不要 markdown。\n"
            "3) 保持原意，不要删减事实。\n\n"
            f"文本：\n{original}"
        )
        translated = self.llm.complete_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=self.max_tokens,
            max_retries=llm_max_retries,
            retry_backoff=llm_retry_backoff,
            backoff_factor=llm_backoff_factor,
            backoff_max=llm_backoff_max,
            retry_jitter=llm_retry_jitter,
        )
        normalized = _strip_markdown_code_blocks(translated).strip() or original
        if len(self._cache) >= self.cache_limit:
            self._cache.clear()
        self._cache[original] = normalized
        return normalized

    def _normalize_value(
        self,
        value: Any,
        *,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> Any:
        if isinstance(value, str):
            return self._translate_text(
                value,
                llm_max_retries=llm_max_retries,
                llm_retry_backoff=llm_retry_backoff,
                llm_backoff_factor=llm_backoff_factor,
                llm_backoff_max=llm_backoff_max,
                llm_retry_jitter=llm_retry_jitter,
            )
        if isinstance(value, list):
            return [
                self._normalize_value(
                    item,
                    llm_max_retries=llm_max_retries,
                    llm_retry_backoff=llm_retry_backoff,
                    llm_backoff_factor=llm_backoff_factor,
                    llm_backoff_max=llm_backoff_max,
                    llm_retry_jitter=llm_retry_jitter,
                )
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: self._normalize_value(
                    item,
                    llm_max_retries=llm_max_retries,
                    llm_retry_backoff=llm_retry_backoff,
                    llm_backoff_factor=llm_backoff_factor,
                    llm_backoff_max=llm_backoff_max,
                    llm_retry_jitter=llm_retry_jitter,
                )
                for key, item in value.items()
            }
        return value

    def normalize_hard_pack(
        self,
        hard_pack: dict[str, Any],
        *,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> dict[str, Any]:
        if not self.enabled:
            return hard_pack

        normalized = copy.deepcopy(hard_pack)
        hard_rules = normalized.get("hard_rules")
        if isinstance(hard_rules, list):
            for item in hard_rules:
                if not isinstance(item, dict):
                    continue
                predicate = str(item.get("predicate") or "").strip().lower()
                if predicate in self.fields:
                    item["value"] = self._normalize_value(
                        item.get("value"),
                        llm_max_retries=llm_max_retries,
                        llm_retry_backoff=llm_retry_backoff,
                        llm_backoff_factor=llm_backoff_factor,
                        llm_backoff_max=llm_backoff_max,
                        llm_retry_jitter=llm_retry_jitter,
                    )

        prev_context = normalized.get("prev_context")
        if isinstance(prev_context, dict):
            character_state = prev_context.get("character_state")
            if isinstance(character_state, list):
                for row in character_state:
                    if not isinstance(row, dict):
                        continue
                    state = row.get("state")
                    if not isinstance(state, dict):
                        continue
                    for key, raw_value in list(state.items()):
                        key_l = str(key).strip().lower()
                        if key_l in self.fields:
                            state[key] = self._normalize_value(
                                raw_value,
                                llm_max_retries=llm_max_retries,
                                llm_retry_backoff=llm_retry_backoff,
                                llm_backoff_factor=llm_backoff_factor,
                                llm_backoff_max=llm_backoff_max,
                                llm_retry_jitter=llm_retry_jitter,
                            )
        return normalized


class TemplateMemoryReader:
    """Read-only template extraction from Book A memory stores."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        semantic_search_enabled: bool = True,
        semantic_model_name: str = "",
    ):
        self.store = store
        self._asset_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.semantic_search_enabled = semantic_search_enabled
        self.semantic_model_name = semantic_model_name.strip()
        self._semantic_model: Any | None = None
        self._semantic_model_loaded = False
        self._vector_config_cache: dict[str, Any] | None = None
        self._semantic_model_warning = ""
        self._entity_name_index_cache: dict[str, str] | None = None

    def _qdrant_headers(self) -> dict[str, str]:
        return {"api-key": self.store.qdrant_api_key} if self.store.qdrant_api_key else {}

    def _load_vector_config(self) -> dict[str, Any]:
        if self._vector_config_cache is not None:
            return self._vector_config_cache
        self._vector_config_cache = {}
        if not self.store.qdrant_url:
            return self._vector_config_cache
        try:
            response = httpx.get(
                f"{self.store.qdrant_url.rstrip('/')}/collections/{self.store.qdrant_collection}",
                headers=self._qdrant_headers(),
                timeout=20.0,
                trust_env=False,
            )
            response.raise_for_status()
            vectors = (
                response.json()
                .get("result", {})
                .get("config", {})
                .get("params", {})
                .get("vectors", {})
            )
            if isinstance(vectors, dict):
                self._vector_config_cache = vectors
        except Exception:
            self._vector_config_cache = {}
        return self._vector_config_cache

    @staticmethod
    def _pop_proxy_env() -> dict[str, str]:
        backup: dict[str, str] = {}
        for key in (
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
        ):
            value = os.environ.pop(key, None)
            if value is not None:
                backup[key] = value
        return backup

    @staticmethod
    def _restore_proxy_env(backup: dict[str, str]) -> None:
        for key, value in backup.items():
            os.environ[key] = value

    def _resolve_vector_target(self) -> tuple[str | None, int | None]:
        vectors = self._load_vector_config()
        if not vectors:
            return (None, None)
        if "size" in vectors:
            try:
                return (None, int(vectors["size"]))
            except Exception:
                return (None, None)
        for name, cfg in vectors.items():
            if isinstance(cfg, dict) and "size" in cfg:
                try:
                    return (str(name), int(cfg["size"]))
                except Exception:
                    continue
        return (None, None)

    @staticmethod
    def _default_semantic_model_for_dim(vector_dim: int | None) -> str:
        if vector_dim == 1024:
            return "BAAI/bge-large-zh-v1.5"
        if vector_dim == 768:
            return "moka-ai/m3e-base"
        if vector_dim == 384:
            return "paraphrase-multilingual-MiniLM-L12-v2"
        return "moka-ai/m3e-base"

    def _get_semantic_model(self, expected_dim: int | None) -> Any | None:
        if self._semantic_model_loaded:
            return self._semantic_model
        self._semantic_model_loaded = True
        if not self.semantic_search_enabled:
            return None
        alias_map = {
            "chinese": "moka-ai/m3e-base",
            "chinese-large": "BAAI/bge-large-zh-v1.5",
            "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
        }
        requested = self.semantic_model_name.strip()
        model_name = alias_map.get(requested, requested) if requested else self._default_semantic_model_for_dim(expected_dim)
        use_flag_model = model_name == "BAAI/bge-large-zh-v1.5" and FLAG_MODEL_AVAILABLE
        proxy_backup = self._pop_proxy_env()
        try:
            model_dim: int | None = None
            if use_flag_model:
                model = FlagModel(
                    model_name,
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True,
                )
                try:
                    model_dim = len(model.encode(["维度探测"])[0])
                except Exception:
                    model_dim = None
                loaded: dict[str, Any] = {"type": "flag", "model": model, "name": model_name, "dim": model_dim}
            else:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name)
                try:
                    model_dim = int(model.get_sentence_embedding_dimension())
                except Exception:
                    model_dim = None
                loaded = {"type": "sentence_transformer", "model": model, "name": model_name, "dim": model_dim}
            if expected_dim is not None and model_dim is not None and model_dim != expected_dim:
                fallback_name = self._default_semantic_model_for_dim(expected_dim)
                # If caller explicitly requested a model alias/name, keep strict behavior:
                # return None so retrieval does not silently switch to another embedding family.
                if requested:
                    self._semantic_model_warning = (
                        "template semantic model dimension mismatch: "
                        f"requested={model_name} ({model_dim}), expected={expected_dim}"
                    )
                    print(f"[warn] {self._semantic_model_warning}", flush=True)
                    self._semantic_model = None
                    return self._semantic_model
                if fallback_name != model_name:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(fallback_name)
                    loaded = {
                        "type": "sentence_transformer",
                        "model": model,
                        "name": fallback_name,
                        "dim": int(model.get_sentence_embedding_dimension()),
                    }
                    self.semantic_model_name = fallback_name
            else:
                self.semantic_model_name = model_name
            self._semantic_model = loaded
        except Exception:
            self._semantic_model = None
        finally:
            self._restore_proxy_env(proxy_backup)
        return self._semantic_model

    @staticmethod
    def _chapter_sort_key(chapter_value: Any) -> tuple[int, str]:
        text = str(chapter_value or "")
        match = re.search(r"(\d+)", text)
        if not match:
            return (10**9, text)
        return (int(match.group(1)), text)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        if not text:
            return set()
        lowered = text.lower()
        tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered))
        return {token for token in tokens if token.strip()}

    @staticmethod
    def _dedupe_templates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for item in items:
            key = f"{item.get('chapter','')}::{item.get('text','')}"
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _load_entity_name_index(self) -> dict[str, str]:
        if self._entity_name_index_cache is not None:
            return self._entity_name_index_cache

        index: dict[str, str] = {}
        path = Path(self.store.canon_db_path).expanduser()
        if not path.exists():
            self._entity_name_index_cache = index
            return self._entity_name_index_cache

        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                """
                SELECT entity_id, canonical_name
                FROM entity_registry
                WHERE canonical_name IS NOT NULL AND canonical_name <> ''
                """
            ).fetchall()
            for entity_id, canonical_name in rows:
                if isinstance(entity_id, str):
                    cleaned_id = entity_id.strip()
                else:
                    cleaned_id = ""
                name = _coerce_name(canonical_name)
                if cleaned_id and name:
                    index[cleaned_id] = name
        except Exception:
            index = {}
        finally:
            conn.close()

        self._entity_name_index_cache = index
        return self._entity_name_index_cache

    def _sanitize_template_item_for_prompt(self, item: dict[str, Any]) -> dict[str, Any]:
        entity_name_index = dict(self._load_entity_name_index())
        _collect_entity_name_index(item, entity_name_index)
        sanitized = _replace_entity_ids_with_names(item, entity_name_index)
        if not isinstance(sanitized, dict):
            return {
                "text": str(item.get("text") or "").strip(),
                "chapter": str(item.get("chapter") or "").strip(),
            }
        sanitized.pop("chapter_sort_key", None)
        if "text" in sanitized:
            sanitized["text"] = str(sanitized.get("text") or "").strip()
        if "chapter" in sanitized:
            sanitized["chapter"] = str(sanitized.get("chapter") or "").strip()
        return sanitized

    def sanitize_template_payload_for_prompt(self, payload: Any) -> Any:
        if isinstance(payload, list):
            sanitized_list = [self.sanitize_template_payload_for_prompt(item) for item in payload]
            return [item for item in sanitized_list if not _is_empty_value(item)]

        if isinstance(payload, dict):
            if "text" in payload and ("metadata" in payload or "chapter" in payload):
                return self._sanitize_template_item_for_prompt(payload)
            sanitized: dict[str, Any] = {}
            preserve_empty_keys = {
                "plot_templates",
                "style_templates",
                "narrative_templates",
                "fact_templates",
                "relation_templates",
            }
            for key, value in payload.items():
                if key == "chapter_sort_key":
                    continue
                child = self.sanitize_template_payload_for_prompt(value)
                if _is_empty_value(child) and key not in preserve_empty_keys:
                    continue
                if key in preserve_empty_keys and not isinstance(child, list):
                    child = []
                sanitized[key] = child
            return sanitized

        return payload

    def _qdrant_scroll_all(self, book_id: str, asset_type: str, limit: int = 5000) -> list[dict[str, Any]]:
        if not self.store.qdrant_url:
            return []
        cache_key = (book_id, asset_type)
        cached = self._asset_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            rows: list[dict[str, Any]] = []
            offset: Any | None = None
            page_limit = 256
            while len(rows) < max(limit, 1):
                request_payload: dict[str, Any] = {
                    "filter": {
                        "must": [
                            {"key": "book_id", "match": {"value": book_id}},
                            {"key": "asset_type", "match": {"value": asset_type}},
                        ]
                    },
                    "limit": min(page_limit, max(limit, 1) - len(rows)),
                    "with_payload": True,
                    "with_vector": False,
                }
                if offset is not None:
                    request_payload["offset"] = offset
                response = httpx.post(
                    f"{self.store.qdrant_url.rstrip('/')}/collections/{self.store.qdrant_collection}/points/scroll",
                    headers=self._qdrant_headers(),
                    json=request_payload,
                    timeout=20.0,
                    trust_env=False,
                )
                response.raise_for_status()
                result = response.json().get("result", {})
                points = result.get("points", [])
                if not points:
                    break
                for row in points:
                    payload = row.get("payload", {}) or {}
                    chapter = payload.get("chapter")
                    rows.append(
                        {
                            "text": payload.get("text", ""),
                            "chapter": chapter,
                            "metadata": payload.get("metadata", {}),
                            "chapter_sort_key": self._chapter_sort_key(chapter),
                        }
                    )
                offset = result.get("next_page_offset")
                if offset is None:
                    break
            rows = self._dedupe_templates(rows)
            self._asset_cache[cache_key] = rows
            return rows
        except Exception:
            self._asset_cache[cache_key] = []
            return []

    @staticmethod
    def _pick_evenly(items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        if len(items) <= top_k:
            return items
        if top_k == 1:
            return [items[len(items) // 2]]
        picks: list[dict[str, Any]] = []
        last_index = len(items) - 1
        for idx in range(top_k):
            position = round((idx / (top_k - 1)) * last_index)
            picks.append(items[position])
        return TemplateMemoryReader._dedupe_templates(picks)[:top_k]

    def _select_book_templates(
        self,
        book_id: str,
        asset_type: str,
        top_k: int,
        *,
        query_text: str = "",
    ) -> list[dict[str, Any]]:
        # Prefer semantic retrieval when we have a profile-level query.
        if query_text.strip():
            semantic_rows = self._qdrant_semantic_search(
                book_id=book_id,
                asset_type=asset_type,
                query_text=query_text,
                limit=max(top_k, 1),
            )
            if semantic_rows:
                return semantic_rows
        pool = self._qdrant_scroll_all(book_id, asset_type)
        if not pool:
            return []
        if query_text.strip():
            token_selected = self._select_by_token_overlap(pool, query_text, top_k)
            if token_selected:
                return token_selected
        sorted_pool = sorted(pool, key=lambda item: item.get("chapter_sort_key", (10**9, "")))
        return self._pick_evenly(sorted_pool, top_k)

    def _select_chapter_templates(
        self,
        book_id: str,
        asset_type: str,
        top_k: int,
        *,
        query_text: str,
    ) -> list[dict[str, Any]]:
        semantic_rows = self._qdrant_semantic_search(
            book_id=book_id,
            asset_type=asset_type,
            query_text=query_text,
            limit=top_k,
        )
        if semantic_rows:
            return semantic_rows

        pool = self._qdrant_scroll_all(book_id, asset_type)
        if not pool:
            return []
        token_selected = self._select_by_token_overlap(pool, query_text, top_k)
        if token_selected:
            return token_selected
        if not self._tokenize(query_text):
            return self._select_book_templates(book_id, asset_type, top_k)
        return self._select_book_templates(book_id, asset_type, top_k)

    def _select_by_token_overlap(
        self,
        pool: list[dict[str, Any]],
        query_text: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []

        scored: list[tuple[int, tuple[int, str], dict[str, Any]]] = []
        for item in pool:
            text = str(item.get("text") or "")
            metadata = item.get("metadata")
            metadata_text = json.dumps(metadata, ensure_ascii=False) if metadata else ""
            item_tokens = self._tokenize(f"{text} {metadata_text}")
            overlap = len(query_tokens & item_tokens)
            if overlap <= 0:
                continue
            scored.append((overlap, item.get("chapter_sort_key", (10**9, "")), item))
        if not scored:
            return []

        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        selected = [item for _, _, item in scored[: max(top_k * 3, top_k)]]
        selected = self._dedupe_templates(selected)
        return selected[:top_k]

    def _qdrant_semantic_search(
        self,
        *,
        book_id: str,
        asset_type: str,
        query_text: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = query_text.strip()
        if not query or not self.store.qdrant_url:
            return []
        vector_name, vector_dim = self._resolve_vector_target()
        if vector_dim is None:
            return []
        model = self._get_semantic_model(vector_dim)
        if model is None:
            return []
        try:
            if isinstance(model, dict) and model.get("type") == "flag":
                query_vector = model["model"].encode([query])[0].tolist()
            elif isinstance(model, dict):
                query_vector = model["model"].encode(query).tolist()
            else:
                query_vector = model.encode(query).tolist()
            payload: dict[str, Any] = {
                "limit": max(limit, 1),
                "with_payload": True,
                "with_vector": False,
                "filter": {
                    "must": [
                        {"key": "book_id", "match": {"value": book_id}},
                        {"key": "asset_type", "match": {"value": asset_type}},
                    ]
                },
            }
            if vector_name:
                payload["vector"] = {"name": vector_name, "vector": query_vector}
            else:
                payload["vector"] = query_vector

            response = httpx.post(
                f"{self.store.qdrant_url.rstrip('/')}/collections/{self.store.qdrant_collection}/points/search",
                headers=self._qdrant_headers(),
                json=payload,
                timeout=20.0,
                trust_env=False,
            )
            response.raise_for_status()
            rows = response.json().get("result", [])
            numeric_scores = [float(row.get("score")) for row in rows if isinstance(row.get("score"), (int, float))]
            if len(numeric_scores) >= 2 and (max(numeric_scores) - min(numeric_scores)) < 1e-9:
                # Usually indicates low-signal vectors (e.g., placeholder zeros); let lexical fallback handle it.
                return []
            result: list[dict[str, Any]] = []
            for row in rows:
                row_payload = row.get("payload", {}) or {}
                chapter = row_payload.get("chapter")
                result.append(
                    {
                        "text": row_payload.get("text", ""),
                        "chapter": chapter,
                        "metadata": row_payload.get("metadata", {}),
                        "score": row.get("score"),
                        "retrieval": "semantic_search",
                        "chapter_sort_key": self._chapter_sort_key(chapter),
                    }
                )
            return self._dedupe_templates(result)[: max(limit, 1)]
        except Exception:
            return []

    def _canon_recent(self, book_id: str, limit: int) -> dict[str, Any]:
        path = Path(self.store.canon_db_path).expanduser()
        if not path.exists():
            return {"hard_rules": [], "relations": []}
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            hard_rules = conn.execute(
                """
                SELECT fh.chapter_no, er.canonical_name, fh.predicate, fh.object_json, fh.evidence_chunk_id
                FROM fact_history fh
                JOIN commit_log cl ON cl.commit_id = fh.commit_id
                JOIN entity_registry er ON er.entity_id = fh.subject_id
                WHERE cl.book_id = ? AND fh.tier = 'HARD_RULE'
                ORDER BY fh.chapter_no DESC, fh.created_at DESC
                LIMIT ?
                """,
                (book_id, max(limit, 1)),
            ).fetchall()
            relations = conn.execute(
                """
                SELECT rh.chapter_no, fr.canonical_name AS from_name, tr.canonical_name AS to_name,
                       rh.kind, rh.status, rh.evidence_chunk_id
                FROM relationship_history rh
                JOIN commit_log cl ON cl.commit_id = rh.commit_id
                LEFT JOIN entity_registry fr ON fr.entity_id = rh.from_id
                LEFT JOIN entity_registry tr ON tr.entity_id = rh.to_id
                WHERE cl.book_id = ?
                ORDER BY rh.chapter_no DESC, rh.created_at DESC
                LIMIT ?
                """,
                (book_id, max(limit, 1)),
            ).fetchall()
            return {
                "hard_rules": [
                    {
                        "chapter_no": row["chapter_no"],
                        "subject": row["canonical_name"],
                        "predicate": row["predicate"],
                        "value": json.loads(row["object_json"] or "null"),
                        "evidence": row["evidence_chunk_id"],
                    }
                    for row in hard_rules
                ],
                "relations": [dict(row) for row in relations],
            }
        finally:
            conn.close()

    def _neo4j_relation_kinds(self) -> list[dict[str, Any]]:
        if GraphDatabase is None:
            return []
        try:
            driver = GraphDatabase.driver(self.store.neo4j_uri, auth=(self.store.neo4j_user, self.store.neo4j_pass))
            with driver.session(database=self.store.neo4j_database) as session:
                rows = session.run(
                    """
                    MATCH ()-[r:RELATES]->()
                    RETURN r.kind AS kind, count(*) AS cnt
                    ORDER BY cnt DESC
                    LIMIT 20
                    """
                )
                result = [{"kind": row["kind"], "count": row["cnt"]} for row in rows]
            driver.close()
            return result
        except Exception:
            return []

    def build_book_template_profile(self, book_id: str, top_k: int, *, query_text: str = "") -> dict[str, Any]:
        return {
            "plot_templates": self._select_book_templates(book_id, "plot_beat", top_k, query_text=query_text),
            "style_templates": self._select_book_templates(book_id, "style", top_k, query_text=query_text),
            "conflict_templates": self._select_book_templates(book_id, "conflict", top_k, query_text=query_text),
            # Digest assets usually have much wider chapter coverage than style/plot tags.
            "chapter_digest_templates": self._select_book_templates(
                book_id,
                "chapter_digest",
                top_k,
                query_text=query_text,
            ),
            "fact_digest_templates": self._select_book_templates(
                book_id,
                "fact_digest",
                max(top_k // 2, 2),
                query_text=query_text,
            ),
            "relation_digest_templates": self._select_book_templates(
                book_id,
                "relation_digest",
                max(top_k // 2, 2),
                query_text=query_text,
            ),
            "canon": self._canon_recent(book_id, top_k),
            "neo4j_relation_kinds": self._neo4j_relation_kinds(),
        }

    def build_chapter_template_pack(self, book_id: str, chapter_plan: dict[str, Any], top_k: int) -> dict[str, Any]:
        query_text = " ".join(
            [
                str(chapter_plan.get("goal", "")),
                str(chapter_plan.get("conflict", "")),
                " ".join(str(item) for item in chapter_plan.get("beat_outline", []) if str(item).strip()),
            ]
        ).strip()
        digest_top_k = max(top_k // 2, 2)
        return {
            "chapter_goal": chapter_plan.get("goal", ""),
            "chapter_conflict": chapter_plan.get("conflict", ""),
            "plot_templates": self._select_chapter_templates(
                book_id,
                "plot_beat",
                top_k,
                query_text=query_text,
            ),
            "style_templates": self._select_chapter_templates(
                book_id,
                "style",
                top_k,
                query_text=query_text,
            ),
            "narrative_templates": self._select_chapter_templates(
                book_id,
                "chapter_digest",
                top_k,
                query_text=query_text,
            ),
            "fact_templates": self._select_chapter_templates(
                book_id,
                "fact_digest",
                digest_top_k,
                query_text=query_text,
            ),
            "relation_templates": self._select_chapter_templates(
                book_id,
                "relation_digest",
                digest_top_k,
                query_text=query_text,
            ),
        }


class TargetMemoryReader:
    """Read Book B hard context before generating each chapter."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self._entity_name_index_cache: dict[str, str] | None = None

    def _load_entity_name_index(self) -> dict[str, str]:
        if self._entity_name_index_cache is not None:
            return self._entity_name_index_cache

        index: dict[str, str] = {}
        path = Path(self.store.canon_db_path).expanduser()
        if not path.exists():
            self._entity_name_index_cache = index
            return self._entity_name_index_cache

        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                """
                SELECT entity_id, canonical_name
                FROM entity_registry
                WHERE canonical_name IS NOT NULL AND canonical_name <> ''
                """
            ).fetchall()
            for entity_id, canonical_name in rows:
                if isinstance(entity_id, str):
                    cleaned_id = entity_id.strip()
                else:
                    cleaned_id = ""
                name = _coerce_name(canonical_name)
                if cleaned_id and name:
                    index[cleaned_id] = name
        except Exception:
            index = {}
        finally:
            conn.close()

        self._entity_name_index_cache = index
        return self._entity_name_index_cache

    @staticmethod
    def _sanitize_prev_context_for_prompt(prev_context: dict[str, Any]) -> dict[str, Any]:
        """Keep human-readable context fields only (no internal entity ids)."""
        clean_context: dict[str, Any] = {
            "character_state": [],
            "recent_relations": [],
            "open_threads": [],
        }

        for row in prev_context.get("character_state", []) or []:
            if not isinstance(row, dict):
                continue
            canonical_name = str(row.get("name") or "").strip()
            if not canonical_name:
                continue
            clean_context["character_state"].append(
                {
                    "canonical_name": canonical_name,
                    "name": canonical_name,
                    "state": row.get("state", {}) if isinstance(row.get("state"), dict) else {},
                    "updated_chapter": row.get("updated_chapter"),
                }
            )

        for row in prev_context.get("recent_relations", []) or []:
            if not isinstance(row, dict):
                continue
            from_name = str(row.get("from_name") or "").strip()
            to_name = str(row.get("to_name") or "").strip()
            if not from_name or not to_name:
                continue
            clean_context["recent_relations"].append(
                {
                    "from_name": from_name,
                    "to_name": to_name,
                    "kind": row.get("kind"),
                    "status": row.get("status"),
                    "valid_from": row.get("valid_from"),
                    "valid_to": row.get("valid_to"),
                    "evidence_chunk_id": row.get("evidence_chunk_id"),
                }
            )

        for row in prev_context.get("open_threads", []) or []:
            if not isinstance(row, dict):
                continue
            clean_context["open_threads"].append(
                {
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "priority": row.get("priority"),
                    "planned_window": row.get("planned_window"),
                    "notes": row.get("notes"),
                    "updated_chapter": row.get("updated_chapter"),
                }
            )

        return clean_context

    def sanitize_hard_pack_for_prompt(self, hard_pack: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(hard_pack, dict):
            return {"hard_rules": [], "prev_context": {"character_state": [], "recent_relations": [], "open_threads": []}}

        pack = dict(hard_pack)
        prev_context = pack.get("prev_context")
        if isinstance(prev_context, dict):
            pack["prev_context"] = TargetMemoryReader._sanitize_prev_context_for_prompt(prev_context)

        entity_name_index = dict(self._load_entity_name_index())
        _collect_entity_name_index(pack, entity_name_index)
        cleaned = _replace_entity_ids_with_names(pack, entity_name_index)
        if not isinstance(cleaned, dict):
            return {"hard_rules": [], "prev_context": {"character_state": [], "recent_relations": [], "open_threads": []}}

        hard_rules = cleaned.get("hard_rules")
        if isinstance(hard_rules, list):
            for row in hard_rules:
                if isinstance(row, dict):
                    row.pop("evidence", None)
                    row.pop("evidence_chunk_id", None)

        prev_context_clean = cleaned.get("prev_context")
        if isinstance(prev_context_clean, dict):
            for row in prev_context_clean.get("recent_relations", []) or []:
                if isinstance(row, dict):
                    row.pop("evidence_chunk_id", None)
        return cleaned

    def build_hard_pack(self, book_id: str, chapter_no: str, top_k: int) -> dict[str, Any]:
        from canon_db_v2 import CanonDBV2

        db = CanonDBV2(str(Path(self.store.canon_db_path).expanduser()))
        try:
            prev_context = db.get_prev_context_snapshot(
                chapter_no,
                state_limit=max(top_k, 12),
                relation_limit=max(top_k, 12),
                thread_limit=max(top_k, 8),
            )
            prev_context = self._sanitize_prev_context_for_prompt(prev_context)
        finally:
            db.close()

        path = Path(self.store.canon_db_path).expanduser()
        if not path.exists():
            return {"hard_rules": [], "prev_context": prev_context}

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT fh.chapter_no, er.canonical_name, fh.predicate, fh.object_json, fh.evidence_chunk_id
                FROM fact_history fh
                JOIN commit_log cl ON cl.commit_id = fh.commit_id
                JOIN entity_registry er ON er.entity_id = fh.subject_id
                WHERE cl.book_id = ? AND fh.tier = 'HARD_RULE' AND fh.chapter_no < ?
                ORDER BY fh.chapter_no DESC, fh.created_at DESC
                LIMIT ?
                """,
                (book_id, chapter_no, max(top_k, 1)),
            ).fetchall()
            hard_rules = [
                {
                    "chapter_no": row["chapter_no"],
                    "subject": row["canonical_name"],
                    "predicate": row["predicate"],
                    "value": json.loads(row["object_json"] or "null"),
                    "evidence": row["evidence_chunk_id"],
                }
                for row in rows
            ]
        finally:
            conn.close()
        return {"hard_rules": hard_rules, "prev_context": prev_context}


def _load_all_done_chapters(canon_db_path: str, book_id: str) -> set[str]:
    path = Path(canon_db_path).expanduser()
    if not path.exists():
        return set()
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            """
            SELECT chapter_no
            FROM commit_log
            WHERE book_id = ? AND status = 'ALL_DONE'
            """,
            (book_id,),
        ).fetchall()
        return {str(row[0]) for row in rows if row and row[0]}
    finally:
        conn.close()


def _load_run_report_summaries(output_dir: Path, target_book_id: str) -> dict[str, str]:
    summaries, _ = _load_run_report_context(output_dir, target_book_id)
    return summaries


def _normalize_chapter_no(chapter_no: Any) -> str:
    text = str(chapter_no or "").strip()
    if not text:
        return ""
    match = re.search(r"(\d+)", text)
    if not match:
        return text
    return match.group(1).zfill(4)


def _parse_resume_overwrite_range(raw: str) -> set[str]:
    values = str(raw or "").strip()
    if not values:
        return set()

    chapters: set[str] = set()
    for token in values.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            left_raw, right_raw = [item.strip() for item in part.split("-", 1)]
            if not left_raw.isdigit() or not right_raw.isdigit():
                raise ValueError(f"Invalid range token in --resume-overwrite-range: {part}")
            left = int(left_raw)
            right = int(right_raw)
            if left <= 0 or right <= 0:
                raise ValueError(f"Invalid chapter range in --resume-overwrite-range: {part}")
            start = min(left, right)
            end = max(left, right)
            for chapter_int in range(start, end + 1):
                chapters.add(f"{chapter_int:04d}")
            continue
        if not part.isdigit():
            raise ValueError(f"Invalid chapter token in --resume-overwrite-range: {part}")
        chapter_int = int(part)
        if chapter_int <= 0:
            raise ValueError(f"Invalid chapter token in --resume-overwrite-range: {part}")
        chapters.add(f"{chapter_int:04d}")
    return chapters


def _extract_chapter_title_and_text(chapter_markdown: str, fallback_title: str = "") -> tuple[str, str]:
    text = str(chapter_markdown or "")
    lines = text.splitlines()
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("#"):
            title = first_line.lstrip("#").strip() or str(fallback_title or "").strip()
            chapter_text = "\n".join(lines[1:]).strip()
            return title, chapter_text
    return str(fallback_title or "").strip(), text.strip()


def _read_chapter_opening_excerpt(chapter_path: Path, fallback_title: str = "") -> str:
    if not chapter_path.exists():
        return ""
    try:
        markdown = chapter_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    _, chapter_text = _extract_chapter_title_and_text(markdown, fallback_title=fallback_title)
    return _extract_opening_excerpt(chapter_text)


def _load_previous_chapter_title(*, output_dir: Path, target_book_id: str, chapter_no: int) -> str:
    prev_no = int(chapter_no or 0) - 1
    if prev_no <= 0:
        return ""
    chapter_path = output_dir / f"{target_book_id}_chapter_{prev_no:04d}.md"
    if not chapter_path.exists():
        return ""
    try:
        markdown = chapter_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    title, _ = _extract_chapter_title_and_text(markdown, fallback_title="")
    return str(title or "").strip()


def _load_run_report_context(output_dir: Path, target_book_id: str) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    report_path = output_dir / f"{target_book_id}_run_report.json"
    if not report_path.exists():
        return ({}, {})
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return ({}, {})
    items = data.get("items", [])
    summary_map: dict[str, str] = {}
    continuity_capsule_map: dict[str, dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            chapter_no = _normalize_chapter_no(item.get("chapter_no"))
            summary = str(item.get("summary") or "").strip()
            if chapter_no and summary:
                summary_map[chapter_no] = summary
            raw_capsule = item.get("continuity_capsule")
            if isinstance(raw_capsule, dict):
                sanitized = _sanitize_continuity_capsules_for_prompt([raw_capsule], window=1)
                if sanitized:
                    capsule = sanitized[0]
                    capsule_chapter_no = _normalize_chapter_no(capsule.get("chapter_no")) or chapter_no
                    if capsule_chapter_no:
                        continuity_capsule_map[capsule_chapter_no] = capsule
    return summary_map, continuity_capsule_map


def _preload_recent_continuity_capsules(
    *,
    output_dir: Path,
    target_book_id: str,
    start_chapter: int,
    continuity_window: int,
    blueprint_chapters: list[dict[str, Any]],
    summary_map: dict[str, str],
    run_report_capsules: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window = max(int(continuity_window or 0), 1)
    history_end = max(int(start_chapter or 1) - 1, 0)
    history_start = max(1, history_end - window + 1)
    preload_meta = {
        "history_start": history_start if history_end > 0 else None,
        "history_end": history_end if history_end > 0 else None,
        "from_run_report": 0,
        "from_chapter_files": 0,
        "preloaded_count": 0,
    }
    if history_end <= 0:
        return ([], preload_meta)

    chapter_plan_map: dict[str, dict[str, Any]] = {}
    for chapter in blueprint_chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_no = _normalize_chapter_no(chapter.get("chapter_no"))
        if chapter_no:
            chapter_plan_map[chapter_no] = chapter

    capsule_by_chapter: dict[str, dict[str, Any]] = {}

    for chapter_no, capsule in run_report_capsules.items():
        normalized_no = _normalize_chapter_no(chapter_no)
        if not normalized_no.isdigit():
            continue
        chapter_int = int(normalized_no)
        if chapter_int < history_start or chapter_int > history_end:
            continue
        sanitized = _sanitize_continuity_capsules_for_prompt([capsule], window=1)
        if sanitized:
            capsule_by_chapter[normalized_no] = sanitized[0]
    preload_meta["from_run_report"] = len(capsule_by_chapter)

    for chapter_int in range(history_start, history_end + 1):
        chapter_no = f"{chapter_int:04d}"
        if chapter_no in capsule_by_chapter:
            continue
        chapter_path = output_dir / f"{target_book_id}_chapter_{chapter_no}.md"
        if not chapter_path.exists():
            continue
        chapter_markdown = chapter_path.read_text(encoding="utf-8")
        fallback_title = str((chapter_plan_map.get(chapter_no) or {}).get("title") or "").strip()
        chapter_title, chapter_text = _extract_chapter_title_and_text(chapter_markdown, fallback_title=fallback_title)
        compact_text = chapter_text.replace("\n", " ").strip()
        chapter_summary = summary_map.get(chapter_no) or (compact_text[:220] if compact_text else chapter_title)
        capsule_by_chapter[chapter_no] = _build_fallback_continuity_capsule(
            chapter_no=chapter_no,
            chapter_title=chapter_title or fallback_title or chapter_no,
            chapter_plan=chapter_plan_map.get(chapter_no) or {},
            chapter_text=chapter_text,
            chapter_summary=chapter_summary,
            hard_pack_b={},
        )
        preload_meta["from_chapter_files"] += 1

    sorted_chapters = sorted(
        (int(chapter_no), chapter_no)
        for chapter_no in capsule_by_chapter
        if chapter_no.isdigit()
    )
    selected = [capsule_by_chapter[chapter_no] for _, chapter_no in sorted_chapters][-window:]
    preloaded = _sanitize_continuity_capsules_for_prompt(selected, window=window)
    preload_meta["preloaded_count"] = len(preloaded)
    return preloaded, preload_meta


def _resolve_locked_until_from_files(
    *,
    output_dir: Path,
    target_book_id: str,
    initial_locked_until: int,
    scan_limit: int | None = None,
) -> int:
    locked = max(int(initial_locked_until or 0), 0)
    upper = max(int(scan_limit or 0), 0)
    while True:
        nxt = locked + 1
        if upper and nxt > upper:
            break
        chapter_path = output_dir / f"{target_book_id}_chapter_{nxt:04d}.md"
        if not chapter_path.exists():
            break
        locked = nxt
    return locked


def _extract_anchor_from_capsule(capsule: dict[str, Any] | None) -> list[str]:
    if not isinstance(capsule, dict):
        return []
    anchor: list[str] = []
    anchor.extend(_sanitize_text_list(capsule.get("next_chapter_must_answer"), max_items=3, max_len=240))
    anchor.extend(_sanitize_text_list(capsule.get("open_threads"), max_items=3, max_len=240))
    normalized, _ = _sanitize_cross_batch_anchor_items(anchor, max_items=3, max_len=320)
    return normalized


def _load_cross_batch_anchor(
    *,
    output_dir: Path,
    target_book_id: str,
    start_chapter: int,
    run_report_capsules: dict[str, dict[str, Any]],
) -> tuple[list[str], str, dict[str, Any]]:
    meta: dict[str, Any] = {
        "raw": [],
        "normalized": [],
        "truncated": False,
    }
    prev_no = max(int(start_chapter or 1) - 1, 0)
    if prev_no <= 0:
        return ([], "book_start", meta)
    prev_key = f"{prev_no:04d}"
    from_report = _extract_anchor_from_capsule(run_report_capsules.get(prev_key))
    if from_report:
        normalized, truncated = _sanitize_cross_batch_anchor_items(from_report, max_items=3, max_len=320)
        meta = {
            "raw": from_report,
            "normalized": normalized,
            "truncated": truncated,
        }
        return (normalized, "run_report_capsule", meta)
    chapter_path = output_dir / f"{target_book_id}_chapter_{prev_key}.md"
    if chapter_path.exists():
        text = chapter_path.read_text(encoding="utf-8")
        title, chapter_text = _extract_chapter_title_and_text(text, fallback_title=prev_key)
        tail_excerpt, tail_truncated = _build_tail_anchor_excerpt(chapter_text, min_chars=80, max_chars=260)
        if tail_excerpt:
            raw_anchor = [f"{title}结尾承接：{tail_excerpt}"]
            normalized, normalized_truncated = _sanitize_cross_batch_anchor_items(
                raw_anchor, max_items=1, max_len=320
            )
            if normalized:
                meta = {
                    "raw": raw_anchor,
                    "normalized": normalized,
                    "truncated": bool(tail_truncated or normalized_truncated),
                }
                return (normalized, "chapter_file_tail", meta)
    return ([], "none", meta)


def _derive_link_items_from_recent_capsule(capsule: dict[str, Any] | None) -> list[str]:
    if not isinstance(capsule, dict):
        return []
    candidates: list[str] = []
    candidates.extend(_sanitize_text_list(capsule.get("next_chapter_must_answer"), max_items=6, max_len=120))
    candidates.extend(_sanitize_text_list(capsule.get("open_threads"), max_items=6, max_len=120))
    candidates.extend(_sanitize_text_list(capsule.get("state_deltas"), max_items=4, max_len=120))
    return _sanitize_text_list(candidates, max_items=3, max_len=120)


def _resolve_chapter_linkage(
    *,
    chapter_no: str,
    idx: int,
    start_chapter: int,
    chapter_plan: dict[str, Any],
    prev_chapter_plan: dict[str, Any],
    recent_continuity_capsules: list[dict[str, Any]],
    hard_pack_b: dict[str, Any],
) -> dict[str, Any]:
    previous_chapter_carry_over = (
        _normalize_link_items(prev_chapter_plan.get("carry_over_to_next")) if isinstance(prev_chapter_plan, dict) else []
    )
    current_chapter_open_with = _normalize_link_items(chapter_plan.get("open_with"))
    carry_source = "blueprint_prev_chapter" if previous_chapter_carry_over else "none"
    open_with_source = "chapter_plan_open_with" if current_chapter_open_with else "none"

    normalized_no = _normalize_chapter_no(chapter_no)
    chapter_int = (
        int(normalized_no)
        if normalized_no.isdigit()
        else max(int(start_chapter or 1), 1) + max(int(idx or 1) - 1, 0)
    )
    is_true_book_start = chapter_int <= 1
    is_batch_first = int(idx or 1) == 1 and not is_true_book_start

    if is_true_book_start:
        return {
            "previous_chapter_carry_over": [],
            "current_chapter_open_with": [],
            "is_batch_first": False,
            "is_true_book_start": True,
            "carry_source": "book_start",
            "open_with_source": "book_start",
        }

    if is_batch_first and not previous_chapter_carry_over:
        previous_chapter_carry_over = _derive_link_items_from_recent_capsule(
            recent_continuity_capsules[-1] if recent_continuity_capsules else None
        )
        if previous_chapter_carry_over:
            carry_source = "recent_continuity_capsule"
        else:
            previous_chapter_carry_over = _sanitize_text_list(
                _collect_prev_threads_from_hard_pack(hard_pack_b),
                max_items=3,
                max_len=120,
            )
            if previous_chapter_carry_over:
                carry_source = "hard_context_open_threads"

    if is_batch_first and not previous_chapter_carry_over and current_chapter_open_with:
        previous_chapter_carry_over = _sanitize_text_list(current_chapter_open_with, max_items=3, max_len=120)
        if previous_chapter_carry_over:
            carry_source = "chapter_plan_open_with_fallback"

    if not current_chapter_open_with and previous_chapter_carry_over:
        current_chapter_open_with = _sanitize_text_list(previous_chapter_carry_over, max_items=3, max_len=120)
        open_with_source = "derived_from_previous_carry_over"
    elif current_chapter_open_with and previous_chapter_carry_over:
        has_overlap = any(
            _link_items_match(carry_item, open_item)
            for carry_item in previous_chapter_carry_over
            for open_item in current_chapter_open_with
        )
        if not has_overlap:
            anchored = [previous_chapter_carry_over[0], *current_chapter_open_with]
            current_chapter_open_with = _sanitize_text_list(anchored, max_items=3, max_len=120)
            open_with_source = "anchored_with_previous_carry_over"

    return {
        "previous_chapter_carry_over": previous_chapter_carry_over,
        "current_chapter_open_with": current_chapter_open_with,
        "is_batch_first": is_batch_first,
        "is_true_book_start": is_true_book_start,
        "carry_source": carry_source,
        "open_with_source": open_with_source,
    }


def _redact_store_for_log(store: MemoryStore) -> dict[str, Any]:
    return {
        "canon_db_path": store.canon_db_path,
        "neo4j_uri": store.neo4j_uri,
        "neo4j_user": store.neo4j_user,
        "neo4j_database": store.neo4j_database,
        "qdrant_url": store.qdrant_url,
        "qdrant_collection": store.qdrant_collection,
    }


class OneClickBookGenerator:
    def __init__(self, llm_config: dict[str, Any]):
        self.llm = LLMClient(llm_config=llm_config)

    def build_master_arc(
        self,
        *,
        target_book_id: str,
        world_spec: dict[str, Any],
        template_profile: dict[str, Any],
        total_chapters: int,
        stage_size: int,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> dict[str, Any]:
        chapter_total = max(int(total_chapters or 0), 1)
        stage_span = max(int(stage_size or 1), 1)
        stage_count = max((chapter_total + stage_span - 1) // stage_span, 1)
        prompt = (
            "你是长篇小说总策划，请输出“全书总纲”JSON。\n"
            "只输出 JSON，不要 markdown。\n"
            "schema:\n"
            "{"
            '"book_title":"string","genre":"string","global_arc":["string"],'
            '"ending_anchor":["string"],'
            '"stage_outline":[{"stage_id":"0001_0100","stage_goal":"string","main_conflict":"string","must_keep_threads":["string"]}]'
            "}\n"
            "要求：\n"
            "1) 所有文本使用简体中文；\n"
            f"2) stage_outline 长度约为 {stage_count}，stage_id 按 {stage_span} 章分段；\n"
            "3) global_arc/ending_anchor 必须可执行，不可使用空泛占位句。\n\n"
            f"target_book_id: {target_book_id}\n"
            f"chapter_total: {chapter_total}\n"
            f"stage_size: {stage_span}\n"
            f"world_spec: {json.dumps(world_spec, ensure_ascii=False)}\n"
            f"template_profile_from_bookA: {json.dumps(template_profile, ensure_ascii=False)}"
        )
        try:
            raw = self.llm.complete_with_retry(
                prompt,
                temperature=0.35,
                max_tokens=3072,
                max_retries=llm_max_retries,
                retry_backoff=llm_retry_backoff,
                backoff_factor=llm_backoff_factor,
                backoff_max=llm_backoff_max,
                retry_jitter=llm_retry_jitter,
            )
            parsed = _parse_json_object(raw)
        except Exception:
            parsed = {}
        global_arc = _sanitize_text_list(parsed.get("global_arc"), max_items=10, max_len=140)
        ending_anchor = _sanitize_text_list(parsed.get("ending_anchor"), max_items=8, max_len=140)
        stage_outline_raw = parsed.get("stage_outline")
        stage_outline: list[dict[str, Any]] = []
        if isinstance(stage_outline_raw, list):
            for row in stage_outline_raw:
                if not isinstance(row, dict):
                    continue
                stage_outline.append(
                    {
                        "stage_id": str(row.get("stage_id") or "").strip(),
                        "stage_goal": str(row.get("stage_goal") or "").strip(),
                        "main_conflict": str(row.get("main_conflict") or "").strip(),
                        "must_keep_threads": _sanitize_text_list(row.get("must_keep_threads"), max_items=6, max_len=120),
                    }
                )
        if not global_arc:
            global_arc = _sanitize_text_list(world_spec.get("global_arc"), max_items=8, max_len=140)
        if not ending_anchor:
            ending_anchor = _sanitize_text_list(world_spec.get("ending_anchor"), max_items=6, max_len=140)
        return {
            "target_book_id": target_book_id,
            "book_title": str(parsed.get("book_title") or world_spec.get("book_title") or target_book_id).strip(),
            "genre": str(parsed.get("genre") or world_spec.get("genre") or "").strip(),
            "global_arc": global_arc,
            "ending_anchor": ending_anchor,
            "stage_outline": stage_outline,
            "chapter_total": chapter_total,
            "stage_size": stage_span,
            "generated_at": datetime.now().isoformat(),
        }

    def build_stage_arc(
        self,
        *,
        stage_start: int,
        stage_end: int,
        world_spec: dict[str, Any],
        master_arc: dict[str, Any],
        cross_batch_anchor: list[str],
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> dict[str, Any]:
        sid = _stage_id(stage_start, stage_end)
        prompt = (
            "你是分卷策划，请输出阶段蓝图 JSON。\n"
            "只输出 JSON，不要 markdown。\n"
            "schema:\n"
            "{"
            '"stage_id":"0001_0100","stage_goal":"string","main_conflict":"string",'
            '"must_keep_entities":["string"],"must_keep_threads":["string"],"batch_guidance":["string"]'
            "}\n"
            "要求：所有文本使用简体中文；batch_guidance 给出对后续20章批次细纲的硬约束。\n\n"
            f"stage_id: {sid}\n"
            f"stage_range: {stage_start}-{stage_end}\n"
            f"cross_batch_anchor: {json.dumps(cross_batch_anchor, ensure_ascii=False)}\n"
            f"master_arc: {json.dumps(master_arc, ensure_ascii=False)}\n"
            f"world_spec: {json.dumps(world_spec, ensure_ascii=False)}"
        )
        try:
            raw = self.llm.complete_with_retry(
                prompt,
                temperature=0.3,
                max_tokens=2048,
                max_retries=llm_max_retries,
                retry_backoff=llm_retry_backoff,
                backoff_factor=llm_backoff_factor,
                backoff_max=llm_backoff_max,
                retry_jitter=llm_retry_jitter,
            )
            parsed = _parse_json_object(raw)
        except Exception:
            parsed = {}
        sanitized_cross_batch_anchor, _ = _sanitize_cross_batch_anchor_items(
            cross_batch_anchor, max_items=3, max_len=320
        )
        return {
            "stage_id": sid,
            "stage_start": int(stage_start),
            "stage_end": int(stage_end),
            "stage_goal": str(parsed.get("stage_goal") or "").strip(),
            "main_conflict": str(parsed.get("main_conflict") or "").strip(),
            "must_keep_entities": _sanitize_text_list(parsed.get("must_keep_entities"), max_items=12, max_len=64),
            "must_keep_threads": _sanitize_text_list(parsed.get("must_keep_threads"), max_items=10, max_len=120),
            "batch_guidance": _sanitize_text_list(parsed.get("batch_guidance"), max_items=10, max_len=120),
            "cross_batch_anchor": sanitized_cross_batch_anchor,
            "generated_at": datetime.now().isoformat(),
        }

    def _refine_chapter_titles(
        self,
        *,
        chapters: list[dict[str, Any]],
        world_spec: dict[str, Any],
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> list[dict[str, str]]:
        chapter_briefs = []
        for chapter in chapters:
            chapter_briefs.append(
                {
                    "chapter_no": chapter.get("chapter_no"),
                    "title_seed": _strip_chapter_title_prefix(chapter.get("title", "")),
                    "goal": chapter.get("goal", ""),
                    "conflict": chapter.get("conflict", ""),
                    "ending_hook": chapter.get("ending_hook", ""),
                }
            )
        prompt = (
            "你是网文编辑，负责把章节名改得更有文学感和钩子感。\n"
            "只输出 JSON，不要 markdown。\n"
            "输出 schema:\n"
            '{"titles":[{"chapter_no":"0001","title":"文学感自由标题"}]}\n\n'
            "硬约束：\n"
            "1) title 只写冒号后的标题，不要写“第X章：”前缀。\n"
            "2) 标题长度建议 8~18 个中文字符。\n"
            "3) 避免模板化、避免多章重复同一结构（例如连续大量“X与Y”）。\n"
            "4) 每章标题要体现该章核心冲突或悬念。\n\n"
            f"world_spec: {json.dumps(world_spec, ensure_ascii=False)}\n"
            f"chapters: {json.dumps(chapter_briefs, ensure_ascii=False)}"
        )
        raw = self.llm.complete_with_retry(
            prompt,
            temperature=0.75,
            max_tokens=2048,
            max_retries=llm_max_retries,
            retry_backoff=llm_retry_backoff,
            backoff_factor=llm_backoff_factor,
            backoff_max=llm_backoff_max,
            retry_jitter=llm_retry_jitter,
        )
        parsed = _parse_json_object(raw)
        titles = parsed.get("titles")
        if not isinstance(titles, list):
            return []
        result: list[dict[str, str]] = []
        for row in titles:
            if not isinstance(row, dict):
                continue
            chapter_no = str(row.get("chapter_no") or "").strip()
            title = _strip_chapter_title_prefix(str(row.get("title") or ""))
            if chapter_no and title:
                result.append({"chapter_no": chapter_no, "title": title})
        return result

    def _repair_blueprint_pair_with_llm(
        self,
        *,
        chapters: list[dict[str, Any]],
        pair: dict[str, Any],
        style: str,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        from_no = str(pair.get("from_chapter") or "").zfill(4)
        to_no = str(pair.get("to_chapter") or "").zfill(4)
        pair_issues = [str(item) for item in (pair.get("issues") or []) if str(item).strip()]
        from_idx = next((idx for idx, row in enumerate(chapters) if str(row.get("chapter_no") or "").zfill(4) == from_no), -1)
        to_idx = next((idx for idx, row in enumerate(chapters) if str(row.get("chapter_no") or "").zfill(4) == to_no), -1)
        if from_idx < 0 or to_idx < 0 or to_idx != from_idx + 1:
            return {
                "from_chapter": from_no,
                "to_chapter": to_no,
                "status": "skip_invalid_pair",
                "changed": False,
                "issues": pair_issues,
            }

        current = chapters[from_idx]
        nxt = chapters[to_idx]
        prompt = (
            "你是章节蓝图连续性修复编辑。请只修复相邻两章承接关系，不改章节编号，不改章节顺序。\n"
            "只输出 JSON，不要 markdown，不要解释。\n"
            "输出 schema:\n"
            "{"
            '"from_chapter":"0143","to_chapter":"0144",'
            '"from_patch":{"open_with":["..."],"carry_over_to_next":["..."],"beat_outline":["..."]},'
            '"to_patch":{"open_with":["..."],"carry_over_to_next":["..."],"beat_outline":["..."]}'
            "}\n"
            "硬约束：\n"
            "1) from_patch.carry_over_to_next 至少一项必须在 to_patch.open_with 中被承接；\n"
            "2) 两章 beat_outline 第一条都必须是可见开场动作，并分别与各自 open_with 对齐；\n"
            "3) 不得把同一事件改写成平行版本，不得照抄上一章大段文本；\n"
            "4) 仅返回这两章 patch 字段，文本为简体中文；\n"
            f"5) ending_style={style}（closure 收束优先，soft 轻留白，hook 强钩子）。\n\n"
            f"pair_issues: {json.dumps(pair_issues, ensure_ascii=False)}\n"
            f"from_chapter_current: {json.dumps(current, ensure_ascii=False)}\n"
            f"to_chapter_current: {json.dumps(nxt, ensure_ascii=False)}"
        )
        try:
            repaired_raw = self.llm.complete_with_retry(
                prompt,
                temperature=0.1,
                max_tokens=max_tokens,
                max_retries=llm_max_retries,
                retry_backoff=llm_retry_backoff,
                backoff_factor=llm_backoff_factor,
                backoff_max=llm_backoff_max,
                retry_jitter=llm_retry_jitter,
            )
            repaired_obj = _parse_json_object(repaired_raw)
        except Exception as exc:
            return {
                "from_chapter": from_no,
                "to_chapter": to_no,
                "status": f"llm_error:{type(exc).__name__}",
                "changed": False,
                "issues": pair_issues,
            }

        from_patch = repaired_obj.get("from_patch")
        to_patch = repaired_obj.get("to_patch")
        if not isinstance(from_patch, dict) or not isinstance(to_patch, dict):
            return {
                "from_chapter": from_no,
                "to_chapter": to_no,
                "status": "invalid_patch_payload",
                "changed": False,
                "issues": pair_issues,
            }

        changed = False
        from_open = _normalize_link_items(from_patch.get("open_with"))
        from_carry = _normalize_link_items(from_patch.get("carry_over_to_next"))
        to_open = _normalize_link_items(to_patch.get("open_with"))
        to_carry = _normalize_link_items(to_patch.get("carry_over_to_next"))
        from_beats = _sanitize_text_list(from_patch.get("beat_outline"), max_items=8, max_len=120)
        to_beats = _sanitize_text_list(to_patch.get("beat_outline"), max_items=8, max_len=120)

        if from_open:
            sanitized = _sanitize_text_list(from_open, max_items=3, max_len=120)
            if sanitized != _normalize_link_items(current.get("open_with")):
                current["open_with"] = sanitized
                changed = True
        if from_carry:
            sanitized = _sanitize_text_list(from_carry, max_items=3, max_len=120)
            if sanitized != _normalize_link_items(current.get("carry_over_to_next")):
                current["carry_over_to_next"] = sanitized
                changed = True
        if to_open:
            sanitized = _sanitize_text_list(to_open, max_items=3, max_len=120)
            if sanitized != _normalize_link_items(nxt.get("open_with")):
                nxt["open_with"] = sanitized
                changed = True
        if to_carry:
            sanitized = _sanitize_text_list(to_carry, max_items=3, max_len=120)
            if sanitized != _normalize_link_items(nxt.get("carry_over_to_next")):
                nxt["carry_over_to_next"] = sanitized
                changed = True
        if from_beats:
            if from_beats != _sanitize_text_list(current.get("beat_outline"), max_items=8, max_len=120):
                current["beat_outline"] = from_beats
                changed = True
        if to_beats:
            if to_beats != _sanitize_text_list(nxt.get("beat_outline"), max_items=8, max_len=120):
                nxt["beat_outline"] = to_beats
                changed = True

        return {
            "from_chapter": from_no,
            "to_chapter": to_no,
            "status": "patched" if changed else "no_change",
            "changed": changed,
            "issues": pair_issues,
        }

    def build_blueprint(
        self,
        *,
        target_book_id: str,
        world_spec: dict[str, Any],
        template_profile: dict[str, Any],
        chapter_count: int,
        start_chapter: int,
        previous_title_hint: str = "",
        ending_style: str,
        max_tokens: int,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> dict[str, Any]:
        style = str(ending_style or "closure").strip().lower()
        if style not in ENDING_STYLES:
            style = "closure"
        has_template_profile = bool(template_profile)
        if has_template_profile:
            role_line = "你是长篇小说策划编辑。根据世界观与模板参考，输出 Book B 的章节蓝图。\n"
            template_source_block = f"template_profile_from_bookA: {json.dumps(template_profile, ensure_ascii=False)}\n"
            template_note = "可参考模板节拍，但不得照抄。\n"
        else:
            role_line = "你是长篇小说策划编辑。根据世界观设定，输出 Book B 的章节蓝图。\n"
            template_source_block = ""
            template_note = "本次不提供 BookA 模板，请仅依据 world_spec 生成可连载蓝图。\n"
        prompt = (
            role_line
            + "只返回 JSON，不要 markdown。\n\n"
            f"target_book_id: {target_book_id}\n"
            f"start_chapter: {start_chapter}\n"
            f"chapter_count: {chapter_count}\n"
            f"world_spec: {json.dumps(world_spec, ensure_ascii=False)}\n"
            f"{template_source_block}"
            f"{template_note}\n"
            "JSON schema:\n"
            "{\n"
            '  "book_title": "string",\n'
            '  "genre": "string",\n'
            '  "global_arc": ["string"],\n'
            '  "chapters": [{"chapter_no":"0001","title":"string","goal":"string","conflict":"string","beat_outline":["string"],"ending_hook":"string","carry_over_to_next":["string"],"open_with":["string"]}]\n'
            "}\n"
            f"要求 chapters 数组长度必须是 {chapter_count}，且所有文本字段使用简体中文。\n"
            "chapter.title 只写标题正文，不要写“第X章：”前缀。\n"
            "相邻章节标题避免同一语法骨架重复（例如连续“X的Y：Z”）。\n"
            "标题起句避免模板化：连续章节不要反复使用“关于……”或“在……处……”。\n"
            "硬约束：\n"
            "1) ending_hook 可选；若填写必须具体，禁止使用“留下悬念/推动下一章”这类通用占位句。\n"
            "2) 每章必须给出 carry_over_to_next（遗留到下一章的具体事项）。\n"
            "3) 从第二章开始必须给出 open_with（开篇承接上一章事项）。\n"
            "4) 第N章 carry_over_to_next 至少一项应在第N+1章 open_with 对应。\n"
            "5) 每章 beat_outline 第一条必须是可见的开场动作/事件，不得写成“已发生结论”。\n"
            "6) 章末风格按 ending_style 执行：closure=阶段收束优先且不强行抛悬念，soft=轻留白，hook=强钩子。"
            f"\nending_style: {style}"
        )
        raw = self.llm.complete_with_retry(
            prompt,
            temperature=0.4,
            max_tokens=max_tokens,
            max_retries=llm_max_retries,
            retry_backoff=llm_retry_backoff,
            backoff_factor=llm_backoff_factor,
            backoff_max=llm_backoff_max,
            retry_jitter=llm_retry_jitter,
        )

        parsed: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                parsed = _parse_json_object(raw)
                break
            except Exception as exc:
                last_error = exc
                if attempt >= 3:
                    break
                repair_prompt = (
                    "把下面内容修复为严格合法 JSON 对象，只输出 JSON。\n"
                    f"要求 chapters 数组长度必须是 {chapter_count}。\n"
                    "文本字段必须是简体中文。chapter.title 只写标题正文，不要写“第X章：”前缀。\n"
                    "不要补充解释，不要 markdown。\n\n"
                    f"待修复内容:\n{raw}"
                )
                raw = self.llm.complete_with_retry(
                    repair_prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    max_retries=llm_max_retries,
                    retry_backoff=llm_retry_backoff,
                    backoff_factor=llm_backoff_factor,
                    backoff_max=llm_backoff_max,
                    retry_jitter=llm_retry_jitter,
                )
        if parsed is None:
            raise RuntimeError(f"Failed to parse blueprint JSON after repair attempts: {last_error}")
        chapters_raw = parsed.get("chapters")
        if not isinstance(chapters_raw, list):
            chapters_raw = []
        normalized_chapters = _normalize_chapter_plan(
            chapters_raw,
            chapter_count,
            start_chapter,
            ending_style=style,
        )
        try:
            refined_titles = self._refine_chapter_titles(
                chapters=normalized_chapters,
                world_spec=world_spec,
                llm_max_retries=llm_max_retries,
                llm_retry_backoff=llm_retry_backoff,
                llm_backoff_factor=llm_backoff_factor,
                llm_backoff_max=llm_backoff_max,
                llm_retry_jitter=llm_retry_jitter,
            )
            title_map = {row["chapter_no"]: row["title"] for row in refined_titles if row.get("chapter_no")}
            for chapter in normalized_chapters:
                chapter_no = str(chapter.get("chapter_no") or "")
                match = re.search(r"(\d+)", chapter_no)
                chapter_int = int(match.group(1)) if match else 1
                chapter["title"] = _format_chapter_title(
                    chapter_int,
                    title_map.get(chapter_no) or chapter.get("title", ""),
                    chapter.get("goal", ""),
                )
        except Exception:
            # Title polish failure should not block the whole generation.
            for chapter in normalized_chapters:
                chapter_no = str(chapter.get("chapter_no") or "")
                chapter_int = int(chapter_no) if chapter_no.isdigit() else 1
                chapter["title"] = _format_chapter_title(
                    chapter_int,
                    chapter.get("title", ""),
                    chapter.get("goal", ""),
                )
        continuity_report = _validate_blueprint_continuity(
            normalized_chapters,
            start_chapter=start_chapter,
            ending_style=style,
        )
        continuity_repair_attempts = 0
        blueprint_auto_link_fix_count = 0
        blueprint_auto_opening_anchor_fix_count = 0
        if not continuity_report.get("ok"):
            auto_patched_chapters, auto_link_fix_count = _auto_patch_blueprint_links(normalized_chapters)
            auto_patched_chapters, auto_opening_fix_count = _auto_patch_opening_beats(
                auto_patched_chapters,
                start_chapter=start_chapter,
            )
            if auto_link_fix_count or auto_opening_fix_count:
                normalized_chapters = auto_patched_chapters
                blueprint_auto_link_fix_count += auto_link_fix_count
                blueprint_auto_opening_anchor_fix_count += auto_opening_fix_count
                continuity_report = _validate_blueprint_continuity(
                    normalized_chapters,
                    start_chapter=start_chapter,
                    ending_style=style,
                )
        for repair_round in range(1, 7):
            if continuity_report.get("ok"):
                break
            continuity_repair_attempts = repair_round
            issue_preview = continuity_report.get("issues", [])[:24]
            pair_preview = continuity_report.get("pairs", [])[:10]
            repair_prompt = (
                "你是小说蓝图修复器。请修复下面 chapters 的跨章连续性问题。\n"
                "只输出 JSON 对象，不要 markdown，不要解释。\n"
                "输出 schema:\n"
                '{"chapters":[{"chapter_no":"0001","title":"文学感自由标题","goal":"...","conflict":"...","beat_outline":["..."],"ending_hook":"具体钩子","carry_over_to_next":["..."],"open_with":["..."]}]}\n'
                f"硬约束：chapters 数组长度必须是 {chapter_count}，chapter_no 顺序与数量不可改变。\n"
                "ending_hook 可选；若填写禁止通用空钩子；必须确保第N章 carry_over_to_next 至少一项在第N+1章 open_with 承接。\n"
                "每章 beat_outline 第一条必须是可见开场动作，并与本章 open_with 对齐。\n"
                f"章末风格 ending_style={style}（closure 收束优先，soft 轻留白，hook 强钩子）。\n"
                "所有文本字段用简体中文。\n\n"
                f"问题列表: {json.dumps(issue_preview, ensure_ascii=False)}\n"
                f"承接诊断样例: {json.dumps(pair_preview, ensure_ascii=False)}\n"
                f"当前 chapters: {json.dumps(normalized_chapters, ensure_ascii=False)}"
            )
            repaired_raw = self.llm.complete_with_retry(
                repair_prompt,
                temperature=0.1,
                max_tokens=max_tokens,
                max_retries=llm_max_retries,
                retry_backoff=llm_retry_backoff,
                backoff_factor=llm_backoff_factor,
                backoff_max=llm_backoff_max,
                retry_jitter=llm_retry_jitter,
            )
            repaired_obj = _parse_json_object(repaired_raw)
            repaired_chapters_raw = repaired_obj.get("chapters")
            if not isinstance(repaired_chapters_raw, list):
                continue
            normalized_chapters = _normalize_chapter_plan(
                repaired_chapters_raw,
                chapter_count,
                start_chapter,
                ending_style=style,
            )
            auto_patched_chapters, auto_link_fix_count = _auto_patch_blueprint_links(normalized_chapters)
            auto_patched_chapters, auto_opening_fix_count = _auto_patch_opening_beats(
                auto_patched_chapters,
                start_chapter=start_chapter,
            )
            if auto_link_fix_count or auto_opening_fix_count:
                normalized_chapters = auto_patched_chapters
                blueprint_auto_link_fix_count += auto_link_fix_count
                blueprint_auto_opening_anchor_fix_count += auto_opening_fix_count
            continuity_report = _validate_blueprint_continuity(
                normalized_chapters,
                start_chapter=start_chapter,
                ending_style=style,
            )

        title_diversity_alerts = _diversify_adjacent_titles(
            normalized_chapters,
            previous_title=previous_title_hint,
        )

        blueprint_opening_llm_repair_attempts = 0
        blueprint_pair_llm_repair_rounds = 0
        blueprint_pair_llm_repair_trace: list[dict[str, Any]] = []
        if not continuity_report.get("ok"):
            for repair_round in range(1, 5):
                blueprint_opening_llm_repair_attempts = repair_round
                issue_preview = continuity_report.get("issues", [])[:24]
                repair_prompt = (
                    "你是章节连续性修复编辑。请只修复跨章承接，不改 chapter_no 与章节数量。\n"
                    "只输出 JSON 对象，不要 markdown，不要解释。\n"
                    "输出 schema:\n"
                    '{"chapters":[{"chapter_no":"0001","title":"文学感自由标题","goal":"...","conflict":"...",'
                    '"beat_outline":["..."],"ending_hook":"具体钩子","carry_over_to_next":["..."],"open_with":["..."]}]}\n'
                    f"硬约束：chapters 数组长度必须是 {chapter_count}，chapter_no 顺序不可变化。\n"
                    "强约束：\n"
                    "1) 第N章 carry_over_to_next 至少一项在第N+1章 open_with 对应；\n"
                    "2) 每章 beat_outline 第一条必须与本章 open_with 对齐，且是可见开场动作（不得写成已发生结论）；\n"
                    "3) 不得把同一事件改写成平行版本，不得整段复述上一章；\n"
                    "4) 标题保留文学感并避免相邻重复骨架。\n\n"
                    f"上一章标题提示: {previous_title_hint}\n"
                    f"当前问题: {json.dumps(issue_preview, ensure_ascii=False)}\n"
                    f"当前 chapters: {json.dumps(normalized_chapters, ensure_ascii=False)}"
                )
                repaired_raw = self.llm.complete_with_retry(
                    repair_prompt,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    max_retries=llm_max_retries,
                    retry_backoff=llm_retry_backoff,
                    backoff_factor=llm_backoff_factor,
                    backoff_max=llm_backoff_max,
                    retry_jitter=llm_retry_jitter,
                )
                try:
                    repaired_obj = _parse_json_object(repaired_raw)
                except Exception:
                    continue
                repaired_chapters_raw = repaired_obj.get("chapters")
                if not isinstance(repaired_chapters_raw, list):
                    continue
                normalized_chapters = _normalize_chapter_plan(
                    repaired_chapters_raw,
                    chapter_count,
                    start_chapter,
                    ending_style=style,
                )
                auto_patched_chapters, auto_link_fix_count = _auto_patch_blueprint_links(normalized_chapters)
                auto_patched_chapters, auto_opening_fix_count = _auto_patch_opening_beats(
                    auto_patched_chapters,
                    start_chapter=start_chapter,
                )
                if auto_link_fix_count or auto_opening_fix_count:
                    normalized_chapters = auto_patched_chapters
                    blueprint_auto_link_fix_count += auto_link_fix_count
                    blueprint_auto_opening_anchor_fix_count += auto_opening_fix_count
                _diversify_adjacent_titles(normalized_chapters, previous_title=previous_title_hint)
                continuity_report = _validate_blueprint_continuity(
                    normalized_chapters,
                    start_chapter=start_chapter,
                    ending_style=style,
                )
                if continuity_report.get("ok"):
                    break

        if not continuity_report.get("ok"):
            for repair_round in range(1, 9):
                failed_pairs = [
                    pair
                    for pair in (continuity_report.get("pairs") or [])
                    if isinstance(pair, dict) and not bool(pair.get("ok"))
                ]
                if not failed_pairs:
                    break
                blueprint_pair_llm_repair_rounds = repair_round
                round_trace: dict[str, Any] = {
                    "round": repair_round,
                    "failed_pair_count_before": len(failed_pairs),
                    "pairs": [],
                }
                round_changed = False
                for pair in failed_pairs:
                    trace_item = self._repair_blueprint_pair_with_llm(
                        chapters=normalized_chapters,
                        pair=pair,
                        style=style,
                        llm_max_retries=llm_max_retries,
                        llm_retry_backoff=llm_retry_backoff,
                        llm_backoff_factor=llm_backoff_factor,
                        llm_backoff_max=llm_backoff_max,
                        llm_retry_jitter=llm_retry_jitter,
                        max_tokens=max_tokens,
                    )
                    round_trace["pairs"].append(trace_item)
                    if trace_item.get("changed"):
                        round_changed = True
                auto_patched_chapters, auto_link_fix_count = _auto_patch_blueprint_links(normalized_chapters)
                auto_patched_chapters, auto_opening_fix_count = _auto_patch_opening_beats(
                    auto_patched_chapters,
                    start_chapter=start_chapter,
                )
                if auto_link_fix_count or auto_opening_fix_count:
                    normalized_chapters = auto_patched_chapters
                    blueprint_auto_link_fix_count += auto_link_fix_count
                    blueprint_auto_opening_anchor_fix_count += auto_opening_fix_count
                continuity_report = _validate_blueprint_continuity(
                    normalized_chapters,
                    start_chapter=start_chapter,
                    ending_style=style,
                )
                round_trace["failed_pair_count_after"] = int(
                    continuity_report.get("summary", {}).get("broken_link_count") or 0
                )
                round_trace["gate_ok_after_round"] = bool(continuity_report.get("ok"))
                blueprint_pair_llm_repair_trace.append(round_trace)
                if continuity_report.get("ok"):
                    break
                if not round_changed:
                    break

        failed_pairs_final = [
            {
                "from_chapter": str(pair.get("from_chapter") or "").zfill(4),
                "to_chapter": str(pair.get("to_chapter") or "").zfill(4),
                "issues": [str(item) for item in (pair.get("issues") or []) if str(item).strip()],
            }
            for pair in (continuity_report.get("pairs") or [])
            if isinstance(pair, dict) and not bool(pair.get("ok"))
        ]
        summary = continuity_report.get("summary")
        if isinstance(summary, dict):
            summary["pair_llm_repair_rounds"] = blueprint_pair_llm_repair_rounds
            summary["pair_llm_repair_trace_len"] = len(blueprint_pair_llm_repair_trace)
            summary["failed_pairs_final_count"] = len(failed_pairs_final)
            summary["auto_link_fix_count"] = blueprint_auto_link_fix_count
            summary["auto_opening_anchor_fix_count"] = blueprint_auto_opening_anchor_fix_count

        if not continuity_report.get("ok"):
            raise RuntimeError(
                "Blueprint continuity gate failed: "
                + "; ".join(str(item) for item in continuity_report.get("issues", [])[:8])
            )

        for chapter in normalized_chapters:
            if isinstance(chapter, dict):
                chapter.pop("needs_repair", None)
        parsed["chapters"] = normalized_chapters
        parsed["_blueprint_continuity_report"] = continuity_report
        parsed["_blueprint_continuity_repair_attempts"] = continuity_repair_attempts
        parsed["_blueprint_auto_link_fix_count"] = blueprint_auto_link_fix_count
        parsed["_blueprint_auto_opening_anchor_fix_count"] = blueprint_auto_opening_anchor_fix_count
        parsed["_blueprint_opening_llm_repair_attempts"] = blueprint_opening_llm_repair_attempts
        parsed["_blueprint_pair_llm_repair_rounds"] = blueprint_pair_llm_repair_rounds
        parsed["_blueprint_pair_llm_repair_trace"] = blueprint_pair_llm_repair_trace
        parsed["_blueprint_failed_pairs_final"] = failed_pairs_final
        parsed["_title_diversity_alerts"] = title_diversity_alerts
        return parsed

    def generate_chapter_text(
        self,
        *,
        world_spec: dict[str, Any],
        blueprint_meta: dict[str, Any],
        chapter_plan: dict[str, Any],
        template_chapter_pack: dict[str, Any],
        hard_pack_b: dict[str, Any],
        previous_chapter_carry_over: list[str],
        current_chapter_open_with: list[str],
        recent_continuity_capsules: list[dict[str, Any]],
        reference_profile: dict[str, Any] | None,
        continuity_window: int,
        continuity_min_entities: int,
        continuity_min_open_threads: int,
        ending_style: str,
        chapter_min_chars: int,
        chapter_max_chars: int,
        temperature: float,
        max_tokens: int,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> str:
        style = str(ending_style or "closure").strip().lower()
        if style not in ENDING_STYLES:
            style = "closure"
        reference_profile = reference_profile if isinstance(reference_profile, dict) else {}
        anchor_entities = _sanitize_text_list(reference_profile.get("anchor_entities"), max_items=6, max_len=24)
        pronoun_hints = [
            f"{name}:{hint}"
            for name, hint in (reference_profile.get("entity_pronoun_hint") or {}).items()
            if str(name).strip() and str(hint).strip() in {"他", "她", "TA"}
        ][:6]
        pov_hint = str(reference_profile.get("pov_hint") or "").strip()
        if pov_hint == "first_person":
            pov_hint_text = "第一人称主叙"
        elif pov_hint == "third_person":
            pov_hint_text = "第三人称主叙"
        else:
            pov_hint_text = "mixed"
        if style == "hook":
            ending_rule_line = "- 章节结尾呼应 ending_hook，可制造明确钩子推进下一章\n"
        elif style == "soft":
            ending_rule_line = "- 章节结尾优先阶段收束，可保留轻微未决点，但避免硬反转式强钩子\n"
        else:
            ending_rule_line = "- 章节结尾优先阶段收束，不强行抛悬念；若 ending_hook 为空则禁止硬钩子\n"
        hard_rules_text = (
            f"硬约束：\n- 字数尽量在 {chapter_min_chars} 到 {chapter_max_chars} 中文字符之间\n"
            "- 绝不违背 hard_context_from_bookB 的硬事实\n"
            f"{ending_rule_line}"
            "- 可借鉴模板，不可照抄文本\n"
            "- 所有叙事文本、规则描述、状态描述均使用简体中文（专有ID/缩写可保留）\n"
            "- 开篇前15%必须承接 current_chapter_open_with，且回应 previous_chapter_carry_over\n"
            "- chapter_plan.beat_outline 的第1条必须在开篇20%内落地为可见动作/对话，不得只写回忆或结论\n"
            "- 不得把承接事项写成读者未见过程的既成事实，至少写出一个现场推进片段\n"
            "- 关键人物首次出现优先“名字+动作/对话”，不要仅用裸代词起句\n"
            "- 已建立人物的代词不得反向漂移（他/她保持一致）\n"
            f"- 必须延续最近上下文中的关键实体，尽量提及不少于 {max(continuity_min_entities, 0)} 个\n"
            f"- 必须推进最近开放线索，尽量覆盖不少于 {max(continuity_min_open_threads, 0)} 条"
        )
        prompt = (
            "你是中文长篇小说作者。请生成完整章节正文。\n"
            "Book B 需要保持自身硬一致性，并参考 Book A 模板节拍与风格。\n"
            "输出纯正文，不要 JSON，不要代码块。\n\n"
            f"world_spec: {json.dumps(world_spec, ensure_ascii=False)}\n"
            f"book_blueprint_meta: {json.dumps(blueprint_meta, ensure_ascii=False)}\n"
            f"chapter_plan: {json.dumps(chapter_plan, ensure_ascii=False)}\n"
            f"template_pack_from_bookA: {json.dumps(template_chapter_pack, ensure_ascii=False)}\n"
            f"hard_context_from_bookB: {json.dumps(hard_pack_b, ensure_ascii=False)}\n"
            f"previous_chapter_carry_over: {json.dumps(previous_chapter_carry_over, ensure_ascii=False)}\n"
            f"current_chapter_open_with: {json.dumps(current_chapter_open_with, ensure_ascii=False)}\n"
            f"opening_reference_profile: {json.dumps(reference_profile, ensure_ascii=False)}\n"
            f"opening_anchor_entities: {json.dumps(anchor_entities, ensure_ascii=False)}\n"
            f"opening_pronoun_hints: {json.dumps(pronoun_hints, ensure_ascii=False)}\n"
            f"opening_pov_hint: {pov_hint_text}\n"
            f"recent_continuity_capsules_bookB: {json.dumps(recent_continuity_capsules[-max(continuity_window, 1):], ensure_ascii=False)}\n\n"
            f"{hard_rules_text}"
        )
        text = self.llm.complete_with_retry(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=llm_max_retries,
            retry_backoff=llm_retry_backoff,
            backoff_factor=llm_backoff_factor,
            backoff_max=llm_backoff_max,
            retry_jitter=llm_retry_jitter,
        )
        return _strip_markdown_code_blocks(text)

    def rewrite_chapter_opening_for_continuity(
        self,
        *,
        chapter_title: str,
        chapter_plan: dict[str, Any],
        chapter_text: str,
        previous_chapter_carry_over: list[str],
        current_chapter_open_with: list[str],
        recent_continuity_capsules: list[dict[str, Any]],
        continuity_issues: list[str],
        previous_chapter_opening_text: str,
        first_beat_expected: str,
        reference_profile: dict[str, Any] | None,
        opening_max_chars: int,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> str:
        body = str(chapter_text or "").strip()
        if not body:
            return body
        opening_limit = max(int(opening_max_chars or 0), 260)
        opening_text, remainder_text = _split_chapter_opening_window(
            body,
            max_chars=opening_limit,
            min_chars=min(220, max(120, opening_limit // 2)),
        )
        if not opening_text:
            return body
        profile = reference_profile if isinstance(reference_profile, dict) else _build_opening_reference_profile(
            previous_chapter_carry_over=previous_chapter_carry_over,
            current_chapter_open_with=current_chapter_open_with,
            previous_chapter_opening_text=previous_chapter_opening_text,
            chapter_opening_text=opening_text,
            recent_continuity_capsules=recent_continuity_capsules,
        )
        reference_attempts = max(OPENING_REFERENCE_REWRITE_MAX_ATTEMPTS, 1)
        rewrite_issues = [str(item) for item in continuity_issues[:8]]
        for _attempt in range(1, reference_attempts + 1):
            prompt = (
                "你是小说连载连续性编辑。请只重写“章节开头片段”，用于承接上一章，其他正文剧情不改。\n"
                "只输出 JSON，不要 markdown，不要解释。\n"
                'schema: {"opening":"重写后的开头正文"}\n'
                "硬约束：\n"
                "1) opening 必须是中文叙事正文，不得输出标题、编号、注释；\n"
                "2) opening 必须显式承接 previous_chapter_carry_over 与 current_chapter_open_with 至少各1项；\n"
                "3) 必须在开头落地 first_beat_expected（写成可见动作/对话，不写成既成事实）；\n"
                "4) 禁止照抄 previous_chapter_opening_text；\n"
                "5) 不得新增世界观硬设定，只做承接修补；\n"
                "6) 保持角色代词一致，不得把“他/她”反向漂移；\n"
                "7) 开头首次提及关键角色时优先“名字+动作/对话”，禁止裸代词主导。\n\n"
                f"chapter_title: {chapter_title}\n"
                f"chapter_plan: {json.dumps(chapter_plan, ensure_ascii=False)}\n"
                f"continuity_issues: {json.dumps(rewrite_issues[:8], ensure_ascii=False)}\n"
                f"reference_profile: {json.dumps(profile, ensure_ascii=False)}\n"
                f"previous_chapter_carry_over: {json.dumps(previous_chapter_carry_over, ensure_ascii=False)}\n"
                f"current_chapter_open_with: {json.dumps(current_chapter_open_with, ensure_ascii=False)}\n"
                f"first_beat_expected: {first_beat_expected}\n"
                f"recent_continuity_capsules_bookB: {json.dumps(recent_continuity_capsules[-3:], ensure_ascii=False)}\n"
                f"previous_chapter_opening_text: {previous_chapter_opening_text[:700]}\n\n"
                f"待重写 opening 原文:\n{opening_text}"
            )
            rewritten = self.llm.complete_with_retry(
                prompt,
                temperature=0.35,
                max_tokens=max(512, min(1600, opening_limit * 2)),
                max_retries=llm_max_retries,
                retry_backoff=llm_retry_backoff,
                backoff_factor=llm_backoff_factor,
                backoff_max=llm_backoff_max,
                retry_jitter=llm_retry_jitter,
            )
            candidate = ""
            try:
                parsed = _parse_json_object(rewritten)
                candidate = str(parsed.get("opening") or "").strip()
            except Exception:
                candidate = _strip_markdown_code_blocks(rewritten).strip()
            candidate = _strip_markdown_code_blocks(candidate)
            candidate = _sanitize_generated_chapter_text(candidate, chapter_title)
            if len(re.findall(r"[\u4e00-\u9fff]", candidate)) < 80:
                rewrite_issues = ["重写开头字数不足，需补足完整现场叙事"]
                continue
            merged = candidate.strip()
            if remainder_text:
                merged = f"{merged}\n\n{remainder_text.strip()}"
            reference_validation = _validate_opening_reference_consistency(
                chapter_text=merged,
                reference_profile=profile,
                opening_max_chars=opening_limit,
                min_anchor_hits=OPENING_REFERENCE_MIN_ANCHOR_HITS,
            )
            if reference_validation.get("ok"):
                return merged.strip()
            rewrite_issues = [str(item) for item in reference_validation.get("issues") or []]
        return body

    def _normalize_continuity_capsule(
        self,
        *,
        raw: dict[str, Any] | None,
        chapter_no: str,
        chapter_title: str,
        chapter_plan: dict[str, Any],
        chapter_text: str,
        chapter_summary: str,
        hard_pack_b: dict[str, Any],
    ) -> dict[str, Any]:
        fallback = _build_fallback_continuity_capsule(
            chapter_no=chapter_no,
            chapter_title=chapter_title,
            chapter_plan=chapter_plan,
            chapter_text=chapter_text,
            chapter_summary=chapter_summary,
            hard_pack_b=hard_pack_b,
        )
        if not isinstance(raw, dict):
            return fallback

        normalized = {
            "chapter_no": str(raw.get("chapter_no") or chapter_no).strip().zfill(4),
            "chapter_title": str(raw.get("chapter_title") or chapter_title).strip(),
            "chapter_summary": str(raw.get("chapter_summary") or chapter_summary).replace("\n", " ").strip()[:300],
            "carry_entities": _sanitize_text_list(raw.get("carry_entities"), max_items=10, max_len=32),
            "state_deltas": _sanitize_text_list(raw.get("state_deltas"), max_items=8, max_len=120),
            "open_threads": _sanitize_text_list(raw.get("open_threads"), max_items=8, max_len=120),
            "hard_rules_touched": _sanitize_text_list(raw.get("hard_rules_touched"), max_items=8, max_len=120),
            "next_chapter_must_answer": _sanitize_text_list(raw.get("next_chapter_must_answer"), max_items=6, max_len=120),
        }
        if not normalized["chapter_summary"]:
            normalized["chapter_summary"] = fallback["chapter_summary"]
        if not normalized["carry_entities"]:
            normalized["carry_entities"] = fallback["carry_entities"]
        if not normalized["open_threads"]:
            normalized["open_threads"] = fallback["open_threads"]
        if not normalized["state_deltas"]:
            normalized["state_deltas"] = fallback["state_deltas"]
        if not normalized["hard_rules_touched"]:
            normalized["hard_rules_touched"] = fallback["hard_rules_touched"]
        if not normalized["next_chapter_must_answer"]:
            normalized["next_chapter_must_answer"] = fallback["next_chapter_must_answer"]
        return normalized

    def build_continuity_capsule(
        self,
        *,
        chapter_no: str,
        chapter_title: str,
        chapter_plan: dict[str, Any],
        chapter_text: str,
        hard_pack_b: dict[str, Any],
        summary_style: str,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> dict[str, Any]:
        fallback_summary = self.summarize_chapter(
            chapter_title=chapter_title,
            chapter_text=chapter_text,
            llm_max_retries=llm_max_retries,
            llm_retry_backoff=llm_retry_backoff,
            llm_backoff_factor=llm_backoff_factor,
            llm_backoff_max=llm_backoff_max,
            llm_retry_jitter=llm_retry_jitter,
        )
        if summary_style != "structured":
            return self._normalize_continuity_capsule(
                raw=None,
                chapter_no=chapter_no,
                chapter_title=chapter_title,
                chapter_plan=chapter_plan,
                chapter_text=chapter_text,
                chapter_summary=fallback_summary,
                hard_pack_b=hard_pack_b,
            )

        prompt = (
            "你是连续性编辑，请把当前章节提炼成“连续性胶囊”JSON。\n"
            "只输出 JSON，不要 markdown。\n"
            "schema:\n"
            "{"
            '"chapter_no":"0001","chapter_title":"标题","chapter_summary":"80-180字中文摘要",'
            '"carry_entities":["角色/组织"],"state_deltas":["状态变化"],'
            '"open_threads":["未闭合线索"],"hard_rules_touched":["触及硬规则"],'
            '"next_chapter_must_answer":["下一章必须回应的问题"]'
            "}\n"
            "要求：\n"
            "1) 所有文本使用简体中文；\n"
            "2) 列表字段尽量精炼，每项不超过一行；\n"
            "3) 仅提炼事实，不补新设定。\n\n"
            f"chapter_no: {chapter_no}\n"
            f"chapter_title: {chapter_title}\n"
            f"chapter_plan: {json.dumps(chapter_plan, ensure_ascii=False)}\n"
            f"hard_context_from_bookB: {json.dumps(hard_pack_b, ensure_ascii=False)}\n"
            f"chapter_text:\n{chapter_text[:5500]}"
        )
        raw_capsule: dict[str, Any] | None = None
        try:
            raw = self.llm.complete_with_retry(
                prompt,
                temperature=0.1,
                max_tokens=1024,
                max_retries=llm_max_retries,
                retry_backoff=llm_retry_backoff,
                backoff_factor=llm_backoff_factor,
                backoff_max=llm_backoff_max,
                retry_jitter=llm_retry_jitter,
            )
            raw_capsule = _parse_json_object(raw)
        except Exception:
            raw_capsule = None

        return self._normalize_continuity_capsule(
            raw=raw_capsule,
            chapter_no=chapter_no,
            chapter_title=chapter_title,
            chapter_plan=chapter_plan,
            chapter_text=chapter_text,
            chapter_summary=fallback_summary,
            hard_pack_b=hard_pack_b,
        )

    def summarize_chapter(
        self,
        *,
        chapter_title: str,
        chapter_text: str,
        llm_max_retries: int,
        llm_retry_backoff: float,
        llm_backoff_factor: float,
        llm_backoff_max: float,
        llm_retry_jitter: float,
    ) -> str:
        prompt = f"请用1-2句总结章节。标题：{chapter_title}\n正文：\n{chapter_text[:4000]}"
        summary = self.llm.complete_with_retry(
            prompt,
            temperature=0.2,
            max_tokens=256,
            max_retries=llm_max_retries,
            retry_backoff=llm_retry_backoff,
            backoff_factor=llm_backoff_factor,
            backoff_max=llm_backoff_max,
            retry_jitter=llm_retry_jitter,
        )
        return summary.replace("\n", " ").strip()[:300]


def _upsert_index_records(
    records: list[dict[str, Any]],
    *,
    match_keys: tuple[str, ...],
    payload: dict[str, Any],
) -> None:
    matched = False
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        if all(row.get(key) == payload.get(key) for key in match_keys):
            records[idx] = payload
            matched = True
            break
    if not matched:
        records.append(payload)


def _build_hierarchical_blueprint(
    *,
    args: argparse.Namespace,
    generator: OneClickBookGenerator,
    target_book_id: str,
    world_spec: dict[str, Any],
    template_profile: dict[str, Any],
    output_dir: Path,
    freeze_published_blueprint: bool,
    stage_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    batch_start = max(int(args.start_chapter or 1), 1)
    batch_end = batch_start + max(int(args.chapter_count or 0), 1) - 1
    previous_title_hint = _load_previous_chapter_title(
        output_dir=output_dir,
        target_book_id=target_book_id,
        chapter_no=batch_start,
    )
    blueprint_root_arg = str(getattr(args, "blueprint_root", "") or "").strip()
    blueprint_root = (
        Path(blueprint_root_arg).expanduser()
        if blueprint_root_arg
        else output_dir / "blueprints" / target_book_id
    )
    store = BlueprintStore(
        root_dir=blueprint_root,
        target_book_id=target_book_id,
        stage_size=stage_size,
        freeze_published=freeze_published_blueprint,
    )
    index = store.load_index()
    existing_locked = max(int(index.get("locked_until") or 0), 0)
    scan_limit = max(batch_end, int(index.get("book_total_chapters") or 0), existing_locked)
    locked_until = _resolve_locked_until_from_files(
        output_dir=output_dir,
        target_book_id=target_book_id,
        initial_locked_until=existing_locked,
        scan_limit=scan_limit,
    )
    requested_total = max(int(getattr(args, "book_total_chapters", 0) or 0), 0)
    total_chapters = max(requested_total, int(index.get("book_total_chapters") or 0), batch_end, locked_until)
    index["locked_until"] = locked_until
    index["book_total_chapters"] = total_chapters
    index["stage_size"] = int(stage_size)
    index["freeze_published_blueprint"] = bool(freeze_published_blueprint)

    _, run_report_capsules = _load_run_report_context(output_dir, target_book_id)
    cross_batch_anchor, cross_batch_anchor_source, cross_batch_anchor_meta = _load_cross_batch_anchor(
        output_dir=output_dir,
        target_book_id=target_book_id,
        start_chapter=batch_start,
        run_report_capsules=run_report_capsules,
    )

    master_arc = store.load_master_arc()
    if not isinstance(master_arc, dict):
        master_arc = generator.build_master_arc(
            target_book_id=target_book_id,
            world_spec=world_spec,
            template_profile=template_profile,
            total_chapters=total_chapters,
            stage_size=stage_size,
            llm_max_retries=args.llm_max_retries,
            llm_retry_backoff=args.llm_retry_backoff,
            llm_backoff_factor=args.llm_backoff_factor,
            llm_backoff_max=args.llm_backoff_max,
            llm_retry_jitter=args.llm_retry_jitter,
        )
        store.save_master_arc(master_arc)

    touched_stage_bounds: list[tuple[int, int]] = []
    stage_cursor = batch_start
    while stage_cursor <= batch_end:
        stage_start, stage_end = _stage_bounds_for_chapter(stage_cursor, stage_size)
        touched_stage_bounds.append((stage_start, stage_end))
        stage_cursor = stage_end + 1
    touched_stage_bounds = list(dict.fromkeys(touched_stage_bounds))

    stage_arcs: list[dict[str, Any]] = []
    for stage_start, stage_end in touched_stage_bounds:
        stage_arc = store.load_stage(stage_start, stage_end)
        if not isinstance(stage_arc, dict):
            stage_arc = generator.build_stage_arc(
                stage_start=stage_start,
                stage_end=stage_end,
                world_spec=world_spec,
                master_arc=master_arc,
                cross_batch_anchor=cross_batch_anchor,
                llm_max_retries=args.llm_max_retries,
                llm_retry_backoff=args.llm_retry_backoff,
                llm_backoff_factor=args.llm_backoff_factor,
                llm_backoff_max=args.llm_backoff_max,
                llm_retry_jitter=args.llm_retry_jitter,
            )
            store.save_stage(stage_start, stage_end, stage_arc)
        stage_arcs.append(stage_arc)
        _upsert_index_records(
            index.setdefault("stages", []),
            match_keys=("stage_id",),
            payload={
                "stage_id": _stage_id(stage_start, stage_end),
                "stage_start": stage_start,
                "stage_end": stage_end,
                "path": str(store.stage_path(stage_start, stage_end)),
                "updated_at": datetime.now().isoformat(),
            },
        )

    batch_path = store.batch_path(batch_start, batch_end)
    batch_blueprint: dict[str, Any] | None = None
    batch_plan_source = "generated"
    if freeze_published_blueprint and batch_path.exists():
        existing_batch = store.load_batch(batch_start, batch_end)
        if isinstance(existing_batch, dict):
            cached_chapters_raw = existing_batch.get("chapters")
            if isinstance(cached_chapters_raw, list):
                cached_chapters = _normalize_chapter_plan(
                    cached_chapters_raw,
                    int(args.chapter_count),
                    batch_start,
                    ending_style=getattr(args, "ending_style", "closure"),
                )
                _diversify_adjacent_titles(cached_chapters, previous_title=previous_title_hint)
                cached_report = _validate_blueprint_continuity(
                    cached_chapters,
                    start_chapter=batch_start,
                    ending_style=getattr(args, "ending_style", "closure"),
                )
                if cached_report.get("ok"):
                    existing_batch["chapters"] = cached_chapters
                    existing_batch["_blueprint_continuity_report"] = cached_report
                    existing_batch["_blueprint_auto_link_fix_count"] = 0
                    existing_batch["_blueprint_opening_llm_repair_attempts"] = int(
                        existing_batch.get("_blueprint_opening_llm_repair_attempts", 0) or 0
                    )
                    batch_blueprint = existing_batch
                    batch_plan_source = "reuse_batch_cache"
                else:
                    print(
                        f"[warn] cached batch {batch_start:04d}-{batch_end:04d} failed continuity gate; regenerate",
                        flush=True,
                    )
            else:
                print(
                    f"[warn] cached batch {batch_start:04d}-{batch_end:04d} has invalid chapters payload; regenerate",
                    flush=True,
                )
    if (
        batch_blueprint is None
        and freeze_published_blueprint
        and batch_start <= locked_until
        and not batch_path.exists()
    ):
        raise ValueError(
            f"batch {batch_start:04d}-{batch_end:04d} overlaps locked chapters (<= {locked_until:04d}) "
            "but no cached batch blueprint exists"
        )

    if batch_blueprint is None:
        world_spec_for_blueprint = copy.deepcopy(world_spec)
        world_spec_for_blueprint["hierarchical_blueprint_context"] = {
            "mode": "hierarchical",
            "batch_range": [batch_start, batch_end],
            "locked_until": locked_until,
            "cross_batch_anchor": cross_batch_anchor,
            "cross_batch_anchor_raw": cross_batch_anchor_meta.get("raw", []),
            "cross_batch_anchor_normalized": cross_batch_anchor_meta.get("normalized", cross_batch_anchor),
            "cross_batch_anchor_truncated": bool(cross_batch_anchor_meta.get("truncated", False)),
            "cross_batch_anchor_source": cross_batch_anchor_source,
            "master_arc": master_arc,
            "stage_arcs": stage_arcs,
            "freeze_published_blueprint": bool(freeze_published_blueprint),
        }
        template_profile_for_blueprint: dict[str, Any] = (
            copy.deepcopy(template_profile) if isinstance(template_profile, dict) else {}
        )
        if template_profile_for_blueprint:
            template_profile_for_blueprint["hierarchical_hint"] = {
                "batch_range": [batch_start, batch_end],
                "cross_batch_anchor": cross_batch_anchor,
            }
        batch_blueprint = generator.build_blueprint(
            target_book_id=target_book_id,
            world_spec=world_spec_for_blueprint,
            template_profile=template_profile_for_blueprint,
            chapter_count=args.chapter_count,
            start_chapter=args.start_chapter,
            previous_title_hint=previous_title_hint,
            ending_style=getattr(args, "ending_style", "closure"),
            max_tokens=args.plan_max_tokens,
            llm_max_retries=args.llm_max_retries,
            llm_retry_backoff=args.llm_retry_backoff,
            llm_backoff_factor=args.llm_backoff_factor,
            llm_backoff_max=args.llm_backoff_max,
            llm_retry_jitter=args.llm_retry_jitter,
        )
        store.save_batch(batch_start, batch_end, batch_blueprint)

    _upsert_index_records(
        index.setdefault("batches", []),
        match_keys=("batch_start", "batch_end"),
        payload={
            "batch_start": batch_start,
            "batch_end": batch_end,
            "path": str(batch_path),
            "source": batch_plan_source,
            "stage_ids": [_stage_id(start, end) for start, end in touched_stage_bounds],
            "updated_at": datetime.now().isoformat(),
        },
    )
    store.save_index(index)
    meta = {
        "blueprint_mode": "hierarchical",
        "blueprint_root": str(blueprint_root),
        "stage_size": int(stage_size),
        "batch_size": int(getattr(args, "batch_size", args.chapter_count) or args.chapter_count),
        "batch_plan_source": batch_plan_source,
        "locked_until": int(index.get("locked_until") or 0),
        "cross_batch_anchor_source": cross_batch_anchor_source,
        "cross_batch_anchor": cross_batch_anchor,
        "cross_batch_anchor_raw": cross_batch_anchor_meta.get("raw", []),
        "cross_batch_anchor_normalized": cross_batch_anchor_meta.get("normalized", cross_batch_anchor),
        "cross_batch_anchor_truncated": bool(cross_batch_anchor_meta.get("truncated", False)),
        "stage_ids": [_stage_id(start, end) for start, end in touched_stage_bounds],
        "freeze_published_blueprint": bool(freeze_published_blueprint),
        "blueprint_index_path": str(store.index_path),
    }
    return batch_blueprint, meta


def _make_store(args: argparse.Namespace, prefix: str, fallback: str = "") -> MemoryStore:
    def _read(name: str, default: str) -> str:
        value = getattr(args, f"{prefix}_{name}", "")
        if value:
            return value
        if fallback:
            fallback_value = getattr(args, f"{fallback}_{name}", "")
            if fallback_value:
                return fallback_value
        return default

    return MemoryStore(
        canon_db_path=_read("canon_db_path", str(Path("~/.nanobot/workspace/canon_v2_reprocessed.db"))),
        neo4j_uri=_read("neo4j_uri", "bolt://localhost:7687"),
        neo4j_user=_read("neo4j_user", "neo4j"),
        neo4j_pass=_read("neo4j_pass", "novel123"),
        neo4j_database=_read("neo4j_database", "neo4j"),
        qdrant_url=_read("qdrant_url", ""),
        qdrant_collection=_read("qdrant_collection", "novel_assets_v2"),
        qdrant_api_key=_read("qdrant_api_key", ""),
    )


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    target_book_id = (args.target_book_id or args.book_id).strip()
    if not target_book_id:
        raise ValueError("book_id or target_book_id is required")

    llm_config_obj = getattr(args, "llm_config_obj", None)
    if llm_config_obj is not None:
        if not isinstance(llm_config_obj, dict):
            raise ValueError("llm_config_obj must be a JSON object")
        llm_config = llm_config_obj
    else:
        llm_config_path = args.llm_config_path or args.llm_config
        if not llm_config_path:
            raise ValueError("--llm-config or --llm-config-path is required")
        llm_config = json.loads(Path(llm_config_path).expanduser().read_text(encoding="utf-8"))
    world_spec = _load_world_spec(args.world, args.world_config)
    continuity_mode = str(getattr(args, "continuity_mode", "strict_gate") or "strict_gate")
    if continuity_mode not in {"strict_gate", "warn_only", "off"}:
        raise ValueError("--continuity-mode must be one of: strict_gate, warn_only, off")
    continuity_window = max(int(getattr(args, "continuity_window", 12) or 12), 1)
    continuity_retry = max(int(getattr(args, "continuity_retry", 3) or 3), 1)
    raw_min_entities = getattr(args, "continuity_min_entities", None)
    raw_min_open_threads = getattr(args, "continuity_min_open_threads", None)
    raw_min_chapter_chars = getattr(args, "continuity_min_chars", None)
    continuity_min_entities = max(3 if raw_min_entities is None else int(raw_min_entities), 0)
    continuity_min_open_threads = max(1 if raw_min_open_threads is None else int(raw_min_open_threads), 0)
    continuity_min_chars = max(2600 if raw_min_chapter_chars is None else int(raw_min_chapter_chars), 0)
    raw_max_opening_overlap = getattr(args, "continuity_max_opening_overlap", DEFAULT_CONTINUITY_MAX_OPENING_OVERLAP)
    continuity_max_opening_overlap = min(max(float(raw_max_opening_overlap or 0.0), 0.0), 0.99)
    opening_rewrite_by_llm = bool(getattr(args, "opening_rewrite_by_llm", True))
    opening_rewrite_max_attempts = max(int(getattr(args, "opening_rewrite_max_attempts", 2) or 0), 0)
    opening_rewrite_max_chars = max(int(getattr(args, "opening_rewrite_max_chars", 820) or 820), 260)
    batch_boundary_gate = bool(getattr(args, "batch_boundary_gate", True))
    raw_resume_overwrite_range = str(getattr(args, "resume_overwrite_range", "") or "").strip()
    try:
        resume_overwrite_chapters = _parse_resume_overwrite_range(raw_resume_overwrite_range)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    chapter_summary_style = str(getattr(args, "chapter_summary_style", "structured") or "structured")
    if chapter_summary_style not in {"structured", "short"}:
        raise ValueError("--chapter-summary-style must be one of: structured, short")
    ending_style = str(getattr(args, "ending_style", "closure") or "closure").strip().lower()
    if ending_style not in ENDING_STYLES:
        allowed = ", ".join(sorted(ENDING_STYLES))
        raise ValueError(f"--ending-style must be one of: {allowed}")
    blueprint_template_source = str(getattr(args, "blueprint_template_source", "none") or "none").strip().lower()
    if blueprint_template_source not in BLUEPRINT_TEMPLATE_SOURCES:
        allowed = ", ".join(sorted(BLUEPRINT_TEMPLATE_SOURCES))
        raise ValueError(f"--blueprint-template-source must be one of: {allowed}")
    blueprint_mode = str(getattr(args, "blueprint_mode", "hierarchical") or "hierarchical").strip().lower()
    if blueprint_mode not in BLUEPRINT_MODES:
        allowed = ", ".join(sorted(BLUEPRINT_MODES))
        raise ValueError(f"--blueprint-mode must be one of: {allowed}")
    stage_size = max(int(getattr(args, "stage_size", 100) or 100), 1)
    batch_size = max(int(getattr(args, "batch_size", args.chapter_count) or args.chapter_count), 1)
    freeze_published_blueprint = bool(getattr(args, "freeze_published_blueprint", True))
    use_template_profile_for_blueprint = blueprint_template_source == "booka"

    target_store = _make_store(args, "target", fallback="legacy")
    template_store = _make_store(args, "template", fallback="legacy") if args.template_book_id else None
    if args.enforce_isolation and template_store is not None:
        _assert_isolation(template_store, target_store)
    if args.commit_memory:
        if not target_store.neo4j_database.strip():
            raise ValueError("target_neo4j_database is required when --commit-memory is enabled")
        if template_store is not None:
            isolation_violations = []
            if template_store.canon_norm() == target_store.canon_norm():
                isolation_violations.append("Canon DB path is identical")
            if template_store.qdrant_norm() == target_store.qdrant_norm():
                isolation_violations.append("Qdrant target (url+collection) is identical")
            if template_store.neo4j_norm() == target_store.neo4j_norm():
                isolation_violations.append("Neo4j target (uri+database) is identical")
            if isolation_violations:
                raise ValueError(
                    "Commit-memory isolation requires physically separate A/B stores: "
                    + "; ".join(isolation_violations)
                )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(getattr(args, "log_dir", str(_default_log_dir()))).expanduser()
    injection_logger = (
        InjectionLogger(log_root, target_book_id) if getattr(args, "log_injections", True) else None
    )

    generator = OneClickBookGenerator(llm_config)
    enforce_chinese_fields = _parse_enforce_chinese_fields(getattr(args, "enforce_chinese_fields", None))
    injection_chinese_normalizer = ChineseContextNormalizer(
        llm=generator.llm,
        fields=enforce_chinese_fields,
        enabled=getattr(args, "enforce_chinese_on_injection", True),
    )
    template_reader = (
        TemplateMemoryReader(
            template_store,
            semantic_search_enabled=getattr(args, "template_semantic_search", True),
            semantic_model_name=getattr(args, "template_semantic_model", ""),
        )
        if template_store
        else None
    )
    target_reader = TargetMemoryReader(target_store)
    profile_query_text = json.dumps(world_spec, ensure_ascii=False)[:4000]
    raw_template_profile = (
        template_reader.build_book_template_profile(
            args.template_book_id,
            args.reference_top_k,
            query_text=profile_query_text,
        )
        if template_reader and use_template_profile_for_blueprint
        else {}
    )
    template_profile = (
        template_reader.sanitize_template_payload_for_prompt(raw_template_profile)
        if template_reader and use_template_profile_for_blueprint
        else raw_template_profile
    )
    if injection_logger is not None:
        clean_payload = {
            "target_book_id": target_book_id,
            "template_book_id": args.template_book_id,
            "world_spec": world_spec,
            "blueprint_mode": blueprint_mode,
            "blueprint_template_source": blueprint_template_source,
            "stage_size": stage_size,
            "batch_size": batch_size,
            "freeze_published_blueprint": freeze_published_blueprint,
            "book_total_chapters": int(getattr(args, "book_total_chapters", 0) or 0),
            "blueprint_template_profile_included": bool(template_profile),
            "template_profile_from_bookA": template_profile,
            "template_store": _redact_store_for_log(template_store) if template_store else None,
            "target_store": _redact_store_for_log(target_store),
            "reference_top_k": args.reference_top_k,
            "template_semantic_search": getattr(args, "template_semantic_search", True),
            "template_semantic_model": getattr(args, "template_semantic_model", ""),
            "enforce_chinese_on_injection": getattr(args, "enforce_chinese_on_injection", True),
            "enforce_chinese_on_commit": getattr(args, "enforce_chinese_on_commit", True),
            "enforce_chinese_fields": sorted(enforce_chinese_fields),
            "continuity_mode": continuity_mode,
            "continuity_window": continuity_window,
            "continuity_retry": continuity_retry,
            "continuity_min_entities": continuity_min_entities,
            "continuity_min_open_threads": continuity_min_open_threads,
            "continuity_min_chars": continuity_min_chars,
            "continuity_max_opening_overlap": continuity_max_opening_overlap,
            "opening_rewrite_by_llm": opening_rewrite_by_llm,
            "opening_rewrite_max_attempts": opening_rewrite_max_attempts,
            "opening_rewrite_max_chars": opening_rewrite_max_chars,
            "batch_boundary_gate": batch_boundary_gate,
            "resume_overwrite_range": sorted(resume_overwrite_chapters),
            "chapter_summary_style": chapter_summary_style,
            "ending_style": ending_style,
        }
        raw_payload = {
            **clean_payload,
            "template_profile_from_bookA": raw_template_profile,
        }
        path, raw_path = injection_logger.write_with_raw(
            "injection_book_level_template_profile.json",
            clean_payload,
            raw_payload=raw_payload,
        )
        print(f"[log] book-level injections -> {path}", flush=True)
        if raw_path is not None:
            print(f"[log] book-level injections raw -> {raw_path}", flush=True)
    blueprint_runtime_meta: dict[str, Any] = {
        "blueprint_mode": blueprint_mode,
        "blueprint_root": "",
        "stage_size": stage_size,
        "batch_size": batch_size,
        "batch_plan_source": "generated",
        "locked_until": 0,
        "cross_batch_anchor_source": "none",
        "cross_batch_anchor": [],
        "cross_batch_anchor_raw": [],
        "cross_batch_anchor_normalized": [],
        "cross_batch_anchor_truncated": False,
        "stage_ids": [],
        "freeze_published_blueprint": freeze_published_blueprint,
        "blueprint_index_path": "",
    }
    previous_title_hint = _load_previous_chapter_title(
        output_dir=output_dir,
        target_book_id=target_book_id,
        chapter_no=int(args.start_chapter or 1),
    )
    if blueprint_mode == "hierarchical":
        blueprint, hierarchy_meta = _build_hierarchical_blueprint(
            args=args,
            generator=generator,
            target_book_id=target_book_id,
            world_spec=world_spec,
            template_profile=template_profile,
            output_dir=output_dir,
            freeze_published_blueprint=freeze_published_blueprint,
            stage_size=stage_size,
        )
        blueprint_runtime_meta.update(hierarchy_meta)
    else:
        blueprint = generator.build_blueprint(
            target_book_id=target_book_id,
            world_spec=world_spec,
            template_profile=template_profile,
            chapter_count=args.chapter_count,
            start_chapter=args.start_chapter,
            previous_title_hint=previous_title_hint,
            ending_style=ending_style,
            max_tokens=args.plan_max_tokens,
            llm_max_retries=args.llm_max_retries,
            llm_retry_backoff=args.llm_retry_backoff,
            llm_backoff_factor=args.llm_backoff_factor,
            llm_backoff_max=args.llm_backoff_max,
            llm_retry_jitter=args.llm_retry_jitter,
        )
    blueprint_continuity_report = blueprint.pop("_blueprint_continuity_report", {"ok": True, "issues": [], "pairs": []})
    blueprint_continuity_repair_attempts = int(blueprint.pop("_blueprint_continuity_repair_attempts", 0) or 0)
    blueprint_auto_link_fix_count = int(blueprint.pop("_blueprint_auto_link_fix_count", 0) or 0)
    blueprint_auto_opening_anchor_fix_count = int(blueprint.pop("_blueprint_auto_opening_anchor_fix_count", 0) or 0)
    blueprint_opening_llm_repair_attempts = int(blueprint.pop("_blueprint_opening_llm_repair_attempts", 0) or 0)
    blueprint_pair_llm_repair_rounds = int(blueprint.pop("_blueprint_pair_llm_repair_rounds", 0) or 0)
    blueprint_pair_llm_repair_trace = blueprint.pop("_blueprint_pair_llm_repair_trace", [])
    blueprint_failed_pairs_final = blueprint.pop("_blueprint_failed_pairs_final", [])
    if not isinstance(blueprint_pair_llm_repair_trace, list):
        blueprint_pair_llm_repair_trace = []
    if not isinstance(blueprint_failed_pairs_final, list):
        blueprint_failed_pairs_final = []
    title_diversity_alerts = blueprint.pop("_title_diversity_alerts", [])
    blueprint_path = output_dir / f"{target_book_id}_blueprint.json"
    blueprint_path.write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Blueprint saved: {blueprint_path}", flush=True)
    if injection_logger is not None:
        path = injection_logger.write(
            "injection_blueprint_response.json",
            {
                "target_book_id": target_book_id,
                "blueprint_mode": blueprint_runtime_meta.get("blueprint_mode"),
                "blueprint_template_source": blueprint_template_source,
                "template_profile_included": bool(template_profile),
                "hierarchical_runtime_meta": blueprint_runtime_meta,
                "blueprint": blueprint,
            },
        )
        print(f"[log] blueprint response -> {path}", flush=True)
        continuity_path = injection_logger.write(
            "blueprint_continuity_report.json",
            {
                "target_book_id": target_book_id,
                "blueprint_path": str(blueprint_path),
                "repair_attempts": blueprint_continuity_repair_attempts,
                "auto_link_fix_count": blueprint_auto_link_fix_count,
                "auto_opening_anchor_fix_count": blueprint_auto_opening_anchor_fix_count,
                "opening_llm_repair_attempts": blueprint_opening_llm_repair_attempts,
                "pair_llm_repair_rounds": blueprint_pair_llm_repair_rounds,
                "pair_llm_repair_trace": blueprint_pair_llm_repair_trace,
                "failed_pairs_final": blueprint_failed_pairs_final,
                "title_diversity_alerts": title_diversity_alerts,
                "report": blueprint_continuity_report,
            },
        )
        print(f"[log] blueprint continuity report -> {continuity_path}", flush=True)

    processor = None
    if args.commit_memory:
        from chapter_processor import ChapterProcessor

        delta_debug_dir = str(injection_logger.run_dir / "chapters") if injection_logger is not None else ""
        processor = ChapterProcessor(
            neo4j_uri=target_store.neo4j_uri,
            neo4j_user=target_store.neo4j_user,
            neo4j_pass=target_store.neo4j_pass,
            canon_db_path=str(Path(target_store.canon_db_path).expanduser()),
            neo4j_database=target_store.neo4j_database,
            qdrant_url=target_store.qdrant_url or None,
            qdrant_collection=target_store.qdrant_collection,
            qdrant_api_key=target_store.qdrant_api_key,
            llm_config=llm_config,
            enforce_chinese_on_commit=getattr(args, "enforce_chinese_on_commit", True),
            enforce_chinese_fields=tuple(sorted(enforce_chinese_fields)),
            chinese_llm_max_retries=args.llm_max_retries,
            chinese_retry_backoff=args.llm_retry_backoff,
            chinese_backoff_factor=args.llm_backoff_factor,
            chinese_backoff_max=args.llm_backoff_max,
            chinese_retry_jitter=args.llm_retry_jitter,
            delta_json_repair_attempts=args.delta_json_repair_attempts,
            delta_parse_debug_log=args.delta_parse_debug_log,
            delta_parse_debug_dir=delta_debug_dir,
        )

    run_items: list[dict[str, Any]] = []
    rolling_continuity_capsules: list[dict[str, Any]] = []
    existing_summaries: dict[str, str] = {}
    preload_meta: dict[str, Any] = {
        "history_start": None,
        "history_end": None,
        "from_run_report": 0,
        "from_chapter_files": 0,
        "preloaded_count": 0,
    }
    if args.resume:
        existing_summaries, run_report_capsules = _load_run_report_context(output_dir, target_book_id)
        rolling_continuity_capsules, preload_meta = _preload_recent_continuity_capsules(
            output_dir=output_dir,
            target_book_id=target_book_id,
            start_chapter=int(args.start_chapter or 1),
            continuity_window=continuity_window,
            blueprint_chapters=blueprint.get("chapters", []),
            summary_map=existing_summaries,
            run_report_capsules=run_report_capsules,
        )
        if preload_meta.get("preloaded_count", 0):
            history_start = preload_meta.get("history_start")
            history_end = preload_meta.get("history_end")
            range_text = (
                f"{int(history_start):04d}-{int(history_end):04d}"
                if isinstance(history_start, int) and isinstance(history_end, int)
                else "n/a"
            )
            print(
                (
                    f"[resume] preloaded continuity capsules={preload_meta.get('preloaded_count', 0)} "
                    f"(range={range_text}, from_report={preload_meta.get('from_run_report', 0)}, "
                    f"from_files={preload_meta.get('from_chapter_files', 0)})"
                ),
                flush=True,
            )
    all_done_chapters: set[str] = set()
    if args.resume and processor is not None:
        all_done_chapters = _load_all_done_chapters(target_store.canon_db_path, target_book_id)
    if args.resume and resume_overwrite_chapters:
        first = min(resume_overwrite_chapters)
        last = max(resume_overwrite_chapters)
        print(
            (
                f"[resume] force-overwrite chapters={len(resume_overwrite_chapters)} "
                f"(range={first}-{last}) from --resume-overwrite-range"
            ),
            flush=True,
        )
    started = time.monotonic()
    ok_count = 0
    fail_count = 0
    memory_failed_count = 0
    opening_rewrite_attempts_total = 0
    opening_rewrite_success_total = 0
    terminated = False
    blueprint_meta = {k: v for k, v in blueprint.items() if k != "chapters"}
    chapter_opening_excerpt_cache: dict[str, str] = {}
    if int(args.start_chapter or 1) > 1:
        previous_no = int(args.start_chapter) - 1
        previous_key = f"{previous_no:04d}"
        previous_path = output_dir / f"{target_book_id}_chapter_{previous_key}.md"
        previous_opening = _read_chapter_opening_excerpt(previous_path, fallback_title=previous_key)
        if previous_opening:
            chapter_opening_excerpt_cache[previous_key] = previous_opening

    try:
        for idx, chapter in enumerate(blueprint["chapters"], 1):
            chapter_no = chapter["chapter_no"]
            chapter_title = chapter["title"]
            chapter_path = output_dir / f"{target_book_id}_chapter_{chapter_no}.md"
            force_overwrite = bool(args.resume and chapter_no in resume_overwrite_chapters)

            if args.resume and chapter_path.exists() and not force_overwrite:
                if processor is None or chapter_no in all_done_chapters:
                    chapter_markdown = chapter_path.read_text(encoding="utf-8")
                    _, chapter_text = _extract_chapter_title_and_text(chapter_markdown, fallback_title=chapter_title)
                    chapter_opening_excerpt_cache[chapter_no] = _extract_opening_excerpt(chapter_text)
                    summary = existing_summaries.get(chapter_no) or chapter_title
                    rolling_continuity_capsules.append(
                        _build_fallback_continuity_capsule(
                            chapter_no=chapter_no,
                            chapter_title=chapter_title,
                            chapter_plan=chapter,
                            chapter_text=chapter_text,
                            chapter_summary=summary,
                            hard_pack_b={},
                        )
                    )
                    run_items.append({"chapter_no": chapter_no, "status": "skipped", "path": str(chapter_path)})
                    elapsed = time.monotonic() - started
                    eta = (elapsed / idx) * (len(blueprint["chapters"]) - idx) if idx else 0
                    print(
                        f"[{idx}/{len(blueprint['chapters'])}] skip {chapter_no} "
                        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                        flush=True,
                    )
                    continue

                # Resume mode with memory commit enabled: recover missing commit from existing chapter file.
                chapter_markdown = chapter_path.read_text(encoding="utf-8")
                _, chapter_text = _extract_chapter_title_and_text(chapter_markdown, fallback_title=chapter_title)
                chapter_opening_excerpt_cache[chapter_no] = _extract_opening_excerpt(chapter_text)
                fallback_summary = chapter_text.replace("\n", " ").strip()[:220]
                summary = existing_summaries.get(chapter_no) or fallback_summary or chapter_title
                rolling_continuity_capsules.append(
                    _build_fallback_continuity_capsule(
                        chapter_no=chapter_no,
                        chapter_title=chapter_title,
                        chapter_plan=chapter,
                        chapter_text=chapter_text,
                        chapter_summary=summary,
                        hard_pack_b={},
                    )
                )
                item = {
                    "chapter_no": chapter_no,
                    "title": chapter_title,
                    "status": "resumed_commit",
                    "summary": summary,
                    "path": str(chapter_path),
                }
                try:
                    memory_commit = processor.process_chapter(
                        book_id=target_book_id,
                        chapter_no=chapter_no,
                        chapter_title=chapter_title,
                        chapter_summary=summary,
                        chapter_text=chapter_text,
                        assets=None,
                        mode="llm",
                    )
                    item["memory_commit"] = memory_commit
                    if memory_commit.get("status") == "blocked":
                        raise RuntimeError(f"blocking conflicts: {memory_commit.get('conflicts')}")
                    all_done_chapters.add(chapter_no)
                except Exception as exc:
                    if args.delta_json_fail_policy == "mark_warning" and _is_delta_json_parse_error(exc):
                        item["status"] = "resumed_commit_memory_failed"
                        item["memory_commit_status"] = "failed"
                        item["error_code"] = _classify_runtime_error(exc)
                        item["error"] = str(exc)
                        debug_log_path = getattr(exc, "debug_log_path", None)
                        if debug_log_path:
                            item["debug_log_path"] = str(debug_log_path)
                        memory_failed_count += 1
                    else:
                        raise
                run_items.append(item)
                ok_count += 1
                elapsed = time.monotonic() - started
                eta = (elapsed / idx) * (len(blueprint["chapters"]) - idx) if idx else 0
                print(
                    f"[{idx}/{len(blueprint['chapters'])}] resume-commit {chapter_no} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} ok={ok_count} fail={fail_count}",
                    flush=True,
                )
                continue
            elif force_overwrite and chapter_path.exists():
                print(
                    f"[resume] overwrite chapter {chapter_no} due to --resume-overwrite-range",
                    flush=True,
                )

            try:
                prev_chapter_plan = blueprint["chapters"][idx - 2] if idx > 1 else {}
                chapter_beats = _sanitize_text_list(chapter.get("beat_outline"), max_items=8, max_len=120)
                first_beat_expected = chapter_beats[0] if chapter_beats else ""
                raw_template_pack = (
                    template_reader.build_chapter_template_pack(args.template_book_id, chapter, args.reference_top_k)
                    if template_reader
                    else {}
                )
                template_pack = (
                    template_reader.sanitize_template_payload_for_prompt(raw_template_pack)
                    if template_reader
                    else raw_template_pack
                )
                raw_hard_pack_b = target_reader.build_hard_pack(target_book_id, chapter_no, args.reference_top_k)
                hard_pack_b = target_reader.sanitize_hard_pack_for_prompt(raw_hard_pack_b)
                hard_pack_b = _ensure_hard_context_shape(hard_pack_b)
                hard_pack_b = injection_chinese_normalizer.normalize_hard_pack(
                    hard_pack_b,
                    llm_max_retries=args.llm_max_retries,
                    llm_retry_backoff=args.llm_retry_backoff,
                    llm_backoff_factor=args.llm_backoff_factor,
                    llm_backoff_max=args.llm_backoff_max,
                    llm_retry_jitter=args.llm_retry_jitter,
                )
                recent_capsules_for_prompt = _sanitize_continuity_capsules_for_prompt(
                    rolling_continuity_capsules,
                    continuity_window,
                )
                linkage = _resolve_chapter_linkage(
                    chapter_no=chapter_no,
                    idx=idx,
                    start_chapter=int(args.start_chapter or 1),
                    chapter_plan=chapter,
                    prev_chapter_plan=prev_chapter_plan,
                    recent_continuity_capsules=recent_capsules_for_prompt,
                    hard_pack_b=hard_pack_b,
                )
                previous_chapter_carry_over = _normalize_link_items(linkage.get("previous_chapter_carry_over"))
                current_chapter_open_with = _normalize_link_items(linkage.get("current_chapter_open_with"))
                carry_over_required_for_gate = (
                    previous_chapter_carry_over if (batch_boundary_gate and linkage.get("is_batch_first")) else []
                )
                previous_opening_text = ""
                normalized_chapter_no = _normalize_chapter_no(chapter_no)
                if normalized_chapter_no.isdigit():
                    chapter_int = int(normalized_chapter_no)
                    if chapter_int > 1:
                        prev_key = f"{chapter_int - 1:04d}"
                        previous_opening_text = chapter_opening_excerpt_cache.get(prev_key, "")
                        if not previous_opening_text:
                            prev_path = output_dir / f"{target_book_id}_chapter_{prev_key}.md"
                            previous_opening_text = _read_chapter_opening_excerpt(prev_path, fallback_title=prev_key)
                            if previous_opening_text:
                                chapter_opening_excerpt_cache[prev_key] = previous_opening_text
                opening_reference_profile = _build_opening_reference_profile(
                    previous_chapter_carry_over=previous_chapter_carry_over,
                    current_chapter_open_with=current_chapter_open_with,
                    previous_chapter_opening_text=previous_opening_text,
                    chapter_opening_text="",
                    recent_continuity_capsules=recent_capsules_for_prompt,
                )
                if injection_logger is not None:
                    clean_payload = {
                        "target_book_id": target_book_id,
                        "template_book_id": args.template_book_id,
                        "chapter_no": chapter_no,
                        "blueprint_mode": blueprint_runtime_meta.get("blueprint_mode", blueprint_mode),
                        "stage_ids": blueprint_runtime_meta.get("stage_ids", []),
                        "cross_batch_anchor_source": blueprint_runtime_meta.get("cross_batch_anchor_source", "none"),
                        "cross_batch_anchor": blueprint_runtime_meta.get("cross_batch_anchor", []),
                        "cross_batch_anchor_raw": blueprint_runtime_meta.get("cross_batch_anchor_raw", []),
                        "cross_batch_anchor_normalized": blueprint_runtime_meta.get(
                            "cross_batch_anchor_normalized", []
                        ),
                        "cross_batch_anchor_truncated": bool(
                            blueprint_runtime_meta.get("cross_batch_anchor_truncated", False)
                        ),
                        "world_spec": world_spec,
                        "book_blueprint_meta": blueprint_meta,
                        "chapter_plan": chapter,
                        "template_pack_from_bookA": template_pack,
                        "hard_context_from_bookB": hard_pack_b,
                        "previous_chapter_carry_over": previous_chapter_carry_over,
                        "current_chapter_open_with": current_chapter_open_with,
                        "is_batch_first": bool(linkage.get("is_batch_first")),
                        "is_true_book_start": bool(linkage.get("is_true_book_start")),
                        "resolved_carry_source": linkage.get("carry_source"),
                        "resolved_previous_carry_source": linkage.get("carry_source"),
                        "resolved_open_with_source": linkage.get("open_with_source"),
                        "carry_over_required_for_gate": carry_over_required_for_gate,
                        "recent_continuity_capsules_bookB": recent_capsules_for_prompt,
                        "continuity_mode": continuity_mode,
                        "continuity_window": continuity_window,
                        "continuity_retry": continuity_retry,
                        "continuity_min_entities": continuity_min_entities,
                        "continuity_min_open_threads": continuity_min_open_threads,
                        "continuity_min_chars": continuity_min_chars,
                        "continuity_max_opening_overlap": continuity_max_opening_overlap,
                        "opening_rewrite_by_llm": opening_rewrite_by_llm,
                        "opening_rewrite_max_attempts": opening_rewrite_max_attempts,
                        "opening_rewrite_max_chars": opening_rewrite_max_chars,
                        "opening_reference_profile": opening_reference_profile,
                        "batch_boundary_gate": batch_boundary_gate,
                        "chapter_summary_style": chapter_summary_style,
                        "ending_style": ending_style,
                    }
                    raw_payload = {
                        **clean_payload,
                        "template_pack_from_bookA": raw_template_pack,
                        "hard_context_from_bookB": raw_hard_pack_b,
                        "recent_continuity_capsules_bookB": rolling_continuity_capsules[-continuity_window:],
                    }
                    path, raw_path = injection_logger.write_with_raw(
                        f"chapters/{chapter_no}_pre_generation_injection.json",
                        clean_payload,
                        raw_payload=raw_payload,
                    )
                    print(f"[log] chapter {chapter_no} injections -> {path}", flush=True)
                    if raw_path is not None:
                        print(f"[log] chapter {chapter_no} injections raw -> {raw_path}", flush=True)
                generation_attempts = continuity_retry if continuity_mode == "strict_gate" else 1
                if continuity_mode == "off":
                    generation_attempts = 1
                chapter_text = ""
                summary = ""
                capsule: dict[str, Any] | None = None
                validation_result = {"ok": True, "issues": [], "metrics": {}}
                last_validation_error: str | None = None

                for attempt in range(1, generation_attempts + 1):
                    chapter_text_raw = generator.generate_chapter_text(
                        world_spec=world_spec,
                        blueprint_meta=blueprint_meta,
                        chapter_plan=chapter,
                        template_chapter_pack=template_pack,
                        hard_pack_b=hard_pack_b,
                        previous_chapter_carry_over=previous_chapter_carry_over,
                        current_chapter_open_with=current_chapter_open_with,
                        recent_continuity_capsules=recent_capsules_for_prompt,
                        reference_profile=opening_reference_profile,
                        continuity_window=continuity_window,
                        continuity_min_entities=continuity_min_entities,
                        continuity_min_open_threads=continuity_min_open_threads,
                        ending_style=ending_style,
                        chapter_min_chars=args.chapter_min_chars,
                        chapter_max_chars=args.chapter_max_chars,
                        temperature=args.temperature,
                        max_tokens=args.chapter_max_tokens,
                        llm_max_retries=args.llm_max_retries,
                        llm_retry_backoff=args.llm_retry_backoff,
                        llm_backoff_factor=args.llm_backoff_factor,
                        llm_backoff_max=args.llm_backoff_max,
                        llm_retry_jitter=args.llm_retry_jitter,
                    )
                    chapter_text = _sanitize_generated_chapter_text(chapter_text_raw, chapter_title)
                    capsule = generator.build_continuity_capsule(
                        chapter_no=chapter_no,
                        chapter_title=chapter_title,
                        chapter_plan=chapter,
                        chapter_text=chapter_text,
                        hard_pack_b=hard_pack_b,
                        summary_style=chapter_summary_style,
                        llm_max_retries=args.llm_max_retries,
                        llm_retry_backoff=args.llm_retry_backoff,
                        llm_backoff_factor=args.llm_backoff_factor,
                        llm_backoff_max=args.llm_backoff_max,
                        llm_retry_jitter=args.llm_retry_jitter,
                    )
                    summary = str(capsule.get("chapter_summary") or "").strip()[:300]
                    if not summary:
                        summary = chapter_text.replace("\n", " ").strip()[:300]
                        capsule["chapter_summary"] = summary

                    if continuity_mode == "off":
                        validation_result = {"ok": True, "issues": [], "metrics": {}}
                        break

                    validation_result = _validate_chapter_continuity(
                        chapter_text=chapter_text,
                        recent_capsules=rolling_continuity_capsules,
                        hard_pack_b=hard_pack_b,
                        open_with_expected=current_chapter_open_with,
                        required_carry_over=carry_over_required_for_gate,
                        previous_chapter_opening_text=previous_opening_text,
                        max_opening_overlap=continuity_max_opening_overlap,
                        first_beat_expected=first_beat_expected,
                        continuity_window=continuity_window,
                        min_entities=continuity_min_entities,
                        min_open_threads=continuity_min_open_threads,
                        min_chapter_chars=continuity_min_chars,
                    )
                    reference_validation = _validate_opening_reference_consistency(
                        chapter_text=chapter_text,
                        reference_profile=opening_reference_profile,
                        opening_max_chars=opening_rewrite_max_chars,
                        min_anchor_hits=OPENING_REFERENCE_MIN_ANCHOR_HITS,
                    )
                    if not reference_validation.get("ok"):
                        validation_result = {
                            "ok": False,
                            "issues": list(validation_result.get("issues") or [])
                            + [str(item) for item in (reference_validation.get("issues") or [])],
                            "metrics": {
                                **(validation_result.get("metrics") or {}),
                                "opening_reference_validation": reference_validation,
                            },
                        }
                    if validation_result.get("ok"):
                        break

                    issues = validation_result.get("issues") or []
                    last_validation_error = "; ".join(str(issue) for issue in issues) if issues else "continuity check failed"
                    print(
                        f"[warn] chapter {chapter_no} continuity check failed on attempt {attempt}/{generation_attempts}: "
                        f"{last_validation_error}",
                        flush=True,
                    )
                    if continuity_mode != "strict_gate":
                        break
                    if (
                        opening_rewrite_by_llm
                        and opening_rewrite_max_attempts > 0
                        and _issues_need_opening_rewrite(issues)
                    ):
                        for rewrite_round in range(1, opening_rewrite_max_attempts + 1):
                            opening_rewrite_attempts_total += 1
                            rewritten_text_raw = generator.rewrite_chapter_opening_for_continuity(
                                chapter_title=chapter_title,
                                chapter_plan=chapter,
                                chapter_text=chapter_text,
                                previous_chapter_carry_over=previous_chapter_carry_over,
                                current_chapter_open_with=current_chapter_open_with,
                                recent_continuity_capsules=recent_capsules_for_prompt,
                                continuity_issues=[str(issue) for issue in issues],
                                previous_chapter_opening_text=previous_opening_text,
                                first_beat_expected=first_beat_expected,
                                reference_profile=opening_reference_profile,
                                opening_max_chars=opening_rewrite_max_chars,
                                llm_max_retries=args.llm_max_retries,
                                llm_retry_backoff=args.llm_retry_backoff,
                                llm_backoff_factor=args.llm_backoff_factor,
                                llm_backoff_max=args.llm_backoff_max,
                                llm_retry_jitter=args.llm_retry_jitter,
                            )
                            rewritten_text = _sanitize_generated_chapter_text(rewritten_text_raw, chapter_title)
                            if not rewritten_text or rewritten_text == chapter_text:
                                continue
                            rewritten_capsule = generator.build_continuity_capsule(
                                chapter_no=chapter_no,
                                chapter_title=chapter_title,
                                chapter_plan=chapter,
                                chapter_text=rewritten_text,
                                hard_pack_b=hard_pack_b,
                                summary_style=chapter_summary_style,
                                llm_max_retries=args.llm_max_retries,
                                llm_retry_backoff=args.llm_retry_backoff,
                                llm_backoff_factor=args.llm_backoff_factor,
                                llm_backoff_max=args.llm_backoff_max,
                                llm_retry_jitter=args.llm_retry_jitter,
                            )
                            rewritten_summary = str(rewritten_capsule.get("chapter_summary") or "").strip()[:300]
                            if not rewritten_summary:
                                rewritten_summary = rewritten_text.replace("\n", " ").strip()[:300]
                                rewritten_capsule["chapter_summary"] = rewritten_summary
                            rewritten_validation = _validate_chapter_continuity(
                                chapter_text=rewritten_text,
                                recent_capsules=rolling_continuity_capsules,
                                hard_pack_b=hard_pack_b,
                                open_with_expected=current_chapter_open_with,
                                required_carry_over=carry_over_required_for_gate,
                                previous_chapter_opening_text=previous_opening_text,
                                max_opening_overlap=continuity_max_opening_overlap,
                                first_beat_expected=first_beat_expected,
                                continuity_window=continuity_window,
                                min_entities=continuity_min_entities,
                                min_open_threads=continuity_min_open_threads,
                                min_chapter_chars=continuity_min_chars,
                            )
                            rewritten_reference_validation = _validate_opening_reference_consistency(
                                chapter_text=rewritten_text,
                                reference_profile=opening_reference_profile,
                                opening_max_chars=opening_rewrite_max_chars,
                                min_anchor_hits=OPENING_REFERENCE_MIN_ANCHOR_HITS,
                            )
                            if not rewritten_reference_validation.get("ok"):
                                rewritten_validation = {
                                    "ok": False,
                                    "issues": list(rewritten_validation.get("issues") or [])
                                    + [str(item) for item in (rewritten_reference_validation.get("issues") or [])],
                                    "metrics": {
                                        **(rewritten_validation.get("metrics") or {}),
                                        "opening_reference_validation": rewritten_reference_validation,
                                    },
                                }
                            if rewritten_validation.get("ok"):
                                chapter_text = rewritten_text
                                capsule = rewritten_capsule
                                summary = rewritten_summary
                                validation_result = rewritten_validation
                                opening_rewrite_success_total += 1
                                print(
                                    (
                                        f"[info] chapter {chapter_no} opening continuity repaired by LLM "
                                        f"(attempt {attempt}, rewrite {rewrite_round}/{opening_rewrite_max_attempts})"
                                    ),
                                    flush=True,
                                )
                                break
                            rewritten_issues = rewritten_validation.get("issues") or []
                            print(
                                (
                                    f"[warn] chapter {chapter_no} opening LLM rewrite failed "
                                    f"{rewrite_round}/{opening_rewrite_max_attempts}: "
                                    f"{'; '.join(str(item) for item in rewritten_issues[:6])}"
                                ),
                                flush=True,
                            )
                        if validation_result.get("ok"):
                            break
                    if attempt >= generation_attempts:
                        raise RuntimeError(f"chapter {chapter_no} continuity gate failed: {last_validation_error}")
                    time.sleep(min(2.0 * attempt, 8.0))

                if capsule is None:
                    raise RuntimeError(f"chapter {chapter_no} continuity capsule is empty")
                if not summary:
                    raise RuntimeError(f"chapter {chapter_no} summary is empty")
                if continuity_mode == "warn_only" and not validation_result.get("ok"):
                    print(
                        f"[warn] chapter {chapter_no} generated with continuity warnings: "
                        f"{'; '.join(validation_result.get('issues') or [])}",
                        flush=True,
                    )
                opening_reference_validation = _validate_opening_reference_consistency(
                    chapter_text=chapter_text,
                    reference_profile=opening_reference_profile,
                    opening_max_chars=opening_rewrite_max_chars,
                    min_anchor_hits=OPENING_REFERENCE_MIN_ANCHOR_HITS,
                )

                rolling_continuity_capsules.append(capsule)
                chapter_opening_excerpt_cache[chapter_no] = _extract_opening_excerpt(chapter_text)
                chapter_path.write_text(f"# {chapter_title}\n\n{chapter_text.strip()}\n", encoding="utf-8")
                item = {
                    "chapter_no": chapter_no,
                    "title": chapter_title,
                    "status": "generated",
                    "summary": summary,
                    "path": str(chapter_path),
                    "continuity_validation": validation_result,
                    "continuity_capsule": capsule,
                    "opening_reference_profile": opening_reference_profile,
                    "opening_reference_validation": opening_reference_validation,
                }

                if processor is not None:
                    try:
                        memory_commit = processor.process_chapter(
                            book_id=target_book_id,
                            chapter_no=chapter_no,
                            chapter_title=chapter_title,
                            chapter_summary=summary,
                            chapter_text=chapter_text,
                            assets=None,
                            mode="llm",
                        )
                        item["memory_commit"] = memory_commit
                        if memory_commit.get("status") == "blocked":
                            raise RuntimeError(f"blocking conflicts: {memory_commit.get('conflicts')}")
                        all_done_chapters.add(chapter_no)
                    except Exception as exc:
                        if args.delta_json_fail_policy == "mark_warning" and _is_delta_json_parse_error(exc):
                            item["status"] = "generated_memory_failed"
                            item["memory_commit_status"] = "failed"
                            item["error_code"] = _classify_runtime_error(exc)
                            item["error"] = str(exc)
                            debug_log_path = getattr(exc, "debug_log_path", None)
                            if debug_log_path:
                                item["debug_log_path"] = str(debug_log_path)
                            memory_failed_count += 1
                        else:
                            raise

                run_items.append(item)
                ok_count += 1
            except Exception as exc:  # pragma: no cover - operational path
                run_items.append(
                    {
                        "chapter_no": chapter_no,
                        "title": chapter_title,
                        "status": "failed",
                        "error": str(exc),
                        "error_code": _classify_runtime_error(exc),
                        "debug_log_path": getattr(exc, "debug_log_path", None),
                    }
                )
                fail_count += 1
                if args.consistency_policy == "strict_blocking":
                    terminated = True

            elapsed = time.monotonic() - started
            eta = (elapsed / idx) * (len(blueprint["chapters"]) - idx) if idx else 0
            print(
                f"[{idx}/{len(blueprint['chapters'])} {(idx/len(blueprint['chapters']))*100:5.1f}%] "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} ok={ok_count} fail={fail_count} "
                f"last={chapter_no}",
                flush=True,
            )
            if terminated:
                print("Terminated due to strict_blocking policy.", flush=True)
                break
    finally:
        if processor is not None:
            processor.close()

    title_style_metrics = _compute_title_style_metrics(
        blueprint.get("chapters", []),
        previous_title=previous_title_hint,
    )
    carry_open_checks = 0
    carry_open_hits = 0
    for item in run_items:
        if not isinstance(item, dict):
            continue
        validation = item.get("continuity_validation")
        if not isinstance(validation, dict):
            continue
        metrics = validation.get("metrics")
        if not isinstance(metrics, dict):
            continue
        expected = int(metrics.get("open_with_expected") or 0)
        opening_hits = int(metrics.get("open_with_opening_hits") or 0)
        if expected <= 0:
            continue
        carry_open_checks += 1
        if opening_hits > 0:
            carry_open_hits += 1
    carry_open_hit_rate = (carry_open_hits / carry_open_checks) if carry_open_checks else None
    if blueprint_runtime_meta.get("blueprint_mode") == "hierarchical":
        locked_after = _resolve_locked_until_from_files(
            output_dir=output_dir,
            target_book_id=target_book_id,
            initial_locked_until=int(blueprint_runtime_meta.get("locked_until") or 0),
        )
        blueprint_runtime_meta["locked_until"] = locked_after
        blueprint_root_runtime = str(blueprint_runtime_meta.get("blueprint_root") or "").strip()
        if blueprint_root_runtime:
            store = BlueprintStore(
                root_dir=Path(blueprint_root_runtime),
                target_book_id=target_book_id,
                stage_size=int(blueprint_runtime_meta.get("stage_size") or stage_size),
                freeze_published=bool(blueprint_runtime_meta.get("freeze_published_blueprint", True)),
            )
            idx_payload = store.load_index()
            idx_payload["locked_until"] = max(int(idx_payload.get("locked_until") or 0), int(locked_after))
            idx_payload["book_total_chapters"] = max(
                int(idx_payload.get("book_total_chapters") or 0),
                int(getattr(args, "book_total_chapters", 0) or 0),
                int(args.start_chapter) + int(args.chapter_count) - 1,
                int(locked_after),
            )
            store.save_index(idx_payload)

    report = {
        "target_book_id": target_book_id,
        "template_book_id": args.template_book_id,
        "chapter_count": args.chapter_count,
        "generated_at": datetime.now().isoformat(),
        "injection_log_dir": str(injection_logger.run_dir) if injection_logger is not None else None,
        "target_store": target_store.__dict__,
        "template_store": template_store.__dict__ if template_store else None,
        "continuity_mode": continuity_mode,
        "continuity_window": continuity_window,
        "continuity_retry": continuity_retry,
        "continuity_min_entities": continuity_min_entities,
        "continuity_min_open_threads": continuity_min_open_threads,
        "continuity_min_chars": continuity_min_chars,
        "continuity_max_opening_overlap": continuity_max_opening_overlap,
        "delta_json_repair_attempts": args.delta_json_repair_attempts,
        "delta_json_fail_policy": args.delta_json_fail_policy,
        "delta_parse_debug_log": bool(args.delta_parse_debug_log),
        "opening_rewrite_by_llm": opening_rewrite_by_llm,
        "opening_rewrite_max_attempts": opening_rewrite_max_attempts,
        "opening_rewrite_max_chars": opening_rewrite_max_chars,
        "opening_rewrite_attempts_total": opening_rewrite_attempts_total,
        "opening_rewrite_success_total": opening_rewrite_success_total,
        "batch_boundary_gate": batch_boundary_gate,
        "resume_overwrite_range": sorted(resume_overwrite_chapters),
        "preloaded_continuity_capsules": int(preload_meta.get("preloaded_count") or 0),
        "preloaded_history_range": [
            preload_meta.get("history_start"),
            preload_meta.get("history_end"),
        ],
        "preloaded_from_run_report": int(preload_meta.get("from_run_report") or 0),
        "preloaded_from_chapter_files": int(preload_meta.get("from_chapter_files") or 0),
        "chapter_summary_style": chapter_summary_style,
        "ending_style": ending_style,
        "blueprint_mode": blueprint_runtime_meta.get("blueprint_mode", blueprint_mode),
        "stage_size": int(blueprint_runtime_meta.get("stage_size") or stage_size),
        "batch_size": int(blueprint_runtime_meta.get("batch_size") or batch_size),
        "freeze_published_blueprint": bool(
            blueprint_runtime_meta.get("freeze_published_blueprint", freeze_published_blueprint)
        ),
        "blueprint_root": blueprint_runtime_meta.get("blueprint_root"),
        "blueprint_index_path": blueprint_runtime_meta.get("blueprint_index_path"),
        "published_blueprint_locked_until": int(blueprint_runtime_meta.get("locked_until") or 0),
        "batch_plan_source": blueprint_runtime_meta.get("batch_plan_source", "generated"),
        "stage_ids": blueprint_runtime_meta.get("stage_ids", []),
        "cross_batch_anchor_source": blueprint_runtime_meta.get("cross_batch_anchor_source", "none"),
        "cross_batch_anchor": blueprint_runtime_meta.get("cross_batch_anchor", []),
        "cross_batch_anchor_raw": blueprint_runtime_meta.get("cross_batch_anchor_raw", []),
        "cross_batch_anchor_normalized": blueprint_runtime_meta.get("cross_batch_anchor_normalized", []),
        "cross_batch_anchor_truncated": bool(blueprint_runtime_meta.get("cross_batch_anchor_truncated", False)),
        "blueprint_template_source": blueprint_template_source,
        "blueprint_gate_passed": bool(blueprint_continuity_report.get("ok", False)),
        "blueprint_repair_attempts": blueprint_continuity_repair_attempts,
        "blueprint_auto_link_fix_count": blueprint_auto_link_fix_count,
        "blueprint_auto_opening_anchor_fix_count": blueprint_auto_opening_anchor_fix_count,
        "blueprint_opening_llm_repair_attempts": blueprint_opening_llm_repair_attempts,
        "blueprint_pair_llm_repair_rounds": blueprint_pair_llm_repair_rounds,
        "blueprint_pair_llm_repair_trace": blueprint_pair_llm_repair_trace,
        "blueprint_failed_pairs_final": blueprint_failed_pairs_final,
        "title_diversity_alerts": title_diversity_alerts,
        "title_pattern_distribution": title_style_metrics.get("title_pattern_distribution", {}),
        "adjacent_title_dup_count": int(title_style_metrics.get("adjacent_title_dup_count") or 0),
        "carry_open_hit_rate": carry_open_hit_rate,
        "carry_open_checks": carry_open_checks,
        "blueprint_continuity_summary": blueprint_continuity_report.get("summary", {}),
        "ok": ok_count,
        "memory_failed": memory_failed_count,
        "failed": fail_count,
        "terminated": terminated,
        "items": run_items,
    }
    report_path = output_dir / f"{target_book_id}_run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok" if fail_count == 0 else "partial",
        "target_book_id": target_book_id,
        "template_book_id": args.template_book_id,
        "ok": ok_count,
        "failed": fail_count,
        "terminated": terminated,
        "continuity_mode": continuity_mode,
        "continuity_min_chars": continuity_min_chars,
        "ending_style": ending_style,
        "continuity_max_opening_overlap": continuity_max_opening_overlap,
        "delta_json_repair_attempts": args.delta_json_repair_attempts,
        "delta_json_fail_policy": args.delta_json_fail_policy,
        "delta_parse_debug_log": bool(args.delta_parse_debug_log),
        "opening_rewrite_by_llm": opening_rewrite_by_llm,
        "opening_rewrite_max_attempts": opening_rewrite_max_attempts,
        "opening_rewrite_max_chars": opening_rewrite_max_chars,
        "opening_rewrite_attempts_total": opening_rewrite_attempts_total,
        "opening_rewrite_success_total": opening_rewrite_success_total,
        "batch_boundary_gate": batch_boundary_gate,
        "resume_overwrite_range": sorted(resume_overwrite_chapters),
        "preloaded_continuity_capsules": int(preload_meta.get("preloaded_count") or 0),
        "blueprint_mode": blueprint_runtime_meta.get("blueprint_mode", blueprint_mode),
        "stage_size": int(blueprint_runtime_meta.get("stage_size") or stage_size),
        "batch_size": int(blueprint_runtime_meta.get("batch_size") or batch_size),
        "batch_plan_source": blueprint_runtime_meta.get("batch_plan_source", "generated"),
        "published_blueprint_locked_until": int(blueprint_runtime_meta.get("locked_until") or 0),
        "blueprint_root": blueprint_runtime_meta.get("blueprint_root"),
        "blueprint_index_path": blueprint_runtime_meta.get("blueprint_index_path"),
        "blueprint_template_source": blueprint_template_source,
        "blueprint_gate_passed": bool(blueprint_continuity_report.get("ok", False)),
        "blueprint_repair_attempts": blueprint_continuity_repair_attempts,
        "blueprint_auto_link_fix_count": blueprint_auto_link_fix_count,
        "blueprint_auto_opening_anchor_fix_count": blueprint_auto_opening_anchor_fix_count,
        "blueprint_opening_llm_repair_attempts": blueprint_opening_llm_repair_attempts,
        "blueprint_pair_llm_repair_rounds": blueprint_pair_llm_repair_rounds,
        "blueprint_failed_pairs_final": blueprint_failed_pairs_final,
        "adjacent_title_dup_count": int(title_style_metrics.get("adjacent_title_dup_count") or 0),
        "carry_open_hit_rate": carry_open_hit_rate,
        "memory_failed": memory_failed_count,
        "blueprint_path": str(blueprint_path),
        "report_path": str(report_path),
        "output_dir": str(output_dir),
        "injection_log_dir": str(injection_logger.run_dir) if injection_logger is not None else None,
    }


def default_run_options() -> dict[str, Any]:
    """Programmatic defaults mirroring CLI parse_args."""
    return {
        "book_id": "",
        "target_book_id": "",
        "template_book_id": "",
        "world": "",
        "world_config": "",
        "chapter_count": 0,
        "start_chapter": 1,
        "output_dir": "",
        "llm_config": "",
        "llm_config_path": "",
        "llm_config_obj": None,
        "temperature": 0.8,
        "plan_max_tokens": 4096,
        "chapter_max_tokens": 4096,
        "chapter_min_chars": 2800,
        "chapter_max_chars": 4200,
        "llm_max_retries": 3,
        "llm_retry_backoff": 3.0,
        "llm_backoff_factor": 2.0,
        "llm_backoff_max": 60.0,
        "llm_retry_jitter": 0.5,
        "reference_top_k": 12,
        "blueprint_mode": "hierarchical",
        "stage_size": 100,
        "batch_size": 20,
        "freeze_published_blueprint": True,
        "blueprint_root": "",
        "book_total_chapters": 0,
        "blueprint_template_source": "none",
        "consistency_policy": "strict_blocking",
        "continuity_mode": "strict_gate",
        "continuity_retry": 3,
        "continuity_window": 12,
        "continuity_min_entities": 3,
        "continuity_min_open_threads": 1,
        "continuity_min_chars": 2600,
        "continuity_max_opening_overlap": DEFAULT_CONTINUITY_MAX_OPENING_OVERLAP,
        "delta_json_repair_attempts": 2,
        "delta_json_fail_policy": "mark_warning",
        "delta_parse_debug_log": True,
        "opening_rewrite_by_llm": True,
        "opening_rewrite_max_attempts": 2,
        "opening_rewrite_max_chars": 820,
        "batch_boundary_gate": True,
        "resume_overwrite_range": "",
        "chapter_summary_style": "structured",
        "ending_style": "closure",
        "enforce_isolation": True,
        "resume": False,
        "commit_memory": False,
        "legacy_canon_db_path": str(Path("~/.nanobot/workspace/canon_v2_reprocessed.db")),
        "legacy_neo4j_uri": "bolt://localhost:7687",
        "legacy_neo4j_user": "neo4j",
        "legacy_neo4j_pass": "novel123",
        "legacy_neo4j_database": "neo4j",
        "legacy_qdrant_url": "",
        "legacy_qdrant_collection": "novel_assets_v2",
        "legacy_qdrant_api_key": "",
        "target_canon_db_path": "",
        "target_neo4j_uri": "",
        "target_neo4j_user": "",
        "target_neo4j_pass": "",
        "target_neo4j_database": "",
        "target_qdrant_url": "",
        "target_qdrant_collection": "",
        "target_qdrant_api_key": "",
        "template_canon_db_path": "",
        "template_neo4j_uri": "",
        "template_neo4j_user": "",
        "template_neo4j_pass": "",
        "template_neo4j_database": "",
        "template_qdrant_url": "",
        "template_qdrant_collection": "",
        "template_qdrant_api_key": "",
        "template_semantic_search": True,
        "template_semantic_model": "",
        "enforce_chinese_on_injection": True,
        "enforce_chinese_on_commit": True,
        "enforce_chinese_fields": ",".join(DEFAULT_ENFORCE_CHINESE_FIELDS),
        "log_dir": str(_default_log_dir()),
        "log_injections": True,
    }


def run_generation_with_options(**overrides: Any) -> dict[str, Any]:
    """Programmatic entrypoint for tools/tests (without CLI argv parsing)."""
    options = default_run_options()
    options.update(overrides)
    return run_generation(argparse.Namespace(**options))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A(read)-B(write) one-click novel generation")
    parser.add_argument("--book-id", default="", help="Target Book B id (legacy alias)")
    parser.add_argument("--target-book-id", default="", help="Target Book B id")
    parser.add_argument("--template-book-id", default="", help="Book A template id")
    parser.add_argument("--world", default="")
    parser.add_argument("--world-config", default="")
    parser.add_argument("--chapter-count", type=int, required=True)
    parser.add_argument("--start-chapter", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-config", default="")
    parser.add_argument("--llm-config-path", default="")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--plan-max-tokens", type=int, default=4096)
    parser.add_argument("--chapter-max-tokens", type=int, default=4096)
    parser.add_argument("--chapter-min-chars", type=int, default=2800)
    parser.add_argument("--chapter-max-chars", type=int, default=4200)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=3.0)
    parser.add_argument("--llm-backoff-factor", type=float, default=2.0)
    parser.add_argument("--llm-backoff-max", type=float, default=60.0)
    parser.add_argument("--llm-retry-jitter", type=float, default=0.5)
    parser.add_argument("--reference-top-k", type=int, default=12)
    parser.add_argument(
        "--blueprint-mode",
        choices=sorted(BLUEPRINT_MODES),
        default="hierarchical",
        help="Blueprint planner mode: hierarchical (master/stage/batch) or flat (single-run batch blueprint).",
    )
    parser.add_argument("--stage-size", type=int, default=100, help="Stage span in chapters for hierarchical mode.")
    parser.add_argument("--batch-size", type=int, default=20, help="Expected batch size for hierarchical metadata.")
    parser.add_argument("--blueprint-root", default="", help="Optional directory for persisted hierarchical blueprints.")
    parser.add_argument("--book-total-chapters", type=int, default=0, help="Optional full-book chapter target.")
    parser.add_argument("--freeze-published-blueprint", action="store_true", default=True)
    parser.add_argument("--no-freeze-published-blueprint", action="store_false", dest="freeze_published_blueprint")
    parser.add_argument(
        "--blueprint-template-source",
        choices=sorted(BLUEPRINT_TEMPLATE_SOURCES),
        default="none",
        help="Blueprint stage template source: none (world-spec only) or booka (use Book A profile).",
    )
    parser.add_argument("--consistency-policy", choices=["strict_blocking", "warn_only"], default="strict_blocking")
    parser.add_argument("--continuity-mode", choices=["strict_gate", "warn_only", "off"], default="strict_gate")
    parser.add_argument("--continuity-retry", type=int, default=3)
    parser.add_argument("--continuity-window", type=int, default=12)
    parser.add_argument("--continuity-min-entities", type=int, default=3)
    parser.add_argument("--continuity-min-open-threads", type=int, default=1)
    parser.add_argument("--continuity-min-chars", type=int, default=2600)
    parser.add_argument(
        "--continuity-max-opening-overlap",
        type=float,
        default=DEFAULT_CONTINUITY_MAX_OPENING_OVERLAP,
        help="Max allowed opening overlap ratio vs previous chapter opening (Jaccard on Chinese bigrams).",
    )
    parser.add_argument(
        "--delta-json-repair-attempts",
        type=int,
        default=2,
        help="Max LLM repair rounds for malformed delta JSON before raising parse error.",
    )
    parser.add_argument(
        "--delta-json-fail-policy",
        choices=["mark_warning", "fail"],
        default="mark_warning",
        help="When delta JSON parse fails during memory commit: mark warning and continue, or fail the chapter.",
    )
    parser.add_argument("--delta-parse-debug-log", action="store_true", default=True)
    parser.add_argument("--no-delta-parse-debug-log", action="store_false", dest="delta_parse_debug_log")
    parser.add_argument("--opening-rewrite-by-llm", action="store_true", default=True)
    parser.add_argument("--no-opening-rewrite-by-llm", action="store_false", dest="opening_rewrite_by_llm")
    parser.add_argument("--opening-rewrite-max-attempts", type=int, default=2)
    parser.add_argument("--opening-rewrite-max-chars", type=int, default=820)
    parser.add_argument("--batch-boundary-gate", action="store_true", default=True)
    parser.add_argument("--no-batch-boundary-gate", action="store_false", dest="batch_boundary_gate")
    parser.add_argument("--chapter-summary-style", choices=["structured", "short"], default="structured")
    parser.add_argument(
        "--ending-style",
        choices=sorted(ENDING_STYLES),
        default="closure",
        help="Chapter ending style: closure(收束优先), soft(轻留白), hook(强钩子).",
    )
    parser.add_argument("--enforce-isolation", action="store_true", default=True)
    parser.add_argument("--no-enforce-isolation", action="store_false", dest="enforce_isolation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-overwrite-range",
        default="",
        help="In --resume mode, force regenerate listed chapter numbers/ranges (e.g. 0081-0100,0123).",
    )
    parser.add_argument("--commit-memory", action="store_true")

    # legacy shared store args
    parser.add_argument("--legacy-canon-db-path", default=str(Path("~/.nanobot/workspace/canon_v2_reprocessed.db")))
    parser.add_argument("--legacy-neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--legacy-neo4j-user", default="neo4j")
    parser.add_argument("--legacy-neo4j-pass", default="novel123")
    parser.add_argument("--legacy-neo4j-database", default="neo4j")
    parser.add_argument("--legacy-qdrant-url", default="")
    parser.add_argument("--legacy-qdrant-collection", default="novel_assets_v2")
    parser.add_argument("--legacy-qdrant-api-key", default="")

    # target Book B store
    parser.add_argument("--target-canon-db-path", default="")
    parser.add_argument("--target-neo4j-uri", default="")
    parser.add_argument("--target-neo4j-user", default="")
    parser.add_argument("--target-neo4j-pass", default="")
    parser.add_argument("--target-neo4j-database", default="")
    parser.add_argument("--target-qdrant-url", default="")
    parser.add_argument("--target-qdrant-collection", default="")
    parser.add_argument("--target-qdrant-api-key", default="")

    # template Book A store
    parser.add_argument("--template-canon-db-path", default="")
    parser.add_argument("--template-neo4j-uri", default="")
    parser.add_argument("--template-neo4j-user", default="")
    parser.add_argument("--template-neo4j-pass", default="")
    parser.add_argument("--template-neo4j-database", default="")
    parser.add_argument("--template-qdrant-url", default="")
    parser.add_argument("--template-qdrant-collection", default="")
    parser.add_argument("--template-qdrant-api-key", default="")
    parser.add_argument("--template-semantic-search", action="store_true", default=True)
    parser.add_argument("--no-template-semantic-search", action="store_false", dest="template_semantic_search")
    parser.add_argument("--template-semantic-model", default="")
    parser.add_argument("--enforce-chinese-on-injection", action="store_true", default=True)
    parser.add_argument("--no-enforce-chinese-on-injection", action="store_false", dest="enforce_chinese_on_injection")
    parser.add_argument("--enforce-chinese-on-commit", action="store_true", default=True)
    parser.add_argument("--no-enforce-chinese-on-commit", action="store_false", dest="enforce_chinese_on_commit")
    parser.add_argument("--enforce-chinese-fields", default=",".join(DEFAULT_ENFORCE_CHINESE_FIELDS))
    parser.add_argument("--log-dir", default=str(_default_log_dir()))
    parser.add_argument("--log-injections", action="store_true", default=True)
    parser.add_argument("--no-log-injections", action="store_false", dest="log_injections")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_generation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
