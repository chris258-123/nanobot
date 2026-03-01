from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 停用词 / 通用名词 / 过渡标记 ──────────────────────────────────────────────

STOPWORDS: set[str] = {
    "继续", "随后", "然后", "开始", "事情", "问题", "局势",
    "行动", "计划", "目标", "线索", "承接", "他们", "我们",
    "你们", "一个", "一种", "已经", "没有", "自己",
}

TRANSITION_MARKERS: tuple[str, ...] = (
    "与此同时", "同一时刻", "几分钟后", "片刻后", "当晚",
    "当天夜里", "第二天", "回到", "此时", "另一边",
    "不久后", "很快", "刚刚", "随后", "然后",
)

GENERIC_NAMES: set[str] = {
    "船员", "矿工", "众人", "全员", "骑士", "修女", "守卫",
    "学徒", "祭司", "审计员", "舰长", "神父", "医生",
    "少年", "少女", "男人", "女人", "队长", "老板", "老板娘", "黑袍人",
}

# ── 基础工具函数 ──────────────────────────────────────────────────────────────

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text or "")
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def normalize_for_match(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower())


def char_ngrams(text: str, size: int = 2) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", text or "")
    if len(chars) < size:
        return set()
    return {"".join(chars[i : i + size]) for i in range(len(chars) - size + 1)}


def overlap_ratio(a: str, b: str) -> float:
    a_grams = char_ngrams(a)
    b_grams = char_ngrams(b)
    if not a_grams or not b_grams:
        return 0.0
    union = a_grams | b_grams
    return len(a_grams & b_grams) / len(union)


def link_items_match(left: str, right: str) -> bool:
    l_norm = normalize_for_match(left)
    r_norm = normalize_for_match(right)
    if not l_norm or not r_norm:
        return False
    if l_norm in r_norm or r_norm in l_norm:
        return True
    left_tokens = {t for t in extract_keywords(left) if t not in STOPWORDS}
    right_tokens = {t for t in extract_keywords(right) if t not in STOPWORDS}
    if left_tokens & right_tokens:
        return True
    return len(char_ngrams(left) & char_ngrams(right)) >= 2


def chapter_body(text: str) -> str:
    lines = (text or "").splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).strip()
    return (text or "").strip()


def chapter_title(text: str) -> str:
    lines = (text or "").splitlines()
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip()
    return ""


def opening_text(text: str) -> str:
    return (text or "")[: max(450, int(len(text or "") * 0.15))]


def ending_text(text: str) -> str:
    return (text or "")[-max(420, int(len(text or "") * 0.12)) :]


def load_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def first_beat(payload: dict[str, Any]) -> str:
    plan = payload.get("chapter_plan")
    if not isinstance(plan, dict):
        return ""
    beats = load_list(plan.get("beat_outline"))
    return beats[0] if beats else ""


def has_transition_marker(text: str) -> bool:
    snippet = (text or "")[:180]
    return any(mark in snippet for mark in TRANSITION_MARKERS)


def find_latest_injection(chapter_no: int, gen_log_root: Path | None, book_id: str) -> Path | None:
    if gen_log_root is None:
        return None
    needle = f"{chapter_no:04d}_pre_generation_injection.json"
    cands: list[Path] = [
        run_dir / "chapters" / needle
        for run_dir in gen_log_root.glob(f"{book_id}_*")
        if (run_dir / "chapters" / needle).exists()
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


# ── 检测函数：文本完整性 ──────────────────────────────────────────────────────

def detect_text_integrity(
    chapters: dict[int, dict[str, Any]],
    start: int,
    end: int,
    book_id: str,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    prev_title = ""
    prev_no: int | None = None
    for no in range(start, end + 1):
        item = chapters.get(no)
        if not item:
            issues.append({"chapter": no, "code": "missing_file", "detail": "章节文件缺失"})
            continue
        text = str(item.get("text") or "")
        body = str(item.get("body") or "")
        title = str(item.get("title") or "")
        zh_count = len(re.findall(r"[\u4e00-\u9fff]", body))
        if not title:
            issues.append({"chapter": no, "code": "missing_title", "detail": "缺少标题行(# ...)"})
        if zh_count < 1400:
            issues.append({"chapter": no, "code": "short_length", "detail": f"正文中文字符偏少: {zh_count}"})
        if body.count("\u201c") != body.count("\u201d") or body.count("《") != body.count("》"):
            issues.append({"chapter": no, "code": "quote_mismatch", "detail": "引号或书名号未闭合"})
        if "\x1b" in text or "tokens truncated" in text or "…32 tokens truncated…" in text:
            issues.append({"chapter": no, "code": "artifact_marker", "detail": "检测到截断/终端转义痕迹"})
        tail = body.rstrip()[-1:] if body.rstrip() else ""
        if tail and tail in {"，", "、", "：", ":", "（", "(", "【", "[", "\u201c", "\u2018", "—", "-"}:
            issues.append({"chapter": no, "code": "hanging_ending", "detail": f"结尾可能截断，尾字符={tail}"})
        if prev_title and title and title == prev_title and prev_no is not None:
            issues.append({"chapter": no, "code": "adjacent_duplicate_title", "detail": f"与上一章{prev_no:04d}标题重复"})
        prev_title = title
        prev_no = no
    return {
        "book_id": book_id,
        "range": f"{start:04d}-{end:04d}",
        "generated_at": datetime.now().isoformat(),
        "issue_count": len(issues),
        "issues": issues,
    }


# ── 检测函数：章间连续性 ──────────────────────────────────────────────────────

def detect_continuity(
    chapters: dict[int, dict[str, Any]],
    names_for_overlap: list[str],
    start: int,
    end: int,
    gen_log_root: Path | None,
    book_id: str,
) -> dict[str, Any]:
    pair_results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for left in range(start, end):
        right = left + 1
        if left not in chapters or right not in chapters:
            row = {
                "pair": f"{left:04d}->{right:04d}",
                "ok": False,
                "issues": ["missing_chapter_file"],
                "metrics": {"left_exists": left in chapters, "right_exists": right in chapters},
            }
            pair_results.append(row)
            failed.append(row)
            continue

        left_body = str(chapters[left].get("body") or "")
        right_body = str(chapters[right].get("body") or "")
        left_open = opening_text(left_body)
        right_open = opening_text(right_body)
        left_tail = ending_text(left_body)

        opening_overlap = overlap_ratio(left_open, right_open)
        tail_open_overlap = overlap_ratio(left_tail, right_open)
        shared_entities = [n for n in names_for_overlap if n in left_tail and n in right_open]

        inj_path = find_latest_injection(right, gen_log_root=gen_log_root, book_id=book_id)
        carry_items: list[str] = []
        open_items: list[str] = []
        beat0 = ""
        if inj_path is not None:
            try:
                payload = json.loads(inj_path.read_text(encoding="utf-8"))
                carry_items = load_list(payload.get("previous_chapter_carry_over"))
                open_items = load_list(payload.get("current_chapter_open_with"))
                beat0 = first_beat(payload)
            except Exception:
                pass

        carry_hits = sum(1 for x in carry_items if link_items_match(x, right_open))
        open_hits = sum(1 for x in open_items if link_items_match(x, right_open))
        beat_hit = 1 if beat0 and link_items_match(beat0, right_open) else 0

        issues: list[str] = []
        if carry_items and carry_hits < 1:
            issues.append("carry_over_not_landed")
        if open_items and open_hits < 1:
            issues.append("open_with_not_landed")
        if beat0 and beat_hit < 1:
            issues.append("first_beat_not_landed")
        if opening_overlap >= 0.78:
            issues.append(f"opening_overlap_too_high({opening_overlap:.2f})")
        if not issues and tail_open_overlap < 0.015 and not shared_entities and not has_transition_marker(right_open):
            issues.append("weak_tail_to_head_bridge")

        row = {
            "pair": f"{left:04d}->{right:04d}",
            "ok": not issues,
            "issues": issues,
            "metrics": {
                "opening_overlap": round(opening_overlap, 6),
                "tail_open_overlap": round(tail_open_overlap, 6),
                "shared_entities": shared_entities[:8],
                "carry_expected": len(carry_items),
                "carry_hits": carry_hits,
                "open_expected": len(open_items),
                "open_hits": open_hits,
                "first_beat_expected": 1 if beat0 else 0,
                "first_beat_hits": beat_hit,
                "injection_path": str(inj_path) if inj_path else None,
            },
        }
        pair_results.append(row)
        if issues:
            failed.append(row)

    return {
        "book_id": book_id,
        "range": f"{start:04d}-{end:04d}",
        "generated_at": datetime.now().isoformat(),
        "checked_pairs": len(pair_results),
        "failed_pairs": len(failed),
        "failed_ratio": round((len(failed) / len(pair_results)) if pair_results else 0.0, 6),
        "failures": failed,
        "pairs": pair_results,
    }


# ── 检测函数：人物性别一致性 ──────────────────────────────────────────────────

def _gender_hits(text: str, name: str) -> tuple[list[str], list[str]]:
    male_pat = re.compile(rf"{re.escape(name)}[^。！？\n]{{0,10}}他|他[^。！？\n]{{0,10}}{re.escape(name)}")
    female_pat = re.compile(rf"{re.escape(name)}[^。！？\n]{{0,10}}她|她[^。！？\n]{{0,10}}{re.escape(name)}")
    return [m.group(0) for m in male_pat.finditer(text)], [m.group(0) for m in female_pat.finditer(text)]


def _pick_focus_names(chapters: dict[int, dict[str, Any]], start: int, end: int) -> list[str]:
    corpus = "\n".join(str(chapters[n].get("body") or "") for n in range(start, end + 1) if n in chapters)
    counts = Counter(re.findall(r"[\u4e00-\u9fff]{2,4}", corpus))
    cands = [
        (name, freq) for name, freq in counts.items()
        if name not in GENERIC_NAMES and name not in STOPWORDS and freq >= 18
    ]
    cands.sort(key=lambda x: (-x[1], len(x[0]), x[0]))
    return [name for name, _ in cands[:60]]


def detect_character_consistency(
    chapters: dict[int, dict[str, Any]],
    start: int,
    end: int,
    book_id: str,
) -> dict[str, Any]:
    focus_names = _pick_focus_names(chapters, start, end)
    issues: list[dict[str, Any]] = []
    per_name: dict[str, Any] = {}

    for name in focus_names:
        male_count = 0
        female_count = 0
        male_refs: list[dict[str, Any]] = []
        female_refs: list[dict[str, Any]] = []
        chapter_hits: defaultdict[int, dict[str, int]] = defaultdict(lambda: {"male": 0, "female": 0})

        for no in range(start, end + 1):
            item = chapters.get(no)
            if not item:
                continue
            m_hits, f_hits = _gender_hits(str(item.get("body") or ""), name)
            if m_hits:
                male_count += len(m_hits)
                chapter_hits[no]["male"] += len(m_hits)
                if len(male_refs) < 5:
                    male_refs.append({"chapter": no, "snippet": m_hits[0][:80]})
            if f_hits:
                female_count += len(f_hits)
                chapter_hits[no]["female"] += len(f_hits)
                if len(female_refs) < 5:
                    female_refs.append({"chapter": no, "snippet": f_hits[0][:80]})

        if male_count == 0 and female_count == 0:
            continue

        dominant = "male" if male_count >= female_count else "female"
        majority = male_count if dominant == "male" else female_count
        minority = female_count if dominant == "male" else male_count
        mixed_chapters = [ch for ch, cnt in chapter_hits.items() if cnt["male"] > 0 and cnt["female"] > 0]

        per_name[name] = {
            "male_hits": male_count,
            "female_hits": female_count,
            "dominant": dominant,
            "male_examples": male_refs,
            "female_examples": female_refs,
            "chapter_mixed": [f"{ch:04d}" for ch in sorted(mixed_chapters)],
        }

        if mixed_chapters:
            issues.append({
                "name": name,
                "code": "same_chapter_gender_mix",
                "detail": f"同章出现他/她混用: {', '.join(f'{x:04d}' for x in sorted(mixed_chapters)[:10])}",
            })
        if majority >= 3 and minority >= 2 and (minority / max(majority, 1)) >= 0.2:
            issues.append({
                "name": name,
                "code": "cross_chapter_gender_drift",
                "detail": f"跨章性别指代漂移: male={male_count}, female={female_count}",
            })

    return {
        "book_id": book_id,
        "range": f"{start:04d}-{end:04d}",
        "generated_at": datetime.now().isoformat(),
        "focus_names": focus_names,
        "issue_count": len(issues),
        "issues": issues,
        "per_name": per_name,
    }


# ── 检测函数：事实一致性（Canon DB）────────────────────────────────────────────

def _empty_fact_result(start: int, end: int, book_id: str) -> dict[str, Any]:
    return {
        "book_id": book_id,
        "range": f"{start:04d}-{end:04d}",
        "generated_at": datetime.now().isoformat(),
        "issue_count": 0,
        "issues": [],
        "commit": {"missing_chapters": [], "failed_chapters": [], "done_count": 0},
        "memory": {"fact_missing_chapters": [], "relationship_missing_chapters": [], "unstable_fact_candidates": []},
    }


def detect_fact_consistency(
    db_path: Path | None,
    start: int,
    end: int,
    book_id: str,
) -> dict[str, Any]:
    report = _empty_fact_result(start, end, book_id)
    if db_path is None:
        return report
    if not db_path.exists():
        report["issues"].append({"code": "db_path_missing", "detail": f"数据库文件不存在: {db_path}"})
        report["issue_count"] = 1
        return report

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        report["issues"].append({"code": "db_open_failed", "detail": str(exc)})
        report["issue_count"] = 1
        return report

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT chapter_no, status, commit_type, created_at "
            "FROM commit_log ORDER BY CAST(chapter_no AS INTEGER), created_at"
        )
        latest: dict[str, dict[str, Any]] = {}
        for chapter_no, status, commit_type, created_at in cur.fetchall():
            latest[str(chapter_no).zfill(4)] = {"status": status, "commit_type": commit_type, "created_at": created_at}

        commit_missing, commit_failed, commit_done = [], [], []
        for no in range(start, end + 1):
            ch = f"{no:04d}"
            info = latest.get(ch)
            if info is None:
                commit_missing.append(ch)
            elif info.get("status") != "ALL_DONE":
                commit_failed.append({"chapter": ch, **info})
            else:
                commit_done.append(ch)

        cur.execute("SELECT DISTINCT chapter_no FROM fact_history")
        fact_chapters = {str(r[0]).zfill(4) for r in cur.fetchall()}
        cur.execute("SELECT DISTINCT chapter_no FROM relationship_history")
        rel_chapters = {str(r[0]).zfill(4) for r in cur.fetchall()}

        fact_missing = [f"{no:04d}" for no in range(start, end + 1) if f"{no:04d}" not in fact_chapters]
        rel_missing = [f"{no:04d}" for no in range(start, end + 1) if f"{no:04d}" not in rel_chapters]

        cur.execute(
            "SELECT subject_id, predicate, COUNT(*), COUNT(DISTINCT object_json), "
            "MIN(CAST(chapter_no AS INTEGER)), MAX(CAST(chapter_no AS INTEGER)) "
            "FROM fact_history GROUP BY subject_id, predicate "
            "HAVING COUNT(*) >= 4 AND COUNT(DISTINCT object_json) >= 3 "
            "ORDER BY COUNT(DISTINCT object_json) DESC, COUNT(*) DESC LIMIT 200"
        )
        cur2 = conn.cursor()
        cur2.execute("SELECT entity_id, canonical_name FROM entity_registry WHERE type='character'")
        name_map = {eid: (nm or "") for eid, nm in cur2.fetchall()}
        unstable = [
            {
                "subject_id": sid,
                "subject_name": name_map.get(sid, sid),
                "predicate": pred,
                "events": int(total),
                "distinct_values": int(distinct),
                "chapter_span": f"{int(min_ch):04d}-{int(max_ch):04d}",
            }
            for sid, pred, total, distinct, min_ch, max_ch in cur.fetchall()
        ]
    except sqlite3.Error as exc:
        report["issues"].append({"code": "db_query_failed", "detail": str(exc)})
        report["issue_count"] = 1
        return report
    finally:
        conn.close()

    issues: list[dict[str, Any]] = []
    if commit_missing:
        issues.append({"code": "commit_log_missing", "detail": f"无commit记录章节数={len(commit_missing)}"})
    if commit_failed:
        issues.append({"code": "latest_commit_failed", "detail": f"latest状态非ALL_DONE章节数={len(commit_failed)}"})
    if fact_missing:
        issues.append({"code": "fact_history_missing", "detail": f"无fact_history章节数={len(fact_missing)}"})
    if rel_missing:
        issues.append({"code": "relationship_history_missing", "detail": f"无relationship_history章节数={len(rel_missing)}"})
    if unstable:
        issues.append({"code": "high_variance_facts", "detail": f"高频多值事实条目={len(unstable)}"})

    report["issue_count"] = len(issues)
    report["issues"] = issues
    report["commit"] = {"missing_chapters": commit_missing, "failed_chapters": commit_failed, "done_count": len(commit_done)}
    report["memory"] = {"fact_missing_chapters": fact_missing, "relationship_missing_chapters": rel_missing, "unstable_fact_candidates": unstable}
    return report
