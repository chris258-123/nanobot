#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

BOOK_ID = "novel_04_b_full_1350_v4"
START = 1
END = 338
BOOK_DIR = Path("/home/chris/Desktop/my_workspace/novel_data/04/new_book/full_1350_v4")
LOG_DIR = Path("/home/chris/Desktop/my_workspace/novel_data/04/new_book/log")
GEN_LOG_ROOT = LOG_DIR / "generate_book_ab"
DB_PATH = Path("/home/chris/Desktop/my_workspace/novel_data/04/new_book/canon_novel_04_b_full_1350_v4.db")

TEXT_INTEGRITY_JSON = LOG_DIR / "full_audit_v4_0001_0338_text_integrity.json"
CONTINUITY_JSON = LOG_DIR / "full_audit_v4_0001_0338_continuity.json"
CONTINUITY_FAILED_MD = LOG_DIR / "full_audit_v4_0001_0338_continuity_failed.md"
CHAR_CONSISTENCY_JSON = LOG_DIR / "full_audit_v4_0001_0338_character_consistency.json"
GENDER_ISSUES_MD = LOG_DIR / "full_audit_v4_0001_0338_gender_issues.md"
FACT_CONSISTENCY_JSON = LOG_DIR / "full_audit_v4_0001_0338_fact_consistency.json"
MEMORY_GAP_MD = LOG_DIR / "full_audit_v4_0001_0338_memory_gap.md"
SUMMARY_JSON = LOG_DIR / "full_audit_v4_0001_0338_summary.json"
SUMMARY_MD = LOG_DIR / "full_audit_v4_0001_0338_summary.md"

STOPWORDS = {
    "继续",
    "随后",
    "然后",
    "开始",
    "事情",
    "问题",
    "局势",
    "行动",
    "计划",
    "目标",
    "线索",
    "承接",
    "他们",
    "我们",
    "你们",
    "一个",
    "一种",
    "已经",
    "没有",
    "自己",
}

TRANSITION_MARKERS = (
    "与此同时",
    "同一时刻",
    "几分钟后",
    "片刻后",
    "当晚",
    "当天夜里",
    "第二天",
    "回到",
    "此时",
    "另一边",
    "不久后",
    "很快",
    "刚刚",
    "随后",
    "然后",
)

GENERIC_NAMES = {
    "船员",
    "矿工",
    "众人",
    "全员",
    "骑士",
    "修女",
    "守卫",
    "学徒",
    "祭司",
    "审计员",
    "舰长",
    "神父",
    "医生",
    "少年",
    "少女",
    "男人",
    "女人",
    "队长",
    "老板",
    "老板娘",
    "黑袍人",
}


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
    if not union:
        return 0.0
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


def chapter_path(no: int) -> Path:
    return BOOK_DIR / f"{BOOK_ID}_chapter_{no:04d}.md"


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


def find_latest_injection(chapter_no: int) -> Path | None:
    needle = f"{chapter_no:04d}_pre_generation_injection.json"
    cands: list[Path] = []
    for run_dir in GEN_LOG_ROOT.glob(f"{BOOK_ID}_*"):
        p = run_dir / "chapters" / needle
        if p.exists():
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def load_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        txt = str(item or "").strip()
        if txt:
            out.append(txt)
    return out


def first_beat(payload: dict[str, Any]) -> str:
    plan = payload.get("chapter_plan")
    if not isinstance(plan, dict):
        return ""
    beats = load_list(plan.get("beat_outline"))
    return beats[0] if beats else ""


def has_transition_marker(text: str) -> bool:
    snippet = (text or "")[:180]
    return any(mark in snippet for mark in TRANSITION_MARKERS)


def detect_text_integrity(chapters: dict[int, dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    prev_title = ""
    prev_no = None
    for no in range(START, END + 1):
        item = chapters.get(no)
        if not item:
            issues.append({"chapter": no, "code": "missing_file", "detail": "章节文件缺失"})
            continue
        text = item["text"]
        body = item["body"]
        title = item["title"]
        zh_count = len(re.findall(r"[\u4e00-\u9fff]", body))
        if not title:
            issues.append({"chapter": no, "code": "missing_title", "detail": "缺少标题行(# ...)"})
        if zh_count < 1400:
            issues.append({"chapter": no, "code": "short_length", "detail": f"正文中文字符偏少: {zh_count}"})
        if body.count("“") != body.count("”") or body.count("《") != body.count("》"):
            issues.append({"chapter": no, "code": "quote_mismatch", "detail": "引号或书名号未闭合"})
        if "\x1b" in text or "tokens truncated" in text or "…32 tokens truncated…" in text:
            issues.append({"chapter": no, "code": "artifact_marker", "detail": "检测到截断/终端转义痕迹"})
        tail = body.rstrip()[-1:] if body.rstrip() else ""
        if tail and tail in {"，", "、", "：", ":", "（", "(", "【", "[", "“", "‘", "—", "-"}:
            issues.append({"chapter": no, "code": "hanging_ending", "detail": f"结尾可能截断，尾字符={tail}"})
        if prev_title and title and title == prev_title:
            issues.append({"chapter": no, "code": "adjacent_duplicate_title", "detail": f"与上一章{prev_no:04d}标题重复"})
        prev_title = title
        prev_no = no
    return {
        "book_id": BOOK_ID,
        "range": f"{START:04d}-{END:04d}",
        "generated_at": datetime.now().isoformat(),
        "issue_count": len(issues),
        "issues": issues,
    }


def detect_continuity(chapters: dict[int, dict[str, Any]], names_for_overlap: list[str]) -> dict[str, Any]:
    pair_results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for left in range(START, END):
        right = left + 1
        if left not in chapters or right not in chapters:
            failed.append(
                {
                    "pair": f"{left:04d}->{right:04d}",
                    "issues": ["missing_chapter_file"],
                    "metrics": {"left_exists": left in chapters, "right_exists": right in chapters},
                }
            )
            continue

        left_body = chapters[left]["body"]
        right_body = chapters[right]["body"]
        left_open = opening_text(left_body)
        right_open = opening_text(right_body)
        left_tail = ending_text(left_body)

        opening_overlap = overlap_ratio(left_open, right_open)
        tail_open_overlap = overlap_ratio(left_tail, right_open)
        shared_entities = [name for name in names_for_overlap if name in left_tail and name in right_open]

        inj_path = find_latest_injection(right)
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
        if not issues:
            if tail_open_overlap < 0.015 and len(shared_entities) == 0 and not has_transition_marker(right_open):
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
        "book_id": BOOK_ID,
        "range": f"{START:04d}-{END:04d}",
        "generated_at": datetime.now().isoformat(),
        "checked_pairs": len(pair_results),
        "failed_pairs": len(failed),
        "failed_ratio": round((len(failed) / len(pair_results)) if pair_results else 0.0, 6),
        "failures": failed,
        "pairs": pair_results,
    }


def load_db_names(conn: sqlite3.Connection) -> dict[str, str]:
    cur = conn.cursor()
    cur.execute("SELECT entity_id, canonical_name FROM entity_registry WHERE type='character'")
    return {entity_id: (name or "") for entity_id, name in cur.fetchall()}


def pick_focus_names(chapters: dict[int, dict[str, Any]], db_names: dict[str, str]) -> list[str]:
    corpus = "\n".join(chapters[n]["body"] for n in sorted(chapters))
    cands: list[tuple[str, int]] = []
    for name in db_names.values():
        if not re.fullmatch(r"[\u4e00-\u9fff]{2,4}", name or ""):
            continue
        if name in GENERIC_NAMES:
            continue
        freq = corpus.count(name)
        if freq >= 18:
            cands.append((name, freq))
    cands.sort(key=lambda x: (-x[1], len(x[0]), x[0]))
    return [x[0] for x in cands[:60]]


def gender_hits_for_name(text: str, name: str) -> tuple[list[str], list[str]]:
    male_pattern = re.compile(rf"{re.escape(name)}[^。！？\n]{{0,10}}他|他[^。！？\n]{{0,10}}{re.escape(name)}")
    female_pattern = re.compile(rf"{re.escape(name)}[^。！？\n]{{0,10}}她|她[^。！？\n]{{0,10}}{re.escape(name)}")
    males = [m.group(0) for m in male_pattern.finditer(text)]
    females = [m.group(0) for m in female_pattern.finditer(text)]
    return males, females


def detect_character_consistency(chapters: dict[int, dict[str, Any]], conn: sqlite3.Connection) -> dict[str, Any]:
    db_names = load_db_names(conn)
    focus_names = pick_focus_names(chapters, db_names)

    expected_gender: dict[str, str] = {}
    cur = conn.cursor()
    cur.execute("SELECT subject_id, predicate, object_json FROM fact_history WHERE lower(predicate) IN ('gender','sex','性别')")
    for subject_id, _predicate, object_json in cur.fetchall():
        nm = db_names.get(subject_id)
        if not nm:
            continue
        txt = str(object_json or "")
        if "女" in txt or "female" in txt.lower():
            expected_gender[nm] = "female"
        elif "男" in txt or "male" in txt.lower():
            expected_gender[nm] = "male"

    per_name: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []

    for name in focus_names:
        male_count = 0
        female_count = 0
        male_refs: list[dict[str, Any]] = []
        female_refs: list[dict[str, Any]] = []
        chapter_hits = defaultdict(lambda: {"male": 0, "female": 0})

        for no in range(START, END + 1):
            item = chapters.get(no)
            if not item:
                continue
            text = item["body"]
            m_hits, f_hits = gender_hits_for_name(text, name)
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
        minority = female_count if dominant == "male" else male_count
        majority = male_count if dominant == "male" else female_count

        per_name[name] = {
            "male_hits": male_count,
            "female_hits": female_count,
            "dominant": dominant,
            "expected_gender": expected_gender.get(name),
            "male_examples": male_refs,
            "female_examples": female_refs,
            "chapter_mixed": [
                f"{ch:04d}"
                for ch, cnt in sorted(chapter_hits.items())
                if cnt["male"] > 0 and cnt["female"] > 0
            ],
        }

        mixed_chapters = [ch for ch, cnt in chapter_hits.items() if cnt["male"] > 0 and cnt["female"] > 0]
        if mixed_chapters:
            issues.append(
                {
                    "name": name,
                    "code": "same_chapter_gender_mix",
                    "detail": f"同章出现他/她混用: {', '.join(f'{x:04d}' for x in sorted(mixed_chapters)[:10])}",
                }
            )

        if majority >= 3 and minority >= 2 and (minority / max(majority, 1)) >= 0.2:
            issues.append(
                {
                    "name": name,
                    "code": "cross_chapter_gender_drift",
                    "detail": f"跨章性别指代漂移: male={male_count}, female={female_count}",
                }
            )

        exp = expected_gender.get(name)
        if exp == "male" and female_count >= 2:
            issues.append({"name": name, "code": "conflict_with_db_expected_gender", "detail": f"记忆库预期男，但检测到女性指代 {female_count} 次"})
        if exp == "female" and male_count >= 2:
            issues.append({"name": name, "code": "conflict_with_db_expected_gender", "detail": f"记忆库预期女，但检测到男性指代 {male_count} 次"})

    return {
        "book_id": BOOK_ID,
        "range": f"{START:04d}-{END:04d}",
        "generated_at": datetime.now().isoformat(),
        "focus_names": focus_names,
        "issue_count": len(issues),
        "issues": issues,
        "per_name": per_name,
    }


def detect_fact_consistency(chapters: dict[int, dict[str, Any]], conn: sqlite3.Connection) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT chapter_no, status, commit_type, created_at FROM commit_log ORDER BY CAST(chapter_no AS INTEGER), created_at")
    latest: dict[str, dict[str, Any]] = {}
    for chapter_no, status, commit_type, created_at in cur.fetchall():
        latest[str(chapter_no).zfill(4)] = {"status": status, "commit_type": commit_type, "created_at": created_at}

    commit_missing: list[str] = []
    commit_failed: list[dict[str, Any]] = []
    commit_done: list[str] = []
    for no in range(START, END + 1):
        ch = f"{no:04d}"
        info = latest.get(ch)
        if info is None:
            commit_missing.append(ch)
            continue
        if info.get("status") != "ALL_DONE":
            commit_failed.append({"chapter": ch, **info})
        else:
            commit_done.append(ch)

    cur.execute("SELECT DISTINCT chapter_no FROM fact_history")
    fact_chapters = {str(r[0]).zfill(4) for r in cur.fetchall()}
    cur.execute("SELECT DISTINCT chapter_no FROM relationship_history")
    rel_chapters = {str(r[0]).zfill(4) for r in cur.fetchall()}

    fact_missing = [f"{no:04d}" for no in range(START, END + 1) if f"{no:04d}" not in fact_chapters]
    rel_missing = [f"{no:04d}" for no in range(START, END + 1) if f"{no:04d}" not in rel_chapters]

    cur.execute(
        "SELECT subject_id, predicate, COUNT(*), COUNT(DISTINCT object_json), MIN(CAST(chapter_no AS INTEGER)), MAX(CAST(chapter_no AS INTEGER)) "
        "FROM fact_history GROUP BY subject_id, predicate HAVING COUNT(*) >= 4 AND COUNT(DISTINCT object_json) >= 3 "
        "ORDER BY COUNT(DISTINCT object_json) DESC, COUNT(*) DESC LIMIT 200"
    )
    name_map = load_db_names(conn)
    unstable = []
    for subject_id, predicate, total, distinct_values, min_ch, max_ch in cur.fetchall():
        unstable.append(
            {
                "subject_id": subject_id,
                "subject_name": name_map.get(subject_id, subject_id),
                "predicate": predicate,
                "events": int(total),
                "distinct_values": int(distinct_values),
                "chapter_span": f"{int(min_ch):04d}-{int(max_ch):04d}",
            }
        )

    issues = []
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

    return {
        "book_id": BOOK_ID,
        "range": f"{START:04d}-{END:04d}",
        "generated_at": datetime.now().isoformat(),
        "issue_count": len(issues),
        "issues": issues,
        "commit": {
            "missing_chapters": commit_missing,
            "failed_chapters": commit_failed,
            "done_count": len(commit_done),
        },
        "memory": {
            "fact_missing_chapters": fact_missing,
            "relationship_missing_chapters": rel_missing,
            "unstable_fact_candidates": unstable,
        },
    }


def write_continuity_failed_md(report: dict[str, Any]) -> None:
    lines = [f"# 连续性问题清单 ({START:04d}-{END:04d})", "", f"- 检查相邻章对: {report.get('checked_pairs', 0)}", f"- 问题章对: {report.get('failed_pairs', 0)}", ""]
    for row in report.get("failures", []):
        issues = ", ".join(row.get("issues", []))
        metrics = row.get("metrics", {})
        lines.append(
            f"- `{row.get('pair')}` | {issues} | overlap={metrics.get('opening_overlap')} tail_open={metrics.get('tail_open_overlap')}"
        )
    CONTINUITY_FAILED_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_gender_issues_md(report: dict[str, Any]) -> None:
    lines = [f"# 人物一致性问题清单 ({START:04d}-{END:04d})", "", f"- 问题条数: {report.get('issue_count', 0)}", ""]
    for item in report.get("issues", []):
        lines.append(f"- `{item.get('name')}` | {item.get('code')} | {item.get('detail')}")
    GENDER_ISSUES_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_memory_gap_md(report: dict[str, Any]) -> None:
    commit = report.get("commit", {})
    memory = report.get("memory", {})
    lines = [
        f"# 事实一致性/记忆库问题清单 ({START:04d}-{END:04d})",
        "",
        f"- latest commit 失败章节: {len(commit.get('failed_chapters', []))}",
        f"- commit 缺失章节: {len(commit.get('missing_chapters', []))}",
        f"- fact_history 缺失章节: {len(memory.get('fact_missing_chapters', []))}",
        f"- relationship_history 缺失章节: {len(memory.get('relationship_missing_chapters', []))}",
        "",
    ]

    failed = commit.get("failed_chapters", [])
    if failed:
        lines.append("## latest commit failed 章节")
        for row in failed:
            lines.append(f"- {row.get('chapter')} | status={row.get('status')} | at={row.get('created_at')}")
        lines.append("")

    missing = commit.get("missing_chapters", [])
    if missing:
        lines.extend(["## commit 缺失章节", "- " + ", ".join(missing), ""])

    fact_missing = memory.get("fact_missing_chapters", [])
    if fact_missing:
        lines.extend(["## fact_history 缺失章节", "- " + ", ".join(fact_missing), ""])

    rel_missing = memory.get("relationship_missing_chapters", [])
    if rel_missing:
        lines.extend(["## relationship_history 缺失章节", "- " + ", ".join(rel_missing), ""])

    unstable = memory.get("unstable_fact_candidates", [])
    if unstable:
        lines.append("## 高频多值事实候选（前50）")
        for row in unstable[:50]:
            lines.append(
                f"- {row.get('subject_name')}[{row.get('subject_id')}] | {row.get('predicate')} | values={row.get('distinct_values')} | events={row.get('events')} | span={row.get('chapter_span')}"
            )

    MEMORY_GAP_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_summary(text_report: dict[str, Any], continuity_report: dict[str, Any], char_report: dict[str, Any], fact_report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    summary = {
        "book_id": BOOK_ID,
        "range": f"{START:04d}-{END:04d}",
        "generated_at": datetime.now().isoformat(),
        "problem_counts": {
            "text_integrity_issues": text_report.get("issue_count", 0),
            "continuity_failed_pairs": continuity_report.get("failed_pairs", 0),
            "character_consistency_issues": char_report.get("issue_count", 0),
            "fact_consistency_issues": fact_report.get("issue_count", 0),
        },
        "critical": {
            "continuity_failed_pairs_top20": [x.get("pair") for x in continuity_report.get("failures", [])[:20]],
            "gender_issue_names": sorted({x.get("name") for x in char_report.get("issues", []) if x.get("name")}),
            "latest_commit_failed_chapters": [x.get("chapter") for x in fact_report.get("commit", {}).get("failed_chapters", [])],
            "commit_missing_chapters": fact_report.get("commit", {}).get("missing_chapters", []),
        },
        "report_paths": {
            "text_integrity": str(TEXT_INTEGRITY_JSON),
            "continuity": str(CONTINUITY_JSON),
            "continuity_failed_md": str(CONTINUITY_FAILED_MD),
            "character_consistency": str(CHAR_CONSISTENCY_JSON),
            "gender_issues_md": str(GENDER_ISSUES_MD),
            "fact_consistency": str(FACT_CONSISTENCY_JSON),
            "memory_gap_md": str(MEMORY_GAP_MD),
        },
    }

    md_lines = [
        f"# 全量问题清单汇总 ({START:04d}-{END:04d})",
        "",
        "- 仅列问题，不列通过项。",
        f"- 文本完整性问题: {summary['problem_counts']['text_integrity_issues']}",
        f"- 章间连续性问题章对: {summary['problem_counts']['continuity_failed_pairs']}",
        f"- 人物一致性问题: {summary['problem_counts']['character_consistency_issues']}",
        f"- 事实/记忆库问题: {summary['problem_counts']['fact_consistency_issues']}",
        "",
    ]

    failed_pairs = summary["critical"]["continuity_failed_pairs_top20"]
    if failed_pairs:
        md_lines.extend(["## 连续性问题章对（前20）", "- " + ", ".join(failed_pairs), ""])

    gender_names = summary["critical"]["gender_issue_names"]
    if gender_names:
        md_lines.extend(["## 性别/指代问题人物", "- " + ", ".join(gender_names), ""])

    commit_failed = summary["critical"]["latest_commit_failed_chapters"]
    if commit_failed:
        md_lines.extend(["## latest commit failed 章节", "- " + ", ".join(commit_failed), ""])

    commit_missing = summary["critical"]["commit_missing_chapters"]
    if commit_missing:
        md_lines.extend(["## commit 缺失章节", "- " + ", ".join(commit_missing), ""])

    md_lines.append("## 详细报告路径")
    for key, path in summary["report_paths"].items():
        md_lines.append(f"- {key}: `{path}`")

    return summary, "\n".join(md_lines).strip() + "\n"


def main() -> int:
    chapters: dict[int, dict[str, Any]] = {}
    for no in range(START, END + 1):
        p = chapter_path(no)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        body = chapter_body(text)
        chapters[no] = {"path": str(p), "text": text, "body": body, "title": chapter_title(text)}

    text_report = detect_text_integrity(chapters)

    conn = sqlite3.connect(DB_PATH)
    try:
        db_names = load_db_names(conn)
        overlap_names = [
            nm
            for nm in db_names.values()
            if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", nm or "") and nm not in GENERIC_NAMES
        ]
        continuity_report = detect_continuity(chapters, overlap_names[:120])
        char_report = detect_character_consistency(chapters, conn)
        fact_report = detect_fact_consistency(chapters, conn)
    finally:
        conn.close()

    write_json(TEXT_INTEGRITY_JSON, text_report)
    write_json(CONTINUITY_JSON, continuity_report)
    write_json(CHAR_CONSISTENCY_JSON, char_report)
    write_json(FACT_CONSISTENCY_JSON, fact_report)

    write_continuity_failed_md(continuity_report)
    write_gender_issues_md(char_report)
    write_memory_gap_md(fact_report)

    summary_json, summary_md = build_summary(text_report, continuity_report, char_report, fact_report)
    write_json(SUMMARY_JSON, summary_json)
    SUMMARY_MD.write_text(summary_md, encoding="utf-8")

    print(
        json.dumps(
            {
                "text_integrity_issues": text_report.get("issue_count", 0),
                "continuity_failed_pairs": continuity_report.get("failed_pairs", 0),
                "character_consistency_issues": char_report.get("issue_count", 0),
                "fact_consistency_issues": fact_report.get("issue_count", 0),
                "summary": str(SUMMARY_JSON),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
