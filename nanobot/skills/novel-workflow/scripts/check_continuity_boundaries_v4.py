#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
}


@dataclass
class BoundaryResult:
    pair: str
    ok: bool
    reasons: list[str]
    metrics: dict[str, Any]


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", str(text or ""))
    seen: set[str] = set()
    output: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower())


def _char_ngrams(text: str, size: int = 2) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", str(text or ""))
    if len(chars) < size:
        return set()
    return {"".join(chars[idx : idx + size]) for idx in range(len(chars) - size + 1)}


def _opening_overlap_ratio(left: str, right: str) -> float:
    left_bigrams = _char_ngrams(left)
    right_bigrams = _char_ngrams(right)
    if not left_bigrams or not right_bigrams:
        return 0.0
    union = left_bigrams | right_bigrams
    if not union:
        return 0.0
    return len(left_bigrams & right_bigrams) / len(union)


def _link_items_match(left: str, right: str) -> bool:
    l_norm = _normalize_for_match(left)
    r_norm = _normalize_for_match(right)
    if not l_norm or not r_norm:
        return False
    if l_norm in r_norm or r_norm in l_norm:
        return True
    left_tokens = {token for token in _extract_keywords(left) if token not in STOPWORDS}
    right_tokens = {token for token in _extract_keywords(right) if token not in STOPWORDS}
    if left_tokens & right_tokens:
        return True
    return len(_char_ngrams(left) & _char_ngrams(right)) >= 2


def _chapter_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def _opening_text(text: str) -> str:
    limit = max(400, int(len(text) * 0.15))
    return text[:limit]


def _find_latest_injection(log_root: Path, book_id: str, chapter_no: int) -> Path | None:
    pattern = f"{book_id}_*"
    needle = f"{chapter_no:04d}_pre_generation_injection.json"
    candidates: list[Path] = []
    for run_dir in log_root.glob(pattern):
        p = run_dir / "chapters" / needle
        if p.exists():
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        txt = str(item or "").strip()
        if txt:
            out.append(txt)
    return out


def _collect_pairs(max_chapter: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = [(138, 139), (139, 140), (143, 144), (144, 145)]
    cursor = 148
    while cursor < max_chapter:
        pairs.append((cursor, cursor + 1))
        cursor += 10
    dedup: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        dedup.append(pair)
    return dedup


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser()
    log_root = Path(args.log_root).expanduser()
    report_path = Path(args.report_path).expanduser()

    chapter_files = sorted(output_dir.glob(f"{args.book_id}_chapter_*.md"))
    chapter_nos: list[int] = []
    for path in chapter_files:
        m = re.search(r"chapter_(\d{4})\.md$", path.name)
        if m:
            chapter_nos.append(int(m.group(1)))
    max_chapter = max(chapter_nos) if chapter_nos else 0

    boundaries = _collect_pairs(max_chapter)
    results: list[BoundaryResult] = []

    for left_no, right_no in boundaries:
        left_path = output_dir / f"{args.book_id}_chapter_{left_no:04d}.md"
        right_path = output_dir / f"{args.book_id}_chapter_{right_no:04d}.md"
        if not left_path.exists() or not right_path.exists():
            continue

        left_body = _chapter_body(left_path)
        right_body = _chapter_body(right_path)
        left_open = _opening_text(left_body)
        right_open = _opening_text(right_body)

        inj_path = _find_latest_injection(log_root, args.book_id, right_no)
        carry_items: list[str] = []
        open_items: list[str] = []
        first_beat = ""
        if inj_path is not None:
            payload = json.loads(inj_path.read_text(encoding="utf-8"))
            carry_items = _load_list(payload.get("previous_chapter_carry_over"))
            open_items = _load_list(payload.get("current_chapter_open_with"))
            chapter_plan = payload.get("chapter_plan")
            if isinstance(chapter_plan, dict):
                beats = _load_list(chapter_plan.get("beat_outline"))
                first_beat = beats[0] if beats else ""

        carry_hits = sum(1 for item in carry_items if _link_items_match(item, right_open))
        open_hits = sum(1 for item in open_items if _link_items_match(item, right_open))
        beat_hit = 1 if first_beat and _link_items_match(first_beat, right_open) else 0
        overlap = _opening_overlap_ratio(left_open, right_open)

        reasons: list[str] = []
        if carry_items and carry_hits < 1:
            reasons.append("carry_over 未在下一章开头命中")
        if open_items and open_hits < 1:
            reasons.append("open_with 未在开头命中")
        if first_beat and beat_hit < 1:
            reasons.append("beat_outline[0] 未在开头落地")
        prev_zh = len(re.findall(r"[\u4e00-\u9fff]", left_open))
        curr_zh = len(re.findall(r"[\u4e00-\u9fff]", right_open))
        if prev_zh >= 60 and curr_zh >= 60 and overlap >= args.max_overlap:
            reasons.append(f"开头重叠过高({overlap:.2f})")

        metrics = {
            "left": left_no,
            "right": right_no,
            "carry_expected": len(carry_items),
            "carry_hits": carry_hits,
            "open_expected": len(open_items),
            "open_hits": open_hits,
            "first_beat_expected": 1 if first_beat else 0,
            "first_beat_hits": beat_hit,
            "opening_overlap": round(overlap, 6),
            "injection_path": str(inj_path) if inj_path else None,
        }
        results.append(BoundaryResult(pair=f"{left_no:04d}->{right_no:04d}", ok=not reasons, reasons=reasons, metrics=metrics))

    passed = [
        {"pair": row.pair, "metrics": row.metrics}
        for row in results
        if row.ok
    ]
    failed = [
        {"pair": row.pair, "reasons": row.reasons, "metrics": row.metrics}
        for row in results
        if not row.ok
    ]

    fail_rate = (len(failed) / len(results)) if results else 0.0
    risk = "low"
    if fail_rate > 0.2:
        risk = "high"
    elif failed:
        risk = "medium"

    run_report = output_dir / f"{args.book_id}_run_report.json"
    run_summary: dict[str, Any] = {}
    if run_report.exists():
        try:
            payload = json.loads(run_report.read_text(encoding="utf-8"))
            run_summary = {
                "carry_open_hit_rate": payload.get("carry_open_hit_rate"),
                "opening_rewrite_attempts_total": payload.get("opening_rewrite_attempts_total"),
                "opening_rewrite_success_total": payload.get("opening_rewrite_success_total"),
                "blueprint_failed_pairs_final": payload.get("blueprint_failed_pairs_final"),
            }
        except Exception:
            run_summary = {}

    report = {
        "book_id": args.book_id,
        "checked_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "log_root": str(log_root),
        "checked_pairs": len(results),
        "passed_pairs": len(passed),
        "failed_pairs": len(failed),
        "risk": risk,
        "run_summary": run_summary,
        "passed_boundaries": passed,
        "failed_boundaries": failed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boundary continuity checker for novel_04_b_full_1350_v4")
    parser.add_argument("--book-id", default="novel_04_b_full_1350_v4")
    parser.add_argument(
        "--output-dir",
        default="/home/chris/Desktop/my_workspace/novel_data/04/new_book/full_1350_v4",
    )
    parser.add_argument(
        "--log-root",
        default="/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/generate_book_ab",
    )
    parser.add_argument(
        "--report-path",
        default="/home/chris/Desktop/my_workspace/novel_data/04/new_book/log/continuity_boundary_check_v4.json",
    )
    parser.add_argument("--max-overlap", type=float, default=0.72)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_check(args)
    print(
        f"continuity_check: checked={report[checked_pairs]} failed={report[failed_pairs]} risk={report[risk]} report={args.report_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
