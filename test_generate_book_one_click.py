import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parent
        / "nanobot"
        / "skills"
        / "novel-workflow"
        / "scripts"
        / "generate_book_one_click.py"
    )
    spec = importlib.util.spec_from_file_location("generate_book_one_click", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_chapter_plan_fills_defaults() -> None:
    module = _load_module()
    plan = module._normalize_chapter_plan(
        raw_chapters=[
            {"title": "开局", "goal": "建立主角", "conflict": "身份危机", "beat_outline": ["事故", "追捕"]},
            {"title": "转折"},
        ],
        chapter_count=3,
        start_chapter=10,
    )

    assert len(plan) == 3
    assert plan[0]["chapter_no"] == "0010"
    assert plan[0]["title"] == "开局"
    assert plan[0]["beat_outline"] == ["事故", "追捕"]
    assert plan[1]["chapter_no"] == "0011"
    assert plan[1]["goal"]  # default filled
    assert plan[2]["chapter_no"] == "0012"
    assert plan[2]["title"] == "第12章"


def test_load_world_spec_requires_input(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="Provide --world or --world-config"):
        module._load_world_spec("", "")

    config_path = tmp_path / "world.json"
    config_path.write_text('{"genre":"科幻","core":"环世界"}', encoding="utf-8")
    loaded = module._load_world_spec("补充规则", str(config_path))
    assert loaded["genre"] == "科幻"
    assert loaded["world_appendix"] == "补充规则"
