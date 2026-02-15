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
        / "generate_book_ab.py"
    )
    spec = importlib.util.spec_from_file_location("generate_book_ab", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_chapter_plan_fills_defaults() -> None:
    module = _load_module()
    plan = module._normalize_chapter_plan(
        raw_chapters=[
            {"title": "开局", "goal": "主线启动", "conflict": "资源不足", "beat_outline": ["事故", "追踪"]},
            {"title": "升级"},
        ],
        chapter_count=3,
        start_chapter=21,
    )

    assert len(plan) == 3
    assert plan[0]["chapter_no"] == "0021"
    assert plan[0]["title"] == "开局"
    assert plan[0]["beat_outline"] == ["事故", "追踪"]
    assert plan[1]["chapter_no"] == "0022"
    assert plan[1]["goal"]
    assert plan[2]["chapter_no"] == "0023"
    assert plan[2]["title"] == "第23章"


def test_load_world_spec_requires_input(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="Provide --world or --world-config"):
        module._load_world_spec("", "")

    config_path = tmp_path / "world.json"
    config_path.write_text('{"genre":"科幻","core":"环世界"}', encoding="utf-8")
    loaded = module._load_world_spec("补充设定", str(config_path))
    assert loaded["genre"] == "科幻"
    assert loaded["world_appendix"] == "补充设定"


def test_assert_isolation_detects_store_clashes() -> None:
    module = _load_module()
    store_a = module.MemoryStore(
        canon_db_path="/tmp/a.db",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pass="pwd",
        neo4j_database="neo4j",
        qdrant_url="http://localhost:6333",
        qdrant_collection="novel_assets_v2",
    )
    store_b = module.MemoryStore(
        canon_db_path="/tmp/a.db",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pass="pwd",
        neo4j_database="neo4j",
        qdrant_url="http://localhost:6333",
        qdrant_collection="novel_assets_v2",
    )
    with pytest.raises(ValueError, match="Isolation check failed"):
        module._assert_isolation(store_a, store_b)


def test_default_run_options_enable_isolation() -> None:
    module = _load_module()
    defaults = module.default_run_options()
    assert defaults["enforce_isolation"] is True
