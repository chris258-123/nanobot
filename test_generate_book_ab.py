import argparse
import importlib.util
import json
import sqlite3
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
    assert defaults["log_injections"] is True
    assert str(defaults["log_dir"]).endswith("/logs")


def test_load_all_done_chapters(tmp_path: Path) -> None:
    module = _load_module()
    db_path = tmp_path / "canon.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE commit_log (
                commit_id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL,
                chapter_no TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO commit_log (commit_id, book_id, chapter_no, status) VALUES (?, ?, ?, ?)",
            [
                ("c1", "book_b", "0001", "ALL_DONE"),
                ("c2", "book_b", "0002", "FAILED"),
                ("c3", "book_b", "0003", "ALL_DONE"),
                ("c4", "book_a", "0001", "ALL_DONE"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    done = module._load_all_done_chapters(str(db_path), "book_b")
    assert done == {"0001", "0003"}


def test_load_run_report_summaries(tmp_path: Path) -> None:
    module = _load_module()
    report_path = tmp_path / "book_b_run_report.json"
    report_path.write_text(
        json.dumps(
            {
                "items": [
                    {"chapter_no": "0001", "summary": "第一章摘要"},
                    {"chapter_no": "0002", "summary": "第二章摘要"},
                    {"chapter_no": "0003", "status": "failed"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summaries = module._load_run_report_summaries(tmp_path, "book_b")
    assert summaries == {"0001": "第一章摘要", "0002": "第二章摘要"}


def test_injection_logger_writes_payloads(tmp_path: Path) -> None:
    module = _load_module()
    logger = module.InjectionLogger(tmp_path, "book_b")
    written = logger.write("chapters/0001_pre_generation_injection.json", {"foo": "bar"})
    assert written.exists()
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["foo"] == "bar"
    assert logger.run_dir.parent.name == "generate_book_ab"


def test_injection_logger_can_write_clean_and_raw_payloads(tmp_path: Path) -> None:
    module = _load_module()
    logger = module.InjectionLogger(tmp_path, "book_b")
    clean_path, raw_path = logger.write_with_raw(
        "chapters/0001_pre_generation_injection.json",
        {"mode": "clean"},
        raw_payload={"mode": "raw"},
    )
    assert clean_path.exists()
    assert raw_path is not None and raw_path.exists()
    assert clean_path.name.endswith(".json")
    assert raw_path.name.endswith(".raw.json")
    assert json.loads(clean_path.read_text(encoding="utf-8"))["mode"] == "clean"
    assert json.loads(raw_path.read_text(encoding="utf-8"))["mode"] == "raw"


def test_is_delta_json_parse_error_detects_jsondecodeerror() -> None:
    module = _load_module()
    exc = json.JSONDecodeError("bad json", "{", 1)
    assert module._is_delta_json_parse_error(exc) is True


def test_classify_runtime_error_marks_delta_parse() -> None:
    module = _load_module()
    delta_parse_error = type("DeltaParseError", (Exception,), {})
    exc = delta_parse_error("Failed to parse delta JSON for chapter 0001")
    assert module._classify_runtime_error(exc) == "DELTA_JSON_PARSE_FAILED"


def test_redact_store_for_log_hides_credentials() -> None:
    module = _load_module()
    store = module.MemoryStore(
        canon_db_path="/tmp/a.db",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pass="secret-pass",
        neo4j_database="neo4j",
        qdrant_url="http://localhost:6333",
        qdrant_collection="book_a",
        qdrant_api_key="secret-key",
    )
    redacted = module._redact_store_for_log(store)
    assert "neo4j_pass" not in redacted
    assert "qdrant_api_key" not in redacted
    assert redacted["qdrant_collection"] == "book_a"


def test_template_reader_chapter_pack_uses_query_specific_selection() -> None:
    module = _load_module()
    store = module.MemoryStore(
        canon_db_path="/tmp/a.db",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pass="pwd",
        neo4j_database="neo4j",
        qdrant_url="http://localhost:6333",
        qdrant_collection="book_a",
    )
    reader = module.TemplateMemoryReader(store)

    pool = [
        {"text": "都市调查与线索拼接", "chapter": "第1章", "metadata": {}, "chapter_sort_key": (1, "第1章")},
        {"text": "飞船追逐与太空战斗", "chapter": "第50章", "metadata": {}, "chapter_sort_key": (50, "第50章")},
        {"text": "法庭对峙与政治博弈", "chapter": "第120章", "metadata": {}, "chapter_sort_key": (120, "第120章")},
        {"text": "战斗后的心理修复", "chapter": "第200章", "metadata": {}, "chapter_sort_key": (200, "第200章")},
    ]
    reader._asset_cache[("book_a", "plot_beat")] = pool
    reader._asset_cache[("book_a", "style")] = pool
    reader._asset_cache[("book_a", "conflict")] = pool

    profile = reader.build_book_template_profile("book_a", top_k=3)
    chapter_pack = reader.build_chapter_template_pack(
        "book_a",
        {"goal": "飞船战斗", "conflict": "太空追逐", "beat_outline": ["交火", "脱离"]},
        top_k=3,
    )

    assert len(profile["plot_templates"]) == 3
    assert 1 <= len(chapter_pack["plot_templates"]) <= 3
    assert chapter_pack["plot_templates"][0]["text"] == "飞船追逐与太空战斗"
    assert chapter_pack["plot_templates"] != profile["plot_templates"]


def test_template_reader_sanitize_pack_replaces_entity_ids_with_names(tmp_path: Path) -> None:
    module = _load_module()
    canon_db_path = tmp_path / "canon.db"
    conn = sqlite3.connect(canon_db_path)
    try:
        conn.execute(
            """
            CREATE TABLE entity_registry (
                entity_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO entity_registry (entity_id, canonical_name) VALUES (?, ?)",
            [
                ("character_a1", "李雷"),
                ("character_b2", "韩梅梅"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    store = module.MemoryStore(
        canon_db_path=str(canon_db_path),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pass="pwd",
        neo4j_database="neo4j",
        qdrant_url="http://localhost:6333",
        qdrant_collection="book_a",
    )
    reader = module.TemplateMemoryReader(store)
    raw_pack = {
        "plot_templates": [
            {
                "text": "角色关系变化",
                "chapter": "0004",
                "metadata": {
                    "subject_id": "character_a1",
                    "event": "两人达成同盟",
                    "relation": {"from_id": "character_a1", "to_id": "character_b2", "kind": "ALLY"},
                    "evidence_chunk_id": "0004#c00",
                },
                "retrieval": "semantic_search",
                "score": 0.91,
                "chapter_sort_key": (4, "0004"),
            }
        ],
        "fact_templates": [
            {
                "text": "未知角色信息",
                "chapter": "0005",
                "metadata": {"subject_id": "character_unknown", "fact": "身份待揭晓"},
            }
        ],
    }

    sanitized = reader.sanitize_template_payload_for_prompt(raw_pack)
    payload_text = json.dumps(sanitized, ensure_ascii=False)
    assert "subject_id" not in payload_text
    assert "from_id" not in payload_text
    assert "to_id" not in payload_text
    assert "evidence_chunk_id" not in payload_text
    assert sanitized["plot_templates"][0]["metadata"]["subject_name"] == "李雷"
    assert sanitized["plot_templates"][0]["metadata"]["relation"]["from_name"] == "李雷"
    assert sanitized["plot_templates"][0]["metadata"]["relation"]["to_name"] == "韩梅梅"
    assert sanitized["plot_templates"][0]["retrieval"] == "semantic_search"
    assert sanitized["fact_templates"][0]["metadata"]["subject_name"] == "character_unknown"
    assert "chapter_sort_key" not in sanitized["plot_templates"][0]


def test_target_reader_sanitize_hard_pack_removes_noise_ids() -> None:
    module = _load_module()
    raw_pack = {
        "hard_rules": [
            {
                "chapter_no": "0003",
                "subject": "李雷",
                "predicate": "立场",
                "value": {"actor_id": "character_a1", "actor_name": "李雷"},
                "evidence": "0003#c08",
                "evidence_chunk_id": "0003#c08",
            }
        ],
        "prev_context": {
            "character_state": [{"name": "李雷", "state": {"mood": "紧张"}, "updated_chapter": "0003"}],
            "recent_relations": [
                {
                    "from_name": "李雷",
                    "to_name": "韩梅梅",
                    "kind": "ALLY",
                    "status": "active",
                    "evidence_chunk_id": "0003#c06",
                }
            ],
            "open_threads": [],
        },
    }

    cleaned = module.TargetMemoryReader.sanitize_hard_pack_for_prompt(raw_pack)
    payload_text = json.dumps(cleaned, ensure_ascii=False)
    assert "evidence_chunk_id" not in payload_text
    assert "evidence" not in payload_text
    assert "actor_id" not in payload_text
    assert cleaned["hard_rules"][0]["value"]["actor_name"] == "李雷"


def test_commit_memory_requires_physical_store_isolation(tmp_path: Path) -> None:
    module = _load_module()
    output_dir = tmp_path / "out"
    options = module.default_run_options()
    options.update(
        {
            "target_book_id": "book_b",
            "template_book_id": "book_a",
            "world": "测试世界观",
            "chapter_count": 1,
            "output_dir": str(output_dir),
            "llm_config_obj": {
                "providers": {
                    "anthropic": {
                        "apiKey": "test-key",
                        "apiBase": "https://example.invalid/v1",
                        "extraHeaders": None,
                    }
                },
                "model": "test-model",
            },
            "commit_memory": True,
            "enforce_isolation": False,
            "template_canon_db_path": "/tmp/shared.db",
            "target_canon_db_path": "/tmp/shared.db",
            "template_neo4j_uri": "bolt://localhost:7687",
            "target_neo4j_uri": "bolt://localhost:7687",
            "template_neo4j_database": "neo4j",
            "target_neo4j_database": "neo4j",
            "template_qdrant_url": "http://localhost:6333",
            "target_qdrant_url": "http://localhost:6333",
            "template_qdrant_collection": "shared_collection",
            "target_qdrant_collection": "shared_collection",
        }
    )
    with pytest.raises(ValueError, match="physically separate A/B stores"):
        module.run_generation(argparse.Namespace(**options))
