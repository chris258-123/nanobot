import importlib.util
import json
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
        / "delta_extractor_llm.py"
    )
    spec = importlib.util.spec_from_file_location("delta_extractor_llm", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_json_repairs_trailing_commas() -> None:
    module = _load_module()
    extractor = module.DeltaExtractorLLM(
        {"type": "custom", "url": "http://localhost", "model": "dummy", "api_key": "x"},
        json_repair_attempts=0,
    )
    parsed = extractor._parse_json(
        '{"entities_new": [], "fact_changes": [],}',
        chapter_no="0001",
        raw_response='{"entities_new": [], "fact_changes": [],}',
    )
    assert parsed["entities_new"] == []
    assert parsed["fact_changes"] == []


def test_parse_json_uses_llm_repair_round() -> None:
    module = _load_module()

    class DummyExtractor(module.DeltaExtractorLLM):
        def __init__(self):
            super().__init__(
                {"type": "custom", "url": "http://localhost", "model": "dummy", "api_key": "x"},
                json_repair_attempts=1,
            )
            self.calls = 0

        def _call_llm(self, prompt: str, *, temperature: float = 0.1, max_tokens: int | None = None) -> str:
            self.calls += 1
            return '{"entities_new": [], "fact_changes": []}'

    extractor = DummyExtractor()
    parsed = extractor._parse_json(
        '{"entities_new": [] "fact_changes": []}',
        chapter_no="0002",
        raw_response='{"entities_new": [] "fact_changes": []}',
    )
    assert parsed["entities_new"] == []
    assert extractor.calls == 1


def test_parse_json_failure_writes_debug_log(tmp_path: Path) -> None:
    module = _load_module()

    class DummyExtractor(module.DeltaExtractorLLM):
        def _call_llm(self, prompt: str, *, temperature: float = 0.1, max_tokens: int | None = None) -> str:
            return "still not json"

    extractor = DummyExtractor(
        {"type": "custom", "url": "http://localhost", "model": "dummy", "api_key": "x"},
        json_repair_attempts=1,
        parse_debug_dir=tmp_path,
        parse_debug_log=True,
    )

    with pytest.raises(module.DeltaParseError) as exc_info:
        extractor._parse_json(
            "bad json content",
            chapter_no="0099",
            raw_response="bad json content",
        )

    debug_path = Path(exc_info.value.debug_log_path or "")
    assert debug_path.exists()
    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert payload["chapter_no"] == "0099"
    assert payload["parse_attempts"]
