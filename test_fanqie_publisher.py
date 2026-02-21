import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_script_module():
    script_path = (
        Path(__file__).resolve().parent
        / "nanobot"
        / "skills"
        / "fanqie-publisher"
        / "scripts"
        / "publish_fanqie.py"
    )
    spec = spec_from_file_location("fanqie_publish_script", script_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_title_uses_first_h1() -> None:
    mod = load_script_module()
    text = "# Main Chapter\n\n## Section\nBody"
    assert mod.extract_title(text, "fallback") == "Main Chapter"


def test_extract_title_falls_back_to_filename() -> None:
    mod = load_script_module()
    text = "## No H1 Here\nOnly content"
    assert mod.extract_title(text, "chapter_001") == "chapter_001"


def test_markdown_to_editor_text_keeps_structure() -> None:
    mod = load_script_module()
    markdown = """
## Intro

![cover](https://img.test/cover.png)

| col1 | col2 |
| --- | --- |
| v1 | v2 |

```python
print("hello")
```

---
"""
    converted = mod.markdown_to_editor_text(markdown)
    assert "[Intro]" in converted
    assert "[image: cover] (https://img.test/cover.png)" in converted
    assert "[table]" in converted
    assert "col1 | col2" in converted
    assert "[code block: python]" in converted
    assert "print(\"hello\")" in converted
    assert "----------" in converted


def test_build_payload_removes_first_h1(tmp_path: Path) -> None:
    mod = load_script_module()
    md_file = tmp_path / "chapter_001.md"
    md_file.write_text("# Ch Title\n\nBody line", encoding="utf-8")
    payload = mod.build_payload(md_file)
    assert payload.title == "Ch Title"
    assert "Ch Title" not in payload.body
    assert "Body line" in payload.body
