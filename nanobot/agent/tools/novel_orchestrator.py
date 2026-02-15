"""Novel workflow orchestrator tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "skills" / "novel-workflow" / "scripts"


def _load_script_class(module_name: str, class_name: str):
    scripts_dir = _scripts_dir()
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


class NovelOrchestratorTool(Tool):
    """High-level novel workflow orchestration."""

    def __init__(
        self,
        qdrant_tool,
        letta_tool,
        beads_tool,
        neo4j_tool=None,
        canon_tool=None,
        integrations_config=None,
    ):
        self.qdrant = qdrant_tool
        self.letta = letta_tool
        self.beads = beads_tool
        self.neo4j_tool = neo4j_tool
        self.canon_tool = canon_tool
        self.integrations_config = integrations_config

    @property
    def name(self) -> str:
        return "novel_orchestrator"

    @property
    def description(self) -> str:
        return """Orchestrate novel workflows.

Actions:
- init_library: initialize library for a book (requires: book_id)
- generate_chapter: generate chapter via letta writer (requires: template_book_id, new_book_id, chapter_num)
- generate_book: one-click generation (supports A-read/B-write isolation) from world settings
- extract_assets: placeholder orchestration for extraction task tracking (requires: book_id, chapter_path)
- process_chapter: run ChapterProcessor for one chapter (requires: book_id, chapter_no; optional: chapter_text/chapter_path/asset_path/mode)
- assemble_context: build Context Pack v2 (requires: book_id, chapter_no)
- detect_conflicts: run Canon conflict preflight (requires: chapter_no)
"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "init_library",
                        "generate_chapter",
                        "generate_book",
                        "extract_assets",
                        "process_chapter",
                        "assemble_context",
                        "detect_conflicts",
                    ],
                },
                "book_id": {"type": "string"},
                "template_book_id": {"type": "string"},
                "new_book_id": {"type": "string"},
                "chapter_num": {"type": "integer"},
                "chapter_count": {"type": "integer"},
                "target_book_id": {"type": "string"},
                "chapter_no": {"type": "string"},
                "chapter_path": {"type": "string"},
                "chapter_text": {"type": "string"},
                "chapter_title": {"type": "string"},
                "chapter_summary": {"type": "string"},
                "world": {"type": "string"},
                "world_config": {"type": "string"},
                "output_dir": {"type": "string"},
                "reference_top_k": {"type": "integer", "default": 8},
                "consistency_policy": {
                    "type": "string",
                    "enum": ["strict_blocking", "warn_only"],
                    "default": "strict_blocking",
                },
                "enforce_isolation": {"type": "boolean", "default": True},
                "start_chapter": {"type": "integer"},
                "resume": {"type": "boolean"},
                "commit_memory": {"type": "boolean"},
                "temperature": {"type": "number"},
                "asset_path": {"type": "string"},
                "mode": {"type": "string", "enum": ["delta", "llm", "replay"], "default": "delta"},
                "delta": {"type": "object"},
                "assets": {"type": "object"},
                "llm_config": {"type": "object"},
                "llm_config_path": {"type": "string"},
                "llm_max_tokens": {"type": "integer", "default": 4096},
                "outline": {"type": "string"},
                "top_n": {"type": "integer", "default": 15},
                "recall_k": {"type": "integer", "default": 8},
                "output_path": {"type": "string"},
                "proposed_facts": {"type": "array", "items": {"type": "object"}},
                "proposed_relations": {"type": "array", "items": {"type": "object"}},
                "target_canon_db_path": {"type": "string"},
                "target_neo4j_uri": {"type": "string"},
                "target_neo4j_user": {"type": "string"},
                "target_neo4j_pass": {"type": "string"},
                "target_neo4j_database": {"type": "string"},
                "target_qdrant_url": {"type": "string"},
                "target_qdrant_collection": {"type": "string"},
                "target_qdrant_api_key": {"type": "string"},
                "template_canon_db_path": {"type": "string"},
                "template_neo4j_uri": {"type": "string"},
                "template_neo4j_user": {"type": "string"},
                "template_neo4j_pass": {"type": "string"},
                "template_neo4j_database": {"type": "string"},
                "template_qdrant_url": {"type": "string"},
                "template_qdrant_collection": {"type": "string"},
                "template_qdrant_api_key": {"type": "string"},
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        try:
            if action == "init_library":
                return await self._init_library(kwargs.get("book_id", ""))
            if action == "generate_chapter":
                return await self._generate_chapter(
                    template_book_id=kwargs.get("template_book_id", ""),
                    new_book_id=kwargs.get("new_book_id", ""),
                    chapter_num=int(kwargs.get("chapter_num", 0)),
                )
            if action == "generate_book":
                return await self._generate_book(**kwargs)
            if action == "extract_assets":
                return await self._extract_assets(
                    book_id=kwargs.get("book_id", ""),
                    chapter_path=kwargs.get("chapter_path", ""),
                )
            if action == "process_chapter":
                return await self._process_chapter(**kwargs)
            if action == "assemble_context":
                return await self._assemble_context(**kwargs)
            if action == "detect_conflicts":
                return await self._detect_conflicts(**kwargs)
            return f"Unknown action: {action}"
        except Exception as exc:  # pragma: no cover - tool surface
            return f"Error: {exc}"

    async def _init_library(self, book_id: str) -> str:
        if not book_id:
            return "Error: book_id is required"

        task_result = await self.beads.execute(
            action="add",
            title=f"Build library for {book_id}",
            description=f"Extract and index assets for {book_id}",
        )
        collection_result = await self.qdrant.execute(action="info")
        if "Error" in collection_result:
            collection_result = await self.qdrant.execute(action="create_collection")

        return f"Library initialized for {book_id}\n{task_result}\n{collection_result}"

    async def _generate_chapter(self, template_book_id: str, new_book_id: str, chapter_num: int) -> str:
        if not template_book_id or not new_book_id or chapter_num <= 0:
            return "Error: template_book_id, new_book_id and positive chapter_num are required"

        search_result = await self.qdrant.execute(
            action="scroll",
            book_id=template_book_id,
            asset_type="plot_beat",
            limit=20,
        )

        agents_list = await self.letta.execute(action="list_agents")
        if "writer_agent" not in agents_list:
            await self.letta.execute(action="create_agent", agent_type="writer")

        message = (
            f"Generate chapter {chapter_num} for {new_book_id}.\n\n"
            f"Template book context:\n{search_result}\n\n"
            "Write an engaging chapter that follows similar narrative structure."
        )
        writer_response = await self.letta.execute(
            action="send_message",
            agent_id="writer_agent",
            message=message,
        )
        return f"Chapter {chapter_num} generated:\n{writer_response}"

    async def _extract_assets(self, book_id: str, chapter_path: str) -> str:
        if not book_id or not chapter_path:
            return "Error: book_id and chapter_path are required"
        task = await self.beads.execute(
            action="add",
            title=f"Extract assets {book_id}",
            description=f"Run extractor for {chapter_path}",
        )
        return (
            f"Extraction task queued for {book_id}: {chapter_path}\n"
            f"{task}\n"
            "Use scripts/asset_extractor_parallel.py for batch extraction."
        )

    async def _generate_book(self, **kwargs: Any) -> str:
        target_book_id = kwargs.get("target_book_id") or kwargs.get("book_id", "")
        chapter_count = int(kwargs.get("chapter_count", 0))
        output_dir = kwargs.get("output_dir", "")
        if not target_book_id or chapter_count <= 0 or not output_dir:
            return "Error: target_book_id/book_id, chapter_count (>0), and output_dir are required"
        if not kwargs.get("world") and not kwargs.get("world_config"):
            return "Error: world or world_config is required"

        memory_cfg = self._get_memory_config()
        llm_config = kwargs.get("llm_config")
        llm_config_path = kwargs.get("llm_config_path", "")
        if llm_config is None and llm_config_path:
            llm_config = self._read_json_file(llm_config_path)
        if llm_config is None:
            return "Error: llm_config or llm_config_path is required"

        runner = _load_script_class("generate_book_ab", "run_generation_with_options")
        result = runner(
            book_id=kwargs.get("book_id", ""),
            target_book_id=target_book_id,
            template_book_id=kwargs.get("template_book_id", ""),
            world=kwargs.get("world", ""),
            world_config=kwargs.get("world_config", ""),
            chapter_count=chapter_count,
            start_chapter=int(kwargs.get("start_chapter", 1)),
            output_dir=output_dir,
            llm_config_obj=llm_config,
            llm_config="",
            llm_config_path="",
            temperature=float(kwargs.get("temperature", 0.8)),
            plan_max_tokens=int(kwargs.get("plan_max_tokens", 4096)),
            chapter_max_tokens=int(kwargs.get("chapter_max_tokens", 4096)),
            chapter_min_chars=int(kwargs.get("chapter_min_chars", 2800)),
            chapter_max_chars=int(kwargs.get("chapter_max_chars", 4200)),
            llm_max_retries=int(kwargs.get("llm_max_retries", 3)),
            llm_retry_backoff=float(kwargs.get("llm_retry_backoff", 3.0)),
            llm_backoff_factor=float(kwargs.get("llm_backoff_factor", 2.0)),
            llm_backoff_max=float(kwargs.get("llm_backoff_max", 60.0)),
            llm_retry_jitter=float(kwargs.get("llm_retry_jitter", 0.5)),
            reference_top_k=int(kwargs.get("reference_top_k", 8)),
            consistency_policy=kwargs.get("consistency_policy", "strict_blocking"),
            enforce_isolation=bool(kwargs.get("enforce_isolation", True)),
            resume=bool(kwargs.get("resume", False)),
            commit_memory=bool(kwargs.get("commit_memory", False)),
            legacy_canon_db_path=memory_cfg["canon_db_path"],
            legacy_neo4j_uri=memory_cfg["neo4j_uri"],
            legacy_neo4j_user=memory_cfg["neo4j_user"],
            legacy_neo4j_pass=memory_cfg["neo4j_pass"],
            legacy_neo4j_database=memory_cfg["neo4j_database"],
            legacy_qdrant_url=memory_cfg["qdrant_url"],
            legacy_qdrant_collection=memory_cfg["qdrant_collection"],
            legacy_qdrant_api_key="",
            target_canon_db_path=kwargs.get("target_canon_db_path", ""),
            target_neo4j_uri=kwargs.get("target_neo4j_uri", ""),
            target_neo4j_user=kwargs.get("target_neo4j_user", ""),
            target_neo4j_pass=kwargs.get("target_neo4j_pass", ""),
            target_neo4j_database=kwargs.get("target_neo4j_database", ""),
            target_qdrant_url=kwargs.get("target_qdrant_url", ""),
            target_qdrant_collection=kwargs.get("target_qdrant_collection", ""),
            target_qdrant_api_key=kwargs.get("target_qdrant_api_key", ""),
            template_canon_db_path=kwargs.get("template_canon_db_path", ""),
            template_neo4j_uri=kwargs.get("template_neo4j_uri", ""),
            template_neo4j_user=kwargs.get("template_neo4j_user", ""),
            template_neo4j_pass=kwargs.get("template_neo4j_pass", ""),
            template_neo4j_database=kwargs.get("template_neo4j_database", ""),
            template_qdrant_url=kwargs.get("template_qdrant_url", ""),
            template_qdrant_collection=kwargs.get("template_qdrant_collection", ""),
            template_qdrant_api_key=kwargs.get("template_qdrant_api_key", ""),
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _read_json_file(self, path: str) -> dict[str, Any]:
        content = Path(path).expanduser().read_text(encoding="utf-8")
        loaded = json.loads(content)
        if not isinstance(loaded, dict):
            raise ValueError(f"JSON content is not an object: {path}")
        return loaded

    def _get_memory_config(self) -> dict[str, Any]:
        cfg = self.integrations_config
        return {
            "neo4j_uri": getattr(getattr(cfg, "neo4j", None), "uri", "bolt://localhost:7687"),
            "neo4j_user": getattr(getattr(cfg, "neo4j", None), "username", "neo4j"),
            "neo4j_pass": getattr(getattr(cfg, "neo4j", None), "password", ""),
            "neo4j_database": getattr(getattr(cfg, "neo4j", None), "database", "neo4j"),
            "canon_db_path": str(Path(getattr(getattr(cfg, "canon_db", None), "db_path", "~/.nanobot/workspace/canon_v2.db")).expanduser()),
            "qdrant_url": getattr(getattr(cfg, "qdrant", None), "url", ""),
            "qdrant_collection": getattr(getattr(cfg, "qdrant", None), "collection_name", "novel_assets"),
        }

    async def _process_chapter(self, **kwargs: Any) -> str:
        book_id = kwargs.get("book_id", "")
        chapter_no = kwargs.get("chapter_no", "")
        mode = kwargs.get("mode", "delta")

        if not book_id or not chapter_no:
            return "Error: book_id and chapter_no are required"

        memory_cfg = self._get_memory_config()
        llm_config = kwargs.get("llm_config")
        llm_config_path = kwargs.get("llm_config_path", "")
        if llm_config is None and llm_config_path:
            llm_config = self._read_json_file(llm_config_path)

        chapter_processor_cls = _load_script_class("chapter_processor", "ChapterProcessor")
        processor = chapter_processor_cls(
            neo4j_uri=memory_cfg["neo4j_uri"],
            neo4j_user=memory_cfg["neo4j_user"],
            neo4j_pass=memory_cfg["neo4j_pass"],
            canon_db_path=memory_cfg["canon_db_path"],
            qdrant_url=memory_cfg["qdrant_url"],
            llm_config=llm_config,
            llm_max_tokens=int(kwargs.get("llm_max_tokens", 4096)),
        )

        try:
            if mode == "replay":
                commit_id = kwargs.get("commit_id", "")
                if not commit_id:
                    return "Error: commit_id is required for replay mode"
                result = processor.replay_commit(commit_id)
                return json.dumps(result, ensure_ascii=False, indent=2)

            chapter_text = kwargs.get("chapter_text", "")
            chapter_path = kwargs.get("chapter_path", "")
            if not chapter_text and chapter_path:
                chapter_text = Path(chapter_path).expanduser().read_text(encoding="utf-8")

            assets = kwargs.get("assets")
            asset_path = kwargs.get("asset_path", "")
            if assets is None and asset_path:
                assets = self._read_json_file(asset_path)

            delta = kwargs.get("delta")

            result = processor.process_chapter(
                book_id=book_id,
                chapter_no=chapter_no,
                delta=delta,
                chapter_title=kwargs.get("chapter_title", ""),
                chapter_summary=kwargs.get("chapter_summary", ""),
                chapter_text=chapter_text,
                assets=assets,
                mode=mode,
            )
            return json.dumps(result, ensure_ascii=False, indent=2)
        finally:
            processor.close()

    async def _assemble_context(self, **kwargs: Any) -> str:
        book_id = kwargs.get("book_id", "")
        chapter_no = kwargs.get("chapter_no", "")
        if not book_id or not chapter_no:
            return "Error: book_id and chapter_no are required"

        memory_cfg = self._get_memory_config()
        context_assembler_cls = _load_script_class("context_assembler", "ContextAssembler")
        assembler = context_assembler_cls(
            canon_db_path=memory_cfg["canon_db_path"],
            neo4j_uri=memory_cfg["neo4j_uri"],
            neo4j_user=memory_cfg["neo4j_user"],
            neo4j_pass=memory_cfg["neo4j_pass"],
            qdrant_url=memory_cfg["qdrant_url"],
            qdrant_collection=memory_cfg["qdrant_collection"],
        )

        try:
            pack = assembler.assemble_context_pack(
                book_id=book_id,
                chapter_no=chapter_no,
                outline=kwargs.get("outline", ""),
                top_n=int(kwargs.get("top_n", 15)),
                recall_k=int(kwargs.get("recall_k", 8)),
            )
        finally:
            assembler.close()

        output_path = kwargs.get("output_path", "")
        if output_path:
            output = Path(output_path).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"Context pack saved to {output}"
        return json.dumps(pack, ensure_ascii=False, indent=2)

    async def _detect_conflicts(self, **kwargs: Any) -> str:
        chapter_no = kwargs.get("chapter_no", "")
        if not chapter_no:
            return "Error: chapter_no is required"

        proposed_facts = kwargs.get("proposed_facts") or []
        proposed_relations = kwargs.get("proposed_relations") or []

        if self.canon_tool is not None:
            return await self.canon_tool.execute(
                action="detect_conflicts",
                chapter_no=chapter_no,
                proposed_facts=proposed_facts,
                proposed_relations=proposed_relations,
            )

        memory_cfg = self._get_memory_config()
        canon_db_cls = _load_script_class("canon_db_v2", "CanonDBV2")
        db = canon_db_cls(memory_cfg["canon_db_path"])
        try:
            result = db.detect_conflicts(chapter_no, proposed_facts, proposed_relations)
            return json.dumps(result, ensure_ascii=False, indent=2)
        finally:
            db.close()
