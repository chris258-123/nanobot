"""LLM-powered delta extractor for chapter-level structured changes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx


def _strip_markdown_code_blocks(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


class DeltaParseError(ValueError):
    """Raised when chapter delta JSON cannot be parsed after repair attempts."""

    def __init__(
        self,
        message: str,
        *,
        chapter_no: str,
        debug_log_path: str | None = None,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.chapter_no = chapter_no
        self.debug_log_path = debug_log_path
        self.last_error = last_error


class DeltaExtractorLLM:
    """Generate per-chapter delta payloads from chapter text and memory context."""

    def __init__(
        self,
        llm_config: dict[str, Any],
        timeout: float = 120.0,
        max_tokens: int | None = None,
        json_repair_attempts: int = 2,
        parse_debug_dir: str | Path | None = None,
        parse_debug_log: bool = True,
    ):
        self.llm_config = llm_config
        self.timeout = timeout
        config_max_tokens = llm_config.get("max_tokens")
        self.max_tokens = int(max_tokens or config_max_tokens or 4096)
        self.json_repair_attempts = max(int(json_repair_attempts or 0), 0)
        self.parse_debug_log = bool(parse_debug_log)
        self.parse_debug_dir = Path(parse_debug_dir).expanduser() if parse_debug_dir else None

    @staticmethod
    def empty_delta() -> dict[str, list]:
        return {
            "entities_new": [],
            "fact_changes": [],
            "relations_delta": [],
            "events": [],
            "address_changes": [],
            "hooks": [],
            "payoffs": [],
            "knows_updates": [],
        }

    def extract(
        self,
        chapter_no: str,
        chapter_text: str,
        chunks: list[dict[str, Any]],
        assets: dict[str, Any],
        prev_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(
            chapter_no=chapter_no,
            chapter_text=chapter_text,
            chunks=chunks,
            assets=assets,
            prev_context=prev_context or {},
        )
        response = self._call_llm(prompt)
        cleaned = _strip_markdown_code_blocks(response)
        parsed = self._parse_json(cleaned, chapter_no=chapter_no, raw_response=response)
        return self._normalize_output(parsed)

    @staticmethod
    def _make_json_candidates(content: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = [("raw", content)]
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", content)
        if sanitized != content:
            candidates.append(("strip_control_chars", sanitized))

        working = sanitized
        start = working.find("{")
        end = working.rfind("}")
        if start != -1 and end > start:
            sliced = working[start : end + 1]
            if sliced != working:
                candidates.append(("slice_outer_object", sliced))
                working = sliced

        # Remove trailing commas before ] or }, which are common in model output.
        repaired = re.sub(r",\s*([}\]])", r"\1", working)
        if repaired != working:
            candidates.append(("remove_trailing_commas", repaired))

        unique: list[tuple[str, str]] = []
        seen: set[str] = set()
        for label, text in candidates:
            if text in seen:
                continue
            seen.add(text)
            unique.append((label, text))
        return unique

    def _parse_json(self, content: str, *, chapter_no: str, raw_response: str) -> dict[str, Any]:
        """Parse JSON with layered repairs and optional LLM JSON repair rounds."""
        parse_attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None

        def _try_candidates(text: str, source: str) -> dict[str, Any] | None:
            nonlocal last_error
            for stage, candidate in self._make_json_candidates(text):
                try:
                    parsed = json.loads(candidate)
                    if not isinstance(parsed, dict):
                        raise ValueError("Expected JSON object")
                    return parsed
                except Exception as exc:  # pragma: no cover - small branch wrapper
                    last_error = exc
                    parse_attempts.append(
                        {
                            "source": source,
                            "stage": stage,
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                        }
                    )
            return None

        parsed = _try_candidates(content, "initial")
        if parsed is not None:
            return parsed

        llm_repairs: list[dict[str, Any]] = []
        repair_input = content
        for round_idx in range(1, self.json_repair_attempts + 1):
            repair_result = self._repair_json_with_llm(repair_input)
            llm_repairs.append(
                {
                    "round": round_idx,
                    "input_sha256": hashlib.sha256(repair_input.encode("utf-8", errors="ignore")).hexdigest(),
                    "output_sha256": hashlib.sha256(repair_result.encode("utf-8", errors="ignore")).hexdigest(),
                }
            )
            parsed = _try_candidates(repair_result, f"llm_repair_round_{round_idx}")
            if parsed is not None:
                return parsed
            repair_input = repair_result

        debug_log_path = self._write_parse_debug_log(
            chapter_no=chapter_no,
            raw_response=raw_response,
            cleaned_content=content,
            parse_attempts=parse_attempts,
            llm_repairs=llm_repairs,
            last_error=last_error,
        )
        error_suffix = f"; debug={debug_log_path}" if debug_log_path else ""
        raise DeltaParseError(
            f"Failed to parse delta JSON for chapter {chapter_no}: {last_error}{error_suffix}",
            chapter_no=chapter_no,
            debug_log_path=debug_log_path,
            last_error=last_error,
        )

    def _repair_json_with_llm(self, invalid_json: str) -> str:
        """Ask the same model to rewrite malformed JSON into a valid JSON object."""
        snippet = invalid_json.strip()
        if len(snippet) > 20000:
            snippet = snippet[:20000]
        prompt = (
            "你是 JSON 修复器。把下面内容修复为合法 JSON 对象。\n"
            "要求：\n"
            "1) 只能输出 JSON 对象本体，不要解释。\n"
            "2) 保持原有字段语义，不要新增无关字段。\n"
            "3) 允许删除明显损坏且无法修复的片段。\n\n"
            "待修复内容：\n"
            f"{snippet}"
        )
        repaired = self._call_llm(
            prompt,
            temperature=0.0,
            max_tokens=min(max(self.max_tokens, 2048), 8192),
        )
        return _strip_markdown_code_blocks(repaired)

    def _write_parse_debug_log(
        self,
        *,
        chapter_no: str,
        raw_response: str,
        cleaned_content: str,
        parse_attempts: list[dict[str, Any]],
        llm_repairs: list[dict[str, Any]],
        last_error: Exception | None,
    ) -> str | None:
        if not self.parse_debug_log or self.parse_debug_dir is None:
            return None
        try:
            self.parse_debug_dir.mkdir(parents=True, exist_ok=True)
            path = self.parse_debug_dir / f"{chapter_no}_delta_parse_failure.json"

            def _tail(text: str, size: int = 800) -> str:
                return text[-size:] if len(text) > size else text

            payload = {
                "chapter_no": chapter_no,
                "error_type": last_error.__class__.__name__ if last_error else None,
                "error": str(last_error) if last_error else "unknown parse failure",
                "error_line": getattr(last_error, "lineno", None),
                "error_col": getattr(last_error, "colno", None),
                "error_pos": getattr(last_error, "pos", None),
                "raw_sha256": hashlib.sha256(raw_response.encode("utf-8", errors="ignore")).hexdigest(),
                "cleaned_sha256": hashlib.sha256(cleaned_content.encode("utf-8", errors="ignore")).hexdigest(),
                "raw_preview_head": raw_response[:800],
                "raw_preview_tail": _tail(raw_response),
                "cleaned_preview_head": cleaned_content[:800],
                "cleaned_preview_tail": _tail(cleaned_content),
                "parse_attempts": parse_attempts,
                "llm_repairs": llm_repairs,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(path)
        except Exception:
            return None

    def _call_llm(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        requested_max_tokens = int(max_tokens or self.max_tokens)

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

        def _restore_proxy_env(backup: dict[str, str]) -> None:
            for key, value in backup.items():
                os.environ[key] = value

        # Mode A: same as 8-element extractor (llm_config.json custom endpoint)
        if self.llm_config.get("type") == "custom":
            proxy_backup = _pop_proxy_env()
            try:
                response = httpx.post(
                    self.llm_config["url"],
                    json={
                        "model": self.llm_config["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": requested_max_tokens,
                    },
                    headers={"Authorization": f"Bearer {self.llm_config['api_key']}"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            finally:
                _restore_proxy_env(proxy_backup)

        # Mode B: providers + model config (anthropic/openai style)
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
                        max_tokens=requested_max_tokens,
                        temperature=temperature,
                    ),
                    timeout=self.timeout,
                )

            proxy_backup = _pop_proxy_env()
            try:
                try:
                    asyncio.get_running_loop()
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        response = pool.submit(lambda: asyncio.run(_chat_with_timeout())).result()
                except RuntimeError:
                    response = asyncio.run(_chat_with_timeout())
            finally:
                _restore_proxy_env(proxy_backup)
            return response.content or ""

        raise ValueError("Unsupported llm_config: expected {type: custom} or {providers, model}")

    def _build_prompt(
        self,
        chapter_no: str,
        chapter_text: str,
        chunks: list[dict[str, Any]],
        assets: dict[str, Any],
        prev_context: dict[str, Any],
    ) -> str:
        chunk_lines = []
        for chunk in chunks:
            text = (chunk.get("text") or "").replace("\n", " ").strip()
            if len(text) > 180:
                text = f"{text[:180]}..."
            chunk_lines.append(f"- {chunk.get('chunk_id')}: {text}")
        chunk_listing = "\n".join(chunk_lines)

        output_schema = {
            "entities_new": [
                {
                    "name": "str",
                    "type": "character|location|item|concept|organization|rule",
                    "aliases": ["str"],
                    "needs_normalization": False,
                    "evidence_chunk_id": "ch0001#c00",
                }
            ],
            "fact_changes": [
                {
                    "subject": "entity_id_or_name",
                    "predicate": "status|owner|location|rule|trait|goal|secret",
                    "value": "any_json_value",
                    "op": "INSERT|UPDATE|DELETE",
                    "tier": "HARD_RULE|HARD_STATE|SOFT_NOTE",
                    "status": "confirmed|implied|rumor",
                    "confidence": 0.0,
                    "valid_from": chapter_no,
                    "valid_to": None,
                    "evidence_chunk_id": "ch0001#c00",
                    "needs_normalization": False,
                }
            ],
            "relations_delta": [
                {
                    "from": "entity_id_or_name",
                    "to": "entity_id_or_name",
                    "kind": "ALLY|ENEMY|FAMILY|MENTOR|ROMANTIC|HIERARCHY|COLLEAGUE|RIVAL|CO_PARTICIPANT|ASSOCIATE",
                    "op": "INSERT|UPDATE|DELETE",
                    "status": "confirmed|implied|rumor",
                    "confidence": 0.0,
                    "valid_from": chapter_no,
                    "valid_to": None,
                    "evidence_chunk_id": "ch0001#c00",
                    "needs_normalization": False,
                }
            ],
            "events": [
                {
                    "event_id": f"{chapter_no}_evt_00",
                    "type": "plot_beat",
                    "summary": "str",
                    "participants": ["entity_id_or_name"],
                    "location": "entity_id_or_name_or_null",
                    "effects": ["str"],
                    "evidence_chunk_id": "ch0001#c00",
                    "needs_normalization": False,
                }
            ],
            "address_changes": [
                {
                    "from": "entity_id_or_name",
                    "to": "entity_id_or_name",
                    "name_used": "str",
                    "context": "formal|informal|public|private",
                    "valid_from": chapter_no,
                    "valid_to": None,
                    "evidence_chunk_id": "ch0001#c00",
                    "needs_normalization": False,
                }
            ],
            "hooks": [
                {
                    "thread_name": "str",
                    "summary": "str",
                    "priority": 1,
                    "evidence_chunk_id": "ch0001#c00",
                }
            ],
            "payoffs": [
                {
                    "thread_name": "str",
                    "summary": "str",
                    "evidence_chunk_id": "ch0001#c00",
                }
            ],
            "knows_updates": [
                {
                    "subject": "entity_id_or_name",
                    "object": "entity_id_or_name_or_fact_key",
                    "mode": "know|believe|suspect",
                    "op": "INSERT|UPDATE|DELETE",
                    "evidence_chunk_id": "ch0001#c00",
                    "needs_normalization": False,
                }
            ],
        }

        return (
            "You are extracting chapter delta changes for a long-form novel memory engine.\n"
            "Return JSON only. No markdown.\n\n"
            "Rules:\n"
            "1) Output only NEW or CHANGED information for this chapter.\n"
            "2) Every item MUST include evidence_chunk_id.\n"
            "3) Prefer entity_id if known from previous context; otherwise use raw name and set needs_normalization=true.\n"
            "4) Keep output compact and factual.\n"
            "5) For events, output 1 to 5 high-signal events only.\n"
            "6) Natural-language values should use Simplified Chinese whenever possible.\n\n"
            f"Chapter no: {chapter_no}\n\n"
            "Available chunk ids:\n"
            f"{chunk_listing}\n\n"
            "Previous context snapshot:\n"
            f"{json.dumps(prev_context, ensure_ascii=False)}\n\n"
            "Assets JSON:\n"
            f"{json.dumps(assets, ensure_ascii=False)}\n\n"
            "Chapter text:\n"
            f"{chapter_text}\n\n"
            "Output schema (types and keys must match):\n"
            f"{json.dumps(output_schema, ensure_ascii=False)}"
        )

    def _normalize_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self.empty_delta()
        for key in normalized:
            value = payload.get(key, [])
            normalized[key] = value if isinstance(value, list) else []
        return normalized
