"""LLM-powered delta extractor for chapter-level structured changes."""

from __future__ import annotations

import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
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


class DeltaExtractorLLM:
    """Generate per-chapter delta payloads from chapter text and memory context."""

    def __init__(
        self,
        llm_config: dict[str, Any],
        timeout: float = 120.0,
        max_tokens: int | None = None,
    ):
        self.llm_config = llm_config
        self.timeout = timeout
        config_max_tokens = llm_config.get("max_tokens")
        self.max_tokens = int(max_tokens or config_max_tokens or 4096)

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
        parsed = self._parse_json(cleaned)
        return self._normalize_output(parsed)

    def _parse_json(self, content: str) -> dict[str, Any]:
        """Parse JSON with small repairs for common LLM formatting glitches."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            sliced = content[start : end + 1]
            try:
                return json.loads(sliced)
            except json.JSONDecodeError:
                content = sliced

        # Remove trailing commas before ] or }, which are common in model output.
        repaired = re.sub(r",\s*([}\]])", r"\1", content)
        return json.loads(repaired)

    def _call_llm(self, prompt: str) -> str:
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
                        "temperature": 0.1,
                        "max_tokens": self.max_tokens,
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
                        max_tokens=self.max_tokens,
                        temperature=0.1,
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
