# Repository Guidelines

## Project Structure & Module Organization
- `nanobot/` is the core Python package.
  - `agent/`: execution loop, context, and tool orchestration
  - `providers/`: LLM provider integrations
  - `channels/`: Telegram/Discord/WhatsApp/Feishu adapters
  - `cli/`: Typer CLI entrypoints (`nanobot ...`)
  - `config/`: settings schema and loading
- `tests/` is the primary automated test suite (configured in `pyproject.toml`).
- Top-level `test_*.py` files are for focused or experimental checks; keep stable regressions under `tests/`.
- `bridge/` contains integration bridge components; `scripts/` contains utility workflows.

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` — install package plus dev tooling.
- `nanobot onboard` — initialize local runtime config in `~/.nanobot/`.
- `nanobot agent -m "Hello"` — run a quick one-shot local agent check.
- `pytest` — run default tests from `tests/`.
- `pytest tests/test_tool_validation.py` — run one test module.
- `ruff check nanobot tests` — lint.
- `ruff format nanobot tests` — format code.
- `docker build -t nanobot .` — build local container image.

## Coding Style & Naming Conventions
- Target Python 3.11+ and follow Ruff settings in `pyproject.toml`.
- Use 4-space indentation and explicit type hints for new/changed public APIs.
- Keep modules focused and composable.
- Naming: `snake_case` for variables/functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` (`asyncio_mode = auto`).
- Test files: `test_*.py`; test functions: `test_*`.
- For new behavior, add both:
  - success-path coverage
  - edge/failure-path coverage
- Run `pytest` and `ruff check` before opening a PR.

## Commit & Pull Request Guidelines
- Follow Conventional Commits used in history (for example: `feat:`, `fix:`, `docs:`, `refactor:`).
- Keep commits scoped to one logical change.
- PRs should include purpose, key changes, validation commands run, and linked issues.
- For user-facing CLI/channel changes, include sample output or screenshots.

## Security & Configuration Tips
- Never commit API keys, tokens, or local runtime data.
- Store secrets in environment variables or `~/.nanobot/config.json`.
- Redact credentials, chat IDs, and provider tokens in shared logs/configs.
