## Agent skills

### Issue tracker

Issues and specs are tracked as GitHub Issues via `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`GLOSSARY.md` and `docs/adr/` at repo root). See `docs/agents/domain.md`.

### Commands
- Run the full test suite: `uv run python -m unittest`
- Run a specific test: `uv run python -m unittest tests.test_video` (or `uv run python -m unittest tests/test_video.py`)