.PHONY: sync
sync:
	uv sync

.PHONY: check
check:
	uv run python -m pytest tests/
