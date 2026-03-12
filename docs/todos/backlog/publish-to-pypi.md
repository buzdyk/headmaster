# Publish to PyPI for standalone CLI install

## Problem

Headmaster requires cloning the repo and running `uv run hm`. Users can't just `pip install headmaster` or `uv tool install headmaster` to get the `hm` binary on their PATH.

## Approach

The packaging is already correct — `pyproject.toml` has `[project.scripts]` declaring the `hm` entry point. Just needs a PyPI publish step.

## Changes

- Set up PyPI API token
- `uv build && uv publish` (or add a GitHub Actions workflow)
- Add install instructions to README: `pip install headmaster` / `uv tool install headmaster` / `pipx install headmaster`
- Consider renaming the package if `headmaster` is taken on PyPI
