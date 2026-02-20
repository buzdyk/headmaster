---
type: devlog
date: 2026-02-20
---
# Feb 20 - Dev Log

## Summary

**Build & Dependencies**
- Migrated from venv/Makefile to uv
- Added uv.lock with resolved dependencies

**Concurrency**
- Added threaded workers for embedding and hashing
- Created todo for parallel embedding workers

**Docs**
- Updated all docs to use `uv run hm`

**Repo Hygiene**
- Replaced .gitkeep with .gitignore in workspace dirs

## Commits

- 85749d2 Switch from venv/Makefile to uv for dependency management
- 1b50e4e Add threaded workers for embedding and hashing
- ea95022 Replace .gitkeep with .gitignore in workspace dirs
- 9af27ee Add uv.lock with resolved project dependencies
- 944acde Add todo for parallel embedding workers
- fe16d34 Use `uv run hm` consistently in all docs
