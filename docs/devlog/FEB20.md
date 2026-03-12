---
type: devlog
date: 2026-02-20
---
# Feb 20 - Dev Log

## Summary

**Build & Tooling**
- Migrated from venv/Makefile to uv
- Added uv.lock with resolved dependencies

**Training**
- Added dropout layers to binary head
- Renamed checkpoint keys for clarity
- Added --threshold override for training CLI

**Embedding**
- Added threaded workers for embedding and hashing

**Docs**
- Standardized `uv run hm` usage across all docs
- Tracked parallel embedding workers in todos

## Commits

- 85749d2 Switch from venv/Makefile to uv for dependency management
- 1b50e4e Add threaded workers for embedding and hashing
- ea95022 Replace .gitkeep with .gitignore in workspace dirs
- 9af27ee Add uv.lock with resolved project dependencies
- 944acde Add todo for parallel embedding workers
- fe16d34 Use `uv run hm` consistently in all docs
- 25c4ff8 cleanup, bump devlog
- da05f51 Add dropout layers and rename checkpoint keys
- 914b1b9 Add --threshold override for binary head training
