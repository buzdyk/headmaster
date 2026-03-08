# Simple web UI for labeling/reviewing training samples

## Problem

Labeling and reviewing training samples currently happens outside headmaster. A built-in UI would lower the barrier to curating high-quality training sets.

## Approach

Provide a simple web interface (e.g. Flask/FastAPI + minimal frontend) for browsing images, assigning labels, and reviewing the training sample distribution.

## Changes

TBD — needs design for:
- Tech stack (server framework, frontend approach)
- Integration with existing headmaster data layout
- Label management (create/edit/delete labels, reassign samples)
- Export workflow back into the training pipeline
