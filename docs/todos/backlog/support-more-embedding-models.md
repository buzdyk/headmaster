# Support more embedding models beyond CLIP

## Problem

Headmaster currently only supports CLIP for generating embeddings. Users working with other vision models or domain-specific embeddings have no way to plug them in.

## Approach

Abstract the embedding model behind a configurable interface so users can select from multiple supported models (e.g. SigLIP, DINOv2, OpenCLIP variants) or bring their own.

## Changes

TBD — needs design for:
- Model registry / plugin interface
- CLI flag for model selection
- Handling different embedding dimensionalities across models
- Checkpoint compatibility (heads trained on one model's embeddings aren't transferable)
