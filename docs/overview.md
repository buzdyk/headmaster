# Overview

Headmaster trains classifier heads on vision model embeddings. You organize images into folders, run `hm`, and get `.pt` checkpoints.

## Goals

- **Filesystem-as-interface** — directory structure defines heads and classes
- **Model-agnostic** — bring your own embedding model (CLIP, DINOv2, SigLIP, etc.)
- **Embedding cache** — compute once per image per model (keyed by content hash), reuse across heads
- **Both head types** — binary (sigmoid) and multi-class (softmax), determined by number of buckets
- **CLI-driven** — all operations are `hm` subcommands (also available as `make` targets, see [Makefile](../Makefile))

## Non-Goals

- Web UI
- Inference runtime
- Distributed training

## How It Works

```
heads/
├── hotdog/
│   ├── positive/        ← drop images here
│   └── negative/
├── weather/
│   ├── sunny/
│   ├── cloudy/
│   ├── rainy/
│   └── snowy/
└── ...

hm embed                  # compute embeddings for all images
hm train                  # train all heads
hm train --head hotdog    # train one head
```

2 buckets → binary head (sigmoid, BCE loss, threshold optimization).
3+ buckets → multi-class head (softmax, cross-entropy loss).

The number of subdirectories is the entire configuration.
