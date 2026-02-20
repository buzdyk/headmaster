# CLI Reference

All operations are `uv run hm` subcommands. Install with `uv sync`.

## Model Management

```
uv run hm model-add --name clip-vit-l --path openai/clip-vit-large-patch14 --dim 768
uv run hm model-add --name dinov2 --path /local/models/dinov2-vitl14 --dim 1024
uv run hm model-list
uv run hm model-activate --name clip-vit-l
uv run hm model-remove --name dinov2
```

A model must be registered and activated before `embed` or `train`. Removing a model deletes its cached embeddings.

## Embedding & Training

```
uv run hm embed                      # compute embeddings for all images (active model)
uv run hm embed --head hotdog        # compute embeddings for one head's images only

uv run hm train                      # train all heads
uv run hm train --head hotdog        # train one head
uv run hm train --head hotdog --threshold 0.6  # override binary threshold

uv run hm status                     # show all heads: bucket counts, whether trained, metrics summary
uv run hm status --head hotdog       # show one head in detail
```

## Export & Cleanup

```
uv run hm export --dest /path/to/dir # copy all checkpoints to target directory
uv run hm clean                      # remove all checkpoints from out/
uv run hm clean --head hotdog        # remove one checkpoint
```

## Status Output

### Summary (no --head)

```
$ uv run hm status
HEAD            TYPE        BUCKETS                          IMAGES  TRAINED  F1
hotdog          binary      positive(45) negative(312)          357  yes      0.94
weather         multiclass  cloudy(30) rainy(28) snowy(15)...   103  no       —
```

### Detail (--head specified)

```
$ uv run hm status --head hotdog
Head:       hotdog
Type:       binary
Model:      clip-vit-l
Trained:    2026-02-17
Buckets:    positive (45), negative (312)
Accuracy:   0.96
Precision:  0.93
Recall:     0.95
F1:         0.94
Threshold:  0.42
```

For multi-class heads, detail view shows per-class precision/recall/F1 instead of a single threshold.

## Classification

```
uv run hm classify --head hotdog                          # uses inbox/hotdog/ as source
uv run hm classify --head hotdog --src ./unsorted/
uv run hm classify --head hotdog --src ./unsorted/ --dest ./results/
```

Runs a trained head against a flat directory of images. Embeds each image using the active model, classifies it, and copies files into bucket subdirectories.

- If `--src` is omitted, defaults to `inbox/<head>/`.
- Output defaults to `classified/<head>/`, override with `--dest`.

```
classified/hotdog/          # or --dest if specified
├── positive/
│   ├── img001.jpg
│   └── img005.jpg
├── negative/
│   ├── img002.jpg
│   └── img003.jpg
└── uncertain/
    └── img004.jpg
```

For binary heads, images with scores within 0.1 of the threshold go to `uncertain/`. For multi-class heads, images where the top class confidence is below 0.5 go to `uncertain/`.

Requires a trained checkpoint in `out/`. Source images are copied, not moved.

## Behavior

- `uv run hm embed` skips images whose embeddings are already cached for the active model.
- `uv run hm train` runs embedding first (dependency), then trains. Overwrites existing checkpoint in `out/`.
- `uv run hm status` reads the head directories and checkpoint files, prints a summary table.
- Head type (binary vs multi-class) is inferred from bucket count at train time. No configuration.
- Switching models with `uv run hm model-activate` doesn't invalidate anything. Old embeddings stay cached. Training uses whichever model is active.
