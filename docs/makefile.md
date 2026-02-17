# Makefile Targets

## Model Management

```
make model-add NAME=clip-vit-l PATH=openai/clip-vit-large-patch14 DIM=768
make model-add NAME=dinov2 PATH=/local/models/dinov2-vitl14 DIM=1024
make model-list
make model-activate NAME=clip-vit-l
make model-remove NAME=dinov2
```

A model must be registered and activated before `embed` or `train`. Removing a model deletes its cached embeddings.

## Embedding & Training

```
make embed                    # compute embeddings for all images (active model)
make embed HEAD=hotdog          # compute embeddings for one head's images only

make train                    # train all heads
make train HEAD=hotdog          # train one head

make status                   # show all heads: bucket counts, whether trained, metrics summary
make status HEAD=hotdog         # show one head in detail
```

## Export & Cleanup

```
make export DEST=/path/to/dir # copy all checkpoints to target directory
make clean                    # remove all checkpoints from out/
make clean HEAD=hotdog          # remove one checkpoint
```

## Status Output

### Summary (no HEAD specified)

```
$ make status
HEAD            TYPE        BUCKETS                          IMAGES  TRAINED  F1
hotdog          binary      positive(45) negative(312)          357  yes      0.94
weather         multiclass  cloudy(30) rainy(28) snowy(15)...   103  no       —
```

### Detail (HEAD specified)

```
$ make status HEAD=hotdog
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
make classify HEAD=hotdog SRC=./unsorted/
make classify HEAD=hotdog SRC=./unsorted/ DEST=./results/
```

Runs a trained head against a flat directory of images. Embeds each image using the active model, classifies it, and copies files into bucket subdirectories. Output defaults to `classified/<head>/`, override with `DEST`.

```
classified/hotdog/          # or DEST if specified
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

- `make embed` skips images whose embeddings are already cached for the active model.
- `make train` runs `embed` first (dependency), then trains. Overwrites existing checkpoint in `out/`.
- `make status` reads the head directories and checkpoint files, prints a summary table.
- Head type (binary vs multi-class) is inferred from bucket count at train time. No configuration.
- Switching models with `model-activate` doesn't invalidate anything. Old embeddings stay cached. Training uses whichever model is active.
