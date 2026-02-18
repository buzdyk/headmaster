# Headmaster

Train classifier heads on vision model embeddings. Organize images into folders, run `hm`, get `.pt` checkpoints.

## Setup

```bash
make venv
source .venv/bin/activate
```

## Quick Start

```bash
# Register and activate an embedding model
hm model-add --name clip-vit-l --path openai/clip-vit-large-patch14 --dim 768
hm model-activate --name clip-vit-l

# Create a head — directory structure is the config
mkdir -p workspace/heads/hotdog/{positive,negative}
# Drop images into the buckets...

# Embed and train
hm embed
hm train --head hotdog

# Check results
hm status --head hotdog
```

## Models

Any Hugging Face vision model that produces a fixed-size embedding vector works. Models download automatically on first `hm embed`. To pre-download:

```bash
huggingface-cli download openai/clip-vit-large-patch14
```

### Reference Models

| Name | HF Path | Dim | Download | Cache/1k imgs | Notes |
|------|---------|----:|--------:|--------------:|-------|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | 512 | ~600 MB | ~2 MB | Fast, good baseline |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | 768 | ~1.7 GB | ~3 MB | Best general-purpose CLIP |
| SigLIP ViT-B/16 | `google/siglip-base-patch16-224` | 768 | ~400 MB | ~3 MB | Better zero-shot than CLIP, smaller download |
| SigLIP SO400M | `google/siglip-so400m-patch14-384` | 1152 | ~1.8 GB | ~4.5 MB | Highest quality among CLIP-family |
| DINOv2 ViT-S/14 | `facebook/dinov2-small` | 384 | ~90 MB | ~1.5 MB | Tiny, good for fine-grained tasks |
| DINOv2 ViT-B/14 | `facebook/dinov2-base` | 768 | ~350 MB | ~3 MB | Self-supervised, strong on textures/structure |
| DINOv2 ViT-L/14 | `facebook/dinov2-large` | 1024 | ~1.2 GB | ~4 MB | Best DINOv2 quality/size tradeoff |

Cache size is the SQLite embedding storage per 1k images (dim × 4 bytes each). Model weights are cached by Hugging Face in `~/.cache/huggingface/`.

### Registration Examples

```bash
hm model-add --name clip-vit-b  --path openai/clip-vit-base-patch32       --dim 512
hm model-add --name clip-vit-l  --path openai/clip-vit-large-patch14      --dim 768
hm model-add --name siglip-b    --path google/siglip-base-patch16-224     --dim 768
hm model-add --name siglip-so   --path google/siglip-so400m-patch14-384   --dim 1152
hm model-add --name dinov2-s    --path facebook/dinov2-small              --dim 384
hm model-add --name dinov2-b    --path facebook/dinov2-base               --dim 768
hm model-add --name dinov2-l    --path facebook/dinov2-large              --dim 1024

hm model-activate --name clip-vit-l
```

## Goals

- **Filesystem-as-interface** — directory structure defines heads and classes
- **Model-agnostic** — bring your own embedding model (CLIP, DINOv2, SigLIP, etc.)
- **Embedding cache** — compute once per image per model (keyed by content hash), reuse across heads
- **Both head types** — binary (sigmoid) and multi-class (softmax), determined by number of buckets

## How It Works

Each subdirectory under `workspace/heads/` is a head. Each subdirectory within a head is a bucket (class). The number of buckets determines the head type:

- **2 buckets** → binary head (sigmoid)
- **3+ buckets** → multi-class head (softmax)

```
workspace/
├── heads/
│   ├── hotdog/
│   │   ├── positive/        ← drop images here
│   │   └── negative/
│   └── weather/
│       ├── sunny/
│       ├── cloudy/
│       ├── rainy/
│       └── snowy/
├── headmaster.db              # SQLite — model registry + embedding cache
├── models/
└── out/                       # Trained checkpoints
    ├── hotdog.pt
    └── weather.pt
```

2 buckets → binary head (sigmoid, BCE loss, threshold optimization).
3+ buckets → multi-class head (softmax, cross-entropy loss).

The number of subdirectories is the entire configuration.

## Commands

### Model Management

```bash
hm model-list                    # show registered models
hm model-activate --name dinov2-b  # switch active model
hm model-remove --name dinov2-s    # remove model and its cached embeddings
```

A model must be registered and activated before embedding or training (see [Models](#models) above). Removing a model deletes its cached embeddings.

### Embedding & Training

```bash
hm embed                    # compute embeddings for all images (active model)
hm embed --head hotdog      # compute embeddings for one head only

hm train                    # train all heads
hm train --head hotdog      # train one head

hm status                   # show all heads summary
hm status --head hotdog     # show one head in detail
```

### Classification

```bash
hm classify --head hotdog --src ./unsorted/
hm classify --head hotdog --src ./unsorted/ --dest ./results/
```

Runs a trained head against a flat directory of images. Embeds each image using the active model, classifies it, and copies files into bucket subdirectories. Output defaults to `classified/<head>/`, override with `--dest`.

```
classified/hotdog/
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

### Export & Cleanup

```bash
hm export --dest /path/to/dir   # copy all checkpoints to target directory
hm clean                        # remove all checkpoints
hm clean --head hotdog           # remove one checkpoint
```

## Heads

A head is a directory under `workspace/heads/`. Each subdirectory within it is a **bucket** (class). Images go directly in bucket directories.

- The head name is the directory name.
- The bucket names become class labels.
- Bucket count determines head type: 2 = binary, 3+ = multi-class.

### Validation

- **Error**: fewer than 2 buckets, or any bucket with 0 images.
- **Warning**: any bucket with fewer than 20 images.

## Database

`headmaster.db` stores the model registry and embedding cache. Not checked into git.

### Model Registry

```sql
CREATE TABLE models (
    id        INTEGER PRIMARY KEY,
    name      TEXT UNIQUE NOT NULL,  -- user-chosen alias, e.g. 'clip-vit-l'
    path      TEXT NOT NULL,         -- path to model weights or HF identifier
    embed_dim INTEGER NOT NULL,      -- embedding dimension, e.g. 768
    active    INTEGER NOT NULL DEFAULT 0  -- 1 = used for embed/train
);
```

Exactly one model is `active` at a time. Embeddings from inactive models are kept in cache (you can switch back without recomputing).

### Embedding Cache

```sql
CREATE TABLE embeddings (
    hash     TEXT NOT NULL,     -- SHA-256 of file contents
    model_id INTEGER NOT NULL REFERENCES models(id),
    vector   BLOB NOT NULL,     -- float32 tensor, serialized
    PRIMARY KEY (hash, model_id)
);
```

Keyed by content hash. Duplicate images across heads share one embedding per model.

## Training Pipeline

### Head Architecture

Both head types share the same MLP body, differing only in the output layer.

**Binary head (2 buckets):**
```
embed_dim → 256 (ReLU) → 128 (ReLU) → 1 (Sigmoid)
```
- Loss: BCE, weighted by inverse class frequency
- After training, sweep thresholds on validation set to maximize F1
- Checkpoint includes optimal threshold

**Multi-class head (3+ buckets):**
```
embed_dim → 256 (ReLU) → 128 (ReLU) → N (Softmax)
```
- Loss: cross-entropy, weighted by inverse class frequency
- Prediction is argmax of softmax output

### Training Loop

1. Scan head directory for buckets and images
2. Load or compute embeddings for all images
3. Split into train/validation (80/20)
4. Compute class weights inversely proportional to class frequency
5. Train with appropriate loss (BCE or cross-entropy)
6. Evaluate on validation set
7. For binary heads: sweep thresholds, pick optimal F1
8. Save checkpoint to `out/<head_name>.pt`

### Default Hyperparameters

| Param | Default |
|-------|---------|
| Epochs | 50 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Train/val split | 80/20 |
| Optimizer | Adam |

### Checkpoint Format

**Binary:**
```python
{
    "type": "binary",
    "embed_dim": int,
    "model": str,
    "state_dict": model.state_dict(),
    "threshold": float,
    "classes": ["negative", "positive"],
    "sources": {
        "negative": ["hotdog/negative/img003.jpg", ...],
        "positive": ["hotdog/positive/img001.jpg", ...],
    },
    "metadata": {
        "head": str,
        "created_at": str,
        "metrics": {"accuracy": float, "precision": float, "recall": float, "f1": float},
    },
}
```

**Multi-class:**
```python
{
    "type": "multiclass",
    "embed_dim": int,
    "model": str,
    "state_dict": model.state_dict(),
    "classes": ["cloudy", "rainy", "snowy", "sunny"],
    "sources": {
        "cloudy":  ["weather/cloudy/img001.jpg", ...],
        "rainy":   ["weather/rainy/img002.jpg", ...],
        "snowy":   ["weather/snowy/img003.jpg", ...],
        "sunny":   ["weather/sunny/img004.jpg", ...],
    },
    "metadata": {
        "head": str,
        "created_at": str,
        "metrics": {"accuracy": float, "per_class": {str: {"precision": float, "recall": float, "f1": float}}},
    },
}
```

`embed_dim` and `classes` are sufficient to reconstruct the head architecture. `model` records which embedding model was used (the registry name, not the path). `sources` maps each class to image paths (relative to `heads/`) used at train time. Class names are sorted alphabetically for deterministic index mapping.

## Status Output

### Summary (no --head)

```
$ hm status
HEAD            TYPE        BUCKETS                          IMAGES  TRAINED  F1
hotdog          binary      positive(45) negative(312)          357  yes      0.94
weather         multiclass  cloudy(30) rainy(28) snowy(15)...   103  no       —
```

### Detail (--head specified)

```
$ hm status --head hotdog
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

## Behavior Notes

- `hm embed` skips images whose embeddings are already cached for the active model.
- `hm train` embeds first, then trains. Overwrites existing checkpoint in `out/`.
- Head type (binary vs multi-class) is inferred from bucket count at train time. No configuration needed.
- Switching models with `model-activate` doesn't invalidate anything. Old embeddings stay cached.
- The embedding cache is rebuildable — delete the DB and `hm embed` reconstructs it (models need to be re-registered).
- Set `HEADMASTER_WORKSPACE` to override the default workspace directory (`workspace`).
