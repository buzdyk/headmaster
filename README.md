# Headmaster

Train classifier heads on vision model embeddings. Organize images into folders, run make, get `.pt` checkpoints.

## Quick Start

```bash
# Register and activate an embedding model
make model-add NAME=clip-vit-l PATH=openai/clip-vit-large-patch14 DIM=768
make model-activate NAME=clip-vit-l

# Create a head — directory structure is the config
mkdir -p heads/hotdog/{positive,negative}
# Drop images into the buckets...

# Embed and train
make embed
make train HEAD=hotdog

# Check results
make status HEAD=hotdog
```

## How It Works

Each subdirectory under `heads/` is a head. Each subdirectory within a head is a bucket (class). The number of buckets determines the head type:

- **2 buckets** → binary head (sigmoid)
- **3+ buckets** → multi-class head (softmax)

Embeddings are cached in SQLite by content hash — duplicate images across heads are only embedded once.

## Docs

- [Overview](docs/overview.md)
- [Data Model](docs/data-model.md)
- [Training Pipeline](docs/training.md)
- [Makefile Targets](docs/makefile.md)
