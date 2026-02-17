# Data Model

## Filesystem Layout

```
workspace/
├── Makefile
├── heads/                     # One subdir per head
│   ├── hotdog/
│   │   ├── positive/
│   │   │   ├── img001.jpg
│   │   │   └── img002.png
│   │   └── negative/
│   │       ├── img003.jpg
│   │       └── img004.png
│   └── weather/
│       ├── sunny/
│       ├── cloudy/
│       ├── rainy/
│       └── snowy/
├── headmaster.db              # SQLite — model registry + embedding cache
└── out/                       # Trained checkpoints
    ├── hotdog.pt
    └── weather.pt
```

## Heads

A head is a directory under `heads/`. Each subdirectory within it is a **bucket** (class). Images go directly in bucket directories.

- The head name is the directory name.
- The bucket names become class labels.
- Bucket count determines head type: 2 = binary, 3+ = multi-class.

### Validation

- **Error**: fewer than 2 buckets, or any bucket with 0 images.
- **Warning**: any bucket with fewer than 20 images.

## Database (SQLite)

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

Exactly one model is `active` at a time. `make embed` and `make train` use the active model. Embeddings from inactive models are kept in cache (you can switch back without recomputing).

### Embedding Cache

```sql
CREATE TABLE embeddings (
    hash     TEXT NOT NULL,     -- SHA-256 of file contents
    model_id INTEGER NOT NULL REFERENCES models(id),
    vector   BLOB NOT NULL,     -- float32 tensor, serialized
    PRIMARY KEY (hash, model_id)
);
```

Keyed by content hash. Duplicate images across heads share one embedding per model. The cache is rebuildable — delete the DB and `make embed` reconstructs it (models need to be re-registered).

## Checkpoints

Trained checkpoints go in `out/` as `<head_name>.pt`. See [training.md](training.md) for the checkpoint format.
