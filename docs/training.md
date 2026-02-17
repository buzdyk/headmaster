# Training Pipeline

## Head Architecture

Both head types share the same MLP body, differing only in the output layer.

### Binary Head (2 buckets)

```
embed_dim → 256 (ReLU) → 128 (ReLU) → 1 (Sigmoid)
```

- Loss: BCE, weighted by inverse class frequency
- After training, sweep thresholds on validation set to maximize F1
- Checkpoint includes optimal threshold

### Multi-class Head (3+ buckets)

```
embed_dim → 256 (ReLU) → 128 (ReLU) → N (Softmax)
```

- Loss: cross-entropy, weighted by inverse class frequency
- N = number of buckets
- Prediction is argmax of softmax output

## Embedding Computation

- Model is configured via the model registry (see [data-model.md](data-model.md)) — not hardcoded
- `embed_dim` is read from the active model's registry entry
- Each image is hashed (SHA-256) and its embedding cached in `headmaster.db` — duplicates across heads share one embedding
- Switching models invalidates nothing — old embeddings stay cached, new model produces new rows

## Training Loop

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

## Checkpoint Format

### Binary

```python
{
    "type": "binary",
    "embed_dim": int,
    "model": str,
    "state_dict": model.state_dict(),
    "threshold": float,
    "classes": ["negative", "positive"],
    "sources": {                   # images used for training, paths relative to heads/
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

### Multi-class

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

`embed_dim` and `classes` are sufficient to reconstruct the head architecture. `model` records which embedding model was used for training (the registry name, not the path). `sources` maps each class to the list of image paths (relative to `heads/`) used at train time. Class names are sorted alphabetically for deterministic index mapping.
