import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from headmaster import db
from headmaster.embed import embed_head, hash_file, _hash_files, serialize_vector, deserialize_vector
from headmaster.heads import Head


class ClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.head_type = "binary" if num_classes == 2 else "multiclass"
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def _build_dataset(
    workspace: Path, head: Head, model_info: dict, workers: int = 0
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[str]]]:
    """Build feature matrix and label vector from a head.

    Returns (X, y, sources) where sources maps class name to relative image paths.
    """
    classes = head.classes  # sorted alphabetically
    class_to_idx = {c: i for i, c in enumerate(classes)}

    embeddings = embed_head(workspace, head, model_info, workers=workers)

    xs, ys = [], []
    sources: dict[str, list[str]] = {c: [] for c in classes}

    for bucket in head.buckets:
        label = class_to_idx[bucket.name]
        all_paths = list(bucket.images)
        hashes = _hash_files(all_paths, workers)
        for img_path, h in zip(all_paths, hashes):
            if h in embeddings:
                xs.append(embeddings[h])
                ys.append(label)
                rel = img_path.relative_to(workspace / "heads")
                sources[bucket.name].append(str(rel))

    X = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y, sources


def _compute_class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = y.numel() / (num_classes * counts)
    return weights


def _split(X: torch.Tensor, y: torch.Tensor, val_ratio: float = 0.2):
    n = len(X)
    perm = torch.randperm(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = perm[:split], perm[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _sweep_threshold(model: ClassifierHead, X_val: torch.Tensor, y_val: torch.Tensor) -> tuple[float, dict]:
    model.eval()
    with torch.no_grad():
        logits = model(X_val).squeeze(-1)
        probs = torch.sigmoid(logits)

    best_f1, best_thresh = 0.0, 0.5
    best_metrics = {}

    for thresh in [i / 100 for i in range(5, 96)]:
        preds = (probs >= thresh).long()
        tp = ((preds == 1) & (y_val == 1)).sum().item()
        fp = ((preds == 1) & (y_val == 0)).sum().item()
        fn = ((preds == 0) & (y_val == 1)).sum().item()
        tn = ((preds == 0) & (y_val == 0)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_val)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

    return best_thresh, best_metrics


def _eval_multiclass(model: ClassifierHead, X_val: torch.Tensor, y_val: torch.Tensor, classes: list[str]) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        preds = logits.argmax(dim=1)

    accuracy = (preds == y_val).float().mean().item()
    per_class = {}
    for i, cls in enumerate(classes):
        tp = ((preds == i) & (y_val == i)).sum().item()
        fp = ((preds == i) & (y_val != i)).sum().item()
        fn = ((preds != i) & (y_val == i)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {"accuracy": round(accuracy, 4), "per_class": per_class}


def train_head(
    workspace: Path,
    head: Head,
    model_info: dict,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    workers: int = 0,
    threshold: float | None = None,
) -> Path:
    """Train a classifier head and save checkpoint. Returns checkpoint path."""
    classes = head.classes
    num_classes = len(classes)

    print(f"training head '{head.name}' ({head.head_type}, {num_classes} classes)")

    X, y, sources = _build_dataset(workspace, head, model_info, workers=workers)
    X_train, y_train, X_val, y_val = _split(X, y)

    print(f"  train: {len(X_train)}, val: {len(X_val)}")

    embed_dim = model_info["embed_dim"]
    model = ClassifierHead(embed_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    class_weights = _compute_class_weights(y_train, num_classes)

    if head.head_type == "binary":
        pos_weight = class_weights[1] / class_weights[0]
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            if head.head_type == "binary":
                loss = criterion(out.squeeze(-1), yb.float())
            else:
                loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg = total_loss / len(loader)
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss={avg:.4f}")

    # Evaluate
    if head.head_type == "binary":
        swept_threshold, metrics = _sweep_threshold(model, X_val, y_val)
        if threshold is not None:
            print(f"  swept threshold={swept_threshold:.2f}  f1={metrics['f1']:.4f} (overridden to {threshold:.2f})")
        else:
            threshold = swept_threshold
            print(f"  threshold={threshold:.2f}  f1={metrics['f1']:.4f}")
    else:
        metrics = _eval_multiclass(model, X_val, y_val, classes)
        print(f"  accuracy={metrics['accuracy']:.4f}")

    # Save checkpoint
    out_dir = workspace / "out"
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / f"{head.name}.pt"

    checkpoint = {
        "type": head.head_type,
        "input_dim": embed_dim,
        "model": model_info["name"],
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "sources": sources,
        "metadata": {
            "head": head.name,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metrics": metrics,
        },
    }
    if head.head_type == "binary":
        checkpoint["threshold"] = threshold

    torch.save(checkpoint, ckpt_path)
    print(f"  saved {ckpt_path}")

    return ckpt_path
