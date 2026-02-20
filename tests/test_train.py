from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
import torch

from headmaster import db
from headmaster.embed import serialize_vector
from headmaster.heads import scan_head
from headmaster.train import (
    ClassifierHead,
    train_head,
    _compute_class_weights,
    _split,
    _sweep_threshold,
    _eval_multiclass,
)
from tests.conftest import make_dummy_image


def _mock_embed_head(workspace, head, model_info, workers=0):
    """Return fake embeddings keyed by file hash, using deterministic hashes."""
    from headmaster.embed import hash_file

    results = {}
    for bucket in head.buckets:
        for img in bucket.images:
            h = hash_file(img)
            results[h] = torch.randn(model_info["embed_dim"])
    return results


class TestClassifierHead:
    def test_binary_architecture(self):
        model = ClassifierHead(embed_dim=64, num_classes=2)
        assert model.head_type == "binary"
        x = torch.randn(8, 64)
        out = model(x)
        assert out.shape == (8, 1)

    def test_multiclass_architecture(self):
        model = ClassifierHead(embed_dim=128, num_classes=5)
        assert model.head_type == "multiclass"
        x = torch.randn(8, 128)
        out = model(x)
        assert out.shape == (8, 5)

    def test_different_embed_dims(self):
        for dim in [64, 256, 768, 1024]:
            model = ClassifierHead(embed_dim=dim, num_classes=3)
            x = torch.randn(4, dim)
            out = model(x)
            assert out.shape == (4, 3)


class TestComputeClassWeights:
    def test_balanced(self):
        y = torch.tensor([0, 0, 1, 1])
        weights = _compute_class_weights(y, 2)
        assert torch.allclose(weights, torch.tensor([1.0, 1.0]))

    def test_imbalanced(self):
        y = torch.tensor([0, 0, 0, 1])
        weights = _compute_class_weights(y, 2)
        # class 0: 4/(2*3) = 0.667, class 1: 4/(2*1) = 2.0
        assert weights[1] > weights[0]


class TestSplit:
    def test_split_sizes(self):
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        X_train, y_train, X_val, y_val = _split(X, y, val_ratio=0.2)
        assert len(X_train) == 80
        assert len(X_val) == 20
        assert len(y_train) == 80
        assert len(y_val) == 20

    def test_no_overlap(self):
        X = torch.arange(50).unsqueeze(1).float()
        y = torch.zeros(50, dtype=torch.long)
        X_train, _, X_val, _ = _split(X, y, val_ratio=0.2)
        train_vals = set(X_train.squeeze().tolist())
        val_vals = set(X_val.squeeze().tolist())
        assert train_vals.isdisjoint(val_vals)


class TestSweepThreshold:
    def test_returns_valid_threshold_and_metrics(self):
        model = ClassifierHead(embed_dim=64, num_classes=2)
        X_val = torch.randn(50, 64)
        y_val = torch.cat([torch.zeros(25), torch.ones(25)]).long()
        threshold, metrics = _sweep_threshold(model, X_val, y_val)

        assert 0.0 < threshold < 1.0
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(0.0 <= v <= 1.0 for v in metrics.values())


class TestEvalMulticlass:
    def test_returns_per_class_metrics(self):
        model = ClassifierHead(embed_dim=64, num_classes=3)
        X_val = torch.randn(60, 64)
        y_val = torch.cat([torch.zeros(20), torch.ones(20), 2 * torch.ones(20)]).long()
        classes = ["a", "b", "c"]
        metrics = _eval_multiclass(model, X_val, y_val, classes)

        assert "accuracy" in metrics
        assert "per_class" in metrics
        for cls in classes:
            assert cls in metrics["per_class"]
            assert "precision" in metrics["per_class"][cls]
            assert "recall" in metrics["per_class"][cls]
            assert "f1" in metrics["per_class"][cls]


class TestTrainHead:
    @patch("headmaster.train.embed_head")
    def test_binary_train(self, mock_embed, workspace_with_model, binary_head):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        head = scan_head(binary_head)

        mock_embed.side_effect = lambda ws, h, mi, workers=0: _mock_embed_head(ws, h, mi)

        ckpt_path = train_head(ws, head, model_info, epochs=5)

        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["type"] == "binary"
        assert ckpt["input_dim"] == 64
        assert ckpt["model"] == "test-model"
        assert "threshold" in ckpt
        assert 0.0 < ckpt["threshold"] < 1.0
        assert set(ckpt["classes"]) == {"positive", "negative"}
        assert "model_state_dict" in ckpt
        assert "sources" in ckpt
        assert "metadata" in ckpt
        assert "metrics" in ckpt["metadata"]
        assert "f1" in ckpt["metadata"]["metrics"]
        assert "created_at" in ckpt["metadata"]

    @patch("headmaster.train.embed_head")
    def test_multiclass_train(self, mock_embed, workspace_with_model, multiclass_head):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        head = scan_head(multiclass_head)

        mock_embed.side_effect = lambda ws, h, mi, workers=0: _mock_embed_head(ws, h, mi)

        ckpt_path = train_head(ws, head, model_info, epochs=5)

        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["type"] == "multiclass"
        assert ckpt["classes"] == ["cloudy", "rainy", "sunny"]
        assert "threshold" not in ckpt
        assert "per_class" in ckpt["metadata"]["metrics"]

    @patch("headmaster.train.embed_head")
    def test_checkpoint_sources(self, mock_embed, workspace_with_model, binary_head):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        head = scan_head(binary_head)

        mock_embed.side_effect = lambda ws, h, mi, workers=0: _mock_embed_head(ws, h, mi)

        ckpt_path = train_head(ws, head, model_info, epochs=5)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        sources = ckpt["sources"]
        assert "positive" in sources
        assert "negative" in sources
        assert len(sources["positive"]) > 0
        assert len(sources["negative"]) > 0
        # Paths should be relative to heads/
        assert all(s.startswith("hotdog/") for s in sources["positive"])

    @patch("headmaster.train.embed_head")
    def test_checkpoint_loadable(self, mock_embed, workspace_with_model, binary_head):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        head = scan_head(binary_head)

        mock_embed.side_effect = lambda ws, h, mi, workers=0: _mock_embed_head(ws, h, mi)

        ckpt_path = train_head(ws, head, model_info, epochs=5)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Verify we can reconstruct the model from checkpoint
        loaded = ClassifierHead(ckpt["input_dim"], len(ckpt["classes"]))
        loaded.load_state_dict(ckpt["model_state_dict"])
        loaded.eval()

        x = torch.randn(1, ckpt["input_dim"])
        out = loaded(x)
        assert out.shape == (1, 1)  # binary
