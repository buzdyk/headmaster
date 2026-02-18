from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
import torch

from headmaster import db
from headmaster.embed import (
    hash_file,
    serialize_vector,
    deserialize_vector,
    embed_images,
    embed_head,
)
from headmaster.heads import scan_head
from tests.conftest import make_dummy_image


def _fake_load_model(model_path, embed_dim=64):
    """Return a fake model/processor/extract_fn that produces random embeddings."""
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock()

    processor = MagicMock()

    def fake_process(images, return_tensors=None):
        return {"pixel_values": torch.randn(len(images), 3, 32, 32)}

    processor.side_effect = fake_process

    def fake_forward(**kwargs):
        batch_size = kwargs["pixel_values"].shape[0]
        outputs = MagicMock()
        outputs.image_embeds = torch.randn(batch_size, embed_dim)
        return outputs

    model.__call__ = fake_forward
    model.side_effect = fake_forward

    def extract(outputs):
        return outputs.image_embeds

    return model, processor, extract


class TestHashFile:
    def test_deterministic(self, tmp_path):
        img = make_dummy_image(tmp_path / "test.jpg")
        h1 = hash_file(img)
        h2 = hash_file(img)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_different_content_different_hash(self, tmp_path):
        img1 = make_dummy_image(tmp_path / "a.jpg", color=(255, 0, 0))
        img2 = make_dummy_image(tmp_path / "b.jpg", color=(0, 255, 0))
        assert hash_file(img1) != hash_file(img2)

    def test_same_content_same_hash(self, tmp_path):
        img1 = make_dummy_image(tmp_path / "a.jpg", color=(100, 100, 100), size=(10, 10))
        img2 = make_dummy_image(tmp_path / "b.jpg", color=(100, 100, 100), size=(10, 10))
        assert hash_file(img1) == hash_file(img2)


class TestSerializeDeserialize:
    def test_roundtrip(self):
        vec = torch.randn(128)
        data = serialize_vector(vec)
        recovered = deserialize_vector(data, 128)
        assert torch.allclose(vec, recovered)

    def test_preserves_values(self):
        vec = torch.tensor([1.0, 2.0, 3.0, -1.5])
        data = serialize_vector(vec)
        recovered = deserialize_vector(data, 4)
        assert torch.allclose(vec, recovered)


class TestEmbedImages:
    @patch("headmaster.embed._load_model")
    def test_embeds_uncached_images(self, mock_load, workspace_with_model):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        mock_load.return_value = _fake_load_model("fake", model_info["embed_dim"])

        images = [make_dummy_image(ws / f"img_{i}.jpg", color=(i, i, i)) for i in range(5)]
        results = embed_images(ws, images, model_info)

        assert len(results) == 5
        for vec in results.values():
            assert vec.shape == (model_info["embed_dim"],)

    @patch("headmaster.embed._load_model")
    def test_caches_embeddings(self, mock_load, workspace_with_model):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        mock_load.return_value = _fake_load_model("fake", model_info["embed_dim"])

        images = [make_dummy_image(ws / f"img_{i}.jpg", color=(i, i, i)) for i in range(3)]

        # First call computes embeddings
        embed_images(ws, images, model_info)
        assert mock_load.call_count == 1

        # Second call should use cache, no model load
        mock_load.reset_mock()
        results = embed_images(ws, images, model_info)
        mock_load.assert_not_called()
        assert len(results) == 3

    @patch("headmaster.embed._load_model")
    def test_partial_cache_hit(self, mock_load, workspace_with_model):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        mock_load.return_value = _fake_load_model("fake", model_info["embed_dim"])

        img1 = make_dummy_image(ws / "img_0.jpg", color=(10, 10, 10))
        img2 = make_dummy_image(ws / "img_1.jpg", color=(20, 20, 20))

        # Cache img1
        embed_images(ws, [img1], model_info)
        mock_load.reset_mock()
        mock_load.return_value = _fake_load_model("fake", model_info["embed_dim"])

        # Embed both — should only compute img2
        results = embed_images(ws, [img1, img2], model_info)
        assert len(results) == 2
        assert mock_load.call_count == 1  # model loaded for uncached img2

    @patch("headmaster.embed._load_model")
    def test_embed_head(self, mock_load, workspace_with_model, binary_head):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        mock_load.return_value = _fake_load_model("fake", model_info["embed_dim"])

        head = scan_head(binary_head)
        results = embed_head(ws, head, model_info)

        assert len(results) == head.total_images

    @patch("headmaster.embed._load_model")
    def test_duplicate_images_share_embedding(self, mock_load, workspace_with_model):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        mock_load.return_value = _fake_load_model("fake", model_info["embed_dim"])

        # Create two images with identical content
        img1 = make_dummy_image(ws / "a.jpg", color=(42, 42, 42), size=(10, 10))
        img2 = make_dummy_image(ws / "b.jpg", color=(42, 42, 42), size=(10, 10))

        results = embed_images(ws, [img1, img2], model_info)
        # Same content hash -> same entry
        assert len(results) == 1
