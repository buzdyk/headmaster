from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
import torch

from headmaster import db
from headmaster.embed import (
    hash_file,
    _hash_files,
    _load_and_preprocess,
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

    def extract(inputs):
        batch_size = inputs["pixel_values"].shape[0]
        return torch.randn(batch_size, embed_dim)

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


class TestHashFiles:
    def test_sequential_matches_parallel(self, tmp_path):
        paths = [make_dummy_image(tmp_path / f"img_{i}.jpg", color=(i, i, i)) for i in range(10)]
        seq = _hash_files(paths, workers=0)
        par = _hash_files(paths, workers=4)
        assert seq == par

    def test_order_preserved(self, tmp_path):
        paths = [make_dummy_image(tmp_path / f"img_{i}.jpg", color=(i * 10, 0, 0)) for i in range(10)]
        hashes = _hash_files(paths, workers=4)
        for p, h in zip(paths, hashes):
            assert h == hash_file(p)

    def test_empty_list(self):
        assert _hash_files([], workers=0) == []
        assert _hash_files([], workers=4) == []


class TestLoadAndPreprocess:
    def test_sequential_matches_parallel(self, tmp_path):
        paths = [make_dummy_image(tmp_path / f"img_{i}.jpg", color=(i, i, i)) for i in range(5)]
        processor = MagicMock(side_effect=lambda images, return_tensors: {"pixel_values": torch.randn(len(images), 3, 32, 32)})

        _load_and_preprocess(paths, processor, workers=0)
        seq_images = processor.call_args[1]["images"] if processor.call_args[1] else processor.call_args[0][0]

        processor.reset_mock()
        _load_and_preprocess(paths, processor, workers=4)
        par_images = processor.call_args[1]["images"] if processor.call_args[1] else processor.call_args[0][0]

        assert len(seq_images) == len(par_images) == 5

    def test_order_preserved(self, tmp_path):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        paths = [make_dummy_image(tmp_path / f"img_{i}.png", color=c, size=(4, 4)) for i, c in enumerate(colors)]

        captured = {}
        def fake_processor(images, return_tensors=None):
            captured["images"] = images
            return {"pixel_values": torch.randn(len(images), 3, 4, 4)}

        _load_and_preprocess(paths, fake_processor, workers=3)
        # First pixel of each image should match the color we created
        for i, (img, color) in enumerate(zip(captured["images"], colors)):
            assert img.getpixel((0, 0)) == color
