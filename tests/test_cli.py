import os
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
import torch

from headmaster import db
from headmaster.cli import main, build_parser
from headmaster.heads import scan_head
from headmaster.train import ClassifierHead
from headmaster.embed import serialize_vector
from tests.conftest import make_dummy_image


def run_cli(args: list[str], workspace: Path):
    """Run the CLI with given args, setting cwd to workspace."""
    with patch("headmaster.cli.get_workspace", return_value=workspace):
        parser = build_parser()
        parsed = parser.parse_args(args)
        parsed.func(parsed)


class TestModelCLI:
    def test_model_add(self, workspace, capsys):
        run_cli(["model-add", "--name", "clip", "--path", "openai/clip", "--dim", "768"], workspace)
        out = capsys.readouterr().out
        assert "added model 'clip'" in out

    def test_model_list_empty(self, workspace, capsys):
        run_cli(["model-list"], workspace)
        out = capsys.readouterr().out
        assert "no models registered" in out

    def test_model_list(self, workspace, capsys):
        run_cli(["model-add", "--name", "clip", "--path", "openai/clip", "--dim", "768"], workspace)
        run_cli(["model-list"], workspace)
        out = capsys.readouterr().out
        assert "clip" in out
        assert "768" in out

    def test_model_activate(self, workspace, capsys):
        run_cli(["model-add", "--name", "clip", "--path", "openai/clip", "--dim", "768"], workspace)
        run_cli(["model-activate", "--name", "clip"], workspace)
        out = capsys.readouterr().out
        assert "activated" in out

    def test_model_remove(self, workspace, capsys):
        run_cli(["model-add", "--name", "clip", "--path", "openai/clip", "--dim", "768"], workspace)
        run_cli(["model-remove", "--name", "clip"], workspace)
        out = capsys.readouterr().out
        assert "removed" in out

    def test_model_active_marker(self, workspace, capsys):
        run_cli(["model-add", "--name", "clip", "--path", "openai/clip", "--dim", "768"], workspace)
        run_cli(["model-activate", "--name", "clip"], workspace)
        run_cli(["model-list"], workspace)
        out = capsys.readouterr().out
        assert "*" in out


class TestStatusCLI:
    def test_status_no_heads(self, workspace, capsys):
        run_cli(["status"], workspace)
        assert "no heads found" in capsys.readouterr().out

    def test_status_summary(self, workspace, binary_head, capsys):
        run_cli(["status"], workspace)
        out = capsys.readouterr().out
        assert "hotdog" in out
        assert "binary" in out

    def test_status_detail(self, workspace, binary_head, capsys):
        run_cli(["status", "--head", "hotdog"], workspace)
        out = capsys.readouterr().out
        assert "Head:" in out
        assert "hotdog" in out
        assert "binary" in out
        assert "Trained:    no" in out

    def test_status_detail_nonexistent(self, workspace):
        with pytest.raises(SystemExit, match="not found"):
            run_cli(["status", "--head", "nope"], workspace)


class TestEmbedCLI:
    def test_embed_no_active_model(self, workspace, binary_head):
        with pytest.raises(SystemExit, match="no active model"):
            run_cli(["embed"], workspace)

    @patch("headmaster.embed.embed_head")
    def test_embed_all(self, mock_embed, workspace_with_model, binary_head, capsys):
        mock_embed.return_value = {}
        run_cli(["embed"], workspace_with_model)
        out = capsys.readouterr().out
        assert "embedding head 'hotdog'" in out

    @patch("headmaster.embed.embed_head")
    def test_embed_specific_head(self, mock_embed, workspace_with_model, binary_head, capsys):
        mock_embed.return_value = {}
        run_cli(["embed", "--head", "hotdog"], workspace_with_model)
        out = capsys.readouterr().out
        assert "hotdog" in out


class TestTrainCLI:
    @patch("headmaster.train.embed_head")
    def test_train(self, mock_embed, workspace_with_model, binary_head, capsys):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)
        head = scan_head(binary_head)

        from tests.test_train import _mock_embed_head
        mock_embed.side_effect = lambda ws, h, mi, workers=0: _mock_embed_head(ws, h, mi)

        run_cli(["train", "--head", "hotdog"], ws)
        out = capsys.readouterr().out
        assert "training head 'hotdog'" in out
        assert (ws / "out" / "hotdog.pt").exists()


class TestCleanCLI:
    def test_clean_no_checkpoints(self, workspace, capsys):
        run_cli(["clean"], workspace)
        assert "no checkpoints" in capsys.readouterr().out

    def test_clean_specific(self, workspace, capsys):
        (workspace / "out" / "hotdog.pt").write_bytes(b"fake")
        run_cli(["clean", "--head", "hotdog"], workspace)
        assert not (workspace / "out" / "hotdog.pt").exists()

    def test_clean_all(self, workspace, capsys):
        (workspace / "out" / "hotdog.pt").write_bytes(b"fake")
        (workspace / "out" / "weather.pt").write_bytes(b"fake")
        run_cli(["clean"], workspace)
        assert not list((workspace / "out").glob("*.pt"))


class TestExportCLI:
    def test_export_no_checkpoints(self, workspace, capsys):
        run_cli(["export", "--dest", str(workspace / "export_dir")], workspace)
        assert "no checkpoints" in capsys.readouterr().out

    def test_export(self, workspace, capsys):
        (workspace / "out" / "hotdog.pt").write_bytes(b"fake checkpoint")
        dest = workspace / "export_dir"
        run_cli(["export", "--dest", str(dest)], workspace)
        assert (dest / "hotdog.pt").exists()


class TestClassifyCLI:
    def test_classify_no_checkpoint(self, workspace_with_model):
        with pytest.raises(SystemExit, match="no checkpoint"):
            run_cli(["classify", "--head", "hotdog", "--src", "."], workspace_with_model)

    @patch("headmaster.embed.embed_images")
    def test_classify_binary(self, mock_embed, workspace_with_model, capsys):
        ws = workspace_with_model
        model_info = db.get_active_model(ws)

        # Create a fake checkpoint
        model = ClassifierHead(64, 2)
        ckpt = {
            "type": "binary",
            "input_dim": 64,
            "model": "test-model",
            "model_state_dict": model.state_dict(),
            "classes": ["negative", "positive"],
            "threshold": 0.5,
            "sources": {},
            "metadata": {"head": "hotdog", "created_at": "2026-01-01", "metrics": {}},
        }
        torch.save(ckpt, ws / "out" / "hotdog.pt")

        # Create source images
        src_dir = ws / "to_classify"
        for i in range(3):
            make_dummy_image(src_dir / f"img_{i}.jpg")

        # Mock embedding to return random vectors
        from headmaster.embed import hash_file
        def fake_embed(ws, images, model_info):
            return {hash_file(p): torch.randn(64) for p in images}
        mock_embed.side_effect = fake_embed

        run_cli(["classify", "--head", "hotdog", "--src", str(src_dir)], ws)
        out = capsys.readouterr().out
        assert "classifying 3 images" in out
        assert "results in" in out

    @patch("headmaster.embed.embed_images")
    def test_classify_default_inbox(self, mock_embed, workspace_with_model, capsys):
        ws = workspace_with_model

        model = ClassifierHead(64, 2)
        ckpt = {
            "type": "binary",
            "input_dim": 64,
            "model": "test-model",
            "model_state_dict": model.state_dict(),
            "classes": ["negative", "positive"],
            "threshold": 0.5,
            "sources": {},
            "metadata": {"head": "hotdog", "created_at": "2026-01-01", "metrics": {}},
        }
        torch.save(ckpt, ws / "out" / "hotdog.pt")

        # Drop images in default inbox
        inbox = ws / "inbox" / "hotdog"
        for i in range(3):
            make_dummy_image(inbox / f"img_{i}.png", color=(i * 50, 0, 0))

        from headmaster.embed import hash_file
        def fake_embed(ws, images, model_info):
            return {hash_file(p): torch.randn(64) for p in images}
        mock_embed.side_effect = fake_embed

        # No --src specified
        run_cli(["classify", "--head", "hotdog"], ws)
        out = capsys.readouterr().out
        assert "classifying 3 images" in out
        assert (ws / "classified" / "hotdog").exists()

    @patch("headmaster.embed.embed_images")
    def test_classify_with_dest(self, mock_embed, workspace_with_model, capsys):
        ws = workspace_with_model

        model = ClassifierHead(64, 2)
        ckpt = {
            "type": "binary",
            "input_dim": 64,
            "model": "test-model",
            "model_state_dict": model.state_dict(),
            "classes": ["negative", "positive"],
            "threshold": 0.5,
            "sources": {},
            "metadata": {"head": "hotdog", "created_at": "2026-01-01", "metrics": {}},
        }
        torch.save(ckpt, ws / "out" / "hotdog.pt")

        src_dir = ws / "to_classify"
        make_dummy_image(src_dir / "img.jpg")

        from headmaster.embed import hash_file
        def fake_embed(ws, images, model_info):
            return {hash_file(p): torch.randn(64) for p in images}
        mock_embed.side_effect = fake_embed

        dest = ws / "my_results"
        run_cli(["classify", "--head", "hotdog", "--src", str(src_dir), "--dest", str(dest)], ws)
        assert dest.exists()
