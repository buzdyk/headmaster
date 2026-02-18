import pytest
from pathlib import Path
from PIL import Image

from headmaster import db


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with initialized DB and directory structure."""
    (tmp_path / "heads").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "out").mkdir()
    db.init_db(tmp_path)
    return tmp_path


@pytest.fixture
def workspace_with_model(workspace):
    """Workspace with a registered and active fake model."""
    db.model_add(workspace, "test-model", "fake/model-path", 64)
    db.model_activate(workspace, "test-model")
    return workspace


def make_dummy_image(path: Path, color: tuple = (255, 0, 0), size: tuple = (32, 32)):
    """Create a small dummy image file."""
    img = Image.new("RGB", size, color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


@pytest.fixture
def binary_head(workspace):
    """Create a binary head with positive/negative buckets."""
    head_dir = workspace / "heads" / "hotdog"
    for i in range(25):
        make_dummy_image(head_dir / "positive" / f"pos_{i:03d}.png", color=(255, i * 10, 0))
    for i in range(25):
        make_dummy_image(head_dir / "negative" / f"neg_{i:03d}.png", color=(0, i * 10, 255))
    return head_dir


@pytest.fixture
def multiclass_head(workspace):
    """Create a multi-class head with 3 buckets."""
    head_dir = workspace / "heads" / "weather"
    for i in range(25):
        make_dummy_image(head_dir / "sunny" / f"sunny_{i:03d}.png", color=(255, 255, i * 10))
    for i in range(25):
        make_dummy_image(head_dir / "cloudy" / f"cloudy_{i:03d}.png", color=(128, 128, i * 10))
    for i in range(25):
        make_dummy_image(head_dir / "rainy" / f"rainy_{i:03d}.png", color=(0, i * 10, 200))
    return head_dir
