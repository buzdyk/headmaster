import pytest
from pathlib import Path

from headmaster.heads import scan_head, scan_all_heads, validate_head, Bucket, Head
from tests.conftest import make_dummy_image


class TestScanHead:
    def test_binary_head(self, binary_head):
        head = scan_head(binary_head)
        assert head.name == "hotdog"
        assert head.head_type == "binary"
        assert len(head.buckets) == 2
        assert head.total_images == 50
        assert set(head.classes) == {"positive", "negative"}

    def test_multiclass_head(self, multiclass_head):
        head = scan_head(multiclass_head)
        assert head.name == "weather"
        assert head.head_type == "multiclass"
        assert len(head.buckets) == 3
        assert head.total_images == 75
        assert head.classes == ["cloudy", "rainy", "sunny"]

    def test_ignores_hidden_dirs(self, workspace):
        head_dir = workspace / "heads" / "test"
        make_dummy_image(head_dir / "a" / "img.jpg")
        make_dummy_image(head_dir / "b" / "img.jpg")
        (head_dir / ".hidden").mkdir()
        head = scan_head(head_dir)
        assert len(head.buckets) == 2

    def test_ignores_non_image_files(self, workspace):
        head_dir = workspace / "heads" / "test"
        make_dummy_image(head_dir / "a" / "img.jpg")
        make_dummy_image(head_dir / "b" / "img.jpg")
        (head_dir / "a" / "notes.txt").write_text("not an image")
        head = scan_head(head_dir)
        a_bucket = next(b for b in head.buckets if b.name == "a")
        assert len(a_bucket.images) == 1

    def test_classes_sorted_alphabetically(self, workspace):
        head_dir = workspace / "heads" / "test"
        make_dummy_image(head_dir / "zebra" / "img.jpg")
        make_dummy_image(head_dir / "apple" / "img.jpg")
        make_dummy_image(head_dir / "mango" / "img.jpg")
        head = scan_head(head_dir)
        assert head.classes == ["apple", "mango", "zebra"]

    def test_supported_image_extensions(self, workspace):
        head_dir = workspace / "heads" / "test"
        bucket = head_dir / "a"
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
            make_dummy_image(bucket / f"img{ext}")
        make_dummy_image(head_dir / "b" / "img.jpg")
        head = scan_head(head_dir)
        a_bucket = next(b for b in head.buckets if b.name == "a")
        assert len(a_bucket.images) == 6


class TestScanAllHeads:
    def test_finds_all_heads(self, workspace, binary_head, multiclass_head):
        heads = scan_all_heads(workspace)
        names = [h.name for h in heads]
        assert "hotdog" in names
        assert "weather" in names

    def test_empty_heads_dir(self, workspace):
        heads = scan_all_heads(workspace)
        assert heads == []

    def test_no_heads_dir(self, tmp_path):
        heads = scan_all_heads(tmp_path)
        assert heads == []


class TestValidation:
    def test_valid_head(self, binary_head):
        head = scan_head(binary_head)
        errors, warnings = validate_head(head)
        assert errors == []
        assert warnings == []

    def test_fewer_than_2_buckets(self, workspace):
        head_dir = workspace / "heads" / "bad"
        make_dummy_image(head_dir / "only_one" / "img.jpg")
        head = scan_head(head_dir)
        errors, warnings = validate_head(head)
        assert len(errors) == 1
        assert "fewer than 2 buckets" in errors[0]

    def test_empty_bucket(self, workspace):
        head_dir = workspace / "heads" / "bad"
        make_dummy_image(head_dir / "a" / "img.jpg")
        (head_dir / "b").mkdir(parents=True)
        head = scan_head(head_dir)
        errors, warnings = validate_head(head)
        assert any("no images" in e for e in errors)

    def test_low_image_count_warning(self, workspace):
        head_dir = workspace / "heads" / "small"
        for i in range(5):
            make_dummy_image(head_dir / "a" / f"img_{i}.jpg")
        for i in range(5):
            make_dummy_image(head_dir / "b" / f"img_{i}.jpg")
        head = scan_head(head_dir)
        errors, warnings = validate_head(head)
        assert errors == []
        assert len(warnings) == 2
        assert all("recommend >= 20" in w for w in warnings)

    def test_no_warning_at_20_images(self, workspace):
        head_dir = workspace / "heads" / "ok"
        for i in range(20):
            make_dummy_image(head_dir / "a" / f"img_{i}.jpg")
        for i in range(20):
            make_dummy_image(head_dir / "b" / f"img_{i}.jpg")
        head = scan_head(head_dir)
        errors, warnings = validate_head(head)
        assert errors == []
        assert warnings == []
