import sys
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


@dataclass
class Bucket:
    name: str
    images: list[Path]


@dataclass
class Head:
    name: str
    path: Path
    buckets: list[Bucket]

    @property
    def head_type(self) -> str:
        return "binary" if len(self.buckets) == 2 else "multiclass"

    @property
    def total_images(self) -> int:
        return sum(len(b.images) for b in self.buckets)

    @property
    def classes(self) -> list[str]:
        return sorted(b.name for b in self.buckets)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def scan_head(head_dir: Path) -> Head:
    buckets = []
    for entry in sorted(head_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            images = sorted(p for p in entry.iterdir() if _is_image(p))
            buckets.append(Bucket(name=entry.name, images=images))
    return Head(name=head_dir.name, path=head_dir, buckets=buckets)


def scan_all_heads(workspace: Path) -> list[Head]:
    heads_dir = workspace / "heads"
    if not heads_dir.is_dir():
        return []
    heads = []
    for entry in sorted(heads_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            heads.append(scan_head(entry))
    return heads


def validate_head(head: Head) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings)."""
    errors = []
    warnings = []
    if len(head.buckets) < 2:
        errors.append(f"head '{head.name}' has fewer than 2 buckets")
    for b in head.buckets:
        if len(b.images) == 0:
            errors.append(f"bucket '{b.name}' in head '{head.name}' has no images")
        elif len(b.images) < 20:
            warnings.append(
                f"bucket '{b.name}' in head '{head.name}' has only {len(b.images)} images (recommend >= 20)"
            )
    return errors, warnings
