import hashlib
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from headmaster import db
from headmaster.heads import Head


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_files(paths: list[Path], workers: int = 0) -> list[str]:
    """Hash files with optional thread parallelism. Returns hashes in input order."""
    if workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(hash_file, paths))
    return [hash_file(p) for p in paths]


def _load_and_preprocess(paths: list[Path], processor, workers: int = 0):
    """Load and preprocess images with optional thread parallelism.

    Returns processor output dict ready for the model.
    """
    def _open(p: Path):
        return Image.open(p).convert("RGB")

    if workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            images = list(pool.map(_open, paths))
    else:
        images = [_open(p) for p in paths]

    return processor(images=images, return_tensors="pt"), images


def serialize_vector(tensor: torch.Tensor) -> bytes:
    arr = tensor.detach().cpu().float().numpy()
    return arr.tobytes()


def deserialize_vector(data: bytes, dim: int) -> torch.Tensor:
    arr = np.frombuffer(data, dtype=np.float32).copy()
    return torch.from_numpy(arr)


def _load_model(model_path: str):
    """Load a vision model and processor via transformers.

    Returns (model, processor, extract_fn) where extract_fn takes
    a dict of inputs (already on device) and returns the embedding tensor.
    """
    from transformers import AutoModel, AutoProcessor, AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    arch = type(config).__name__.lower()

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    if "clip" in arch or "siglip" in arch:
        def extract(inputs):
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            return model.visual_projection(vision_out.pooler_output)
    elif "dinov2" in arch or "vit" in arch:
        def extract(inputs):
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0]  # CLS token
    else:
        def extract(inputs):
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state[:, 0]

    return model, processor, extract


def embed_images(
    workspace: Path,
    image_paths: list[Path],
    model_info: dict,
    batch_size: int = 32,
    workers: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute embeddings for images, using cache where possible.

    Returns {file_hash: embedding_tensor} for all provided images.
    """
    model_id = model_info["id"]
    embed_dim = model_info["embed_dim"]

    # Hash all images and check cache
    hashes = _hash_files(image_paths, workers)
    path_to_hash: dict[Path, str] = dict(zip(image_paths, hashes))

    cached_hashes = db.get_cached_hashes(workspace, model_id)
    uncached_paths = [p for p, h in path_to_hash.items() if h not in cached_hashes]

    results: dict[str, torch.Tensor] = {}

    # Load cached embeddings
    for p, h in path_to_hash.items():
        if h in cached_hashes:
            data = db.get_embedding(workspace, h, model_id)
            if data is not None:
                results[h] = deserialize_vector(data, embed_dim)

    if not uncached_paths:
        return results

    # Load model and compute uncached embeddings
    model, processor, extract_fn = _load_model(model_info["path"])
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for i in range(0, len(uncached_paths), batch_size):
        batch_paths = uncached_paths[i : i + batch_size]
        inputs, _images = _load_and_preprocess(batch_paths, processor, workers)

        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeds = extract_fn(inputs)

        for j, p in enumerate(batch_paths):
            h = path_to_hash[p]
            vec = embeds[j]
            results[h] = vec.cpu()
            db.put_embedding(workspace, h, model_id, serialize_vector(vec))

    return results


def embed_head(workspace: Path, head: Head, model_info: dict, workers: int = 0) -> dict[str, torch.Tensor]:
    """Embed all images in a head. Returns {hash: tensor}."""
    all_images = []
    for bucket in head.buckets:
        all_images.extend(bucket.images)
    return embed_images(workspace, all_images, model_info, workers=workers)
