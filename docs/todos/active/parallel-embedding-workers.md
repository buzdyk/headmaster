# Parallel workers for embedding pipeline

## Problem

Embedding 1,500+ images is the training bottleneck. File hashing (SHA-256) and image loading (PIL open + decode) are CPU/IO-bound and run sequentially.

## Approach

Use `concurrent.futures.ThreadPoolExecutor` for both hashing and image loading. Threads work because:
- `hashlib` and file reads release the GIL
- PIL decode releases the GIL
- Avoids macOS fork-safety issues with MPS (unlike `ProcessPoolExecutor` or DataLoader `num_workers`)

GPU inference stays single-threaded on a single device.

## Changes

### `src/headmaster/embed.py`
- Add `_hash_files(paths, workers)` — parallel SHA-256 hashing via ThreadPoolExecutor
- Add `_load_and_preprocess(batch_paths, processor, workers)` — parallel PIL image loading
- Update `embed_images()` signature: add `workers: int = 0`
- Update `embed_head()` signature: add `workers: int = 0`, pass through

### `src/headmaster/train.py`
- Update `_build_dataset()`: add `workers` param, pass to `embed_head()`, use `_hash_files()` for the secondary hash loop
- Update `train_head()`: add `workers` param, pass to `_build_dataset()`

### `src/headmaster/cli.py`
- Add `-j` / `--workers` flag to `embed` and `train` subcommands (default 0 = sequential)
- Pass `workers=args.workers` through to `embed_head()` / `train_head()`

### Tests
- Update mock lambdas in `test_train.py` and `test_cli.py` to accept new `workers` kwarg
- Add tests for `_hash_files` and `_load_and_preprocess` (sequential == parallel results, ordering preserved)

## Verification
```
uv run pytest
uv run hm embed -j 4
uv run hm train -j 4
uv run hm status
```
