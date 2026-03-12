# Add `hm confusion-matrix` command

## Problem

There is no built-in way to evaluate a trained head's accuracy against a labeled test set. Users need a first-class subcommand to generate confusion matrices.

## Approach

Add an `hm confusion-matrix` (or `hm eval`) command that loads a trained head checkpoint, runs inference on a test directory organized by class, and prints a confusion matrix with accuracy stats.

## Changes

TBD — needs design for:
- CLI surface (`hm confusion-matrix --head <name> --test-dir <path>`)
- Whether to reuse cached embeddings from the DB or re-embed test images
- Output format (table, CSV, JSON)
- Support for both binary and multi-class heads
