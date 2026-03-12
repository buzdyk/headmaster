# Add `hm zeroshot` command

## Problem

Zero-shot CLIP classification (image vs two text prompts) is only available through ad-hoc demo scripts. Users should be able to run zero-shot comparisons directly from the CLI.

## Approach

Add an `hm zeroshot` command that classifies one or more images by CLIP similarity to a pair of text prompts (positive vs negative), without requiring a trained head.

## Changes

TBD — needs design for:
- CLI surface (`hm zeroshot --positive "Rei Ayanami" --negative "anime character" --src <path>`)
- Single image vs directory of images
- Output format (probabilities per image)
- Whether to support batch comparison against multiple prompt pairs
- Only works with CLIP-family models (not DINOv2) — needs validation
