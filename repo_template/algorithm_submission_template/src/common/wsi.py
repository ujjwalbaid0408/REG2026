"""Memory-safe, fast WSI tiling for REG2026 (OpenSlide, no-pyramid friendly).

These slides are single-level tiled(256) JPEG generic-TIFFs with NO pyramid, so
openslide.get_thumbnail() decodes the whole slide (~80s) and must be avoided. Instead we
grid-sample level-0 windows directly and filter by tissue content inline (~3-5s/slide),
reading only a bounded number of windows. Returns RGB patches for the encoder.
"""
from __future__ import annotations
import numpy as np
import openslide


def _tissue_frac(patch: np.ndarray) -> float:
    gray = patch.mean(axis=2)
    mx = patch.max(axis=2).astype(np.float32)
    mn = patch.min(axis=2).astype(np.float32)
    sat = (mx - mn) / (mx + 1e-6)
    return float(((gray < 220) & (sat > 0.08)).mean())


def sample_tiles(path, tile=224, max_tiles=196, read_budget=900,
                 min_tissue=0.25, seed=0):
    """Return up to max_tiles foreground RGB patches (N, tile, tile, 3) uint8.

    Grid-samples up to read_budget level-0 windows in a seeded-shuffled order, keeps
    those with tissue fraction >= min_tissue. Never decodes the full slide.
    """
    slide = openslide.OpenSlide(str(path))
    W, H = slide.dimensions
    xs = np.arange(0, max(1, W - tile), tile)
    ys = np.arange(0, max(1, H - tile), tile)
    grid = [(int(x), int(y)) for y in ys for x in xs]
    rng = np.random.RandomState(seed)
    rng.shuffle(grid)

    kept, reads = [], 0
    for (x, y) in grid:
        if reads >= read_budget or len(kept) >= max_tiles:
            break
        reads += 1
        patch = np.asarray(slide.read_region((x, y), 0, (tile, tile)).convert("RGB"))
        if _tissue_frac(patch) >= min_tissue:
            kept.append(patch)
    # relax threshold if too few foreground tiles were found
    if len(kept) < min(16, max_tiles):
        for (x, y) in grid:
            if len(kept) >= max_tiles or reads >= read_budget * 2:
                break
            reads += 1
            patch = np.asarray(slide.read_region((x, y), 0, (tile, tile)).convert("RGB"))
            if _tissue_frac(patch) >= 0.05:
                kept.append(patch)
    slide.close()
    if not kept:  # degenerate slide: return a single center patch
        slide = openslide.OpenSlide(str(path))
        cx, cy = max(0, W // 2 - tile // 2), max(0, H // 2 - tile // 2)
        kept = [np.asarray(slide.read_region((cx, cy), 0, (tile, tile)).convert("RGB"))]
        slide.close()
    return np.stack(kept).astype(np.uint8)
