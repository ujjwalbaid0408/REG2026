"""Memory-safe, fast WSI tiling for REG2026 (OpenSlide + bounded tifffile fallback).

Most slides are single-level tiled(256) JPEG generic-TIFFs with NO pyramid, so
openslide.get_thumbnail() decodes the whole slide (~80s) and must be avoided. Instead we
grid-sample level-0 windows directly and filter by tissue content inline (~3-5s/slide),
reading only a bounded number of windows.

~9% of the corpus, however, are STRIPED (non-tiled) Deflate-compressed TIFFs that OpenSlide
cannot open at all (OpenSlideUnsupportedFormatError). The naive fallback is
tifffile.imread() (full-image read), but the largest striped slides are 5-8.5 gigapixels =
23-26 GB in RAM, which OOM-kills the container on the 32 GB A10G GC node. Instead we decode
the TIFF STRIP-BY-STRIP via random-access (~34 MB resident per strip), assembling only the
row-bands the grid sampler actually touches. Memory is bounded to a few strips regardless of
slide size, so any striped slide (train AND test) is read without OOM. Returns RGB patches.
"""
from __future__ import annotations
import numpy as np

# Guard: never let a single strip decode blow past this (pathological single-strip files).
_STRIP_MAX_BYTES = 2_000_000_000  # 2 GB


def _tissue_frac(patch: np.ndarray) -> float:
    gray = patch.mean(axis=2)
    mx = patch.max(axis=2).astype(np.float32)
    mn = patch.min(axis=2).astype(np.float32)
    sat = (mx - mn) / (mx + 1e-6)
    return float(((gray < 220) & (sat > 0.08)).mean())


def _grid_sample(W, H, patch_fn, tile, max_tiles, read_budget, min_tissue, seed):
    """Shared grid-sampling: patch_fn(x, y) -> (tile, tile, 3) uint8 RGB."""
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
        patch = patch_fn(x, y)
        if _tissue_frac(patch) >= min_tissue:
            kept.append(patch)
    if len(kept) < min(16, max_tiles):                 # relax threshold if too few
        for (x, y) in grid:
            if len(kept) >= max_tiles or reads >= read_budget * 2:
                break
            reads += 1
            patch = patch_fn(x, y)
            if _tissue_frac(patch) >= 0.05:
                kept.append(patch)
    if not kept:                                       # degenerate: center patch
        cx, cy = max(0, W // 2 - tile // 2), max(0, H // 2 - tile // 2)
        kept = [patch_fn(cx, cy)]
    return np.stack(kept).astype(np.uint8)


def _openslide_patch_fn(slide, tile):
    def fn(x, y):
        return np.asarray(slide.read_region((x, y), 0, (tile, tile)).convert("RGB"))
    return fn


def _to_rgb_tile(p, tile):
    """Normalize a raw (h, w, samples) crop to a (tile, tile, 3) uint8 RGB patch."""
    if p.ndim == 2:
        p = p[..., None]
    if p.shape[-1] == 1:
        p = np.repeat(p, 3, axis=-1)
    elif p.shape[-1] > 3:
        p = p[..., :3]
    if p.shape[0] != tile or p.shape[1] != tile:       # pad edge tiles
        p = np.pad(p, ((0, tile - p.shape[0]), (0, tile - p.shape[1]), (0, 0)),
                   mode="constant", constant_values=255)
    return p.astype(np.uint8)


def _striped_sample(page, fh, tile, max_tiles, read_budget, min_tissue, seed):
    """Band-major, bounded-memory tiling for striped (non-tiled) TIFFs.

    Decodes individual strips on demand via tifffile's low-level page.decode(); only the
    row-bands hit by the (shuffled) grid are materialized, ~34 MB per strip. We iterate
    grid rows (each a band of `tile` rows) in shuffled order for spatial spread, decoding
    each band's overlapping strips exactly once, then sweep x within the band. This keeps
    resident memory to a handful of strips for slides of ANY pixel count.
    """
    H, W = int(page.shape[0]), int(page.shape[1])
    rps = int(page.rowsperstrip)
    nstrips = len(page.dataoffsets)

    cache = {}

    def strip(si):
        if si in cache:
            return cache[si]
        nbytes = int(page.databytecounts[si])
        # rough decoded-size guard for pathological single-strip files
        if rps * W * 4 > _STRIP_MAX_BYTES:
            return None
        fh.seek(int(page.dataoffsets[si]))
        seg, _, _ = page.decode(fh.read(nbytes), si)   # (1, rows, W, samples)
        seg = np.asarray(seg)[0]
        if len(cache) > 6:                             # bound cache to ~6 strips
            cache.clear()
        cache[si] = seg
        return seg

    def band(y):
        """Assemble rows [y, y+tile) from overlapping strips -> (<=tile, W, samples)."""
        end = min(H, y + tile)
        rows, yy = [], y
        while yy < end:
            si = yy // rps
            if si >= nstrips:
                break
            s = strip(si)
            if s is None:
                break
            r0 = yy - si * rps
            r1 = min(s.shape[0], r0 + (end - yy))
            rows.append(s[r0:r1])
            yy += (r1 - r0)
        if not rows:
            return None
        return np.concatenate(rows, axis=0)

    xs = np.arange(0, max(1, W - tile), tile)
    ys = np.arange(0, max(1, H - tile), tile)
    rng = np.random.RandomState(seed)
    ys = ys.copy(); rng.shuffle(ys)
    xs_shuf = xs.copy(); rng.shuffle(xs_shuf)

    def sweep(min_t, kept, reads, budget):
        for y in ys:
            if reads[0] >= budget or len(kept) >= max_tiles:
                break
            b = band(int(y))
            if b is None:
                continue
            for x in xs_shuf:
                if reads[0] >= budget or len(kept) >= max_tiles:
                    break
                reads[0] += 1
                p = _to_rgb_tile(b[:, int(x):int(x) + tile], tile)
                if _tissue_frac(p) >= min_t:
                    kept.append(p)
        return kept

    kept, reads = [], [0]
    sweep(min_tissue, kept, reads, read_budget)
    if len(kept) < min(16, max_tiles):                 # relax threshold if too few
        sweep(0.05, kept, reads, read_budget * 2)
    if len(kept) < min(8, max_tiles):                  # sparse slide: exhaustive scan
        # Tissue may occupy a tiny fraction of a huge slide and be missed by the
        # budget-limited shuffled sweep. Walk every band IN ORDER (cache-friendly,
        # consecutive bands share strips) so we never zero out a slide that has tissue.
        for y in sorted(ys):
            if len(kept) >= max_tiles:
                break
            b = band(int(y))
            if b is None:
                continue
            for x in xs:
                if len(kept) >= max_tiles:
                    break
                p = _to_rgb_tile(b[:, int(x):int(x) + tile], tile)
                if _tissue_frac(p) >= 0.05:
                    kept.append(p)
    if not kept:                                       # degenerate: center band/patch
        b = band(max(0, H // 2 - tile // 2))
        if b is not None:
            kept = [_to_rgb_tile(b[:, max(0, W // 2 - tile // 2):][:, :tile], tile)]
        else:
            kept = [np.full((tile, tile, 3), 255, np.uint8)]
    return np.stack(kept).astype(np.uint8)


def sample_tiles(path, tile=224, max_tiles=196, read_budget=900,
                 min_tissue=0.25, seed=0):
    """Return up to max_tiles foreground RGB patches (N, tile, tile, 3) uint8.

    OpenSlide windowed reads when possible; bounded strip-by-strip tifffile fallback for
    striped TIFFs OpenSlide can't open. Never uses openslide.get_thumbnail() and never
    does a full-image read, so memory stays bounded on slides of any size.
    """
    try:
        import openslide
        slide = openslide.OpenSlide(str(path))
        W, H = slide.dimensions
        out = _grid_sample(W, H, _openslide_patch_fn(slide, tile),
                           tile, max_tiles, read_budget, min_tissue, seed)
        slide.close()
        return out
    except Exception:
        # Striped/Deflate TIFF (or any OpenSlide failure): bounded strip-by-strip read.
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            return _striped_sample(page, tf.filehandle, tile, max_tiles,
                                   read_budget, min_tissue, seed)
