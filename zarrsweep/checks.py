"""Integrity checks that need only metadata JSON and an object listing (no chunk downloads)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# Blosc/zstd/raw framing never inflates a chunk by more than a small header; anything far above the
# uncompressed chunk size means the object is not a chunk of this array (wrong-shape write, foreign file).
def oversize_slack(raw_chunk_bytes: int) -> int:
    """Codec framing allowance: 256 B + 1 % of the raw chunk."""
    return 256 + raw_chunk_bytes // 100


@dataclass
class Finding:
    store: str
    level: Optional[str]
    code: str
    severity: str  # error | warning | info
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class LevelInfo:
    path: str
    meta: Optional[dict]
    keys: dict  # chunk index tuple -> object size
    stray: list  # keys that are not valid chunk keys
    listing_truncated: bool = False


def _itemsize(dtype: str) -> int:
    return np.dtype(dtype if not isinstance(dtype, list) else [tuple(x) for x in dtype]).itemsize


def grid_shape(meta: dict) -> tuple:  # works on normalised meta (see meta.py)
    return tuple(math.ceil(s / c) for s, c in zip(meta["shape"], meta["chunks"]))


def parse_chunk_key(rel: str, ndim: int, sep: str) -> Optional[tuple]:
    parts = rel.split(sep) if sep != "/" else rel.split("/")
    if len(parts) != ndim or not all(re.fullmatch(r"\d+", p) for p in parts):
        return None
    return tuple(int(p) for p in parts)


def check_level(store: str, lv: LevelInfo) -> list[Finding]:
    out: list[Finding] = []
    if lv.meta is None:
        out.append(Finding(store, lv.path, "LEVEL_META_MISSING", "error",
                           f"multiscales lists dataset '{lv.path}' but '{lv.path}/.zarray' is missing"))
        return out
    m = lv.meta
    grid = grid_shape(m)
    n_grid = int(np.prod(grid))
    raw = m["raw_object_bytes"]
    if not lv.keys and not lv.listing_truncated:
        out.append(Finding(store, lv.path, "LEVEL_EMPTY", "error" if n_grid else "info",
                           f"level has a header but 0 chunk objects (grid {grid}); every read returns fill_value",
                           {"grid": grid, "fill_value": m.get("fill_value")}))
    oob = [k for k in lv.keys if any(i >= g for i, g in zip(k, grid))]
    if oob:
        out.append(Finding(store, lv.path, "CHUNK_OUT_OF_GRID", "warning",
                           f"{len(oob)} chunk objects lie outside the declared grid {grid}",
                           {"examples": [list(k) for k in sorted(oob)[:5]], "count": len(oob)}))
    over = {k: s for k, s in lv.keys.items() if s > raw + oversize_slack(raw)}
    if over:
        ratios = sorted({round(s / raw, 2) for s in over.values()})
        out.append(Finding(store, lv.path, "CHUNK_OVERSIZED", "error",
                           f"{len(over)} of {len(lv.keys)} {'shard' if m.get('sharded') else 'chunk'} objects are "
                           f"larger than an uncompressed {'shard' if m.get('sharded') else 'chunk'} ({raw} B); "
                           f"they cannot decode to the declared shape {m['chunks']}",
                           {"count": len(over), "listed": len(lv.keys), "raw_chunk_bytes": raw,
                            "size_ratios": ratios[:10],
                            "examples": [[list(k), s] for k, s in sorted(over.items())[:5]]}))
    if m.get("uncompressed"):
        bad = {k: s for k, s in lv.keys.items() if s != raw and s <= raw + oversize_slack(raw)}
        if bad:
            out.append(Finding(store, lv.path, "CHUNK_WRONG_SIZE_UNCOMPRESSED", "error",
                               f"{len(bad)} uncompressed chunk objects are not exactly {raw} B",
                               {"count": len(bad), "examples": [[list(k), s] for k, s in sorted(bad.items())[:5]]}))
    if lv.stray:
        out.append(Finding(store, lv.path, "STRAY_OBJECTS", "info",
                           f"{len(lv.stray)} objects under the array are not chunk keys",
                           {"examples": lv.stray[:5]}))
    return out


def check_pyramid(store: str, attrs: Optional[dict], levels: list[LevelInfo]) -> list[Finding]:
    out: list[Finding] = []
    if attrs is None or not attrs.get("multiscales"):
        out.append(Finding(store, None, "NO_MULTISCALES", "warning", "no OME-NGFF multiscales in .zattrs"))
        return out
    ms = attrs["multiscales"][0]
    axes = ms.get("axes") or []
    spatial = [a for a in axes if isinstance(a, dict) and a.get("type") == "space"]
    if spatial and not all(a.get("unit") for a in spatial):
        out.append(Finding(store, None, "AXES_NO_UNIT", "warning",
                           "spatial axes declare no unit; physical scale of this store is undefined",
                           {"axes": axes}))
    scales = []
    for d in ms.get("datasets", []):
        sc = next((t.get("scale") for t in d.get("coordinateTransformations", []) if t.get("type") == "scale"), None)
        scales.append(sc)
    if scales and scales[0] is not None and all(abs(v - 1.0) < 1e-9 for v in scales[0]) and spatial and not all(a.get("unit") for a in spatial):
        out.append(Finding(store, None, "SCALE_UNITLESS_IDENTITY", "warning",
                           "level 0 scale is [1,1,1] with no unit: voxel size is not recorded, so the store "
                           "cannot be registered against its source scan without out-of-band knowledge"))
    metas = [lv.meta for lv in levels]
    for i in range(1, len(levels)):
        a, b = metas[i - 1], metas[i]
        if a is None or b is None:
            continue
        sa = scales[i - 1] if i - 1 < len(scales) else None
        sb = scales[i] if i < len(scales) else None
        # per-axis factor implied by the declared scales (falls back to 2 on every axis)
        if sa and sb and len(sa) == len(sb) == len(a["shape"]):
            factors = [y / x if x else float("nan") for x, y in zip(sa, sb)]
        else:
            factors = [2.0] * len(a["shape"])
        ok = True
        for n_prev, n_cur, f in zip(a["shape"], b["shape"], factors):
            if not (f > 0) or abs(f - round(f)) > 1e-3:
                ok = False
                break
            f = int(round(f))
            if n_cur not in (math.ceil(n_prev / f), n_prev // f):
                ok = False
        if not ok:
            out.append(Finding(store, levels[i].path, "PYRAMID_SHAPE_SCALE_MISMATCH", "warning",
                               f"shape {b['shape']} (prev {a['shape']}) is not consistent with the declared "
                               f"scale step {[round(f, 4) for f in factors]}",
                               {"scale_prev": sa, "scale": sb}))
        if a.get("dtype") != b.get("dtype"):
            out.append(Finding(store, levels[i].path, "PYRAMID_DTYPE_MISMATCH", "warning",
                               f"dtype {b.get('dtype')} differs from level {levels[i-1].path} ({a.get('dtype')})"))
    # occupancy consistency: a coarse chunk should exist wherever a finer level has data under it
    for i in range(1, len(levels)):
        fine, coarse = levels[i - 1], levels[i]
        if not fine.meta or not coarse.meta or fine.listing_truncated or coarse.listing_truncated or not fine.keys:
            continue
        fc, cc = fine.meta["chunks"], coarse.meta["chunks"]
        fac = [max(1, round(p / c)) if c else 2 for p, c in zip(fine.meta["shape"], coarse.meta["shape"])]
        fgrid = grid_shape(fine.meta)
        expected = {tuple((k[d] * fc[d]) // fac[d] // cc[d] for d in range(len(k)))
                    for k in fine.keys if all(i < g for i, g in zip(k, fgrid))}
        missing = expected - set(coarse.keys)
        if missing and len(missing) / len(expected) > 0.01:
            out.append(Finding(store, coarse.path, "PYRAMID_COVERAGE_GAP", "error" if not coarse.keys else "warning",
                               f"{len(missing)} of {len(expected)} chunks expected from level {fine.path}'s "
                               f"occupancy are absent (reads there return fill_value although finer data exists)",
                               {"missing": len(missing), "expected": len(expected),
                                "examples": [list(k) for k in sorted(missing)[:5]]}))
    return out
