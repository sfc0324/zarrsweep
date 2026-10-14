"""Optional value-level confirmation of PYRAMID_COVERAGE_GAP findings (needs `numcodecs`).

For a coarse chunk that is absent although finer chunks exist under it, download those finer chunks and
count voxels that differ from fill_value. Non-zero -> the coarse level really lost data; zero -> the finer
chunks are explicitly written fill and the gap is harmless. Supports zarr v2 (any numcodecs codec) and
unsharded zarr v3 with bytes(+zstd/gzip/blosc); sharded v3 levels are reported as not verifiable."""
from __future__ import annotations

import itertools
import random

import numpy as np

from .checks import parse_chunk_key
from .meta import read_array_meta, read_group


def _decode(raw: bytes, meta: dict) -> np.ndarray:
    import numcodecs

    if meta["zarr_format"] == 2:
        v2 = meta["_v2"]
        if v2.get("compressor") is not None:
            raw = numcodecs.get_codec(v2["compressor"]).decode(raw)
        for flt in reversed(v2.get("filters") or []):
            raw = numcodecs.get_codec(flt).decode(raw)
        return np.frombuffer(raw, dtype=np.dtype(meta["dtype"]))
    if meta["sharded"]:
        raise NotImplementedError("sharded zarr v3")
    for c in reversed(meta["codecs"]):
        name, cfg = c.get("name"), c.get("configuration") or {}
        if name == "bytes":
            continue
        if name in ("zstd", "gzip", "blosc"):
            raw = numcodecs.get_codec({"id": name, **({"level": cfg["level"]} if "level" in cfg and name != "blosc" else {})}).decode(raw)
        else:
            raise NotImplementedError(f"v3 codec {name}")
    return np.frombuffer(raw, dtype=np.dtype(meta["dtype"]))


def _key(prefix, path, meta, idx):
    return f"{prefix}/{path}/{meta['key_prefix']}" + meta["sep"].join(str(i) for i in idx)


def fine_children(coarse_idx, fine_meta, coarse_meta, factors):
    """Finer chunk indices whose data downsamples into coarse chunk coarse_idx."""
    ranges = []
    for d, ci in enumerate(coarse_idx):
        lo = ci * coarse_meta["chunks"][d] * factors[d]
        hi = min((ci + 1) * coarse_meta["chunks"][d] * factors[d], fine_meta["shape"][d])
        c = fine_meta["chunks"][d]
        ranges.append(range(lo // c, max(lo // c + 1, -(-hi // c))))
    return list(itertools.product(*ranges))


def verify_gap(store, prefix, fine_path, coarse_path, missing, fine_keys, fine_meta=None, coarse_meta=None,
               sample=5, seed=0):
    fine_meta = fine_meta or read_array_meta(store, f"{prefix}/{fine_path}")
    coarse_meta = coarse_meta or read_array_meta(store, f"{prefix}/{coarse_path}")
    factors = [max(1, round(p / c)) for p, c in zip(fine_meta["shape"], coarse_meta["shape"])]
    fill = fine_meta.get("fill_value") or 0
    picks = random.Random(seed).sample(sorted(missing), min(sample, len(missing)))
    checked = []
    for cidx in picks:
        nonfill, examined, note = 0, 0, None
        for fidx in fine_children(cidx, fine_meta, coarse_meta, factors):
            if tuple(fidx) not in fine_keys:
                continue
            try:
                arr = _decode(store.get_bytes(_key(prefix, fine_path, fine_meta, fidx)), fine_meta)
            except NotImplementedError as exc:
                note = f"not verifiable: {exc}"
                break
            examined += 1
            nonfill += int(np.count_nonzero(arr != fill))
        checked.append({"coarse_chunk": list(cidx), "fine_chunks_examined": examined, "nonfill_voxels": nonfill,
                        **({"note": note} if note else {})})
    return checked


def find_and_verify_gaps(store, prefix, sample=5, seed=0):
    """Recompute PYRAMID_COVERAGE_GAP for every adjacent level pair of `prefix` and value-check a sample."""
    _, attrs, _ = read_group(store, prefix)
    paths = [d["path"] for d in (attrs or {})["multiscales"][0]["datasets"]]
    metas, keys = {}, {}
    for p in paths:
        m = read_array_meta(store, f"{prefix}/{p}")
        metas[p] = m
        ks = set()
        base = f"{prefix}/{p}/"
        for k, _ in store.list(base):
            rel = k[len(base):]
            if rel.startswith(m["key_prefix"]):
                idx = parse_chunk_key(rel[len(m["key_prefix"]):], len(m["shape"]), m["sep"])
                if idx is not None:
                    ks.add(idx)
        keys[p] = ks
    out = []
    for fp, cp in zip(paths, paths[1:]):
        fm, cm = metas[fp], metas[cp]
        fac = [max(1, round(a / b)) for a, b in zip(fm["shape"], cm["shape"])]
        fgrid = [-(-s // c) for s, c in zip(fm["shape"], fm["chunks"])]
        expected = {tuple((k[d] * fm["chunks"][d]) // fac[d] // cm["chunks"][d] for d in range(len(k)))
                    for k in keys[fp] if all(i < g for i, g in zip(k, fgrid))}
        missing = expected - keys[cp]
        if not missing:
            continue
        checked = verify_gap(store, prefix, fp, cp, missing, keys[fp], fm, cm, sample=sample, seed=seed)
        out.append({"store": prefix, "fine": fp, "coarse": cp, "missing": len(missing), "expected": len(expected),
                    "sampled": checked, "real_data_missing": sum(1 for c in checked if c["nonfill_voxels"] > 0)})
    return out
