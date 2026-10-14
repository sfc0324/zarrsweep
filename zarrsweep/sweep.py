"""Sweep one or many OME-Zarr (v2) stores: metadata + object listing -> findings."""
from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

from .checks import Finding, LevelInfo, check_level, check_pyramid, parse_chunk_key
from .meta import read_array_meta, read_group

_META_FILES = {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}


def _list_level(store, base: str, meta: dict, max_keys: Optional[int]):
    keys, stray, n = {}, [], 0
    nd = len(meta["shape"])
    for key, size in store.list(base + "/", max_keys=max_keys):
        n += 1
        rel = key[len(base) + 1:]
        if rel in _META_FILES:
            continue
        idx = None
        if rel.startswith(meta["key_prefix"]):
            idx = parse_chunk_key(rel[len(meta["key_prefix"]):], nd, meta["sep"])
        if idx is None:
            stray.append(rel)
        else:
            keys[idx] = size
    return keys, stray, max_keys is not None and n >= max_keys


def sweep_store(store, prefix: str, max_keys_per_level: Optional[int] = None) -> dict:
    prefix = prefix.rstrip("/")
    findings: list[Finding] = []
    fmt, attrs, root_meta = read_group(store, prefix)
    if fmt is None:
        findings.append(Finding(prefix, None, "STORE_MISSING", "error",
                                "catalogued store has no .zattrs/.zgroup/.zarray/zarr.json"))
        return {"store": prefix, "zarr_format": None, "levels": [], "findings": [f.to_dict() for f in findings]}
    if root_meta is not None:
        level_paths = [""]
    else:
        ms = (attrs or {}).get("multiscales") or []
        level_paths = [d["path"] for d in ms[0].get("datasets", [])] if ms else []
    levels: list[LevelInfo] = []
    for lp in level_paths:
        base = f"{prefix}/{lp}".rstrip("/")
        meta = root_meta if lp == "" else read_array_meta(store, base)
        keys, stray, truncated = ({}, [], False)
        if meta is not None:
            keys, stray, truncated = _list_level(store, base, meta, max_keys_per_level)
            if lp == "":  # root array: sibling metadata/attrs files are not strays
                stray = [k for k in stray if k.split("/")[0] not in _META_FILES and not k.endswith(".json")]
        lv = LevelInfo(lp or ".", meta, keys, stray, truncated)
        levels.append(lv)
        findings += check_level(prefix, lv)
    if root_meta is None:
        findings += check_pyramid(prefix, attrs, levels)
    return {
        "store": prefix,
        "zarr_format": fmt,
        "levels": [{"path": lv.path, "shape": (lv.meta or {}).get("shape"), "chunks": (lv.meta or {}).get("chunks"),
                    "dtype": (lv.meta or {}).get("dtype"), "sharded": (lv.meta or {}).get("sharded"),
                    "objects": len(lv.keys), "listing_truncated": lv.listing_truncated} for lv in levels],
        "findings": [f.to_dict() for f in findings],
    }


BUCKET_S3_URL = "s3://vesuvius-challenge-open-data"


def catalog_origins(catalog: dict) -> list[tuple[str, list[str]]]:
    """(path, [access-root urls]) for every catalogued origin whose path is a .zarr store."""
    found: dict[str, set] = {}

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("path"), str) and isinstance(o.get("access_roots"), list):
                path = o["path"].rstrip("/")
                if path.endswith(".zarr"):
                    found.setdefault(path, set()).update(r.get("url", "") for r in o["access_roots"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(catalog)
    return sorted((p, sorted(r)) for p, r in found.items())


def catalog_zarr_paths(catalog: dict, bucket_s3_url: str = BUCKET_S3_URL) -> list[str]:
    """Catalogued .zarr stores served from this bucket (origins on other hosts, e.g. data.aws.ash2txt.org,
    are skipped: that host has no S3 listing API)."""
    return [p for p, roots in catalog_origins(catalog) if bucket_s3_url in roots]


def load_catalog(store) -> dict:
    raw = store.get_bytes("metadata.json")
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def sweep_many(store, prefixes: Iterable[str], workers: int = 8, max_keys_per_level: Optional[int] = None,
               progress=None) -> list[dict]:
    prefixes = list(prefixes)
    results = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(sweep_store, store, p, max_keys_per_level): p for p in prefixes}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # network etc: record, keep going
                results.append({"store": p, "levels": [], "findings": [
                    Finding(p, None, "SWEEP_ERROR", "error", f"{type(exc).__name__}: {exc}").to_dict()]})
            if progress:
                progress(i, len(prefixes), p)
    return sorted(results, key=lambda r: r["store"])
