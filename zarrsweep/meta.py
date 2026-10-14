"""Normalise zarr v2 `.zarray` and v3 `zarr.json` array metadata to one dict used by the checks.

Normalised keys: shape, chunks (stored-object grid: the shard shape for sharded v3), dtype (numpy str),
fill_value, sep (chunk key separator), key_prefix ("" for v2, "c/" for v3 default encoding),
raw_object_bytes (upper bound of an uncompressed stored object), uncompressed (bool), zarr_format,
sharded (bool), codecs (original v2 compressor/filters or v3 codec list)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

_V3_DTYPES = {"bool": "|b1", "int8": "|i1", "uint8": "|u1", "int16": "<i2", "uint16": "<u2", "int32": "<i4",
              "uint32": "<u4", "int64": "<i8", "uint64": "<u8", "float16": "<f2", "float32": "<f4",
              "float64": "<f8"}


def from_v2(z: dict) -> dict:
    dtype = z["dtype"]
    item = np.dtype(dtype if not isinstance(dtype, list) else [tuple(x) for x in dtype]).itemsize
    raw = int(np.prod(z["chunks"])) * item
    return {"zarr_format": 2, "shape": list(z["shape"]), "chunks": list(z["chunks"]), "dtype": dtype,
            "fill_value": z.get("fill_value"), "sep": z.get("dimension_separator", "."), "key_prefix": "",
            "raw_object_bytes": raw, "uncompressed": z.get("compressor") is None and not z.get("filters"),
            "sharded": False, "codecs": {"compressor": z.get("compressor"), "filters": z.get("filters")},
            "_v2": z}


def from_v3(z: dict) -> Optional[dict]:
    if z.get("node_type") != "array":
        return None
    grid = z["chunk_grid"]
    if grid.get("name") != "regular":
        raise ValueError(f"unsupported chunk grid {grid.get('name')!r}")
    chunks = list(grid["configuration"]["chunk_shape"])
    dtype = _V3_DTYPES.get(z["data_type"], z["data_type"])
    item = np.dtype(dtype).itemsize
    enc = z.get("chunk_key_encoding", {"name": "default"})
    cfg = enc.get("configuration") or {}
    if enc.get("name") == "v2":
        sep, prefix = cfg.get("separator", "."), ""
    else:
        sep, prefix = cfg.get("separator", "/"), "c" + cfg.get("separator", "/")
    codecs = z.get("codecs", [])
    raw = int(np.prod(chunks)) * item
    sharded = bool(codecs) and codecs[0].get("name") == "sharding_indexed"
    if sharded:
        inner = codecs[0]["configuration"]["chunk_shape"]
        n_inner = int(np.prod([math.ceil(c / i) for c, i in zip(chunks, inner)]))
        raw += n_inner * 16 + 4  # index: (offset, nbytes) uint64 per inner chunk (+ optional crc32c)
        inner_codecs = codecs[0]["configuration"].get("codecs", [])
        uncompressed = [c.get("name") for c in inner_codecs] == ["bytes"]
    else:
        uncompressed = [c.get("name") for c in codecs] == ["bytes"]
    return {"zarr_format": 3, "shape": list(z["shape"]), "chunks": chunks, "dtype": dtype,
            "fill_value": z.get("fill_value"), "sep": sep, "key_prefix": prefix, "raw_object_bytes": raw,
            "uncompressed": uncompressed and not sharded, "sharded": sharded, "codecs": codecs, "_v3": z}


def read_array_meta(store, base: str) -> Optional[dict]:
    z2 = store.get_json(f"{base}/.zarray")
    if z2 is not None:
        return from_v2(z2)
    z3 = store.get_json(f"{base}/zarr.json")
    if z3 is not None:
        return from_v3(z3)
    return None


def read_group(store, prefix: str):
    """-> (zarr_format, attributes dict or None, root-array meta or None) ; (None, None, None) if absent."""
    attrs = store.get_json(f"{prefix}/.zattrs")
    zgroup = store.get_json(f"{prefix}/.zgroup")
    zarray = store.get_json(f"{prefix}/.zarray")
    if attrs is not None or zgroup is not None or zarray is not None:
        return 2, attrs, (from_v2(zarray) if zarray is not None else None)
    z3 = store.get_json(f"{prefix}/zarr.json")
    if z3 is None:
        return None, None, None
    if z3.get("node_type") == "array":
        return 3, z3.get("attributes"), from_v3(z3)
    attrs = z3.get("attributes") or {}
    if "ome" in attrs and isinstance(attrs["ome"], dict):  # NGFF 0.5 nests under "ome"
        attrs = {**attrs, **attrs["ome"]}
    return 3, attrs, None
