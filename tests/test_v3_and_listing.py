import json
import os
from urllib.parse import unquote

import numpy as np
import pytest

from zarrsweep.meta import from_v3
from zarrsweep.store import LocalStore, S3Store
from zarrsweep.sweep import sweep_store


def _w(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)


def make_v3(root, name="v.zarr", shapes=((8, 8), (4, 4)), chunks=(4, 4), sharded=False, write=None, unit="micrometer"):
    base = os.path.join(root, name)
    axes = [{"name": n, "type": "space", **({"unit": unit} if unit else {})} for n in "yx"]
    _w(f"{base}/zarr.json", json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": {
        "version": "0.5", "multiscales": [{"axes": axes, "datasets": [
            {"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": [2.0 ** i] * 2}]}
            for i in range(len(shapes))]}]}}}))
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    if sharded:
        codecs = [{"name": "sharding_indexed", "configuration": {"chunk_shape": [2, 2], "codecs": codecs,
                                                                   "index_codecs": codecs, "index_location": "end"}}]
    for lvl, shp in enumerate(shapes):
        _w(f"{base}/{lvl}/zarr.json", json.dumps({"zarr_format": 3, "node_type": "array", "shape": list(shp),
                                                  "data_type": "uint8", "fill_value": 0,
                                                  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
                                                  "chunk_key_encoding": {"name": "default"}, "codecs": codecs}))
        grid = [-(-s // c) for s, c in zip(shp, chunks)]
        raw = from_v3(json.load(open(f"{base}/{lvl}/zarr.json")))["raw_object_bytes"]
        for idx in (write.get(lvl) if write and lvl in write else list(np.ndindex(*grid))):
            _w(f"{base}/{lvl}/c/" + "/".join(map(str, idx)), np.full(raw, 3, np.uint8).tobytes())
    return name


def codes(res):
    return sorted({f["code"] for f in res["findings"]})


def test_v3_clean_and_metadata_under_ome_key(tmp_path):
    res = sweep_store(LocalStore(str(tmp_path)), make_v3(tmp_path))
    assert res["zarr_format"] == 3 and codes(res) == []
    assert res["levels"][0]["objects"] == 4


def test_v3_empty_level_and_gap(tmp_path):
    name = make_v3(tmp_path, write={1: []})
    assert {"LEVEL_EMPTY", "PYRAMID_COVERAGE_GAP"} <= set(codes(sweep_store(LocalStore(str(tmp_path)), name)))


def test_v3_uncompressed_wrong_size_and_unitless(tmp_path):
    name = make_v3(tmp_path, unit=None)
    _w(f"{tmp_path}/{name}/0/c/0/0", b"\0" * 5)
    assert {"CHUNK_WRONG_SIZE_UNCOMPRESSED", "AXES_NO_UNIT"} <= set(codes(sweep_store(LocalStore(str(tmp_path)), name)))


def test_v3_sharded_oversize_accounts_for_index(tmp_path):
    name = make_v3(tmp_path, sharded=True)
    m = from_v3(json.load(open(f"{tmp_path}/{name}/0/zarr.json")))
    assert m["sharded"] and m["raw_object_bytes"] == 16 + 4 * 16 + 4
    assert codes(sweep_store(LocalStore(str(tmp_path)), name)) == []
    _w(f"{tmp_path}/{name}/0/c/1/1", b"\0" * (m["raw_object_bytes"] * 8))
    assert "CHUNK_OVERSIZED" in codes(sweep_store(LocalStore(str(tmp_path)), name))


class _Resp:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        pass


def _fake_s3(keys, page=3):
    """Emulate ListObjectsV2 (prefix, delimiter, continuation-token, 1000->`page` keys per page)."""
    calls = []

    def get(url, params=None):
        calls.append(params)
        prefix, delim = params["prefix"], params.get("delimiter")
        items, seen = [], set()
        for k in sorted(keys):
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix):]
            if delim and delim in rest:
                p = prefix + rest.split(delim)[0] + delim
                if p not in seen:
                    seen.add(p)
                    items.append(("P", p))
            else:
                items.append(("K", k))
        start = int(params.get("continuation-token", 0))
        chunk = items[start:start + page]
        body = "".join(f"<Contents><Key>{v}</Key><Size>{keys[v]}</Size></Contents>" if t == "K"
                       else f"<CommonPrefixes><Prefix>{v}</Prefix></CommonPrefixes>" for t, v in chunk)
        more = start + page < len(items)
        body += f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
        if more:
            body += f"<NextContinuationToken>{start + page}</NextContinuationToken>"
        return _Resp(f"<ListBucketResult>{body}</ListBucketResult>".encode())

    return get, calls


def test_s3_listing_paginates_and_fans_out(monkeypatch):
    keys = {f"s.zarr/0/{z}/{y}/{x}": 10 for z in range(5) for y in range(3) for x in range(4)}
    keys["s.zarr/0/.zarray"] = 1
    st = S3Store("http://fake", list_workers=3)
    get, calls = _fake_s3(keys)
    monkeypatch.setattr(st, "_get", lambda url, params=None, **kw: get(url, params))
    got = dict(st.list("s.zarr/0/"))
    assert got == keys
    assert any(c.get("delimiter") == "/" for c in calls) and any("continuation-token" in c for c in calls)
    capped = list(st.list("s.zarr/0/", max_keys=7))
    assert len(capped) == 7


def test_truncated_listing_is_reported_not_flagged_empty(tmp_path):
    from zarrsweep.sweep import sweep_store as sw
    import tests.test_zarrsweep as t

    name = t.make_store(tmp_path)
    res = sw(LocalStore(str(tmp_path)), name, max_keys_per_level=2)
    assert all(lv["listing_truncated"] for lv in res["levels"])
    assert "PYRAMID_COVERAGE_GAP" not in codes(res)
