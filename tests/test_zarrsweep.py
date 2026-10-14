import json
import os

import numpy as np
import pytest

from zarrsweep.checks import parse_chunk_key
from zarrsweep.cli import main
from zarrsweep.store import LocalStore
from zarrsweep.sweep import catalog_origins, catalog_zarr_paths, sweep_store


def _w(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode) as f:
        f.write(data)


def make_store(root, name="s.zarr", shapes=((8, 8, 8), (4, 4, 4)), chunks=(4, 4, 4), unit="micrometer",
               scales=None, write=None, dtype="|u1"):
    """Uncompressed zarr v2 OME pyramid; `write` maps level -> list of chunk idx (default: full grid)."""
    base = os.path.join(root, name)
    scales = scales or [[2.0 ** i] * 3 for i in range(len(shapes))]
    axes = [{"name": n, "type": "space", **({"unit": unit} if unit else {})} for n in "zyx"]
    _w(f"{base}/.zgroup", json.dumps({"zarr_format": 2}))
    _w(f"{base}/.zattrs", json.dumps({"multiscales": [{"axes": axes, "datasets": [
        {"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": sc}]} for i, sc in enumerate(scales)]}]}))
    raw = int(np.prod(chunks)) * np.dtype(dtype).itemsize
    for lvl, shp in enumerate(shapes):
        _w(f"{base}/{lvl}/.zarray", json.dumps({"zarr_format": 2, "shape": list(shp), "chunks": list(chunks),
                                                "dtype": dtype, "compressor": None, "fill_value": 0, "order": "C",
                                                "filters": None, "dimension_separator": "/"}))
        grid = [-(-s // c) for s, c in zip(shp, chunks)]
        idxs = write.get(lvl) if write and lvl in write else list(np.ndindex(*grid))
        for idx in idxs:
            _w(f"{base}/{lvl}/" + "/".join(map(str, idx)), np.full(raw, 7, np.uint8).tobytes() if not isinstance(idx, tuple) or True else b"")
    return name


def codes(res):
    return sorted({f["code"] for f in res["findings"]})


def test_clean_store_has_no_findings(tmp_path):
    name = make_store(tmp_path)
    assert codes(sweep_store(LocalStore(str(tmp_path)), name)) == []


def test_empty_level(tmp_path):
    name = make_store(tmp_path, write={1: []})
    res = sweep_store(LocalStore(str(tmp_path)), name)
    assert "LEVEL_EMPTY" in codes(res)


def test_oversized_and_out_of_grid(tmp_path):
    name = make_store(tmp_path)
    big = np.zeros(64 * 8, np.uint8).tobytes()  # 8x the 64-byte chunk
    _w(f"{tmp_path}/{name}/0/0/0/0", big)
    _w(f"{tmp_path}/{name}/0/5/0/0", np.zeros(64, np.uint8).tobytes())
    _w(f"{tmp_path}/{name}/0/1/1/0", np.zeros(10, np.uint8).tobytes())
    c = codes(sweep_store(LocalStore(str(tmp_path)), name))
    assert {"CHUNK_OVERSIZED", "CHUNK_OUT_OF_GRID", "CHUNK_WRONG_SIZE_UNCOMPRESSED"} <= set(c)


def test_coverage_gap(tmp_path):
    # level 0 grid 2x2x2 fully written, level 1 (4^3, one chunk) missing its only chunk -> empty+gap
    name = make_store(tmp_path, shapes=((16, 16, 16), (8, 8, 8)), chunks=(8, 8, 8), write={1: []})
    c = codes(sweep_store(LocalStore(str(tmp_path)), name))
    assert "PYRAMID_COVERAGE_GAP" in c


def test_partial_coverage_gap(tmp_path):
    name = make_store(tmp_path, shapes=((32, 32, 32), (16, 16, 16)), chunks=(4, 4, 4),
                      write={1: [i for i in np.ndindex(4, 4, 4) if i != (0, 0, 0)]})
    res = sweep_store(LocalStore(str(tmp_path)), name)
    gap = [f for f in res["findings"] if f["code"] == "PYRAMID_COVERAGE_GAP"]
    assert gap and gap[0]["details"]["missing"] == 1 and gap[0]["details"]["examples"] == [[0, 0, 0]]


def test_unitless_identity_scale(tmp_path):
    name = make_store(tmp_path, unit=None, scales=[[1.0] * 3, [2.0] * 3])
    assert {"AXES_NO_UNIT", "SCALE_UNITLESS_IDENTITY"} <= set(codes(sweep_store(LocalStore(str(tmp_path)), name)))


def test_anisotropic_pyramid_is_fine_but_inconsistent_scale_is_flagged(tmp_path):
    ok = make_store(tmp_path, "a.zarr", shapes=((8, 8, 8), (8, 4, 4)), scales=[[1, 1, 1], [1, 2, 2]])
    assert codes(sweep_store(LocalStore(str(tmp_path)), ok)) == []
    bad = make_store(tmp_path, "b.zarr", shapes=((8, 8, 8), (4, 4, 4)), scales=[[1, 1, 1], [1, 2, 2]])
    assert "PYRAMID_SHAPE_SCALE_MISMATCH" in codes(sweep_store(LocalStore(str(tmp_path)), bad))


def test_missing_store(tmp_path):
    assert codes(sweep_store(LocalStore(str(tmp_path)), "nope.zarr")) == ["STORE_MISSING"]


def test_parse_chunk_key():
    assert parse_chunk_key("1/2/3", 3, "/") == (1, 2, 3)
    assert parse_chunk_key("1.2.3", 3, ".") == (1, 2, 3)
    assert parse_chunk_key(".zarray", 3, "/") is None
    assert parse_chunk_key("1/2", 3, "/") is None


def test_catalog_only_sweeps_bucket_origins():
    cat = {"samples": {"A": {"data": [
        {"origins": [{"path": "A/volumes/x.zarr/", "access_roots": [{"type": "s3", "url": "s3://vesuvius-challenge-open-data"}]}]},
        {"origins": [{"path": "samples/A/volumes/y.zarr/", "access_roots": [{"type": "https", "url": "https://data.aws.ash2txt.org"}]}]},
        {"origins": [{"path": "A/photos/p.jpg", "access_roots": [{"url": "s3://vesuvius-challenge-open-data"}]}]}]}}}
    assert [p for p, _ in catalog_origins(cat)] == ["A/volumes/x.zarr", "samples/A/volumes/y.zarr"]
    assert catalog_zarr_paths(cat) == ["A/volumes/x.zarr"]


def test_verify_gap_reads_fine_chunks(tmp_path):
    pytest.importorskip("numcodecs")
    from zarrsweep.verify import find_and_verify_gaps

    name = make_store(tmp_path, shapes=((16, 16, 16), (8, 8, 8)), chunks=(8, 8, 8), write={1: []})
    out = find_and_verify_gaps(LocalStore(str(tmp_path)), name)
    assert out[0]["missing"] == 1 and out[0]["real_data_missing"] == 1
    assert out[0]["sampled"][0]["fine_chunks_examined"] == 8


def test_cli_exit_code_and_reports(tmp_path):
    name = make_store(tmp_path, write={1: []})
    j, m = tmp_path / "r.json", tmp_path / "r.md"
    rc = main([name, "--local-root", str(tmp_path), "--json", str(j), "--markdown", str(m)])
    assert rc == 1 and "LEVEL_EMPTY" in m.read_text() and json.loads(j.read_text())[0]["store"] == name
    clean = make_store(tmp_path, "c.zarr")
    assert main([clean, "--local-root", str(tmp_path), "--markdown", str(m)]) == 0
