# zarrsweep

Exhaustive, listing-first integrity sweep for the OME-Zarr stores (zarr **v2 and v3**, incl. sharded v3) in the
Vesuvius Challenge open-data bucket (`s3://vesuvius-challenge-open-data`). CPU-only, anonymous, no chunk
downloads except for the optional gap value check. Dependencies: `numpy`, `requests` (+ `numcodecs` for
`--verify-gaps`). MIT licensed.

## What it checks

From `.zattrs` / `.zarray` plus an S3 `ListObjectsV2` of each level (object keys and sizes only):

| code | severity | meaning |
|---|---|---|
| `STORE_MISSING` | error | a catalogued store has no `.zattrs/.zgroup/.zarray/zarr.json` |
| `LEVEL_META_MISSING` | error | multiscales lists a level whose `.zarray` is absent |
| `LEVEL_EMPTY` | error | a level has a header but zero chunk objects (every read returns `fill_value`) |
| `CHUNK_OVERSIZED` | error | chunk objects larger than an uncompressed chunk (+ codec framing): they can't decode to the declared chunk shape |
| `CHUNK_WRONG_SIZE_UNCOMPRESSED` | error | uncompressed array with chunk objects that aren't exactly one chunk |
| `CHUNK_OUT_OF_GRID` | warning | chunk keys outside the declared grid |
| **`PYRAMID_COVERAGE_GAP`** | warning/error | a coarse level is missing chunks under which the next finer level *has* chunks — i.e. readers that open the coarse level (VC3D overviews, `find_valid_patches`, downsampled training) see fill where the scan has data |
| `PYRAMID_SHAPE_SCALE_MISMATCH` | warning | a level's shape isn't consistent with the per-axis scale step (anisotropic pyramids like segment surface volumes, which only downsample y/x, are handled) |
| `PYRAMID_DTYPE_MISMATCH` | warning | dtype changes across levels |
| `AXES_NO_UNIT`, `SCALE_UNITLESS_IDENTITY` | warning | spatial axes without unit / level-0 scale `[1,1,1]` |
| `STRAY_OBJECTS`, `ZARR_V3_SKIPPED` | info | non-chunk objects in an array; v3 stores (not checked yet) |

`--verify-gaps N` turns each `PYRAMID_COVERAGE_GAP` into evidence: it re-lists the pair of levels, samples up
to N missing coarse chunks, downloads only the finer chunks that downsample into them, and counts non-fill
voxels. Non-zero counts mean the coarse level really lost data; zero means the finer chunks are written fill
and the gap is harmless.

### How it relates to existing audits (honest comparison)

Several community audits already cover much of this ground; zarrsweep overlaps with them and the unit /
empty-level / oversized-chunk findings below were **already reported** by them.

| | scroll-data-audit (Bullo27) | scroll-catalog-audit (gaogao94) | ScrolIQ `scroliq-chunk-audit` (Svyable) | **zarrsweep** |
|---|---|---|---|---|
| catalog ↔ array/filename/scan metadata (shape, dtype, pixel size, energy) | **yes** | partly (pixel size vs manifest) | — | no |
| pyramid value correctness (downsample operator on sampled blocks) | **yes** (`--deep`) | — | — | no |
| missing units / `[1,1,1]` scale | — | **yes** | — | yes |
| declared levels empty / missing | yes | **yes** | — | yes |
| chunk-object sizes vs declared chunk | — | — | **yes** (sampled ≈8 pages/level bucket-wide; exhaustive on 23 prize volumes) | yes, **every object of every level** |
| zarr v3 / sharded stores | v2 | headers | — | **yes** (listing + sizes, shard index accounted) |
| cross-level occupancy (coarse chunk missing under finer data) | — | — | — | **yes**, plus `--verify-gaps` value check |
| anisotropic (y/x-only) pyramids without false positives | ? | ? | — | yes |
| local store before upload / CI exit codes | — | — | — | yes |

What is new here: (1) the cross-level occupancy check and its targeted value confirmation, which found the
two previously unreported defects below; (2) exhaustive, not sampled, object enumeration of all 884 stores
(206 M objects in about an hour) including the 81 v3 ink stores; (3) a small, testable library/CLI usable
as a pre-upload gate. It does **not** replace scroll-data-audit's value-level pyramid checks or the
catalog cross-checks of scroll-catalog-audit.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e '.[verify,test]'
```

## Examples

```bash
# one store
zarrsweep PHerc0343P/volumes/20250521134555-8.640um-1.2m-116keV-masked.zarr --verify-gaps 5

# every .zarr in the bucket's metadata.json, with gap confirmation
zarrsweep --workers 16 --verify-gaps 5 --json reports/full_sweep.json --markdown reports/full_sweep.md --fail-on never

# only surface predictions
zarrsweep --include '*/representations/predictions/surfaces/*'

# a local store before upload (paths relative to --local-root); exit 1 on any error
zarrsweep my_volume.zarr --local-root /data/staging
```

Listing is fully paginated and, for nested chunk layouts, fanned out in parallel over the first chunk axis,
so there is no object cap by default (a 1.35 M-object level lists in ~20 s). `--max-keys-per-level N` caps it
for quick runs; capped levels are marked `listing_truncated` and skip emptiness/occupancy checks.

## Results of a full sweep (2026-10-14, bucket `metadata.json` of 2026-10-12)

884 stores (every `.zarr` origin served from this bucket; 10 origins hosted on `data.aws.ash2txt.org` are
skipped), **206,392,756 objects listed, no level truncated**, 81 zarr v3 stores included; 62 min wall on 8
cores. Full output: `reports/full_sweep.{md,json,log}`. (An earlier capped v0.1 run is kept in
`reports/v0.1-capped/`.)

| code | findings | status |
|---|---|---|
| `CHUNK_OVERSIZED` | 1 — PHerc0343P 8.64 µm level 0, 555/8543 objects | already reported: villa #1965 |
| `PYRAMID_COVERAGE_GAP` | 3 stores (value-checked below) | **2 new** |
| `AXES_NO_UNIT` / `SCALE_UNITLESS_IDENTITY` | 115 each (67 raw volumes, 43 surface predictions, 4 PHerc1447 surface volumes, 1 ink-3d) | already reported: #1951, #1957, #1962 |
| `STRAY_OBJECTS` | 1 (`.chunk_occupancy.npz` in PHercParis4 2.4 µm level 0) | benign |
| everything else (empty levels, missing stores, out-of-grid, wrong-size uncompressed, v3 shard sizes, shape/scale) | 0 | the empty store of #1963 has since been removed |

Coverage gaps, value-checked with `--verify-gaps 5`:

| store | levels | missing coarse chunks | finding |
|---|---|---|---|
| `PHerc0343P/volumes/20250521134555-8.640um-1.2m-116keV-masked.zarr` | 0→1 | 123 / 1737 | **new**: level 1 has holes exactly where level 0 has the #1965 oversized objects (all 123; none elsewhere); the objects under the holes are not empty (0.85 M–134 M non-zero bytes). Fixing only level 0 would leave level 1 broken. |
| `PHerc1299/representations/predictions/surfaces/20260309130042-surface-20260413222639-surface-m7-L2-th0.2.zarr` | 4→5 | 1 / 3 | **new**: level 5 lacks chunk `2/0/0` (its last z slice); level-4 source holds 6,378 surface voxels. Low impact (coarsest level only). |
| `PHerc1667/segments/20260130150000-w039_2026013015_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr` | 0→1, 1→2, 2→3, 4→5 | 3, 1, 1, 1 | 1–2 non-zero voxels at chunk borders; harmless |

Draft upstream reports (not posted): `drafts/1965_comment.md`, `drafts/new_issue_pherc1299.md`.

Limitations: listing can't detect corrupt-but-right-sized chunks (use scroll-data-audit `--deep` for values);
sharded v3 shards aren't decoded by `--verify-gaps`; only the first multiscale of a group is checked.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Synthetic v2 and v3 (incl. sharded) stores built on disk cover every check, the catalog origin filter, the gap verifier, CLI exit codes, and an emulated S3 `ListObjectsV2` (pagination, delimiter fan-out, caps).
