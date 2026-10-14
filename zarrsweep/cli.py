"""zarrsweep CLI."""
from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import sys

from .store import DEFAULT_BUCKET_URL, LocalStore, S3Store
from .sweep import catalog_zarr_paths, load_catalog, sweep_many


def render_markdown(results: list[dict]) -> str:
    counts = collections.Counter()
    stores_by_code = collections.defaultdict(list)
    for r in results:
        for f in r["findings"]:
            counts[(f["severity"], f["code"])] += 1
            stores_by_code[f["code"]].append(f)
    lines = ["# zarrsweep report", "", f"Stores swept: {len(results)}", "",
             "| severity | code | findings |", "|---|---|---|"]
    order = {"error": 0, "warning": 1, "info": 2}
    for (sev, code), n in sorted(counts.items(), key=lambda kv: (order.get(kv[0][0], 9), -kv[1])):
        lines.append(f"| {sev} | {code} | {n} |")
    for code, fs in sorted(stores_by_code.items()):
        lines += ["", f"## {code}", ""]
        for f in fs[:40]:
            lvl = f" level `{f['level']}`" if f["level"] not in (None, ".") else ""
            lines.append(f"- `{f['store']}`{lvl}: {f['message']}")
        if len(fs) > 40:
            lines.append(f"- ... and {len(fs) - 40} more (see JSON)")
    ver = [v for r in results for v in r.get("gap_verification", [])]
    if ver:
        lines += ["", "## Coverage-gap value check", "", "| store | fine -> coarse | missing coarse chunks | sampled | sampled with non-fill data under them |", "|---|---|---|---|---|"]
        for v in ver:
            lines.append(f"| `{v['store']}` | {v['fine']} -> {v['coarse']} | {v['missing']} / {v['expected']} | "
                         f"{len(v['sampled'])} | {v['real_data_missing']} (non-fill voxels: "
                         f"{', '.join(str(c['nonfill_voxels']) for c in v['sampled'])}) |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zarrsweep", description=__doc__)
    ap.add_argument("stores", nargs="*", help="store prefixes (bucket-relative) to sweep; default: every .zarr in metadata.json")
    ap.add_argument("--bucket-url", default=DEFAULT_BUCKET_URL)
    ap.add_argument("--local-root", help="sweep a local directory instead of the bucket")
    ap.add_argument("--include", action="append", default=[], help="glob filter on catalog paths (repeatable)")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--limit", type=int, help="sweep at most N stores")
    ap.add_argument("--max-keys-per-level", type=int, default=0,
                    help="stop listing a level after N objects (default 0 = list everything); truncated levels "
                         "skip emptiness/occupancy checks")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", dest="json_out", help="write full results JSON here")
    ap.add_argument("--markdown", help="write a markdown summary here")
    ap.add_argument("--verify-gaps", type=int, default=0, metavar="N",
                    help="for stores with PYRAMID_COVERAGE_GAP, download up to N finer chunks per gap and report "
                         "whether they hold non-fill data (needs numcodecs)")
    ap.add_argument("--fail-on", choices=["error", "warning", "never"], default="error")
    a = ap.parse_args(argv)

    store = LocalStore(a.local_root) if a.local_root else S3Store(a.bucket_url)
    prefixes = a.stores
    if not prefixes:
        if a.local_root:
            ap.error("give store paths when using --local-root")
        prefixes = catalog_zarr_paths(load_catalog(store))
    if a.include:
        prefixes = [p for p in prefixes if any(fnmatch.fnmatch(p, g) for g in a.include)]
    if a.exclude:
        prefixes = [p for p in prefixes if not any(fnmatch.fnmatch(p, g) for g in a.exclude)]
    if a.limit:
        prefixes = prefixes[: a.limit]
    mk = a.max_keys_per_level or None
    results = sweep_many(store, prefixes, workers=a.workers, max_keys_per_level=mk,
                         progress=lambda i, n, p: print(f"[{i}/{n}] {p}", file=sys.stderr))
    if a.verify_gaps:
        from .verify import find_and_verify_gaps
        for r in results:
            if any(f["code"] == "PYRAMID_COVERAGE_GAP" for f in r["findings"]):
                r["gap_verification"] = find_and_verify_gaps(store, r["store"], sample=a.verify_gaps)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(results, f, indent=1)
    md = render_markdown(results)
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write(md)
    else:
        print(md)
    sevs = {f["severity"] for r in results for f in r["findings"]}
    if a.fail_on == "error" and "error" in sevs:
        return 1
    if a.fail_on == "warning" and sevs & {"error", "warning"}:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
