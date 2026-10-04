#!/usr/bin/env python3
"""Pin the BANC deposit's files once (a developer tool; the two-flies work, docs/TWO_FLIES_PLAN.md 9.3): read the Harvard
Dataverse listing of one published version of the deposit (doi:10.7910/DVN/7WTH1N), download the files the kit builds the
female-with-a-nerve-cord from, check each one's size and MD5 against the listing, compute its SHA-256, and print the
``SOURCES`` block that virtual_fly/banc.py carries. Nothing here is committed: the files land in the kit's data folder
(``connectome.DATA_DIR / "banc-src"``) and are fetched again on another machine by the loader, pinned by these hashes.

    python tools/pin_banc.py                         # the latest published version, the four files, the block printed
    python tools/pin_banc.py --version 1.0 --files meta,edges_v3 --list-only

The deposit is CC BY 4.0 (Bates et al. 2026). The Dataverse API needs no key for these files; a `User-Agent` names the kit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from virtual_fly import __version__                               # noqa: E402
from virtual_fly.connectome import DATA_DIR, data_folder          # noqa: E402

DOI = "doi:10.7910/DVN/7WTH1N"
API = "https://dataverse.harvard.edu/api"
AGENT = f"virtual-fly/{__version__} (tools/pin_banc.py)"
WANTED = {                                   # key: the file's name in the deposit's compiled_data folder
    "meta": "banc_888_meta.feather",
    "edges_v3": "banc_888_edgelist_simple_v3.feather",
    "edges_v2": "banc_888_edgelist_simple_v2.feather",
    "transmitters": "banc_888_neurotransmitter_prediction_v2.csv",
}
DEFAULT_FILES = ("meta", "edges_v3", "transmitters")
SOURCE_DIR = DATA_DIR / "banc-src"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def listing(version: str) -> dict:
    """The published version's listing: {version, released, files: {key: {id, file, bytes, md5, restricted, directory}}}."""
    d = fetch_json(f"{API}/datasets/:persistentId/versions/{version}?persistentId={DOI}")
    if d.get("status") != "OK":
        raise SystemExit(f"Dataverse answered {d.get('status')}: {d.get('message')}")
    v = d["data"]
    files = {}
    for f in v["files"]:
        df = f["dataFile"]
        for key, name in WANTED.items():
            if df["filename"] == name and f.get("directoryLabel", "") == "compiled_data":
                ck = df.get("checksum") or {}
                files[key] = {"id": int(df["id"]), "file": name, "bytes": int(df["filesize"]),
                              "md5": ck.get("value") if ck.get("type") == "MD5" else None, "checksum_type": ck.get("type"),
                              "restricted": bool(f.get("restricted")), "directory": f.get("directoryLabel", "")}
    title = next((x["value"] for x in v.get("metadataBlocks", {}).get("citation", {}).get("fields", []) if x["typeName"] == "title"), "")
    return {"version": f"{v['versionNumber']}.{v['versionMinorNumber']}", "released": v.get("releaseTime"),
            "state": v.get("versionState"), "license": (v.get("license") or {}).get("name"), "title": title,
            "guestbook": v.get("guestbook"), "files": files}


def _digests(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def download(entry: dict, folder: Path, quiet: bool = False) -> Path:
    """One file into the folder (skipped when a file of the right MD5 is there); size and MD5 checked; no .part left."""
    path = folder / entry["file"]
    if path.exists() and path.stat().st_size == entry["bytes"] and _digests(path)[0] == entry["md5"]:
        if not quiet:
            print(f"  {entry['file']}: already here, MD5 matches", file=sys.stderr)
        return path
    url = f"{API}/access/datafile/{entry['id']}"
    if not quiet:
        print(f"  {entry['file']}: downloading {entry['bytes'] / 1e6:,.1f} MB from {url}", file=sys.stderr)
    tmp = path.with_suffix(path.suffix + ".part")
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": AGENT})
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except BaseException:                            # Ctrl+C or a network error: leave no half file behind
        tmp.unlink(missing_ok=True)
        raise
    size = tmp.stat().st_size
    if size != entry["bytes"]:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{entry['file']}: Dataverse sent {size:,} bytes, the listing says {entry['bytes']:,}")
    md5, _ = _digests(tmp)
    if entry["md5"] and md5 != entry["md5"]:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{entry['file']}: MD5 {md5} does not match the listing's {entry['md5']}")
    tmp.replace(path)
    if not quiet:
        print(f"    {size:,} bytes in {time.time() - t0:.0f} s, MD5 matches", file=sys.stderr)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=":latest-published", help="the deposit's published version (1.0, 2.0, 2.1, 3.0, ...; "
                                                                   "default: the latest published)")
    ap.add_argument("--files", default=",".join(DEFAULT_FILES), help="comma list of " + ", ".join(WANTED))
    ap.add_argument("--dir", type=Path, default=SOURCE_DIR, help=f"where the files go (default {SOURCE_DIR})")
    ap.add_argument("--list-only", action="store_true", help="print the listing and stop (no download)")
    ap.add_argument("--json", type=Path, help="also write the listing and the hashes here")
    args = ap.parse_args(argv)
    keys = [k.strip() for k in args.files.split(",") if k.strip()]
    bad = [k for k in keys if k not in WANTED]
    if bad:
        ap.error(f"unknown file key(s) {', '.join(bad)}: choose from {', '.join(WANTED)}")
    lst = listing(args.version)
    print(f"BANC deposit {DOI}, version {lst['version']} ({lst['state']}, released {lst['released']}), licence {lst['license']}, "
          f"guestbook {lst['guestbook'] or 'none'}: {lst['title']}", file=sys.stderr)
    missing = [k for k in keys if k not in lst["files"]]
    if missing:
        raise SystemExit(f"this version has no {', '.join(WANTED[k] for k in missing)} in compiled_data")
    for k in keys:
        e = lst["files"][k]
        print(f"  {k:12s} {e['file']:48s} id {e['id']:9d}  {e['bytes']:>12,} B  MD5 {e['md5']}  restricted {e['restricted']}", file=sys.stderr)
        if e["restricted"]:
            raise SystemExit(f"{e['file']} is restricted on Dataverse: stop and ask")
    if args.list_only:
        return 0
    folder = data_folder(args.dir)
    out = {}
    for k in keys:
        e = lst["files"][k]
        path = download(e, folder)
        md5, sha = _digests(path)
        out[k] = {**e, "sha256": sha, "md5_measured": md5, "url": f"{API}/access/datafile/{e['id']}",
                  "version": lst["version"], "released": lst["released"]}
    print("\nSOURCES = {   # the BANC deposit on Harvard Dataverse, " + f"version {lst['version']} ({lst['released']}), CC BY 4.0")
    for k, e in out.items():
        print(f'    "{k}": {{"url": "{e["url"]}", "file": "{e["file"]}", "id": {e["id"]}, "version": "{e["version"]}",\n'
              f'        "mb": {round(e["bytes"] / 1e6)}, "bytes": {e["bytes"]}, "md5": "{e["md5"]}",\n'
              f'        "sha256": "{e["sha256"]}"}},')
    print("}")
    if args.json:
        args.json.write_text(json.dumps({"doi": DOI, "listing": lst, "files": out, "when": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
