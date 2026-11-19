"""Download only the MMAD images a split needs, without the 28 GB of zips.

The MMAD zips on Hugging Face support HTTP range requests, so I read each zip's central directory
and then fetch just the members I need: the query images of the split, their single reference
(``random_templates[0]``, strict protocol) and, optionally, their masks (evaluation only).

    python scripts/fetch_mmad_subset.py --split dev               # dev images + references
    python scripts/fetch_mmad_subset.py --split dev --masks       # + masks, to score the expert
    python scripts/fetch_mmad_subset.py --split test --limit 50   # first 50 test images

Files land in ``data/MMAD/<source>/...`` (the same layout as the official zips). Already present
files are skipped, so the script can be re-run after a disconnect. Needs ``data/MMAD/mmad.json``
and ``data/splits/<split>_ids.json`` (committed in the repo; ``inspector.utils.splits`` makes them).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets/jiang-cc/MMAD/resolve/main/{}.zip"
ZIPS = ["DS-MVTec", "MVTec-AD", "VisA", "MVTec-LOCO", "GoodsAD"]  # zip name == top-level folder


def _get(url: str, start: int, end: int) -> tuple[bytes, int]:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "oneref"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"server ignored the range request (HTTP {r.status})")
                return r.read(), int(r.headers["Content-Range"].split("/")[1])
        except Exception:  # noqa: BLE001  (network hiccup: retry, then give up)
            if attempt == 3:
                raise
    raise AssertionError("unreachable")


def central_directory(url: str) -> dict[str, tuple[int, int, int]]:
    """Map member name -> (compression method, compressed size, local header offset)."""
    _, total = _get(url, 0, 0)
    tail, _ = _get(url, max(0, total - 65536), total - 1)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    j = tail.rfind(b"PK\x06\x06")  # ZIP64 end of central directory (the zips are > 4 GB)
    if j != -1:
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd, _ = _get(url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra, q = cd[p + 46 + nlen:p + 46 + nlen + xlen], 0
        while q + 4 <= len(extra):  # ZIP64 extra field holds the real sizes / offset
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, r = [], q + 4
                for v in (usize, csize, off):
                    if v == 0xFFFFFFFF:
                        v, = struct.unpack("<Q", extra[r:r + 8])
                        r += 8
                    vals.append(v)
                usize, csize, off = vals
            q += 4 + hlen
        out[name] = (method, csize, off)
        p += 46 + nlen + xlen + clen
    return out


def extract(url: str, entry: tuple[int, int, int], dest: Path) -> None:
    method, csize, off = entry
    hdr, _ = _get(url, off, off + 29)
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    start = off + 30 + nlen + xlen
    data, _ = _get(url, start, start + csize - 1) if csize else (b"", 0)
    if method == 8:
        data = zlib.decompress(data, -15)
    elif method != 0:
        raise RuntimeError(f"unsupported zip compression {method} for {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def wanted_files(mmad: dict, ids: set[str], masks: bool) -> set[str]:
    """Query image + random_templates[0] (+ mask) for every image whose id is in ``ids``."""
    files: set[str] = set()
    for key, e in mmad.items():
        image_id = "/".join(Path(key).parts[-4:]).rsplit(".", 1)[0]  # same id as inspector.schemas.mmad
        if image_id not in ids:
            continue
        files.add(key)
        files.add(e["random_templates"][0])
        if masks and e.get("mask_path"):
            category_dir = key[: -len(e["image_path"])]  # mask_path is relative to the category folder
            files.add(category_dir + e["mask_path"])     # LOCO: a folder of masks (prefix match below)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images of the split")
    ap.add_argument("--masks", action="store_true", help="also fetch masks (evaluation only, never an input)")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "MMAD")
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    mmad = json.loads((args.root / "mmad.json").read_text(encoding="utf-8"))
    ids = json.loads((args.splits_dir / f"{args.split}_ids.json").read_text(encoding="utf-8"))[: args.limit]
    files = wanted_files(mmad, set(ids), args.masks)
    todo = sorted(f for f in files if not (args.root / f).exists())
    print(f"{args.split}: {len(ids)} images -> {len(files)} files, {len(todo)} to download")

    missing = 0
    for zname in ZIPS:
        mine = [f for f in todo if f.split("/", 1)[0] == zname]
        if not mine:
            continue
        url = HF.format(zname)
        cd = central_directory(url)
        jobs = []
        for f in mine:
            names = [f] if f in cd else [n for n in cd if n.startswith(f.rstrip("/") + "/") and not n.endswith("/")]
            if not names:
                missing += 1
                print(f"  not in {zname}.zip: {f}", file=sys.stderr)
            jobs += [(n, args.root / n) for n in names if not (args.root / n).exists()]
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(lambda job: extract(url, cd[job[0]], job[1]), jobs))
        print(f"  {zname}: {len(jobs)} files")
    print("done" + (f" ({missing} files not found)" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
