"""Download MMAD from Hugging Face (jiang-cc/MMAD): a small sample, or everything.

    python scripts/download_mmad.py --sample                 # mmad.json + DS-MVTec.zip (~1.6 GB)
    python scripts/download_mmad.py --sample --zips VisA.zip
    python scripts/download_mmad.py --full                  # all zips (~28 GB)
    python scripts/download_mmad.py --sample --dest D:/MMAD # then ONEREF_MMAD_ROOT=D:/MMAD

Then run the pipeline with ``--available-only`` so that only the images that exist are used.
Needs HF_TOKEN (env) if the dataset is gated. Zips are deleted after extraction (--keep-zips to keep).
Zip sizes: DS-MVTec 1.6 GB, VisA 1.9 GB, MVTec-AD 5.3 GB, MVTec-LOCO 6.1 GB, GoodsAD 13.3 GB.
"""
import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "jiang-cc/MMAD"
ALL_ZIPS = ["DS-MVTec.zip", "VisA.zip", "MVTec-AD.zip", "MVTec-LOCO.zip", "GoodsAD.zip"]


def download_mmad(dest: str | Path = "data/MMAD", zips: list[str] | None = None, keep_zips: bool = False) -> Path:
    """Fetch ``mmad.json`` and the given zips (default: the smallest, DS-MVTec) into ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    hf_hub_download(REPO, "mmad.json", repo_type="dataset", local_dir=dest, token=token)
    for name in zips or ["DS-MVTec.zip"]:
        z = Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=dest, token=token))
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
        if not keep_zips:
            z.unlink()
        print(f"{name}: extracted to {dest}")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="mmad.json + one small zip (default DS-MVTec.zip)")
    g.add_argument("--full", action="store_true", help="mmad.json + all zips (~28 GB)")
    ap.add_argument("--zips", nargs="+", choices=ALL_ZIPS, help="zips for --sample")
    ap.add_argument("--dest", default=os.environ.get("ONEREF_MMAD_ROOT", "data/MMAD"))
    ap.add_argument("--keep-zips", action="store_true")
    a = ap.parse_args()
    download_mmad(a.dest, ALL_ZIPS if a.full else a.zips, a.keep_zips)
