#!/usr/bin/env python3
"""Extract one pod+server hourly partition from Alibaba OSS via HTTP Range."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from rangezip import HTTPRangeReader

BASE = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data"
URLS = {
    "pod": f"{BASE}/asi_opensource_pod_hourly.zip",
    "server": f"{BASE}/asi_opensource_server_hourly.zip",
}


def member_name(kind: str, day: int, hour: int) -> str:
    return f"asi_opensource_{kind}_hourly/day={day}/hour={hour:02d}/part-000.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--hour", type=int, required=True)
    parser.add_argument("--out", default="out")
    parser.add_argument("--block-mib", type=int, default=8)
    args = parser.parse_args()

    if not 0 <= args.day <= 184:
        raise SystemExit("--day must be in 0..184")
    if not 0 <= args.hour <= 23:
        raise SystemExit("--hour must be in 0..23")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_id = f"d{args.day:03d}h{args.hour:02d}"
    manifest = {"slice_id": slice_id, "files": []}

    for kind, url in URLS.items():
        remote = HTTPRangeReader(url, block_size=args.block_mib * 1024 * 1024)
        with zipfile.ZipFile(remote) as archive:
            member = member_name(kind, args.day, args.hour)
            info = archive.getinfo(member)
            dst = out_dir / f"{kind}_d{args.day:03d}_h{args.hour:02d}_p000.parquet"
            with archive.open(info, "r") as src, dst.open("wb") as sink:
                shutil.copyfileobj(src, sink, 8 * 1024 * 1024)

        manifest["files"].append(
            {
                "kind": kind,
                "source_zip": url,
                "source_zip_bytes": remote.size,
                "member": member,
                "member_compressed_bytes": info.compress_size,
                "member_uncompressed_bytes": info.file_size,
                "range_requests": remote.requests,
                "range_bytes_downloaded": remote.bytes_downloaded,
                "output": dst.name,
                "output_bytes": dst.stat().st_size,
                "sha256": sha256(dst),
            }
        )

    manifest_path = out_dir / f"manifest_{slice_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
