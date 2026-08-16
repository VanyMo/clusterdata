#!/usr/bin/env python3
"""Seekable HTTP Range reader for very large remote ZIP archives."""
from __future__ import annotations

import io
import re
from collections import OrderedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$")


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "alibaba-clustertrace-range-slicer/0.1"
    return s


class HTTPRangeReader(io.RawIOBase):
    """Remote file backed only by byte-range GETs.

    Safety invariant: an origin response other than HTTP 206 aborts.  In
    particular, this refuses the dangerous OSS behavior where an invalid Range
    can fall back to HTTP 200 and return the entire object.
    """

    def __init__(self, url: str, block_size: int = 8 * 1024 * 1024, cache_blocks: int = 4):
        self.url = url
        self.block_size = int(block_size)
        self.cache_blocks = int(cache_blocks)
        self.session = _session()
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

        probe = self.session.get(
            url,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity", "x-oss-range-behavior": "standard"},
            timeout=(20, 120),
        )
        if probe.status_code != 206:
            raise RuntimeError(
                f"Range probe failed for {url}: HTTP {probe.status_code}; refusing full-object fallback"
            )
        match = _CONTENT_RANGE.match(probe.headers.get("Content-Range", ""))
        if not match:
            raise RuntimeError("missing/invalid Content-Range on Range probe")

        self.size = int(match.group(3))
        self.requests = 1
        self.bytes_downloaded = len(probe.content)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self.pos + offset
        elif whence == io.SEEK_END:
            new = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new < 0:
            raise ValueError("negative seek")
        self.pos = min(new, self.size)
        return self.pos

    def _get_block(self, index: int) -> bytes:
        if index in self.cache:
            data = self.cache.pop(index)
            self.cache[index] = data
            return data

        start = index * self.block_size
        if start >= self.size:
            return b""
        end = min(self.size - 1, start + self.block_size - 1)
        response = self.session.get(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "x-oss-range-behavior": "standard",
            },
            timeout=(20, 180),
        )
        if response.status_code != 206:
            raise RuntimeError(
                f"Range bytes={start}-{end} failed: HTTP {response.status_code}; refusing full-object fallback"
            )

        expected = end - start + 1
        data = response.content
        if len(data) != expected:
            raise IOError(f"short Range read: got {len(data)} bytes, expected {expected}")

        self.requests += 1
        self.bytes_downloaded += len(data)
        self.cache[index] = data
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.pos
        else:
            size = min(size, self.size - self.pos)

        out = bytearray()
        while size:
            block_index = self.pos // self.block_size
            in_block = self.pos % self.block_size
            block = self._get_block(block_index)
            take = min(size, len(block) - in_block)
            if take <= 0:
                break
            out += block[in_block : in_block + take]
            self.pos += take
            size -= take
        return bytes(out)

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)
