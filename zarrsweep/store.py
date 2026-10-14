"""Minimal read-only access to a public S3 bucket (HTTP GET + ListObjectsV2) or a local directory."""
from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from typing import Iterator, Optional
from urllib.parse import quote

import requests

DEFAULT_BUCKET_URL = "https://vesuvius-challenge-open-data.s3.amazonaws.com"

_KEY_SIZE = re.compile(rb"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", re.S)
_PREFIX = re.compile(rb"<CommonPrefixes><Prefix>([^<]*)</Prefix></CommonPrefixes>")
_TRUNC = re.compile(rb"<IsTruncated>true</IsTruncated>")
_TOKEN = re.compile(rb"<NextContinuationToken>([^<]*)</NextContinuationToken>")


class S3Store:
    """Anonymous S3 access over HTTPS. Listing is fully paginated (no object cap) and, for nested
    chunk layouts, fanned out over the first chunk axis so big levels list in parallel."""

    def __init__(self, base_url: str = DEFAULT_BUCKET_URL, timeout: float = 60, list_workers: int = 8):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.list_workers = list_workers
        self._local = threading.local()

    @property
    def s(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def _get(self, url, **kw):
        for attempt in range(4):
            try:
                r = self.s.get(url, timeout=self.timeout, **kw)
                if r.status_code >= 500 or r.status_code == 429:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                return r
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                if attempt == 3:
                    raise

    def get_json(self, key: str) -> Optional[dict]:
        r = self._get(f"{self.base}/{quote(key)}")
        if r.status_code in (403, 404):
            return None
        r.raise_for_status()
        return json.loads(r.content)

    def get_bytes(self, key: str) -> bytes:
        r = self._get(f"{self.base}/{quote(key)}")
        r.raise_for_status()
        return r.content

    def _pages(self, prefix: str, delimiter: Optional[str] = None):
        token = None
        while True:
            params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if delimiter:
                params["delimiter"] = delimiter
            if token:
                params["continuation-token"] = token
            r = self._get(self.base + "/", params=params)
            r.raise_for_status()
            yield r.content
            if not _TRUNC.search(r.content):
                return
            token = unescape(_TOKEN.search(r.content).group(1).decode())

    def _flat(self, prefix: str) -> list[tuple[str, int]]:
        out = []
        for page in self._pages(prefix):
            out += [(unescape(k.decode()), int(s)) for k, s in _KEY_SIZE.findall(page)]
        return out

    def list(self, prefix: str, max_keys: Optional[int] = None) -> Iterator[tuple[str, int]]:
        """Yield (key, size) under prefix (all of them unless max_keys is given)."""
        if max_keys is not None:
            n = 0
            for page in self._pages(prefix):
                for k, s in _KEY_SIZE.findall(page):
                    yield unescape(k.decode()), int(s)
                    n += 1
                    if n >= max_keys:
                        return
            return
        # fan out over sub-"directories" (first chunk index for nested layouts)
        files, subs = [], []
        for page in self._pages(prefix, delimiter="/"):
            files += [(unescape(k.decode()), int(s)) for k, s in _KEY_SIZE.findall(page)]
            subs += [unescape(p.decode()) for p in _PREFIX.findall(page)]
        yield from files
        if not subs:
            return
        if len(subs) == 1:  # e.g. zarr v3 "c/" — descend one more level to find the fan-out
            yield from self.list(subs[0])
            return
        with ThreadPoolExecutor(self.list_workers) as ex:
            for chunk in ex.map(self._flat, subs):
                yield from chunk


class LocalStore:
    """Same interface over a local directory (tests, or checking a store before upload)."""

    def __init__(self, root: str):
        self.root = root

    def get_json(self, key: str) -> Optional[dict]:
        p = os.path.join(self.root, key)
        if not os.path.isfile(p):
            return None
        with open(p, "rb") as f:
            return json.loads(f.read())

    def get_bytes(self, key: str) -> bytes:
        with open(os.path.join(self.root, key), "rb") as f:
            return f.read()

    def list(self, prefix: str, max_keys: Optional[int] = None) -> Iterator[tuple[str, int]]:
        n = 0
        base = os.path.join(self.root, prefix)
        start = base if os.path.isdir(base) else os.path.dirname(base)
        for dirpath, _, files in sorted(os.walk(start)):
            for fn in sorted(files):
                full = os.path.join(dirpath, fn)
                key = os.path.relpath(full, self.root).replace(os.sep, "/")
                if key.startswith(prefix):
                    yield key, os.path.getsize(full)
                    n += 1
                    if max_keys is not None and n >= max_keys:
                        return
