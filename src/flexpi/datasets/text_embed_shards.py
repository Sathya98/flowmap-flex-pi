"""Read-only access to a sharded umT5 text-embedding cache.

The RoboTwin 3D release ships its ~1M prompt embeddings as
``shards/shard_NNNNN.safetensors`` (``contexts`` [N, L, D] bf16, ``masks``
[N, L] bool, ``__metadata__["keys"]``) plus ``manifest.txt``: every sha256 key,
sorted, where line ``i`` is row ``i % ROWS`` of shard ``i // ROWS``. Unpacking
it into one ``.pt`` per prompt would cost ~1M inodes, so it is read in place:
one row (one prompt) per lookup, via the memory-mapped safetensors file.
"""
import json
from pathlib import Path

import numpy as np

ROWS_PER_SHARD = 2000


class ShardedTextEmbeds:
    def __init__(self, root):
        self.root = Path(root)
        self._keys = None       # sorted S64 array, loaded lazily (per worker)
        self._handles = {}      # shard index -> (safe_open handle, row keys)

    @staticmethod
    def present(root) -> bool:
        root = Path(root)
        return (root / "manifest.txt").is_file() and (root / "shards").is_dir()

    def _index(self, key: str) -> int:
        if self._keys is None:
            self._keys = np.array((self.root / "manifest.txt").read_text().split(), dtype="S64")
        needle = key.encode()
        i = int(np.searchsorted(self._keys, needle))
        if i >= len(self._keys) or self._keys[i] != needle:
            raise KeyError(key)
        return i

    def _shard(self, s):
        if s not in self._handles:
            from safetensors import safe_open
            f = safe_open(str(self.root / "shards" / f"shard_{s:05d}.safetensors"), framework="pt")
            self._handles[s] = (f, json.loads(f.metadata()["keys"]))
        return self._handles[s]

    def get(self, key: str):
        """``(context [L, D], mask [L] bool)`` for a sha256 prompt key."""
        s, row = divmod(self._index(key), ROWS_PER_SHARD)
        f, keys = self._shard(s)
        if keys[row] != key:
            raise RuntimeError(f"{self.root}: manifest says {key} is shard {s} row {row}, shard has {keys[row]}")
        return f.get_slice("contexts")[row:row + 1][0], f.get_slice("masks")[row:row + 1][0].bool()
