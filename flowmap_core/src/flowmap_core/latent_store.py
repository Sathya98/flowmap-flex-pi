"""Model-agnostic storage for caches of frozen-encoder outputs (latents, features).

A cache is a directory of large ``.npy`` arrays, memory-mapped on read, plus a
``manifest.json``. Every data array has one row per item of some *row kind*
(``manifest["arrays"][name]["rows"]``, e.g. one row per training window, or per
unique frame); ``create`` preallocates them sparse from the per-kind lengths, so
encoding can fill rows in any order and resume. ``done.npy`` ``[num_windows]``
marks flushed windows. Index arrays (``INDEX_ARRAYS`` of a subclass) are small
and written in full at creation.

Tensors are stored in numpy dtypes; bf16 as its int16 bit pattern (numpy has no
bfloat16), cast back on read, optionally to the dtype the encoder produced
(``source_dtype``). The model-specific layout (which arrays, row plans, how a
sample is assembled) lives in the model integration, e.g. FlexPi's
``flexpi.datasets.latent_cache.LatentCache``.
"""
import json
from pathlib import Path

import numpy as np
import torch

NP_DTYPES = {torch.bfloat16: np.int16, torch.float32: np.float32, torch.bool: np.bool_,
             torch.int64: np.int64, torch.int32: np.int32, torch.float16: np.float16}


def fingerprint_diff(a, b, prefix=""):
    """Dotted paths at which two fingerprints differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += fingerprint_diff(a.get(k), b.get(k), f"{prefix}{k}.")
        return out
    return [] if a == b else [prefix.rstrip(".")]


def select_windows(episode_from, episode_to, stride):
    """Window starts at every ``stride``-th frame of each episode (all frames at 1)."""
    return np.concatenate([np.arange(a, b, stride, dtype=np.int64)
                           for a, b in zip(np.asarray(episode_from), np.asarray(episode_to))])


def to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy()
    return t.numpy()


def from_numpy(a: np.ndarray, dtype: str, source_dtype: str = None) -> torch.Tensor:
    t = torch.from_numpy(np.array(a))   # copy out of the read-only memmap
    if dtype == "bfloat16":
        t = t.view(torch.bfloat16)
    return t if source_dtype in (None, dtype) else t.to(getattr(torch, source_dtype))


def load(spec, a) -> torch.Tensor:
    return from_numpy(a, spec["dtype"], spec.get("source_dtype"))


class ArrayStore:
    """Array store. ``create`` preallocates (sparse), ``open`` memory-maps.

    Subclasses set ``VERSION`` (checked on open; None skips the check) and
    ``INDEX_ARRAYS`` (names saved in full by ``create``).
    """
    VERSION = None
    INDEX_ARRAYS = ()

    def __init__(self, root, manifest):
        self.root, self.manifest, self._arrays = Path(root), manifest, {}

    @classmethod
    def create(cls, root, manifest, index, lengths):
        """``index`` holds the ``INDEX_ARRAYS``; ``lengths`` maps each row kind to its row count."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for name, spec in manifest["arrays"].items():
            np.lib.format.open_memmap(root / f"{name}.npy", mode="w+", dtype=np.dtype(spec["np_dtype"]),
                                      shape=(lengths[spec["rows"]], *spec["shape"])).flush()
        for name in cls.INDEX_ARRAYS:
            np.save(root / f"{name}.npy", index[name])
        np.lib.format.open_memmap(root / "done.npy", mode="w+", dtype=np.uint8,
                                  shape=(manifest["num_windows"],)).flush()
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return cls(root, manifest)

    @classmethod
    def open(cls, root):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if cls.VERSION is not None and manifest.get("version") != cls.VERSION:
            raise ValueError(f"{root}: cache version {manifest.get('version')} != {cls.VERSION}")
        return cls(root, manifest)

    def array(self, name, mode="r"):
        key = (name, mode)
        if key not in self._arrays:
            self._arrays[key] = np.load(self.root / f"{name}.npy", mmap_mode=mode)
        return self._arrays[key]

    def done(self, mode="r"):
        return self.array("done", mode)
