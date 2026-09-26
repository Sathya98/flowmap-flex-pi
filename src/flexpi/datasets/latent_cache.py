"""On-disk cache of the frozen-encoder outputs of ``FlexPi.build_inputs``.

``build_inputs`` runs three frozen encoders on every training batch: the Wan2.2
VAE on the RGB composite, the same VAE on the pointmap composite, and DINOv3 on
the per-camera frames. None of them depends on trainable weights, and the
LIBERO/RoboTwin data pipelines have no random augmentation, so their outputs
are a pure function of the dataset index and can be computed once.

A *window* is one training sample: the clip starting at dataset index ``g``.
The cache holds ``M`` windows (``windows.npy``): every index, or with
``window_stride=k`` only the windows starting at every k-th frame of each
episode. Layout (a few large ``.npy`` files, memory-mapped on read):

- ``video_latents``/``pointmap_latents`` ``[M, C, F, H, W]``: per window. The
  Wan VAE is temporally causal, so a window's latents depend on where it starts.
- ``dino_frames`` ``[R, C, T_kept]``: per *frame* (``dino_rows.npy`` lists the
  frames). DINOv3 encodes every frame independently, so a frame shared by
  several windows (as anchor or as a later DINO frame) is stored once;
  ``dino_row_index`` ``[M, F_dino]`` maps a window's DINO frames to rows. Later
  frames are clamped at the episode end exactly as LeRobot clamps the query.
  Tokens identical for every frame (LIBERO's synthetic black camera) are stored
  once in ``dino_const.npy``.
- Small per-window tensors straight from the dataset (``action``, ``proprio``,
  pad flags, intrinsics) and the prompt; the umT5 context is re-read from the
  existing text-embedding cache, which is keyed by prompt.
- ``done`` ``[M]`` marks windows whose arrays were flushed; encoding is resumable.

Encoder outputs are stored as bf16 (its int16 bit pattern; numpy has no
bfloat16) and cast back on read to the dtype ``build_inputs`` produced
(``source_dtype``): DINO comes out float32 under autocast, and the noise and
targets built from it must keep that dtype. The generic store (sparse preallocation, memory maps, dtype round trip) is
``flowmap_core.latent_store.ArrayStore``; ``LatentCache`` adds the DINO row layout. The proprio
token is NOT cached: ``proprio_encoder`` is trainable, so ``build_inputs_from_cache``
appends it at train time exactly like ``build_inputs``.
"""
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flowmap_core.latent_store import (  # noqa: F401  (re-exported for FlexPi callers)
    NP_DTYPES as _NP_DTYPES, ArrayStore, fingerprint_diff, from_numpy, load, select_windows, to_numpy,
)

CACHE_VERSION = 2
PROMPT_BYTES = 512
CONTEXT_MEMO = 256   # prompts held in memory per worker (RoboTwin has ~1M)
# Dataset keys build_inputs consumes that we do not store per window: raw
# pixels/depth are what the cache replaces; context is rebuilt from the prompt.
_RAW_KEYS = {"per_cam", "per_cam_depth", "video", "context", "context_mask", "prompt"}
_ENCODER_MODEL_KEYS = (
    "model_id", "composite_layout", "composite_layout_slot_key_map",
    "dino_model_name", "dino_cam_patches", "dino_cam_regions", "dino_temporal_stride",
    "dino_stride_keep_far", "dino_pool_mode", "dino_pixel_unshuffle", "dino_pool_factor",
    "enable_pointmap", "pointmap_norm_bounds", "pointmap_max_depth_m",
)
# Index arrays saved beside the manifest, fully written at init.
_INDEX_ARRAYS = ("windows", "dino_rows", "dino_row_index", "dino_writer", "dino_row_owner")


def _container(node):
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def encoder_fingerprint(cfg) -> dict:
    """Everything that determines the cached tensors; must match at train time."""
    stats = cfg.data.train.get("pretrained_norm_stats")
    return {
        "data_train": _container(cfg.data.train),
        "model": {k: _container(cfg.model.get(k)) for k in _ENCODER_MODEL_KEYS},
        "fuse_vae_embedding_in_latents": bool(cfg.model.video_dit_config.get("fuse_vae_embedding_in_latents", False)),
        "mixed_precision": str(cfg.mixed_precision),
        "norm_stats_sha256": None if not stats else hashlib.sha256(Path(stats).read_bytes()).hexdigest(),
    }


def dino_frame_offsets(video_sample_indices, temporal_stride, keep_far, global_sample_stride):
    """Raw-frame offset (from the window start) of every DINO frame in a window.

    Mirrors ``DinoEncoder.encode_video``: VAE-aligned slots are every 4th of the
    subsampled video frames, then the aux temporal stride picks among them.
    """
    from flexpi.models.helpers.dino import select_aux_frame_slots
    T = len(video_sample_indices)
    aligned = [min(4 * i, T - 1) for i in range((T - 1) // 4 + 1)]
    slots = select_aux_frame_slots(len(aligned), temporal_stride, keep_far=keep_far)
    return [int(video_sample_indices[aligned[s]]) * int(global_sample_stride) for s in slots]


def window_frames(windows, episode_to, offsets):
    """``[M, len(offsets)]`` global frame shown by each DINO frame of each window."""
    ends = np.asarray(episode_to, dtype=np.int64)
    last = ends[np.searchsorted(ends, windows, side="right")] - 1   # last frame of the episode
    return np.stack([np.minimum(windows + o, last) for o in offsets], axis=1)


def dino_row_plan(frames):
    """Rows (unique frames), each window's row indices, and which window writes each row.

    A row is written by the first (window, DINO frame) that shows it, so every
    row has exactly one writer and nothing is encoded twice into the cache.
    """
    rows, flat_index = np.unique(frames, return_inverse=True)
    row_index = flat_index.reshape(frames.shape)
    _, first = np.unique(row_index.ravel(), return_index=True)
    writer = np.zeros(frames.size, dtype=bool)
    writer[first] = True
    return rows, row_index, writer.reshape(frames.shape), first // frames.shape[1]


class LatentCache(ArrayStore):
    """FlexPi's cache: per-window latents plus DINO rows shared between windows."""
    VERSION = CACHE_VERSION
    INDEX_ARRAYS = _INDEX_ARRAYS

    @classmethod
    def create(cls, root, manifest, index):
        """``index`` holds the ``_INDEX_ARRAYS``; the data arrays start empty."""
        lengths = {"window": manifest["num_windows"], "dino": manifest["num_dino_rows"]}
        return super().create(root, manifest, index, lengths)

    def dino_const(self):
        if "_dino_const" not in self._arrays:
            path = self.root / "dino_const.npy"
            self._arrays["_dino_const"] = from_numpy(np.load(path), "bfloat16") if path.exists() else None
        return self._arrays["_dino_const"]

    def dino_frame(self, row) -> torch.Tensor:
        """``[C, T]`` DINO features of one cached frame, constant tokens restored."""
        spec = self.manifest["dino"]
        kept = from_numpy(self.array("dino_frames")[row], "bfloat16")
        const_idx = spec["const_token_idx"]
        if not const_idx:
            return kept
        out = torch.empty((kept.shape[0], spec["num_tokens"]), dtype=kept.dtype)
        out[:, spec["kept_token_idx"]] = kept
        out[:, const_idx] = self.dino_const()
        return out

    def dino_features(self, m) -> torch.Tensor:
        """``[C, F, T, 1]``: window ``m``'s DINO input, as ``DinoEncoder.encode_video`` returns it."""
        rows = self.array("dino_row_index")[m].tolist()
        out = torch.stack([self.dino_frame(r) for r in rows], dim=1).unsqueeze(-1)
        source = self.manifest["arrays"]["dino_frames"].get("source_dtype", "bfloat16")
        return out.to(getattr(torch, source))

    def ready(self):
        """``[M]`` bool: window encoded and every DINO row it reads written."""
        done = np.asarray(self.done()).astype(bool)
        owner = np.asarray(self.array("dino_row_owner"))
        return done & done[owner[np.asarray(self.array("dino_row_index"))]].all(axis=1)


class CachedLatentDataset(torch.utils.data.Dataset):
    """Serves cached encoder outputs in place of pixels/depth.

    Index ``i`` is the i-th cached window (``windows[i]`` in the raw dataset).
    Wraps the raw dataset it was built from: the raw dataset supplies the text
    context (by prompt) and every attribute the trainer reads (``dataset_weights``,
    ``lerobot_dataset.processor`` ...).
    """

    def __init__(self, cache_dir, raw_dataset, require_complete=True):
        self.cache = LatentCache.open(cache_dir)
        self.raw_dataset = raw_dataset
        m = self.cache.manifest
        if len(raw_dataset) != m["dataset_len"]:
            raise ValueError(f"Cache was built for {m['dataset_len']} dataset indices, dataset has {len(raw_dataset)}")
        if m["num_windows"] != m["dataset_len"] and getattr(raw_dataset, "dataset_weights", None) is not None:
            raise ValueError("Weighted multi-dataset sampling indexes raw frames; it cannot use a strided cache")
        missing = int(m["num_windows"] - self.cache.done().sum())
        if require_complete and missing:
            raise ValueError(f"{cache_dir}: {missing} windows are not encoded yet")
        self._contexts = OrderedDict()

    def __getattr__(self, name):
        raw = self.__dict__.get("raw_dataset")
        if raw is None:
            raise AttributeError(name)
        return getattr(raw, name)

    def __len__(self):
        return self.cache.manifest["num_windows"]

    def _context(self, prompt):
        if prompt in self._contexts:
            self._contexts.move_to_end(prompt)
            return self._contexts[prompt]
        context, mask = self.raw_dataset._get_cached_text_context(prompt)
        context[~mask] = 0.0   # same post-processing as RobotVideoDataset._get
        self._contexts[prompt] = (context, torch.ones_like(mask))
        if len(self._contexts) > CONTEXT_MEMO:
            self._contexts.popitem(last=False)
        return self._contexts[prompt]

    def __getitem__(self, i):
        c = self.cache
        if not c.done()[i]:
            raise RuntimeError(f"Cached window {i} is not encoded yet")
        arrays = c.manifest["arrays"]
        sample = {"idx": int(c.array("windows")[i])}
        for name in c.manifest["window_keys"]:
            sample[name] = load(arrays[name], c.array(name)[i])
        sample["cached_input_latents"] = load(arrays["video_latents"], c.array("video_latents")[i])
        if "pointmap_latents" in arrays:
            sample["cached_pointmap_latents"] = load(arrays["pointmap_latents"], c.array("pointmap_latents")[i])
        sample["cached_dino_features"] = c.dino_features(i)
        prompt = bytes(c.array("prompt")[i]).decode("utf-8")
        sample["prompt"] = prompt
        sample["context"], sample["context_mask"] = self._context(prompt)
        return sample


def build_inputs_from_cache(model, sample):
    """``FlexPi.build_inputs`` for a batch of ``CachedLatentDataset`` samples.

    Mirrors ``FlexPiBackbone.build_inputs`` + ``FlexPi.build_inputs`` minus the
    frozen encoders. ``scripts/cache_latents.py verify`` checks the two agree.
    """
    dev, dt = model.device, model.torch_dtype
    context = sample["context"].to(device=dev, dtype=dt, non_blocking=True)
    context_mask = sample["context_mask"].to(device=dev, dtype=torch.bool, non_blocking=True)
    if model.proprio_encoder is not None:
        proprio = sample.get("proprio")
        if proprio is None or proprio.ndim != 3 or proprio.shape[2] != model.proprio_dim:
            raise ValueError("Cached sample needs `proprio` [B, T, proprio_dim]")
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask,
            proprio=proprio[:, 0, :].to(device=dev, dtype=dt))
    # Encoder outputs keep the dtype build_inputs gave them (restored on read).
    input_latents = sample["cached_input_latents"].to(device=dev, non_blocking=True)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    def flag(key):
        value = sample.get(key)
        return None if value is None else value.to(device=dev, dtype=torch.bool, non_blocking=True)

    inputs = {
        "context": context,
        "context_mask": context_mask,
        "input_latents": input_latents,
        "first_frame_latents": input_latents[:, :, 0:1] if fuse else None,
        "fuse_vae_embedding_in_latents": fuse,
        "action": sample["action"].to(device=dev, dtype=dt, non_blocking=True),
        "action_is_pad": flag("action_is_pad"),
        "action_dim_is_pad": flag("action_dim_is_pad"),
        "image_is_pad": flag("image_is_pad"),
        "dino_features": sample["cached_dino_features"].to(device=dev, non_blocking=True),
    }
    if model._pointmap_globally_off:
        inputs["pointmap_raw"] = None
    else:
        if "cached_pointmap_latents" not in sample:
            raise KeyError("Latent cache has no pointmap latents but the model has a pointmap stream")
        model.set_camera_intrinsics(sample["camera_intrinsics"].to(device=dev))
        inputs["pointmap_raw"] = sample["cached_pointmap_latents"].to(device=dev, non_blocking=True)
    return inputs
