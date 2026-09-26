"""Latent-cache storage, window/DINO-row planning and cached build_inputs."""
import tempfile

import numpy as np
import torch
from test_flowmap import tiny_model

from flexpi.datasets import latent_cache as lc


def test_dino_offsets_and_episode_clamp():
    assert lc.dino_frame_offsets(range(0, 33, 4), 2, True, 1) == [0, 32]
    assert lc.dino_frame_offsets(range(0, 33, 4), 1, False, 1) == [0, 16, 32]
    frames = lc.window_frames(np.arange(12), [5, 12], [0, 3])
    assert frames[:, 1].tolist() == [3, 4, 4, 4, 4, 8, 9, 10, 11, 11, 11, 11]


def test_strided_windows_share_dino_rows():
    # Two episodes [0, 20) and [20, 45); stride 8, DINO offsets 0/16/32.
    windows = lc.select_windows([0, 20], [20, 45], 8)
    assert windows.tolist() == [0, 8, 16, 20, 28, 36, 44]
    frames = lc.window_frames(windows, [20, 45], [0, 16, 32])
    rows, row_index, writer, owner = lc.dino_row_plan(frames)
    # Later frames land on other windows' anchors or on the clamped episode ends.
    assert rows.tolist() == [0, 8, 16, 19, 20, 28, 36, 44]
    assert np.array_equal(rows[row_index], frames)
    assert writer.sum() == len(rows)                      # one writer per row
    assert sorted(frames[writer].tolist()) == rows.tolist()
    assert owner.tolist() == [0, 1, 0, 0, 3, 4, 3, 3]     # first window showing the frame


class _Raw:
    def __init__(self, n):
        self.n, self.dataset_weights = n, None

    def __len__(self):
        return self.n

    def _get_cached_text_context(self, prompt):
        mask = torch.tensor([True, True, False])
        return torch.full((3, 16), float(len(prompt))), mask


def _cache(root, n=6, stride=1, tokens=5, const=(4,)):
    kept = [t for t in range(tokens) if t not in const]
    windows = lc.select_windows([0, 3], [3, n], stride)
    frames = lc.window_frames(windows, [3, n], [0, 2])
    rows, row_index, writer, owner = lc.dino_row_plan(frames)
    manifest = {
        "version": lc.CACHE_VERSION, "dataset_len": n, "num_windows": len(windows),
        "num_dino_rows": len(rows), "window_stride": stride,
        "window_keys": ["action", "action_is_pad", "proprio"],
        "arrays": {
            "video_latents": {"shape": [4, 2, 4, 4], "np_dtype": "<i2", "dtype": "bfloat16", "rows": "window"},
            "pointmap_latents": {"shape": [4, 2, 4, 4], "np_dtype": "<i2", "dtype": "bfloat16", "rows": "window"},
            "dino_frames": {"shape": [8, len(kept)], "np_dtype": "<i2", "dtype": "bfloat16",
                            "source_dtype": "float32", "rows": "dino"},
            "prompt": {"shape": [], "np_dtype": f"S{lc.PROMPT_BYTES}", "dtype": "bytes", "rows": "window"},
            "action": {"shape": [4, 8], "np_dtype": "<f4", "dtype": "float32", "rows": "window"},
            "action_is_pad": {"shape": [4], "np_dtype": "|b1", "dtype": "bool", "rows": "window"},
            "proprio": {"shape": [4, 8], "np_dtype": "<f4", "dtype": "float32", "rows": "window"},
        },
        "dino": {"frame_offsets": [0, 2], "num_tokens": tokens, "const_token_idx": list(const), "kept_token_idx": kept},
    }
    np.save(f"{root}/dino_const.npy", lc.to_numpy(torch.full((8, len(const)), 7., dtype=torch.bfloat16)))
    cache = lc.LatentCache.create(root, manifest, dict(
        windows=windows, dino_rows=rows, dino_row_index=row_index, dino_writer=writer, dino_row_owner=owner))
    frame_feats = torch.randn(n, 8, tokens, dtype=torch.bfloat16)   # per raw frame
    frame_feats[:, :, list(const)] = 7.
    video = torch.randn(len(windows), 4, 2, 4, 4, dtype=torch.bfloat16)
    for p, g in enumerate(windows):
        cache.array("video_latents", "r+")[p] = lc.to_numpy(video[p])
        cache.array("pointmap_latents", "r+")[p] = lc.to_numpy(-video[p])
        for j in np.flatnonzero(writer[p]):
            cache.array("dino_frames", "r+")[row_index[p, j]] = lc.to_numpy(frame_feats[frames[p, j]][:, kept])
        cache.array("action", "r+")[p] = np.full((4, 8), g, np.float32)
        cache.array("proprio", "r+")[p] = np.zeros((4, 8), np.float32)
        cache.array("prompt", "r+")[p] = f"task {g % 2}".encode()
    for name in manifest["arrays"]:
        cache.array(name, "r+").flush()
    cache.done("r+")[:] = 1
    cache.done("r+").flush()
    return frame_feats, video


def test_cached_sample_restores_bf16_constant_tokens_and_far_frame():
    with tempfile.TemporaryDirectory() as root:
        frames, video = _cache(root)
        ds = lc.CachedLatentDataset(root, _Raw(6))
        s = ds[1]
        assert torch.equal(s["cached_input_latents"], video[1])
        assert torch.equal(s["cached_pointmap_latents"], -video[1])
        # window 1 lives in episode [0, 3): its far frame (1 + 2) clamps to 2
        # stored bf16, restored to build_inputs' float32
        assert torch.equal(s["cached_dino_features"][..., 0], torch.stack([frames[1], frames[2]], 1).float())
        assert s["prompt"] == "task 1" and torch.equal(s["context"][2], torch.zeros(16))
        assert bool(s["context_mask"].all()) and float(s["action"][0, 0]) == 1
        assert ds.dataset_weights is None   # attributes fall through to the raw dataset


def test_strided_dataset_indexes_cached_windows():
    with tempfile.TemporaryDirectory() as root:
        frames, _ = _cache(root, n=6, stride=2)     # windows 0, 2 | 3, 5
        ds = lc.CachedLatentDataset(root, _Raw(6))
        assert len(ds) == 4
        s = ds[2]
        assert s["idx"] == 3 and float(s["action"][0, 0]) == 3
        assert torch.equal(s["cached_dino_features"][..., 0], torch.stack([frames[3], frames[5]], 1).float())
        assert ds.cache.ready().all()


def test_build_inputs_from_cache_matches_build_inputs_contract():
    with tempfile.TemporaryDirectory() as root:
        _cache(root)
        ds = lc.CachedLatentDataset(root, _Raw(6))
        batch = torch.utils.data.default_collate([ds[0], ds[4]])
        batch["camera_intrinsics"] = torch.eye(3).expand(2, 1, 3, 3)
        model = tiny_model(streams=('action', 'video', 'dino', 'pointmap'))
        inputs = lc.build_inputs_from_cache(model, batch)
        assert inputs["input_latents"].dtype == torch.bfloat16   # the VAE's dtype, not the model's
        assert torch.equal(inputs["first_frame_latents"], inputs["input_latents"][:, :, :1])
        assert inputs["dino_features"].shape == (2, 8, 2, 5, 1)
        assert inputs["dino_features"].dtype == torch.float32   # not cast to the model dtype
        assert inputs["pointmap_raw"].shape == inputs["input_latents"].shape
        assert inputs["action_is_pad"].dtype == torch.bool and inputs["image_is_pad"] is None
