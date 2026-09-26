"""ArrayStore round trip: sparse preallocation by row kind, index arrays, dtype restore."""
import numpy as np
import pytest
import torch

from flowmap_core import latent_store as ls


class Store(ls.ArrayStore):
    VERSION = 3
    INDEX_ARRAYS = ("windows",)


def manifest():
    arrays = {
        "latents": dict(rows="window", shape=[2, 3], dtype="bfloat16", np_dtype="int16", source_dtype="float32"),
        "frames": dict(rows="frame", shape=[4], dtype="float32", np_dtype="float32"),
    }
    return dict(version=3, num_windows=5, arrays=arrays)


def test_create_open_write_read(tmp_path):
    store = Store.create(tmp_path, manifest(), dict(windows=np.arange(5) * 2), dict(window=5, frame=7))
    assert store.array("latents").shape == (5, 2, 3) and store.array("frames").shape == (7, 4)
    value = torch.randn(2, 3)
    store.array("latents", "r+")[1] = ls.to_numpy(value.bfloat16())
    store.done("r+")[1] = 1
    store.array("latents", "r+").flush()
    store.done("r+").flush()

    reopened = Store.open(tmp_path)
    assert reopened.array("windows").tolist() == [0, 2, 4, 6, 8]
    assert np.asarray(reopened.done()).tolist() == [0, 1, 0, 0, 0]
    spec = reopened.manifest["arrays"]["latents"]
    out = ls.load(spec, reopened.array("latents")[1])
    assert out.dtype == torch.float32                          # restored to the encoder's dtype
    assert torch.equal(out, value.bfloat16().float())


def test_version_mismatch_and_unchecked_base(tmp_path):
    m = manifest()
    m["version"] = 2
    ls.ArrayStore.create(tmp_path, m, {}, dict(window=5, frame=1))
    with pytest.raises(ValueError, match="cache version 2 != 3"):
        Store.open(tmp_path)
    assert ls.ArrayStore.open(tmp_path).manifest["version"] == 2


def test_helpers():
    x = torch.randn(3, 2).bfloat16()
    a = ls.to_numpy(x)
    assert a.dtype == np.int16
    assert torch.equal(ls.from_numpy(a, "bfloat16"), x)
    assert ls.from_numpy(a, "bfloat16", "float32").dtype == torch.float32
    assert ls.select_windows([0, 10], [4, 13], 2).tolist() == [0, 2, 10, 12]
    assert ls.fingerprint_diff(dict(a=1, b=dict(c=2, d=3)), dict(a=1, b=dict(c=5), e=0)) == ["b.c", "b.d", "e"]
