"""One flex-joint regime per microbatch when share_within_microbatch is on."""
import torch

from flexpi.models.helpers.flex_joint import FlexJointConfig, sample_flex_batch_flags

NAMES = ("present_v", "present_d", "present_p", "j_v", "j_d", "j_p")
CFG = dict(enabled=True, p_present_video=.5, p_present_dino=.5, p_present_pointmap=.5,
           p_jv=.5, p_jd=.5, p_jp=.5, cross_modal_predict_video=True,
           cross_modal_predict_dino=True, cross_modal_predict_pointmap=True)


def draw(share, batch, n, seed=0):
    cfg = FlexJointConfig(**CFG, share_within_microbatch=share)
    gen = torch.Generator().manual_seed(seed)
    return [sample_flex_batch_flags(cfg, batch, torch.device("cpu"), generator=gen) for _ in range(n)]


def test_shared_flags_are_identical_within_a_microbatch():
    for flags in draw(True, 4, 50):
        assert flags.B == 4
        for name in NAMES:
            value = getattr(flags, name)
            assert value.shape == (4,) and bool((value == value[0]).all())
        assert bool((flags.present_v | flags.present_d | flags.present_p).all())   # rejection kept


def test_per_sample_default_still_varies_within_a_microbatch():
    assert any(not bool((getattr(f, n) == getattr(f, n)[0]).all()) for f in draw(False, 4, 50) for n in NAMES)


def test_marginals_match_per_sample_sampling():
    shared = torch.stack([torch.stack([getattr(f, n)[0] for n in NAMES]) for f in draw(True, 2, 4000)]).float()
    single = torch.stack([torch.stack([getattr(f, n)[0] for n in NAMES]) for f in draw(False, 1, 4000, 1)]).float()
    torch.testing.assert_close(shared.mean(0), single.mean(0), atol=.04, rtol=0)
