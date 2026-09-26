import torch

from flexpi.models.helpers.residual_sensitivity import directional_scores


def test_linear_jvp_and_unit_random_baseline():
    matrix = torch.diag(torch.tensor([5., 1., 1., 1.]))
    x = torch.zeros(4)
    report, primal = directional_scores(lambda f: matrix @ f, x,
                                        torch.tensor([2., 0., 0., 0.]), probes=4)
    assert report['E'] == 2.
    assert report['S_res'] == 25.
    assert report['S_rand'] == 7.
    assert abs(report['R'] - 25 / 7) < 1e-8
    torch.testing.assert_close(primal, x)


def test_zero_residual_is_not_an_alignment_claim():
    x = torch.zeros(4)
    report, _ = directional_scores(lambda f: 2*f, x, x, probes=1)
    assert report['S_res'] == 0
    assert report['S_rand'] == 4
    assert report['R'] is None


def test_nonlinear_jvp_matches_analytic_derivative():
    x = torch.tensor([1., 2., 3.])
    residual = torch.tensor([1., -2., 2.])
    report, _ = directional_scores(lambda f: f.square(), x, x+residual, probes=2)
    expected = ((2*x*residual/residual.norm()).square()).sum()
    assert abs(report['S_res'] - float(expected)) < 1e-5
