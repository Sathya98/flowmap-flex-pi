import numpy as np
from scripts import summarize_paired_residual as analysis


def test_leave_pair_out_excludes_both_members(monkeypatch):
    rows = [dict(pair_id=p, E=10.**p, S_rand=1., episode_success=(c==0))
            for p in range(4) for c in range(2)]
    folds = []
    def predict(train_x, train_y, test_x):
        assert len(test_x) == 2
        assert len(np.unique(test_x[:,0])) == 1
        assert not set(train_x[:,0]).intersection(test_x[:,0])
        folds.append(test_x[0,0])
        return np.full(2,.5)
    monkeypatch.setattr(analysis, 'logistic_prediction', predict)
    result = analysis.grouped_predictions(rows, ('E','S_rand'), 'episode_success')
    assert len(folds) == 4
    assert result['auroc'] == .5
    assert result['brier'] == .25


def test_auc_ties_and_single_class():
    assert analysis.auc([0,1],[1.,1.]) == .5
    assert analysis.auc([0,1],[0.,1.]) == 1.
    assert analysis.auc([0,0],[0.,1.]) is None


def test_logistic_prediction_separates_simple_signal():
    x = np.array([[-3.],[-2.],[-1.],[1.],[2.],[3.]])
    p = analysis.logistic_prediction(x, [0,0,0,1,1,1], np.array([[-2.],[2.]]))
    assert 0 < p[0] < .5 < p[1] < 1
