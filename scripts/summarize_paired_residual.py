"""Matched perturbation results with pair-grouped out-of-fold prediction."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return None
    delta = positive[:,None] - negative[None,:]
    return float(np.mean(delta > 0) + .5*np.mean(delta == 0))


def logistic_prediction(train_x, train_y, test_x):
    """Fixed L2=1, training-fold standardization; no hyperparameter selection."""
    train_y = np.asarray(train_y, dtype=float)
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), (train_y.sum()+1)/(len(train_y)+2))
    center, scale = train_x.mean(0), train_x.std(0)
    scale = np.maximum(scale, 1e-6)
    x = np.column_stack((np.ones(len(train_x)), (train_x-center)/scale))
    test = np.column_stack((np.ones(len(test_x)), (test_x-center)/scale))
    beta = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1]); penalty[0,0] = 0
    for _ in range(50):
        probability = 1/(1+np.exp(-np.clip(x@beta, -30, 30)))
        gradient = x.T@(probability-train_y) + penalty@beta
        weights = np.maximum(probability*(1-probability), 1e-8)
        hessian = (x.T*weights)@x + penalty + 1e-8*np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.linalg.norm(step) < 1e-7:
            break
    return 1/(1+np.exp(-np.clip(test@beta, -30, 30)))


def grouped_predictions(rows, features, outcome):
    x = np.array([[np.log10(max(r[k], 1e-12)) for k in features] for r in rows])
    y = np.array([not r[outcome] for r in rows], dtype=float)
    group = np.array([r['pair_id'] for r in rows])
    if len(np.unique(y)) < 2 or len(np.unique(group)) < 3:
        return None
    prediction = np.zeros(len(rows))
    for pair in np.unique(group):
        test = group == pair
        prediction[test] = logistic_prediction(x[~test], y[~test], x[test])
    p = np.clip(prediction, 1e-8, 1-1e-8)
    return dict(auroc=auc(y,p), brier=float(np.mean((p-y)**2)),
                log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
                probabilities=prediction.tolist())


def summarize(root):
    completed = json.loads((root/'paired_complete.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    outcomes = [json.loads(line) for line in (root/'paired_outcomes.jsonl').read_text().splitlines()]
    outcomes = [r for r in outcomes if r['phase'] == 'evaluation']
    if len(outcomes) != completed['total_episodes']:
        raise ValueError('Incomplete outcome records')
    by_episode = {r['episode']:r for r in outcomes}
    if len(by_episode) != len(outcomes):
        raise ValueError('Duplicate episode IDs')
    records = [json.loads(line) for line in (root/'residual_sensitivity.jsonl').read_text().splitlines()]
    rows = [r for r in records if r['event'] == 'labeled']
    if len({r['episode'] for r in rows}) != len(rows):
        raise ValueError('Protocol requires exactly one prespecified measured chunk per episode')
    for row in rows:
        outcome = by_episode[row['episode']]
        if row['episode_success'] != outcome['success'] or row['chunk'] != 1:
            raise ValueError('Inconsistent score/outcome linkage')
        row.update(success_by_96=outcome['success_by_96'], executed_actions=outcome['executed_actions'])
    pair_checks = []
    for pair in sorted({r['pair_id'] for r in outcomes}):
        pair_outcomes = {r['condition']:r for r in outcomes if r['pair_id']==pair}
        if set(pair_outcomes) != {'clean','perturbed'}:
            raise ValueError(f'Incomplete pair {pair}')
        a,b = pair_outcomes['clean'],pair_outcomes['perturbed']
        if a['seed'] != b['seed'] or a['initial_state_index'] != b['initial_state_index']:
            raise ValueError('Pair state/seed mismatch')
        score_rows = {r['condition']:r for r in rows if r['pair_id']==pair}
        check = dict(pair_id=pair, initial_state_index=a['initial_state_index'],
            clean_success=a['success'], perturbed_success=b['success'],
            clean_success_by_96=a['success_by_96'], perturbed_success_by_96=b['success_by_96'],
            planned_actions_identical=a['planned_action_sha256']==b['planned_action_sha256'],
            scored_conditions=sorted(score_rows))
        if len(score_rows)==2:
            x,y = score_rows['clean'],score_rows['perturbed']
            check.update(predicted_future_identical=x['prediction_sha256']==y['prediction_sha256'],
                         random_baseline_identical=x['S_rand']==y['S_rand'])
            check['perturbed_over_clean'] = {k:y[k]/max(x[k],1e-12) for k in ('E','S_res','S_rand','R')}
        pair_checks.append(check)
    fields = ['episode','pair_id','condition','initial_state_index','perturbation_kind',
              'perturbation_strength','episode_success','success_by_96','executed_actions',
              'E','S_res','S_rand','R','replay_vs_original_rms','executed_after_observation']
    with (root/'paired_scores.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    valid = [r for r in rows if r['R'] is not None and r['random_baseline_resolved']]
    analyses = {}
    for outcome in ('episode_success','success_by_96'):
        labels = [not r[outcome] for r in valid]
        analyses[outcome] = dict(n=len(valid), failures=sum(labels),
            raw_score_failure_auroc={k:auc(labels,[r[k] for r in valid]) for k in ('E','S_res','S_rand','R')},
            leave_one_pair_out={})
        for name,features in (('E_only',('E',)), ('E_and_S_rand',('E','S_rand')),
                              ('E_S_rand_S_res',('E','S_rand','S_res'))):
            analyses[outcome]['leave_one_pair_out'][name] = grouped_predictions(valid,features,outcome)
    fig, axes = plt.subplots(1,2,figsize=(11,4.7))
    values = [max(r[k],1e-16) for r in rows for k in ('S_res','S_rand')]
    bounds = [min(values)*.5,max(values)*2]
    for ax,outcome,title in zip(axes,('episode_success','success_by_96'),
            ('Terminal task outcome (400 actions)','Completion by action 96')):
        for pair in pair_checks:
            points = [r for r in rows if r['pair_id']==pair['pair_id']]
            if len(points)==2:
                ax.plot([r['S_rand'] for r in points],[r['S_res'] for r in points],
                        color='0.75',linewidth=.8,zorder=1)
        for condition,marker in (('clean','o'),('perturbed','s')):
            for success,color in ((True,'tab:blue'),(False,'tab:red')):
                group = [r for r in rows if r['condition']==condition and r[outcome]==success]
                ax.scatter([r['S_rand'] for r in group],[r['S_res'] for r in group],
                    marker=marker,c=color,label=f'{condition}, {"success" if success else "non-success"} ({len(group)})',zorder=2)
        ax.plot(bounds,bounds,'--',color='gray',linewidth=1,label='R = 1')
        ax.set(xscale='log',yscale='log',xlabel='Random sensitivity',ylabel='Residual sensitivity',title=title)
        ax.legend(fontsize=7)
    fig.suptitle('Flex-π: held-out clean/perturbed pairs; squared JVP norms')
    fig.tight_layout()
    fig.savefig(root/'paired_sensitivity_scatter.png',dpi=180)
    fig.savefig(root/'paired_sensitivity_scatter.pdf')
    plt.close(fig)
    summary = dict(selected_perturbation=selection['selected'],
        mixed_calibration=selection['mixed_calibration'], holdout_pairs=len(pair_checks),
        measured_episodes=len(rows), missing_measurement_episodes=sorted(set(by_episode)-{r['episode'] for r in rows}),
        outcomes={condition:dict(episodes=sum(r['condition']==condition for r in outcomes),
            successes=sum(r['condition']==condition and r['success'] for r in outcomes),
            successes_by_96=sum(r['condition']==condition and r['success_by_96'] for r in outcomes))
            for condition in ('clean','perturbed')},
        pairs=pair_checks, analyses=analyses,
        analysis_episode_order=[r['episode'] for r in valid],
        limitations=['Small exploratory sample; no calibration claim.',
            'S_rand is expected to match within each pair: same predicted future and Jacobian, common probes.',
            'The perturbation is observed only by E and residual-directed sensitivity; this does not prove false certainty.',
            'Clean-visual conditional replay differs from the original joint sampler.',
            'Action-96 non-completion measures delay and must not be equated with terminal failure.',
            'Regression L2=1 fixed; logarithms and standardization fit inside each training fold; both pair members held out.'],
        protocol=protocol)
    (root/'paired_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('selected_perturbation','outcomes','analyses')},indent=2),flush=True)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    summarize(parser.parse_args().directory)
