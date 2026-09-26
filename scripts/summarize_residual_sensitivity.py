"""Plot labeled observed-future probes; episode-level statistics avoid pseudoreplication."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def auc(labels, scores):
    positive = np.asarray(scores)[np.asarray(labels, dtype=bool)]
    negative = np.asarray(scores)[~np.asarray(labels, dtype=bool)]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None, :]
    return float((differences > 0).mean() + .5*(differences == 0).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row['event'] == 'labeled']
    out = args.input.parent
    if not rows:
        raise ValueError('No terminally labeled probes; rollout may be incomplete')
    keys = ['episode', 'chunk', 'episode_success', 'chunk_success', 'E', 'S_res',
            'S_rand', 'R', 'replay_vs_original_rms', 'executed_after_observation']
    with (out/'scores.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6, 5))
    for success, color, marker in ((True, 'tab:blue', 'o'), (False, 'tab:red', 'x')):
        group = [row for row in rows if row['episode_success'] == success]
        ax.scatter([max(r['S_rand'], 1e-16) for r in group],
                   [max(r['S_res'], 1e-16) for r in group], c=color, marker=marker,
                   label=f"{'Success' if success else 'Failure'} ({len(group)} chunks)")
    values = [max(r[k], 1e-16) for r in rows for k in ('S_res', 'S_rand')]
    bounds = [min(values)*.5, max(values)*2]
    ax.plot(bounds, bounds, '--', color='gray', linewidth=1, label='R = 1')
    ax.set(xscale='log', yscale='log', xlabel='Random sensitivity (mean squared JVP norm)',
           ylabel='Residual sensitivity (squared JVP norm)',
           title='Flex-π / LIBERO spatial task 0 / VAE residual')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'sensitivity_scatter.png', dpi=180)
    fig.savefig(out/'sensitivity_scatter.pdf')
    plt.close(fig)
    episodes = []
    for episode in sorted({r['episode'] for r in rows}):
        group = [r for r in rows if r['episode'] == episode]
        record = dict(episode=episode, failure=not group[0]['episode_success'], chunks=len(group))
        for key in ('E', 'S_res', 'S_rand', 'R'):
            values = [r[key] for r in group if r[key] is not None]
            record[key] = float(np.mean(values)) if values else None
        episodes.append(record)
    summary = dict(measurements=len(rows), labeled_episodes=len(episodes),
        successes=sum(not e['failure'] for e in episodes), episodes=episodes,
        episode_mean_failure_auroc={},
        interpretation='Exploratory pilot; terminal failure labels, correlated chunks. '
        'Conditional action replay with clean visual tokens differs from joint sampling. '
        'Chunk non-success means task not yet completed, not necessarily a failed plan.')
    for key in ('E', 'S_res', 'S_rand', 'R'):
        valid = [e for e in episodes if e[key] is not None]
        summary['episode_mean_failure_auroc'][key] = auc([e['failure'] for e in valid],
                                                       [e[key] for e in valid])
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
