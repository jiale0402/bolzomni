"""Summarize saved diagnostic scalars without model execution."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / 'results/causal-diagnostic'
    records = [json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    rows = []
    for r in records:
        if r['kind'] != 'gradient_probe':
            continue
        gram = r['groups']['all']['gram']
        row = dict(id=r['id'], losses=r['losses'],
            cos_folding_design= r['groups']['all']['cosine'][0][2],
            hypothetical_adam_slopes=r['hypothetical_adam_directional_derivatives'],
            hypothetical_adam_delta_norm=r['hypothetical_adam_delta_norm'])
        for target, index in [('folding',0), ('design',2)]:
            contributions = {name:-.5*gram[index][j] for j,name in enumerate(r['terms'])}
            own = contributions[r['terms'][index]]
            total = sum(contributions.values())
            row[target] = dict(sgd_directional_contributions=contributions,
                sgd_joint_directional_derivative=total,
                sgd_joint_to_own_ratio=total/own if own else None,
                joint_points_uphill=total>0,
                hypothetical_adam_points_uphill=r['hypothetical_adam_directional_derivatives'][index]>0)
        rows.append(row)
    trajectories = []
    for r in records:
        if r['kind'] != 'feedback_trajectory':
            continue
        f=r['final']
        trajectories.append(dict(id=r['id'],condition=r['condition'],
            agreement=f['revealed_identity_geometry_agreement'],max_ca_gap=f['max_ca_gap'],
            min_ca_gap=f['min_ca_gap'],motif_ca_distance_rmse=f['motif_ca_distance_rmse'],
            finite=f['finite'], snapshots=r['snapshots']))
    summary = dict(gradient_probes=rows,trajectories=trajectories,
        completion=next((r for r in records if r['kind']=='completion'),None),
        limitations=['Directional derivatives are local first-order evidence, not historical causal attribution.',
            'The Adam direction uses saved moments and a hypothetical paired probe, not the historical minibatch.',
            'Trajectory geometry diagnostics are not new designability/refolding success scores.'])
    (root/'scalar_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
