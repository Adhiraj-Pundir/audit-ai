"""
Generate audit_baseline.json from existing per-pair delta CSV.
No GPU needed. Reads logs/base_model_pair_deltas.csv + dataset/data/counterfactual_pairs.csv.
Output: app/data/audit_baseline.json
Runtime: ~5 seconds.
"""
import json
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DALIT = {'Valmiki', 'Balmiki', 'Khatik', 'Paswan', 'Chamar', 'Dhobi',
         'Gond', 'Meena', 'Lodhi', 'Ram', 'Mondal'}
MUSLIM = {'Malik', 'Siddiqui', 'Rahman', 'Ali', 'Sheikh', 'Khan', 'Hasan',
          'Ansari', 'Shaikh', 'Mirza', 'Pathan', 'Qureshi', 'Abbas', 'Hussain'}
PARSI = {'Mistry'}


def axis_of(surname: str) -> str:
    if surname in DALIT:
        return 'dalit'
    if surname in MUSLIM:
        return 'muslim'
    if surname in PARSI:
        return 'parsi'
    return 'other'


def main():
    pairs_path = REPO_ROOT / 'dataset' / 'data' / 'counterfactual_pairs.csv'
    deltas_path = REPO_ROOT / 'logs' / 'base_model_pair_deltas.csv'
    out_path = REPO_ROOT / 'app' / 'data' / 'audit_baseline.json'

    pairs_df = pd.read_csv(pairs_path)
    deltas_df = pd.read_csv(deltas_path)

    merged = deltas_df.merge(
        pairs_df[['pair_id', 'first_name', 'gender']],
        on='pair_id'
    )
    merged['axis'] = merged['corrupted_surname'].apply(axis_of)

    pairs_out = []
    for _, row in merged.iterrows():
        pairs_out.append({
            'pair_id': int(row['pair_id']),
            'clean_surname': str(row['clean_surname']),
            'corrupted_surname': str(row['corrupted_surname']),
            'axis': str(row['axis']),
            'delta': float(row['delta']),
            'clean_margin': float(row['clean_margin']),
            'corrupted_margin': float(row['corrupted_margin']),
            'cibil': int(row['cibil']),
            'income': int(row['income']),
            'first_name': str(row['first_name']),
            'gender': str(row['gender']),
        })

    by_axis = []
    for ax in ['dalit', 'muslim', 'parsi']:
        sub = merged[merged.axis == ax]
        if len(sub) == 0:
            continue
        by_axis.append({
            'axis': ax,
            'n': int(len(sub)),
            'mean_delta': float(sub.delta.mean()),
            'median_delta': float(sub.delta.median()),
            'std_delta': float(sub.delta.std()),
            'n_strong': int((sub.delta.abs() > 0.5).sum()),
        })

    out = {
        'model': 'google/gemma-2-2b',
        'n_pairs': int(len(merged)),
        'mitigation': 'none',
        'summary': {
            'mean_delta': float(merged.delta.mean()),
            'median_delta': float(merged.delta.median()),
            'std_delta': float(merged.delta.std()),
            'n_strong_bias': int((merged.delta.abs() > 0.5).sum()),
            'n_strong_correct_direction': int((merged.delta > 0.5).sum()),
        },
        'by_axis': by_axis,
        'pairs': pairs_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'Wrote {out_path}')
    print(f'  n_pairs: {out["n_pairs"]}')
    print(f'  mean_delta: {out["summary"]["mean_delta"]:+.4f}')
    print(f'  std_delta: {out["summary"]["std_delta"]:.4f}')
    print(f'  n_strong_bias: {out["summary"]["n_strong_bias"]}')
    for ax in out['by_axis']:
        print(f'    {ax["axis"]:8s} n={ax["n"]:3d} '
              f'mean={ax["mean_delta"]:+.3f} std={ax["std_delta"]:.3f}')


if __name__ == '__main__':
    main()
