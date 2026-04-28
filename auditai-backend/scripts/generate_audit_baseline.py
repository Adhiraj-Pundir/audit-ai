"""
Run on A6000 from ~/auditai/ml:
    python scripts/generate_audit_baseline.py

Reads:
  dataset/data/counterfactual_pairs.csv
  logs/base_model_pair_deltas.csv

Writes:
  app/data/audit_baseline.json
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "app" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DALIT  = {'Valmiki','Balmiki','Khatik','Paswan','Chamar','Dhobi','Gond','Meena','Lodhi','Ram','Mondal'}
MUSLIM = {'Malik','Siddiqui','Rahman','Ali','Sheikh','Khan','Hasan','Ansari','Shaikh','Mirza','Pathan','Qureshi','Abbas','Hussain'}
PARSI  = {'Mistry'}


def categorise(surname: str) -> str:
    if surname in DALIT:  return 'dalit'
    if surname in MUSLIM: return 'muslim'
    if surname in PARSI:  return 'parsi'
    return 'other'


def axis_stats(df: pd.DataFrame, ax: str) -> dict:
    sub = df[df['axis'] == ax]['delta']
    return {
        'axis': ax,
        'n': len(sub),
        'mean_delta':   round(float(sub.mean()),   3),
        'median_delta': round(float(sub.median()), 3),
        'std_delta':    round(float(sub.std()),    3),
        'n_strong':     int((sub.abs() > 0.5).sum()),
    }


def main():
    ml_root = Path('~/auditai/ml').expanduser()
    pairs_df  = pd.read_csv(ml_root / 'dataset/data/counterfactual_pairs.csv')
    deltas_df = pd.read_csv(ml_root / 'logs/base_model_pair_deltas.csv')

    merged = deltas_df.merge(pairs_df[['pair_id', 'first_name', 'gender']], on='pair_id')
    merged['axis'] = merged['corrupted_surname'].apply(categorise)

    pairs_out = merged[[
        'pair_id', 'clean_surname', 'corrupted_surname', 'axis',
        'delta', 'clean_margin', 'corrupted_margin',
        'cibil', 'income', 'first_name', 'gender',
    ]].to_dict('records')

    d = merged['delta']
    out = {
        'model': 'google/gemma-2-2b',
        'n_pairs': len(merged),
        'mitigation': 'none',
        'summary': {
            'mean_delta':              round(float(d.mean()),   3),
            'median_delta':            round(float(d.median()), 3),
            'std_delta':               round(float(d.std()),    3),
            'n_strong_bias':           int((d.abs() > 0.5).sum()),
            'n_strong_correct_direction': int((d > 0.5).sum()),
        },
        'by_axis': [axis_stats(merged, ax) for ax in ('dalit', 'muslim', 'parsi')],
        'pairs': pairs_out,
    }

    out_path = DATA_DIR / 'audit_baseline.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'Wrote {out_path}  ({len(merged)} pairs)')
    print('summary:', out['summary'])


if __name__ == '__main__':
    main()
