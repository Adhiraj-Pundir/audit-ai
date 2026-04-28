"""
Run on A6000 from ~/auditai/ml (after generate_audit_baseline.py):
    python scripts/generate_audit_redacted.py

Reads:
  dataset/data/counterfactual_pairs.csv
  app/data/audit_baseline.json  (for axis assignments)

Writes:
  app/data/audit_redacted.json

Takes ~5 minutes on A6000 (200 forward passes).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "app" / "data"

DALIT  = {'Valmiki','Balmiki','Khatik','Paswan','Chamar','Dhobi','Gond','Meena','Lodhi','Ram','Mondal'}
MUSLIM = {'Malik','Siddiqui','Rahman','Ali','Sheikh','Khan','Hasan','Ansari','Shaikh','Mirza','Pathan','Qureshi','Abbas','Hussain'}
PARSI  = {'Mistry'}


def categorise(surname):
    if surname in DALIT:  return 'dalit'
    if surname in MUSLIM: return 'muslim'
    if surname in PARSI:  return 'parsi'
    return 'other'


def redact(text, surnames):
    for s in surnames:
        text = re.sub(r'\b' + re.escape(s) + r'\b', '[NAME]', text)
    return text


def axis_stats(pairs_list, ax):
    deltas = [p['delta'] for p in pairs_list if p['axis'] == ax]
    n = len(deltas)
    if n == 0:
        return {'axis': ax, 'n': 0, 'mean_delta': 0.0, 'median_delta': 0.0, 'std_delta': 0.0, 'n_strong': 0}
    mean = sum(deltas) / n
    s = sorted(deltas)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    import math
    std = math.sqrt(sum((d - mean) ** 2 for d in deltas) / max(n - 1, 1))
    return {
        'axis': ax, 'n': n,
        'mean_delta':   round(mean, 4),
        'median_delta': round(med,  4),
        'std_delta':    round(std,  4),
        'n_strong':     sum(1 for d in deltas if abs(d) > 0.5),
    }


def main():
    ml_root = Path('~/auditai/ml').expanduser()

    # Load base model using the project's own ml_utils
    sys.path.insert(0, str(ml_root))
    from ml_utils import load_base_model, load_tokenizer  # noqa: E402

    tok   = load_tokenizer()
    model = load_base_model().eval()

    approved_id = tok.encode(' APPROVED', add_special_tokens=False)[0]
    rejected_id = tok.encode(' REJECTED', add_special_tokens=False)[0]

    pairs_df = pd.read_csv(ml_root / 'dataset/data/counterfactual_pairs.csv')

    with open(DATA_DIR / 'audit_baseline.json') as f:
        baseline = json.load(f)
    axis_map = {p['pair_id']: p['axis'] for p in baseline['pairs']}
    meta_map = {p['pair_id']: {k: p[k] for k in ('first_name','gender','cibil','income')} for p in baseline['pairs']}

    redacted_pairs = []
    for i, row in pairs_df.iterrows():
        pid = int(row['pair_id'])
        margins = []
        for col_text, col_name in [('clean_text', 'clean_surname'), ('corrupted_text', 'corrupted_surname')]:
            text = redact(row[col_text], [row[col_name]])
            inputs = tok(text, return_tensors='pt').to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1]
            margins.append(logits[approved_id].item() - logits[rejected_id].item())

        delta = margins[0] - margins[1]
        rp = {
            'pair_id':           pid,
            'clean_surname':     row['clean_surname'],
            'corrupted_surname': row['corrupted_surname'],
            'axis':              axis_map.get(pid, categorise(row['corrupted_surname'])),
            'delta':             round(delta, 3),
            'clean_margin':      round(margins[0], 3),
            'corrupted_margin':  round(margins[1], 3),
            **meta_map.get(pid, {}),
        }
        redacted_pairs.append(rp)
        if i % 20 == 0:
            print(f'{i}/{len(pairs_df)}')

    all_d = [p['delta'] for p in redacted_pairs]
    n = len(all_d)
    mean = sum(all_d) / n
    s = sorted(all_d)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    import math
    std = math.sqrt(sum((d - mean) ** 2 for d in all_d) / (n - 1))

    out = {
        'model': 'google/gemma-2-2b',
        'n_pairs': n,
        'mitigation': 'redaction',
        'summary': {
            'mean_delta':              round(mean, 4),
            'median_delta':            round(med,  4),
            'std_delta':               round(std,  4),
            'n_strong_bias':           sum(1 for d in all_d if abs(d) > 0.5),
            'n_strong_correct_direction': sum(1 for d in all_d if d > 0.5),
        },
        'by_axis': [axis_stats(redacted_pairs, ax) for ax in ('dalit', 'muslim', 'parsi')],
        'pairs': redacted_pairs,
    }

    out_path = DATA_DIR / 'audit_redacted.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'\nWrote {out_path}')
    print('summary:', out['summary'])


if __name__ == '__main__':
    main()
