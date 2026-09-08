"""
Loads whatever results/*_metrics.json files exist and prints/saves the
final cross-task comparison table (Section 7 of the report).

Usage:
    python evaluate.py
"""
import json
from pathlib import Path

import pandas as pd
import yaml


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path):
    return json.load(open(path)) if Path(path).exists() else None


def main():
    cfg = load_config()
    results_dir = Path(cfg['paths']['results_dir'])

    rows = {}

    t1 = load_json(results_dir / 'task1_metrics.json')
    if t1:
        rows['Task 1: BERT (full text corpus)'] = t1['test_tuned']

    t2 = load_json(results_dir / 'task2_metrics.json')
    if t2:
        rows['Task 2: GraphSAGE'] = t2['test_tuned']

    t3 = load_json(results_dir / 'task3_metrics.json')
    if t3:
        for mode, result in t3['ablation'].items():
            label = {'gnn_only': 'Task 3 ablation: GNN-only',
                      'bert_only': 'Task 3 ablation: BERT-only',
                      'concat': 'Task 3 ablation: early concat',
                      'cross_attn': 'Task 3: cross-attention fusion'}.get(mode, mode)
            rows[label] = result

    t4 = load_json(results_dir / 'task4_metrics.json')
    if t4:
        print('\nTask 4 retrieval (bonus, not a tag-metric row):')
        print('  audio -> caption:', t4['retrieval_metrics']['audio_to_caption'])
        print('  caption -> audio:', t4['retrieval_metrics']['caption_to_audio'])
        print('  chance level    :', t4['retrieval_metrics']['chance_level'])

    if not rows:
        print('No results found yet — run train.py for at least one task first.')
        return

    table = pd.DataFrame(rows).T[['macro_f1', 'micro_f1', 'roc_auc', 'pr_auc']].round(4)
    print('\n' + '=' * 70)
    print('CROSS-TASK COMPARISON')
    print('=' * 70)
    print(table.to_string())

    table.to_csv(results_dir / 'final_comparison.csv')
    print(f'\nSaved {results_dir / "final_comparison.csv"}')


if __name__ == '__main__':
    main()
