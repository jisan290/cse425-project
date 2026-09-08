"""
Unified training entrypoint for all four tasks.

Usage:
    python train.py --task 1
    python train.py --task 2
    python train.py --task 3
    python train.py --task 4
    python train.py --task all

Each task reads its hyperparameters from config.yaml and writes its
model checkpoint + metrics json into results/. Tasks 2-4 all depend
on data/processed/task2_graphs/ having been built first (Task 2's own
run does this automatically).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src import audio_features, graph_builder, bert_encoder, gnn_model, fusion_model, contrastive


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(cfg):
    data_dir = Path(cfg['paths']['data_dir'])
    mtat = pd.read_csv(data_dir / 'mtat_master.csv')
    meta = json.load(open(data_dir / 'dataset_meta.json'))
    top50 = meta['top50_tags']
    tr = mtat[mtat.split_folder == 'train'].reset_index(drop=True)
    va = mtat[mtat.split_folder == 'val'].reset_index(drop=True)
    te = mtat[mtat.split_folder == 'test'].reset_index(drop=True)
    return mtat, top50, tr, va, te


def run_task1(cfg, device):
    print('=== TASK 1: BERT tag classifier ===')
    mtat, top50, tr, va, te = load_dataset(cfg)
    c = cfg['task1_bert']

    model, history, tokenizer, (train_dl, val_dl, test_dl) = bert_encoder.train_bert_tagger(
        tr, va, te, top50, model_name=c['model_name'], max_len=c['max_len'],
        batch_size=c['batch_size'], epochs=c['epochs'], device=device)

    Pv, Yv = bert_encoder.predict(model, val_dl, device)
    thr = bert_encoder.tune_thresholds(Pv, Yv)
    Pt, Yt = bert_encoder.predict(model, test_dl, device)
    result = bert_encoder.compute_metrics(Pt, Yt, thr)
    print('Task 1 test (tuned):', result)

    out = Path(cfg['paths']['results_dir'])
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'thresholds': thr, 'tags': top50},
               out / 'task1_bert.pt')
    json.dump({'test_tuned': result, 'val_history': history}, open(out / 'task1_metrics.json', 'w'),
               indent=2, default=float)
    return result


def run_task2(cfg, device):
    print('=== TASK 2: GraphSAGE on segment graphs ===')
    mtat, top50, tr, va, te = load_dataset(cfg)
    c = cfg['task2_gnn']

    mp3_files = audio_features.find_audio_files('data/raw')
    audio_by_id = audio_features.build_clip_id_index(mp3_files)

    graph_dir = cfg['paths']['graph_dir']
    graph_meta = graph_builder.cache_all_splits(tr, va, te, audio_by_id, graph_dir)

    train_ds = graph_builder.MTATGraphDataset(graph_meta[graph_meta.split == 'train'], mtat, top50)
    val_ds = graph_builder.MTATGraphDataset(graph_meta[graph_meta.split == 'val'], mtat, top50)
    test_ds = graph_builder.MTATGraphDataset(graph_meta[graph_meta.split == 'test'], mtat, top50)

    from torch_geometric.loader import DataLoader as GeoDataLoader
    train_loader = GeoDataLoader(train_ds, batch_size=c['batch_size'], shuffle=True)
    val_loader = GeoDataLoader(val_ds, batch_size=c['batch_size'], shuffle=False)
    test_loader = GeoDataLoader(test_ds, batch_size=c['batch_size'], shuffle=False)

    node_dim = train_ds[0].x.shape[1]
    model = gnn_model.GraphSAGETagger(node_dim, c['hidden_channels'], len(top50),
                                       c['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c['lr'])
    crit = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, c['epochs'] + 1):
        model.train()
        for b in train_loader:
            b = b.to(device)
            opt.zero_grad()
            logits = model(b.x, b.edge_index, b.batch)
            loss = crit(logits, b.y.view(-1, len(top50)))
            loss.backward()
            opt.step()
        Pv, Yv = gnn_model.predict_gnn(model, val_loader, device, len(top50))
        m = gnn_model.compute_metrics(Pv, Yv)
        print(f'  epoch {epoch}: {m}')

    Pv, Yv = gnn_model.predict_gnn(model, val_loader, device, len(top50))
    thr = gnn_model.tune_thresholds(Pv, Yv)
    Pt, Yt = gnn_model.predict_gnn(model, test_loader, device, len(top50))
    result = gnn_model.compute_metrics(Pt, Yt, thr)
    print('Task 2 test (tuned):', result)

    out = Path(cfg['paths']['results_dir'])
    torch.save(model.state_dict(), out / 'task2_gnn.pt')
    json.dump({'test_tuned': result}, open(out / 'task2_metrics.json', 'w'), indent=2, default=float)
    return result


def run_task3(cfg, device):
    print('=== TASK 3: GNN-BERT cross-attention fusion + ablation ===')
    mtat, top50, tr, va, te = load_dataset(cfg)
    c = cfg['task3_fusion']

    graph_meta = pd.read_csv(Path(cfg['paths']['graph_dir']) / 'graph_metadata.csv')
    text_cols = ['clip_id', 'caption', 'split_folder'] + top50
    paired = graph_meta.merge(mtat[text_cols], on='clip_id', how='inner')

    from transformers import AutoTokenizer
    from torch_geometric.loader import DataLoader as GeoDataLoader
    tokenizer = AutoTokenizer.from_pretrained(cfg['task1_bert']['model_name'])

    tr3 = paired[paired.split == 'train']
    va3 = paired[paired.split == 'val']
    te3 = paired[paired.split == 'test']

    train_ds = fusion_model.FusionDataset(tr3, top50, tokenizer)
    val_ds = fusion_model.FusionDataset(va3, top50, tokenizer)
    test_ds = fusion_model.FusionDataset(te3, top50, tokenizer)

    train_loader = GeoDataLoader(train_ds, batch_size=c['batch_size'], shuffle=True)
    val_loader = GeoDataLoader(val_ds, batch_size=c['batch_size'], shuffle=False)
    test_loader = GeoDataLoader(test_ds, batch_size=c['batch_size'], shuffle=False)
    node_dim = train_ds[0].x.shape[0] if False else train_ds[0].x.shape[1]

    ablation = {}
    out = Path(cfg['paths']['results_dir'])
    for mode in c['modes']:
        model, history = fusion_model.train_fusion(
            {'node_dim': node_dim, 'n_tags': len(top50), 'hidden': c['hidden'],
             'mode': mode, 'model_name': cfg['task1_bert']['model_name'],
             'dropout': c['dropout']},
            train_loader, val_loader, len(top50), epochs=c['epochs'], device=device)

        Pv, Yv, _, _ = fusion_model.predict(model, val_loader, device, len(top50))
        thr = fusion_model.tune_thresholds(Pv, Yv)
        Pt, Yt, _, _ = fusion_model.predict(model, test_loader, device, len(top50))
        result = fusion_model.compute_metrics(Pt, Yt, thr)
        ablation[mode] = result
        print(f'  {mode}: {result}')

        if mode == 'cross_attn':
            torch.save(model.state_dict(), out / 'task3_fusion.pt')

    json.dump({'ablation': ablation}, open(out / 'task3_metrics.json', 'w'),
               indent=2, default=float)
    return ablation


def run_task4(cfg, device):
    print('=== TASK 4 (bonus): contrastive retrieval + zero-shot ===')
    c = cfg['task4_contrastive']
    data_dir = cfg['paths']['data_dir']
    graph_dir = cfg['paths']['graph_dir']

    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(cfg['task1_bert']['model_name'])
    meta = contrastive.build_paired_meta(Path(data_dir).parent)
    splits = {s: meta[meta.split == s] for s in ('train', 'val', 'test')}

    train_loader = DataLoader(contrastive.PairedDataset(splits['train'], graph_dir, tokenizer),
                               batch_size=c['batch_size'], shuffle=True, collate_fn=contrastive.collate)
    val_loader = DataLoader(contrastive.PairedDataset(splits['val'], graph_dir, tokenizer),
                             batch_size=len(splits['val']), shuffle=False, collate_fn=contrastive.collate)
    test_loader = DataLoader(contrastive.PairedDataset(splits['test'], graph_dir, tokenizer),
                              batch_size=len(splits['test']), shuffle=False, collate_fn=contrastive.collate)

    sample_graph = next(iter(train_loader))[0]
    node_dim = sample_graph.x.shape[1]

    model = contrastive.DualEncoder(node_dim, cfg['task1_bert']['model_name']).to(device)
    history, best_epoch = contrastive.train(model, train_loader, val_loader,
                                             c['epochs'], c['lr'], device)
    metrics, examples = contrastive.evaluate_retrieval(model, test_loader, device, tuple(c['retrieval_ks']))
    print('Retrieval:', metrics)

    out = Path(cfg['paths']['results_dir'])
    torch.save(model.state_dict(), out / 'task4_contrastive.pt')
    json.dump({'history': history, 'best_epoch': best_epoch, 'retrieval_metrics': metrics},
               open(out / 'task4_metrics.json', 'w'), indent=2, default=float)
    json.dump(examples, open(out / 'retrieval_examples' / 'task4_examples.json', 'w'), indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['1', '2', '3', '4', 'all'], required=True)
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    Path(cfg['paths']['results_dir']).mkdir(parents=True, exist_ok=True)
    (Path(cfg['paths']['results_dir']) / 'retrieval_examples').mkdir(exist_ok=True)

    tasks = {'1': run_task1, '2': run_task2, '3': run_task3, '4': run_task4}
    if args.task == 'all':
        for t in ['1', '2', '3', '4']:
            tasks[t](cfg, device)
    else:
        tasks[args.task](cfg, device)


if __name__ == '__main__':
    main()
