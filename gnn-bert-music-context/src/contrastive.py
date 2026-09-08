"""
Task 4 (Member D, bonus) — dual-encoder GNN-BERT contrastive retrieval.

Trains a GraphTower and a TextTower into a shared embedding space with
symmetric InfoNCE, evaluates bidirectional R@K retrieval, and includes
a zero-shot tag-prediction path that needs no classifier head at all —
just nearest-embedding lookup against the 50 tag names — compared
against Task 3's supervised model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import AutoModel
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

EMBED_DIM = 128
TEMPERATURE = 0.07


class GraphTower(nn.Module):
    def __init__(self, in_channels, hidden=128, out_dim=EMBED_DIM):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.proj(global_mean_pool(x, batch))


class TextTower(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', out_dim=EMBED_DIM):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state[:, 0])


class DualEncoder(nn.Module):
    def __init__(self, node_dim, model_name='distilbert-base-uncased'):
        super().__init__()
        self.graph_tower = GraphTower(node_dim)
        self.text_tower = TextTower(model_name)

    def forward(self, graph_batch, input_ids, attention_mask):
        g = F.normalize(self.graph_tower(graph_batch.x, graph_batch.edge_index,
                                          graph_batch.batch), dim=-1)
        t = F.normalize(self.text_tower(input_ids, attention_mask), dim=-1)
        return g, t


def info_nce_loss(g_embed, t_embed, temperature=TEMPERATURE):
    logits = g_embed @ t_embed.T / temperature
    labels = torch.arange(len(g_embed), device=g_embed.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class PairedDataset(Dataset):
    def __init__(self, meta_df, graph_dir, tokenizer, max_len=128):
        self.rows = meta_df.reset_index(drop=True)
        self.graph_dir = Path(graph_dir)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        g = torch.load(self.graph_dir / f'{row.clip_id}.pt', weights_only=False)
        enc = self.tok(str(row.caption), truncation=True, padding='max_length',
                        max_length=self.max_len, return_tensors='pt')
        return {'graph': Data(x=g['x'], edge_index=g['edge_index']),
                'input_ids': enc['input_ids'][0], 'attention_mask': enc['attention_mask'][0],
                'clip_id': int(row.clip_id), 'caption': str(row.caption)}


def collate(batch):
    graphs = Batch.from_data_list([b['graph'] for b in batch])
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    return graphs, input_ids, attention_mask, [b['clip_id'] for b in batch], \
        [b['caption'] for b in batch]


def build_paired_meta(data_dir):
    data_dir = Path(data_dir)
    graph_meta = pd.read_csv(data_dir / 'task2_graphs' / 'graph_metadata.csv')
    mtat = pd.read_csv(data_dir / 'mtat_master.csv')
    return graph_meta.merge(mtat[['clip_id', 'caption']], on='clip_id', how='inner')


def train(model, loader, val_loader, epochs, lr, device):
    """Tracks and restores the best-validation-loss checkpoint — without
    this, training keeps whatever the LAST epoch produced even after
    validation loss starts climbing (overfitting)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history, best_val_loss, best_epoch, best_state = [], float('inf'), -1, None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for graphs, input_ids, attention_mask, _, _ in loader:
            graphs, input_ids, attention_mask = graphs.to(device), input_ids.to(device), attention_mask.to(device)
            opt.zero_grad()
            g, t = model(graphs, input_ids, attention_mask)
            loss = info_nce_loss(g, t)
            loss.backward()
            opt.step()
            total_loss += loss.item() * graphs.num_graphs
        train_loss = total_loss / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            for graphs, input_ids, attention_mask, _, _ in val_loader:
                graphs, input_ids, attention_mask = graphs.to(device), input_ids.to(device), attention_mask.to(device)
                g, t = model(graphs, input_ids, attention_mask)
                val_loss_sum += info_nce_loss(g, t).item() * graphs.num_graphs
            val_loss = val_loss_sum / len(val_loader.dataset)

        print(f'epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f'\nRestored best checkpoint: epoch {best_epoch} (val_loss {best_val_loss:.4f})')
    return history, best_epoch


@torch.no_grad()
def evaluate_retrieval(model, loader, device, ks=(1, 5, 10)):
    model.eval()
    all_g, all_t, all_captions, all_ids = [], [], [], []
    for graphs, input_ids, attention_mask, clip_ids, captions in loader:
        graphs, input_ids, attention_mask = graphs.to(device), input_ids.to(device), attention_mask.to(device)
        g, t = model(graphs, input_ids, attention_mask)
        all_g.append(g.cpu())
        all_t.append(t.cpu())
        all_captions.extend(captions)
        all_ids.extend(clip_ids)
    g, t = torch.cat(all_g), torch.cat(all_t)
    n = len(g)
    ks = [k for k in ks if k <= n]
    sim = g @ t.T

    def recall_at_k(sim_matrix):
        ranks = np.array([(torch.argsort(sim_matrix[i], descending=True) == i)
                           .nonzero(as_tuple=True)[0].item() for i in range(n)])
        return {f'R@{k}': float((ranks < k).mean()) for k in ks}

    g2t, t2g = recall_at_k(sim), recall_at_k(sim.T)
    chance = {f'R@{k}': round(k / n, 4) for k in ks}   # what random guessing would score

    examples = []
    for i in range(min(10, n)):
        order = torch.argsort(sim[:, i], descending=True)[:3]
        examples.append({'query_caption': all_captions[i], 'query_clip_id': all_ids[i],
                          'top3_matched_clip_ids': [all_ids[j] for j in order.tolist()],
                          'correct_in_top3': bool(i in order.tolist())})

    return {'audio_to_caption': g2t, 'caption_to_audio': t2g,
            'chance_level': chance, 'n_test': n}, examples


# ── Zero-shot tag prediction — no classifier head, no label fitting ──

@torch.no_grad()
def embed_graphs(model, loader, device):
    model.eval()
    embs, ids = [], []
    for graphs, _, _, clip_ids, _ in loader:
        graphs = graphs.to(device)
        g = F.normalize(model.graph_tower(graphs.x, graphs.edge_index, graphs.batch), dim=-1)
        embs.append(g.cpu())
        ids.extend(clip_ids)
    return torch.cat(embs), ids


@torch.no_grad()
def embed_tag_names(model, tags, tokenizer, device, max_len=128):
    enc = tokenizer(tags, truncation=True, padding='max_length',
                     max_length=max_len, return_tensors='pt').to(device)
    t = model.text_tower(enc['input_ids'], enc['attention_mask'])
    return F.normalize(t, dim=-1).cpu()


def tune_thresholds_zs(P, Y, grid=np.arange(-0.5, 0.51, 0.05)):
    thr = np.zeros(Y.shape[1])
    for k in range(Y.shape[1]):
        if 0 < Y[:, k].sum() < len(Y):
            scores = [f1_score(Y[:, k], (P[:, k] > t).astype(int), zero_division=0)
                      for t in grid]
            thr[k] = grid[int(np.argmax(scores))]
    return thr


def compute_metrics_zs(P, Y, thr):
    pred = (P > thr[None, :]).astype(int)
    valid = [k for k in range(Y.shape[1]) if 0 < Y[:, k].sum() < len(Y)]
    if not valid:
        return {'macro_f1': 0., 'micro_f1': 0., 'roc_auc': .5, 'pr_auc': 0.}
    return {
        'macro_f1': f1_score(Y, pred, average='macro', zero_division=0),
        'micro_f1': f1_score(Y, pred, average='micro', zero_division=0),
        'roc_auc': roc_auc_score(Y[:, valid], P[:, valid], average='macro'),
        'pr_auc': average_precision_score(Y[:, valid], P[:, valid], average='macro'),
    }


def zero_shot_tag_prediction(model, tokenizer, tags, val_loader, test_loader,
                              Y_val, Y_test, device, max_len=128):
    """Predicts tags purely by cosine similarity between a clip's graph
    embedding and each tag name's text embedding — no training on labels."""
    tag_embeds = embed_tag_names(model, tags, tokenizer, device, max_len)
    g_val, _ = embed_graphs(model, val_loader, device)
    g_test, _ = embed_graphs(model, test_loader, device)

    P_val = (g_val @ tag_embeds.T).numpy()
    P_test = (g_test @ tag_embeds.T).numpy()

    thr = tune_thresholds_zs(P_val, Y_val)
    return compute_metrics_zs(P_test, Y_test, thr), thr
