"""
Task 3 (Member C) — GNN-BERT fusion for multi-context understanding.

Four fusion modes share one model class so the ablation study isolates
just the fusion mechanism: gnn_only, bert_only, concat, cross_attn.

Cross-attention: the graph embedding g queries the BERT token
embeddings H_text.  Q = gW_Q, K = H_text W_K, V = H_text W_V,
A = softmax(QK^T / sqrt(d)), z = CONCAT(g, A V).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import AutoModel
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


class FusionDataset(Dataset):
    """One (graph, tokenized caption, label) triple per clip. Needs
    clips that have BOTH a cached graph and a caption/tag row."""

    def __init__(self, df, tags, tokenizer, graph_meta_col='graph_path', max_len=128):
        self.df = df.reset_index(drop=True)
        self.tags = tags
        self.tok = tokenizer
        self.graph_col = graph_meta_col
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        g = torch.load(row[self.graph_col], map_location='cpu', weights_only=False)
        enc = self.tok(str(row.caption), truncation=True, padding='max_length',
                        max_length=self.max_len, return_tensors='pt')
        data = Data(x=g['x'].float(), edge_index=g['edge_index'].long())
        data.y = torch.tensor(row[self.tags].values.astype(np.float32)).unsqueeze(0)
        data.input_ids = enc['input_ids']
        data.attention_mask = enc['attention_mask']
        data.clip_id = int(row.clip_id)
        return data


class GNNEncoder(nn.Module):
    """Same 3-layer GraphSAGE trunk as Task 2, without the classifier head."""

    def __init__(self, in_channels, hidden=128, dropout=0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)


class CrossAttention(nn.Module):
    """The graph vector queries the text tokens. Padding positions are
    masked out of the softmax so attention can't leak onto [PAD]."""

    def __init__(self, dim_g, dim_t, dim=128):
        super().__init__()
        self.Wq = nn.Linear(dim_g, dim)
        self.Wk = nn.Linear(dim_t, dim)
        self.Wv = nn.Linear(dim_t, dim)
        self.scale = dim ** 0.5

    def forward(self, g, H_text, attn_mask):
        Q = self.Wq(g).unsqueeze(1)                    # [B, 1, d]
        K = self.Wk(H_text)                             # [B, L, d]
        V = self.Wv(H_text)                              # [B, L, d]
        scores = (Q @ K.transpose(1, 2)) / self.scale    # [B, 1, L]
        scores = scores.masked_fill(attn_mask.unsqueeze(1) == 0, -1e4)
        A = torch.softmax(scores, dim=-1)
        ctx = (A @ V).squeeze(1)
        return ctx, A.squeeze(1)


class FusionTagger(nn.Module):
    """mode in {'cross_attn', 'concat', 'gnn_only', 'bert_only'}"""

    def __init__(self, node_dim, n_tags=50, hidden=128, mode='cross_attn',
                 model_name='distilbert-base-uncased', dropout=0.2):
        super().__init__()
        self.mode = mode
        self.gnn = GNNEncoder(node_dim, hidden)
        self.bert = AutoModel.from_pretrained(model_name)
        d_t = self.bert.config.hidden_size

        if mode == 'cross_attn':
            self.xattn = CrossAttention(hidden, d_t, hidden)
            z_dim = hidden + hidden
        elif mode == 'concat':
            z_dim = hidden + d_t
        elif mode == 'gnn_only':
            z_dim = hidden
        elif mode == 'bert_only':
            z_dim = d_t
        else:
            raise ValueError(f'unknown mode: {mode}')

        self.head = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_tags))

    def forward(self, batch, return_attn=False):
        ids = batch.input_ids.view(batch.num_graphs, -1)
        mask = batch.attention_mask.view(batch.num_graphs, -1)
        attn_w = None

        if self.mode != 'bert_only':
            g = self.gnn(batch.x, batch.edge_index, batch.batch)
        if self.mode != 'gnn_only':
            out = self.bert(input_ids=ids, attention_mask=mask)
            H_text = out.last_hidden_state
            t = H_text[:, 0]

        if self.mode == 'cross_attn':
            ctx, attn_w = self.xattn(g, H_text, mask)
            z = torch.cat([g, ctx], dim=-1)
        elif self.mode == 'concat':
            z = torch.cat([g, t], dim=-1)
        elif self.mode == 'gnn_only':
            z = g
        else:
            z = t

        logits = self.head(z)
        return (logits, z, attn_w) if return_attn else (logits, z)


# ── Metrics — identical philosophy to Tasks 1 and 2 ──

def compute_metrics(P, Y, thr=0.5):
    thr = np.full(Y.shape[1], thr) if np.isscalar(thr) else np.asarray(thr)
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


def tune_thresholds(P, Y, grid=np.arange(0.05, 0.96, 0.05)):
    thr = np.full(Y.shape[1], 0.5)
    for k in range(Y.shape[1]):
        s = [f1_score(Y[:, k], (P[:, k] > t).astype(int), zero_division=0) for t in grid]
        thr[k] = grid[int(np.argmax(s))]
    return thr


@torch.no_grad()
def predict(model, loader, device, n_tags):
    model.eval()
    P, Y, Z, ids = [], [], [], []
    for b in loader:
        b = b.to(device)
        logits, z = model(b)
        P.append(torch.sigmoid(logits).float().cpu().numpy())
        Y.append(b.y.view(-1, n_tags).cpu().numpy())
        Z.append(z.float().cpu().numpy())
        ids.extend(np.atleast_1d(b.clip_id.cpu().numpy()).tolist())
    return np.vstack(P), np.vstack(Y), np.vstack(Z), np.asarray(ids)


def train_fusion(model_kwargs, train_loader, val_loader, n_tags,
                  epochs=8, device='cuda', seed=42, verbose=True):
    """Trains one fusion variant, tracking and restoring the best
    validation-PR-AUC checkpoint. model_kwargs must include node_dim
    and mode (one of gnn_only/bert_only/concat/cross_attn)."""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from tqdm.auto import tqdm

    torch.manual_seed(seed)
    model = FusionTagger(**model_kwargs).to(device)

    bert_params = [p for n, p in model.named_parameters() if n.startswith('bert')]
    other_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]
    opt = AdamW([{'params': bert_params, 'lr': 2e-5},
                 {'params': other_params, 'lr': 1e-3}], weight_decay=0.01)
    total = epochs * max(len(train_loader), 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    crit = nn.BCEWithLogitsLoss()

    history, best_pr, best_state = [], -np.inf, None
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for b in tqdm(train_loader, desc=f"[{model_kwargs['mode']}] ep{ep}/{epochs}", leave=False):
            b = b.to(device)
            opt.zero_grad()
            logits, _ = model(b)
            loss = crit(logits, b.y.view(-1, n_tags))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        Pv, Yv, _, _ = predict(model, val_loader, device, n_tags)
        m = compute_metrics(Pv, Yv)
        m['epoch'] = ep
        m['train_loss'] = running / max(len(train_loader), 1)
        history.append(m)
        if verbose:
            print(f"  [{model_kwargs['mode']}] val ep{ep} " +
                  '  '.join(f'{k}={v:.4f}' for k, v in m.items()
                            if k not in ('epoch', 'train_loss')))
        if m['pr_auc'] > best_pr:
            best_pr = m['pr_auc']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history
