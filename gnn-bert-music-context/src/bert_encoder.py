"""
Task 1 (Member A) — DistilBERT multi-label tag classifier.

CLS token -> linear head -> per-tag sigmoid, trained with BCE.
Differential learning rates: the pretrained BERT body gets a small LR
(2e-5), the freshly-initialized head gets a much larger one (1e-3).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from tqdm.auto import tqdm

try:  # torch >= 2.4
    from torch.amp import autocast, GradScaler
    _mk_scaler = lambda: GradScaler('cuda')
    _mk_ctx = lambda: autocast('cuda')
except ImportError:
    _mk_scaler = lambda: torch.cuda.amp.GradScaler()
    _mk_ctx = lambda: torch.cuda.amp.autocast()


class BertTagger(nn.Module):
    def __init__(self, n_tags, model_name='distilbert-base-uncased', dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, n_tags)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                         output_attentions=output_attentions)
        cls = out.last_hidden_state[:, 0]
        logits = self.head(self.drop(cls))       # raw logits on purpose
        return (logits, out.attentions) if output_attentions else logits


class CaptionDS(Dataset):
    """Tokenizes on the fly rather than pre-caching, so any text column
    (raw caption, tag-masked, metadata-only) can reuse this same class."""

    def __init__(self, df, tags, tok, text_col='caption', max_len=128):
        self.texts = df[text_col].fillna('').astype(str).tolist()
        self.y = df[tags].values.astype('float32')
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        e = self.tok(self.texts[i], truncation=True, padding='max_length',
                      max_length=self.max_len, return_tensors='pt')
        return {'input_ids': e['input_ids'][0],
                'attention_mask': e['attention_mask'][0],
                'y': torch.tensor(self.y[i])}


def make_loaders(tr, va, te, tags, tokenizer, text_col='caption',
                  batch_size=32, max_len=128):
    mk = lambda d, sh: DataLoader(CaptionDS(d, tags, tokenizer, text_col, max_len),
                                   batch_size=batch_size, shuffle=sh,
                                   num_workers=0, pin_memory=True)
    return mk(tr, True), mk(va, False), mk(te, False)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    for b in loader:
        lo = model(b['input_ids'].to(device), b['attention_mask'].to(device))
        P.append(torch.sigmoid(lo).float().cpu().numpy())
        Y.append(b['y'].numpy())
    return np.vstack(P), np.vstack(Y)


def compute_metrics(P, Y, thr=0.5):
    thr = np.full(Y.shape[1], thr) if np.isscalar(thr) else thr
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
        scores = [f1_score(Y[:, k], (P[:, k] > t).astype(int), zero_division=0)
                  for t in grid]
        thr[k] = grid[int(np.argmax(scores))]
    return thr


def train_bert_tagger(tr, va, te, tags, model_name='distilbert-base-uncased',
                       text_col='caption', max_len=128, batch_size=32,
                       epochs=5, device='cuda', seed=42, verbose=True):
    """Full Task 1 training loop. Returns (model, history, tokenizer, loaders)."""
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dl, val_dl, test_dl = make_loaders(tr, va, te, tags, tokenizer,
                                              text_col, batch_size, max_len)
    model = BertTagger(len(tags), model_name).to(device)

    opt = AdamW([{'params': model.bert.parameters(), 'lr': 2e-5},
                 {'params': model.head.parameters(), 'lr': 1e-3}], weight_decay=0.01)
    total = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    crit, scaler = nn.BCEWithLogitsLoss(), _mk_scaler()

    history, best_pr, best_state = [], -1, None
    for ep in range(epochs):
        model.train()
        running = 0.0
        for b in tqdm(train_dl, desc=f'[{text_col}] ep{ep+1}/{epochs}', leave=False):
            opt.zero_grad()
            with _mk_ctx():
                loss = crit(model(b['input_ids'].to(device),
                                   b['attention_mask'].to(device)), b['y'].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()

        Pv, Yv = predict(model, val_dl, device)
        m = compute_metrics(Pv, Yv)
        m['epoch'] = ep + 1
        m['train_loss'] = running / len(train_dl)
        history.append(m)
        if verbose:
            print(f'  val ep{ep+1}  ' + '  '.join(f'{k}={v:.4f}' for k, v in m.items()
                                                    if k not in ('epoch', 'train_loss')))
        if m['pr_auc'] > best_pr:
            best_pr = m['pr_auc']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)   # roll back to the best epoch
    return model, history, tokenizer, (train_dl, val_dl, test_dl)
