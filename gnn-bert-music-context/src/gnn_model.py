"""
Task 2 (Member B) — GraphSAGE tagger, plus the CNN mel-spectrogram
baseline (B2) it's compared against.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from .audio_features import SR


class GraphSAGETagger(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_tags=50, dropout=0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, num_tags))

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.classifier(g)


# ── Metrics (identical philosophy to Task 1, for a fair comparison) ──

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


@torch.no_grad()
def predict_gnn(model, loader, device, n_tags):
    model.eval()
    P, Y = [], []
    for b in loader:
        b = b.to(device)
        logits = model(b.x, b.edge_index, b.batch)
        P.append(torch.sigmoid(logits).float().cpu().numpy())
        Y.append(b.y.view(-1, n_tags).cpu().numpy())
    if not P:
        raise RuntimeError('Empty loader — check that this split has cached graphs.')
    return np.vstack(P), np.vstack(Y)


# ── B2: CNN mel-spectrogram baseline (audio only, no graph structure) ──

MEL_N = 128
MEL_FRAMES = 512


class MTATMelDataset(Dataset):
    def __init__(self, df, audio_by_id, top50):
        self.df = df.reset_index(drop=True)
        self.audio_by_id = audio_by_id
        self.top50 = top50

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.audio_by_id[int(row.clip_id)]
        y, sr = librosa.load(path, sr=SR, mono=True)

        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=MEL_N,
                                              n_fft=2048, hop_length=512)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        if mel.shape[1] < MEL_FRAMES:
            mel = np.pad(mel, ((0, 0), (0, MEL_FRAMES - mel.shape[1])))
        else:
            mel = mel[:, :MEL_FRAMES]

        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        y_label = torch.tensor(row[self.top50].values.astype(np.float32))
        return x, y_label


class MelCNN(nn.Module):
    def __init__(self, num_tags=50):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                         nn.Linear(256, num_tags))

    def forward(self, x):
        return self.classifier(self.features(x))
