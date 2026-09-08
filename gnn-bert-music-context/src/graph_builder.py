"""
Task 2 (Member B) — builds a per-clip segment graph from audio features
and caches it to disk as a .pt file, plus a PyTorch Geometric Dataset
that reads those cached graphs for training.

Edges: temporal adjacency between consecutive segments, plus
cosine-similarity edges (threshold 0.75, max 2 per node) between
acoustically similar non-adjacent segments.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

from .audio_features import load_audio, extract_segment_features

SIM_THRESHOLD = 0.75
MAX_SIM_EDGES_PER_NODE = 2


def build_edges(x, similarity_threshold=SIM_THRESHOLD,
                 max_sim_edges=MAX_SIM_EDGES_PER_NODE):
    n = len(x)
    edge_set = set()

    # Temporal adjacency
    for i in range(n - 1):
        edge_set.add((i, i + 1))
        edge_set.add((i + 1, i))

    # Cosine-similarity edges between non-adjacent, acoustically similar nodes
    if n > 1:
        sim = cosine_similarity(x)
        for i in range(n):
            candidates = [(sim[i, j], j) for j in range(n)
                          if j != i and sim[i, j] >= similarity_threshold]
            candidates.sort(reverse=True)
            for _, j in candidates[:max_sim_edges]:
                edge_set.add((i, j))
                edge_set.add((j, i))

    if not edge_set:      # degenerate single-node graphs get a self-loop
        edge_set = {(i, i) for i in range(n)}

    return np.array(list(edge_set), dtype=np.int64).T


def build_graph_from_audio(path):
    """Full pipeline: audio path -> {x, edge_index} dict, or None if
    the clip produced zero usable segments."""
    y = load_audio(path)
    x = extract_segment_features(y)
    if len(x) == 0:
        return None
    return {'x': x, 'edge_index': build_edges(x)}


def cache_split(df, split_name, audio_by_id, graph_dir):
    """Builds and saves one .pt graph file per clip in df that has audio.
    Idempotent — skips clips whose .pt file already exists."""
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    matched = [int(r.clip_id) for _, r in df.iterrows() if int(r.clip_id) in audio_by_id]
    print(f'{split_name}: {len(df)} clips total | {len(matched)} have audio | '
          f'{len(df) - len(matched)} skipped')

    rows = []
    for idx, cid in enumerate(matched):
        out_file = graph_dir / f'{cid}.pt'
        if out_file.exists():
            rows.append({'clip_id': cid, 'split': split_name, 'graph_path': str(out_file)})
            continue
        try:
            graph = build_graph_from_audio(audio_by_id[cid])
            if graph is None:
                continue
            torch.save({'x': torch.tensor(graph['x'], dtype=torch.float32),
                        'edge_index': torch.tensor(graph['edge_index'], dtype=torch.long)},
                       out_file)
            rows.append({'clip_id': cid, 'split': split_name, 'graph_path': str(out_file)})
        except Exception as e:
            print(f'  Failed clip {cid}: {e}')
        if (idx + 1) % 100 == 0:
            print(f'  {split_name}: {idx + 1}/{len(matched)} done')

    print(f'  -> {len(rows)} graphs saved for {split_name}')
    return rows


def cache_all_splits(tr, va, te, audio_by_id, graph_dir):
    """Caches train/val/test and writes graph_metadata.csv. Raises if any
    split ends up with zero graphs (the #1 cause of empty-DataLoader bugs)."""
    rows = (cache_split(tr, 'train', audio_by_id, graph_dir)
            + cache_split(va, 'val', audio_by_id, graph_dir)
            + cache_split(te, 'test', audio_by_id, graph_dir))

    graph_meta = pd.DataFrame(rows)
    graph_meta.to_csv(Path(graph_dir) / 'graph_metadata.csv', index=False)

    counts = graph_meta.groupby('split').size().to_dict()
    missing = [s for s in ('train', 'val', 'test') if counts.get(s, 0) == 0]
    if missing:
        raise RuntimeError(f'Splits with zero graphs: {missing}. Counts: {counts}')

    print('\nAll three splits have graphs.')
    print(counts)
    return graph_meta


class MTATGraphDataset(torch.utils.data.Dataset):
    """Reads cached .pt graphs and attaches the multi-label tag vector."""

    def __init__(self, graph_meta, mtat, top50):
        self.meta = graph_meta.reset_index(drop=True)
        self.top50 = top50
        self.label_lookup = {
            int(row.clip_id): row[top50].values.astype(np.float32)
            for _, row in mtat.iterrows()
        }

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        cid = int(row.clip_id)
        g = torch.load(row.graph_path, map_location='cpu', weights_only=False)
        data = Data(x=g['x'].float(), edge_index=g['edge_index'].long(),
                    y=torch.tensor(self.label_lookup[cid], dtype=torch.float32))
        data.clip_id = cid
        return data
