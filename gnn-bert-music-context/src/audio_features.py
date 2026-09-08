"""
Task 2 (Member B) — audio loading and segment-level feature extraction.

Each clip is resampled to 22,050 Hz, split into fixed 5-second windows,
and each window becomes one graph node with a 64-dim feature vector:
20 MFCC means + 20 MFCC stds + 12 chroma means + 12 chroma stds.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import librosa

SR = 22050
SEGMENT_SECONDS = 5.0
N_MFCC = 20
N_CHROMA = 12
NODE_FEATURE_DIM = N_MFCC * 2 + N_CHROMA * 2   # 64


def find_audio_files(input_root):
    """Searches attached Kaggle datasets for MagnaTagATune mp3 files,
    without assuming a specific directory name."""
    input_root = Path(input_root)
    candidate_dirs = []
    for p in input_root.rglob('*'):
        if p.is_dir() and any(x in p.name.lower()
                               for x in ('magna', 'mtat', 'audio', 'mp3')):
            candidate_dirs.append(p)

    mp3_files = []
    for d in candidate_dirs:
        try:
            mp3_files.extend(d.rglob('*.mp3'))
        except Exception:
            pass
    mp3_files = list(dict.fromkeys(mp3_files))

    if not mp3_files:
        raise FileNotFoundError(
            'No MP3 files found under the input root. Check that the '
            'MagnaTagATune audio dataset is attached / downloaded.')
    return mp3_files


def build_clip_id_index(mp3_files):
    """Maps clip_id -> audio path by extracting numeric tokens from
    each filename (MagnaTagATune filenames embed the clip id)."""
    def possible_clip_ids(path):
        return [int(x) for x in re.findall(r'\d+', path.stem)]

    audio_by_id = {}
    for p in mp3_files:
        for cid in possible_clip_ids(p):
            audio_by_id.setdefault(cid, p)
    return audio_by_id


def load_audio(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs
    return y


def extract_segment_features(y, sr=SR, segment_seconds=SEGMENT_SECONDS,
                              n_mfcc=N_MFCC):
    """One clip -> [n_segments, 64] node feature matrix."""
    samples_per_segment = int(sr * segment_seconds)
    segments = []
    for start in range(0, len(y), samples_per_segment):
        seg = y[start:start + samples_per_segment]
        if len(seg) < int(0.5 * sr):        # drop tiny trailing fragments
            continue
        segments.append(seg)

    features = []
    for seg in segments:
        if len(seg) < samples_per_segment:
            seg = np.pad(seg, (0, samples_per_segment - len(seg)))

        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr)

        feat = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
        ])
        features.append(feat)

    if not features:
        return np.empty((0, NODE_FEATURE_DIM), dtype=np.float32)

    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    return ((features - mean) / std).astype(np.float32)
