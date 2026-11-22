"""Q4(a): chronological splits, boundary-safe windows, train-only z-score, DataLoaders."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from common import PROC

T_IN, T_OUT = 8, 4            # 8 steps (2 h) in, 4 steps (1 h) out, 15-min resolution
N_NODES = 716


def load_processed():
    """Return dict(X (T,N) float32 filled flow, mask (T,N) bool valid-observation, times DatetimeIndex)."""
    X = np.load(os.path.join(PROC, 'sd_flow_15min_filled.npy'))[..., 0]
    mask = np.load(os.path.join(PROC, 'sd_valid_mask.npy'))
    times = pd.DatetimeIndex(np.load(os.path.join(PROC, 'sd_timestamps.npy')))
    assert X.shape == mask.shape and X.shape[0] == len(times) and X.shape[1] == N_NODES
    return dict(X=X, mask=mask, times=times)


def split_slices(times):
    """Contiguous [start, stop) index ranges on the global time axis for each split."""
    y, m = times.year, times.month
    sel = dict(train=(y == 2020),
               val=((y == 2021) & (m <= 3)),
               test=((y == 2021) & (m >= 4)))
    out = {}
    for k, s in sel.items():
        idx = np.where(s)[0]
        assert idx.size > 0 and (np.diff(idx) == 1).all(), f'{k} split must be contiguous'
        out[k] = (int(idx[0]), int(idx[-1]) + 1)
    # splits are disjoint and ordered chronologically
    assert out['train'][1] <= out['val'][0] and out['val'][1] <= out['test'][0]
    return out


class WindowSet(Dataset):
    """Windows fully inside ONE split segment.

    Window i (0-based inside the segment) uses segment rows i..i+7 as input (x_{t-7..t}, t=i+7)
    and rows i+8..i+11 as target (x_{t+1..t+4}).  Number of windows = L - (T_IN+T_OUT) + 1.
    """

    def __init__(self, Xn, mask, start, stop):
        self.start, self.stop = start, stop
        self.X = torch.from_numpy(Xn[start:stop])                    # (L, N), normalised
        self.M = torch.from_numpy(mask[start:stop].astype(np.float32))  # (L, N), 1 = valid observation
        self.L = stop - start
        self.n = self.L - (T_IN + T_OUT) + 1
        assert self.n > 0

    def __len__(self):
        return self.n

    def anchor(self, i):
        """Global index of prediction time t of window i."""
        return self.start + i + T_IN - 1

    def __getitem__(self, i):
        x = self.X[i:i + T_IN]                                       # (8, N)
        y = self.X[i + T_IN:i + T_IN + T_OUT]                        # (4, N)
        m = self.M[i + T_IN:i + T_IN + T_OUT]                        # (4, N) target validity
        return x.unsqueeze(-1), y, m                                 # x: (8, N, 1)


def build_data(batch_size=64, seed=42, subset=None, num_workers=0, verbose=True, norm='sensor'):
    d = load_processed()
    X, mask, times = d['X'], d['mask'], d['times']
    sl = split_slices(times)
    a, b = sl['train']
    # ---- z-score statistics: TRAIN split only, observed (valid) values only ----
    trX, trM = X[a:b].astype(np.float64), mask[a:b]
    g_mean, g_std = float(trX[trM].mean()), float(trX[trM].std())
    if norm == 'global':                                   # one scalar mean/std for every sensor
        mean_v, std_v = np.full(N_NODES, g_mean), np.full(N_NODES, g_std)
    elif norm == 'sensor':                                 # one mean/std per sensor (train-split statistics of that sensor)
        cnt = trM.sum(0)
        mean_v = np.where(cnt > 0, (trX * trM).sum(0) / np.maximum(cnt, 1), g_mean)
        var_v = np.where(cnt > 1, (((trX - mean_v) ** 2) * trM).sum(0) / np.maximum(cnt - 1, 1), g_std ** 2)
        std_v = np.sqrt(var_v)
        fallback = (cnt < 1000) | (std_v < 1e-6)           # sensors with (almost) no training observations -> global statistics
        mean_v = np.where(fallback, g_mean, mean_v); std_v = np.where(fallback, g_std, std_v)
    else:
        raise ValueError(norm)
    Xn = ((X - mean_v[None, :]) / std_v[None, :]).astype(np.float32)
    mean_t, std_t = torch.tensor(mean_v, dtype=torch.float32), torch.tensor(std_v, dtype=torch.float32)   # (N,) broadcast over the last axis
    loss_w = (std_t / g_std)                                # per-sensor weight: loss = |error| in original units / global std
    sets = {k: WindowSet(Xn, mask, *sl[k]) for k in ('train', 'val', 'test')}
    if verbose:
        print(f'z-score mode={norm} (train split only, {int(trM.sum()):,} observed values): global mean={g_mean:.2f} std={g_std:.2f} | per-sensor mean range [{mean_v.min():.1f}, {mean_v.max():.1f}], std range [{std_v.min():.1f}, {std_v.max():.1f}]')
        for k, s in sets.items():
            print(f'{k:5s}: rows [{s.start}, {s.stop}) {times[s.start]} -> {times[s.stop-1]} | rows={s.L} windows={len(s)}')
    # ---- boundary-crossing assertions ----
    for k, s in sets.items():
        lo, hi = sl[k]
        for i in (0, len(s) - 1):                       # first / last window (extremes)
            first_in, last_tgt = s.anchor(i) - (T_IN - 1), s.anchor(i) + T_OUT
            assert lo <= first_in and last_tgt < hi, f'{k}: window {i} crosses split boundary'
        assert s.anchor(len(s) - 1) + T_OUT == hi - 1 and s.anchor(0) - (T_IN - 1) == lo
    g = torch.Generator().manual_seed(seed)
    loaders = {}
    for k, s in sets.items():
        ds = s if subset is None else torch.utils.data.Subset(s, range(min(subset, len(s))))
        loaders[k] = DataLoader(ds, batch_size=batch_size, shuffle=(k == 'train'), generator=g if k == 'train' else None,
                                num_workers=num_workers, drop_last=False)
    return dict(loaders=loaders, sets=sets, mean=mean_t, std=std_t, loss_w=loss_w, norm=norm, global_mean=g_mean, global_std=g_std, mean_np=mean_v, std_np=std_v, slices=sl, times=times, raw=X, mask=mask,
                counts={k: len(s) for k, s in sets.items()})
