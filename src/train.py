"""Training / evaluation (identical protocol for all models)."""
import copy, os, time
import numpy as np
import torch

from common import CKPT, set_seed

CFG = dict(seed=42, batch_size=64, lr=1e-3, weight_decay=0.0, max_epochs=60, patience=6, min_delta=0.0,
           grad_clip=5.0, hidden_d=32, head_hidden=64, d_e=10, top_k=20, loss='masked MAE, weighted by sensor std (= raw-unit MAE / global std)', norm='per-sensor z-score (train only)')


def masked_mae_loss(pred, y, m, w=None):
    """Masked MAE. pred,y,m: (B,4,N); w: optional (N,) per-sensor weights = sensor_std/global_std so that the
    loss equals the raw-unit MAE divided by the global std (i.e. it optimises the same quantity that is reported)."""
    e = (pred - y).abs() * m
    if w is not None:
        e = e * w
    return e.sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def predict(model, loader, mean, std, device):
    """Return denormalised predictions / targets / masks, each (n_windows, 4, N)."""
    model.eval()
    P, Y, M = [], [], []
    for x, y, m in loader:
        p = model(x.to(device)).cpu()
        P.append(p * std + mean); Y.append(y * std + mean); M.append(m)
    return torch.cat(P), torch.cat(Y), torch.cat(M)


def metrics_from(P, Y, M):
    """MAE / RMSE in ORIGINAL flow units (vehicles per 15 min), computed over valid targets only.
    Overall and per horizon (15/30/45/60 min)."""
    err = (P - Y).double()
    cnt_h = M.sum((0, 2)).double()
    mae_h = (err.abs() * M).sum((0, 2)) / cnt_h
    rmse_h = torch.sqrt((err ** 2 * M).sum((0, 2)) / cnt_h)
    cnt = M.sum().double()
    return dict(mae=float(((err.abs() * M).sum() / cnt)), rmse=float(torch.sqrt((err ** 2 * M).sum() / cnt)),
                mae_h=mae_h.tolist(), rmse_h=rmse_h.tolist(), n_valid=int(cnt))


def evaluate(model, loader, mean, std, device):
    P, Y, M = predict(model, loader, mean, std, device)
    return metrics_from(P, Y, M)


def train_model(model, data, name, cfg=CFG, device='cpu', verbose=True):
    """Adam + early stopping on validation MAE (original units); best checkpoint restored at the end."""
    set_seed(cfg['seed'])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    L, mean, std = data['loaders'], data['mean'], data['std']
    lw = data['loss_w'].to(device)
    ckpt = os.path.join(CKPT, f"{name}_seed{cfg['seed']}.pt")
    best, best_ep, bad, hist = float('inf'), -1, 0, []
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, cfg['max_epochs'] + 1):
        te = time.perf_counter()
        model.train(); tot, n = 0.0, 0
        for x, y, m in L['train']:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            loss = masked_mae_loss(model(x), y, m, lw)
            loss.backward()
            if cfg['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        v = evaluate(model, L['val'], mean, std, device)
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - te
        rec = dict(epoch=ep, train_loss_norm=tot / n, val_mae=v['mae'], val_rmse=v['rmse'], epoch_sec=dt)
        if model.graph == 'hybrid':
            rec['alpha'] = float(model.alpha)
        hist.append(rec)
        improved = v['mae'] < best - cfg['min_delta']
        if improved:
            best, best_ep, bad = v['mae'], ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        if verbose:
            print(f"[{name}] ep {ep:02d} train_loss(norm) {rec['train_loss_norm']:.4f} | val MAE {v['mae']:.3f} RMSE {v['rmse']:.3f} "
                  f"| {dt:.1f}s {'*' if improved else ''}", flush=True)
        if bad >= cfg['patience']:
            break
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return dict(history=hist, best_epoch=best_ep, epochs_run=len(hist), train_time_sec=train_time, ckpt=ckpt,
                best_val_mae=best)
