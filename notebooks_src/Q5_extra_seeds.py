# %% [markdown]
# # AI651 Assignment 3 — Q5 (optional): extra seeds for mean ± std
# **Student:** Huzaifa | **Roll No:** 25280009 | AI651 Deep Learning for Space, Time and Graphs, Fall 2026, Assignment 3
#
# The primary table (seed 42) is produced by `Q4_Q5_training.ipynb`. Here the *same* protocol is repeated for seeds 43 and 44 (main-table models and the self-path extension) so that Q6 can report mean ± std. The seed changes: weight initialisation, adaptive-graph embedding initialisation, and the training-batch order. Data splits and normalisation statistics are unchanged (they do not depend on the seed).

# %%
import os, sys, json
sys.path.insert(0, '../src')
import numpy as np, torch, pandas as pd
from common import set_seed, save_json, load_json, PROC, RES, CKPT
from data import build_data, T_IN, T_OUT, N_NODES
from models import STModel, fixed_adjacency, count_params
from train import CFG, train_model, evaluate, predict, metrics_from
import warnings; warnings.filterwarnings('ignore', category=FutureWarning)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
A_fixed = fixed_adjacency(np.load(f'{PROC}/sd_adj.npy'), 'row')
SEEDS = [43, 44]
RUNS = [('temporal_only', 'none', False), ('stgnn_fixed', 'fixed', False), ('stgnn_adaptive', 'adaptive', False), ('stgnn_hybrid', 'hybrid', False),
        ('stgnn_fixed_skip', 'fixed', True), ('stgnn_adaptive_skip', 'adaptive', True), ('stgnn_hybrid_skip', 'hybrid', True)]
print('device', DEVICE, '| seeds', SEEDS, '| runs', [r[0] for r in RUNS])

# %%
for seed in SEEDS:
    for name, graph, sp in RUNS:
        out_path = f'{RES}/train_{name}_seed{seed}.json'
        if os.path.exists(out_path):
            print('skip (exists):', out_path); continue
        cfg = dict(CFG); cfg['seed'] = seed
        d_ = build_data(batch_size=cfg['batch_size'], seed=seed, verbose=False)
        set_seed(seed)
        mdl = STModel(graph, N_NODES, d=cfg['hidden_d'], t_in=T_IN, horizon=T_OUT, A_fixed=A_fixed, d_e=cfg['d_e'], top_k=cfg['top_k'], head_hidden=cfg['head_hidden'], self_path=sp)
        info = train_model(mdl, d_, f'{name}', cfg, device=DEVICE, verbose=False)
        P, Y, M = predict(mdl, d_['loaders']['test'], d_['mean'], d_['std'], DEVICE)
        out = dict(name=name, graph=graph, seed=seed, self_path=sp, params=count_params(mdl), device=DEVICE, **{k: info[k] for k in ('epochs_run', 'best_epoch', 'train_time_sec', 'best_val_mae')},
                   history=info['history'], cfg=cfg, test=metrics_from(P, Y, M), val=evaluate(mdl, d_['loaders']['val'], d_['mean'], d_['std'], DEVICE))
        if graph == 'hybrid': out['alpha'] = float(mdl.alpha)
        if graph in ('adaptive', 'hybrid'):
            with torch.no_grad(): np.save(f'{CKPT}/A_adaptive_{name}_seed{seed}.npy', mdl.adp().cpu().numpy().astype(np.float32))
        save_json(out, out_path)
        print(f"seed {seed} {name:22s} epochs {out['epochs_run']:2d} (best {out['best_epoch']:2d}) time {out['train_time_sec']:6.1f}s | val MAE {out['val']['mae']:.3f} | test MAE {out['test']['mae']:.3f} RMSE {out['test']['rmse']:.3f}"
              + (f" alpha {out['alpha']:.3f}" if 'alpha' in out else ''), flush=True)

# %% [markdown]
# ## Summary over seeds 42, 43, 44 (mean ± sample std, test period, original units)

# %%
rows = []
for name, graph, sp in RUNS:
    rs = [load_json(f'{RES}/train_{name}_seed{s}.json') for s in (42, 43, 44) if os.path.exists(f'{RES}/train_{name}_seed{s}.json')]
    f = lambda k, sub: np.array([r[sub][k] for r in rs])
    rows.append(dict(model=name, graph=graph, self_path=sp, n_seeds=len(rs), test_MAE=f'{f("mae","test").mean():.2f} ± {f("mae","test").std(ddof=1):.2f}', test_RMSE=f'{f("rmse","test").mean():.2f} ± {f("rmse","test").std(ddof=1):.2f}',
                     val_MAE=f'{f("mae","val").mean():.2f} ± {f("mae","val").std(ddof=1):.2f}', time_s=f'{np.mean([r["train_time_sec"] for r in rs]):.0f} ± {np.std([r["train_time_sec"] for r in rs], ddof=1):.0f}',
                     epochs=[r['epochs_run'] for r in rs]))
S = pd.DataFrame(rows); print(S.to_string(index=False)); S.to_csv(f'{RES}/multiseed_summary.csv', index=False)
