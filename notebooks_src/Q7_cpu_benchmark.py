# %% [markdown]
# # AI651 Assignment 3 — Q7: CPU feasibility benchmark
# **Student:** Huzaifa | **Roll No:** 25280009 | AI651 Deep Learning for Space, Time and Graphs, Fall 2026, Assignment 3
#
# All main experiments were trained on the GPU (see the hardware table), but the assignment asks for a CPU-feasible design. Here **one complete training epoch (549 optimisation steps) plus one validation pass is run on the CPU** for each of the four models of the main table, with exactly the same data pipeline, optimiser and batch size, and the wall-clock time is measured with `time.perf_counter`. Nothing else is running on the machine during the measurement. The numbers are saved to `results/cpu_benchmark.json`.

# %%
import os, sys, time, json
sys.path.insert(0, '../src')
import numpy as np, torch
from common import set_seed, save_json, hardware_info, PROC, RES, SEED
from data import build_data, T_IN, T_OUT, N_NODES
from models import STModel, fixed_adjacency, count_params
from train import CFG, masked_mae_loss, evaluate
import warnings; warnings.filterwarnings('ignore', category=FutureWarning)
set_seed(SEED)
import psutil
PHYS = psutil.cpu_count(logical=False)
torch.set_num_threads(PHYS)          # physical cores: using all hyper-threads (24 logical) was measured to be ~5x SLOWER on this machine (oversubscription)
print('seed', SEED, '| device cpu | torch threads', torch.get_num_threads(), '| physical cores', PHYS, '| logical cores', os.cpu_count())
data = build_data(batch_size=CFG['batch_size'], seed=SEED, verbose=False)
A_fixed = fixed_adjacency(np.load(f'{PROC}/sd_adj.npy'), 'row')
lw = data['loss_w']

# %%
out = dict(threads=torch.get_num_threads(), logical_cores=os.cpu_count(), cpu=hardware_info()['cpu_model'], steps_per_epoch=len(data['loaders']['train']), batch_size=CFG['batch_size'], models={})
for name, graph in [('temporal_only', 'none'), ('stgnn_fixed', 'fixed'), ('stgnn_adaptive', 'adaptive'), ('stgnn_hybrid', 'hybrid')]:
    set_seed(SEED)
    mdl = STModel(graph, N_NODES, d=CFG['hidden_d'], t_in=T_IN, horizon=T_OUT, A_fixed=A_fixed, d_e=CFG['d_e'], top_k=CFG['top_k'], head_hidden=CFG['head_hidden'])
    opt = torch.optim.Adam(mdl.parameters(), lr=CFG['lr']); mdl.train()
    t0 = time.perf_counter()
    for x, y, m in data['loaders']['train']:
        opt.zero_grad(set_to_none=True); loss = masked_mae_loss(mdl(x), y, m, lw); loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), CFG['grad_clip']); opt.step()
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter(); v = evaluate(mdl, data['loaders']['val'], data['mean'], data['std'], 'cpu'); t_val = time.perf_counter() - t0
    out['models'][name] = dict(graph=graph, params=count_params(mdl), train_epoch_sec=t_train, val_pass_sec=t_val, epoch_total_sec=t_train + t_val, val_mae_after_1_epoch=v['mae'])
    print(f"{name:15s} CPU: train epoch {t_train:6.1f} s + validation {t_val:5.1f} s = {t_train + t_val:6.1f} s | val MAE after 1 epoch {v['mae']:.2f}", flush=True)
save_json(out, f'{RES}/cpu_benchmark.json')
print('saved results/cpu_benchmark.json')
