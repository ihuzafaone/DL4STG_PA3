# %% [markdown]
# # AI651 Assignment 3 — Q4 & Q5: spatio-temporal forecasting and fixed / learnable / hybrid adjacency
# **Student:** Huzaifa | **Roll No:** 25280009 | AI651 Deep Learning for Space, Time and Graphs, Fall 2026, Assignment 3
#
# This notebook (i) builds the forecasting windows (Q4(a)), (ii) trains the temporal-only baseline (Q4(c)) and the STGNN with **fixed** (Q5(a)), **adaptive** (Q5(b)) and **hybrid** (Q5(c)) adjacency under one identical protocol, and (iii) saves every metric, training time, checkpoint, prediction and learned graph for `Q6_analysis.ipynb`.
# Prerequisite: `Q3_data_and_graph.ipynb` must have been run (it writes `data/processed/*`).

# %%
import os, sys, time, json
sys.path.insert(0, '../src')
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from common import set_seed, hardware_info, save_json, PROC, RES, FIG, CKPT, SEED
from data import build_data, T_IN, T_OUT, N_NODES, split_slices, load_processed
from models import STModel, AdaptiveAdjacency, fixed_adjacency, count_params
from train import CFG, train_model, evaluate, predict, metrics_from
import warnings; warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3})
set_seed(SEED); print('Global seed =', SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HW = hardware_info(); save_json(HW, f'{RES}/hardware.json'); print(json.dumps(HW, indent=1)); print('training device:', DEVICE)
print('CONFIG:', json.dumps(CFG, indent=1))

# %% [markdown]
# ## Q4(a) — Forecasting windows, chronological splits, normalisation
# * Input $X_t=[x_{t-7},\dots,x_t]\in\mathbb R^{8\times716\times1}$, target $Y_t=[x_{t+1},\dots,x_{t+4}]\in\mathbb R^{4\times716\times1}$ (15-min steps: 2 h of history → next hour).
# * Splits (chronological, on the 15-min axis): **train** Jan–Dec 2020, **validation** Jan–Mar 2021, **test** Apr–Dec 2021.
# * Windows are generated **inside each split segment only**, so the first input row and the last target row of every window lie in the same split (a window never straddles a boundary). The code asserts this and an exhaustive check over *all* windows is run below. Number of windows per split $=L-11$ with $L$ the rows of the split.
# * **Z-score normalisation (per sensor, train-only):** for every sensor the mean and std are computed **only over the observed values of the training split (2020)** and applied to all splits (sensors with < 1000 training observations — the 19 sensors that have no 2020 data — use the global training mean/std); predictions are inverted (×std_j+mean_j) before every metric. *Design note (documented iteration):* my first attempt used one global scalar mean/std; with that choice the fixed-graph STGNN reached validation MAE 121.3 vs 56.5 for the temporal-only model (results kept in `results/iter1_global_norm/`) because graph averaging mixes sensors whose flow levels differ by a factor of 40 (per-sensor mean 30–1264). Per-sensor normalisation fixes the scale mismatch and also helps the temporal-only model; it is a data-preparation choice made on **validation** results only, and all four models use it.
* **Loss weighting:** the MAE loss is computed as |error| × (std_j / global std) so that, although each sensor is normalised separately, the loss equals the raw-unit MAE (divided by the global std) that is actually reported.
# * **Batch size 64:** 549 optimisation steps per epoch on 35,125 training windows — enough gradient noise/steps per epoch for fast early stopping, while one step already processes 64×716 = 45,824 sensor series, so memory is small (≈ 0.5 GB) and larger batches would only reduce the number of updates; 64 is also the LargeST benchmark default.

# %%
data = build_data(batch_size=CFG['batch_size'], seed=SEED)
sets, sl, times = data['sets'], data['slices'], data['times']
# exhaustive boundary check over every window of every split
chk = {}
for k, s in sets.items():
    lo, hi = sl[k]
    first_in = np.array([s.anchor(i) - (T_IN - 1) for i in range(len(s))]); last_tgt = np.array([s.anchor(i) + T_OUT for i in range(len(s))])
    assert first_in.min() >= lo and last_tgt.max() < hi and (last_tgt - first_in == T_IN + T_OUT - 1).all()
    chk[k] = dict(windows=len(s), first_input_time=str(times[first_in.min()]), last_target_time=str(times[last_tgt.max()]), rows=int(hi - lo))
print(pd.DataFrame(chk).T.to_string())
assert sl['train'][1] == sl['val'][0] and sl['val'][1] == sl['test'][0]
x, y, m = next(iter(data['loaders']['train']))
print('one training batch: X', tuple(x.shape), '= (B, 8, N, 1) | Y', tuple(y.shape), '= (B, 4, N) | target-valid mask', tuple(m.shape))
print('batches per epoch (train/val/test):', {k: len(v) for k, v in data['loaders'].items()})
Q4a = dict(window_counts=data['counts'], splits=chk, norm='per-sensor z-score, train split only', global_mean=data['global_mean'], global_std=data['global_std'], sensor_mean_min=float(data['mean_np'].min()), sensor_mean_max=float(data['mean_np'].max()), sensor_std_min=float(data['std_np'].min()), sensor_std_max=float(data['std_np'].max()), batch_size=CFG['batch_size'],
           steps_per_epoch=len(data['loaders']['train']), input_shape=[8, 716, 1], target_shape=[4, 716, 1])
mean, std = data['mean'], data['std']

# %% [markdown]
# ### Reference point: "historical last" (persistence) baseline — not a trained model
# Predict $\hat x_{t+h}=x_t$ for all four horizons, evaluated with exactly the same masked, de-normalised metric on the validation and test windows. It is shown only to put the learned models' errors in context.

# %%
def persistence_metrics(split):
    P, Y, M = [], [], []
    for x, y, m in data['loaders'][split]:
        last = x[:, -1, :, 0]                                          # (B, N) normalised last observed (possibly imputed) value
        P.append((last.unsqueeze(1).repeat(1, T_OUT, 1)) * std + mean); Y.append(y * std + mean); M.append(m)
    return metrics_from(torch.cat(P), torch.cat(Y), torch.cat(M))
HL = {s: persistence_metrics(s) for s in ('val', 'test')}
for s in HL: print('Historical-last  %-4s MAE %.2f RMSE %.2f | per-horizon MAE %s' % (s, HL[s]['mae'], HL[s]['rmse'], np.round(HL[s]['mae_h'], 2)))

# %% [markdown]
# ## Q4(b) — Model definition (shared by all four runs)
# **Shapes.** `x: (B, 8, N, 1)` → temporal encoder → `Z: (B, N, d)` → graph conv `H = ReLU(Â Z W_g)` → head → `Ŷ: (B, 4, N)` (the size-1 feature axis F is squeezed; targets are stored as `(B, 4, N)` too).
# 1. **Temporal encoder** (`models.TemporalEncoder`): the 8-step series of every sensor is processed independently by two `Conv1d` layers (kernel 3, same padding, ReLU, 32 channels) followed by a linear read-out over (channels × time) → `d = 32` features per sensor. Implementation detail: `(B,8,N,1)` is reshaped to `(B·N, 1, 8)` so that the convolution acts along time for each sensor.
# 2. **Spatial graph convolution (exactly the assignment formula):** `H = ReLU( Â Z W_g )`, `W_g ∈ ℝ^{32×32}` (+bias). The product `Â Z` is `einsum('nm,bmd->bnd', Â, Z)`, i.e. a dense 716×716 matrix multiplied over the node axis separately for every batch element.
# 3. **Forecast head:** `Linear(32→64) → ReLU → Linear(64→4)` applied per sensor: the 4 outputs are the four 15-min horizons.
# **Loss:** masked **MAE** (L1), weighted so that it equals the raw-unit MAE divided by the global std. MAE is recommended for traffic: flow data contain outages/spikes and heavy tails; L1 is robust to them and directly optimises the reported MAE. Only entries whose *target* is an observed value (mask = 1) enter the loss.
# **Optimiser:** Adam, lr 1e-3, batch 64, gradient-norm clip 5, max 60 epochs, **early stopping on validation MAE** (patience 6), best-validation checkpoint restored for the final evaluation. Same for all four models.
# **Metric convention (all models):** MAE/RMSE are computed in original units (vehicles per 15 min) **only over target entries that are observed** (not NaN-missing, not part of a ≥3 h outage; see Q3(a)). Genuine zeros that are observed are kept. Overall = all valid (window, horizon, sensor) entries; per-horizon = valid entries of that horizon.
# **Temporal-only baseline (Q4(c)):** identical encoder and head; the graph layer is removed: `H = ReLU(Z)`.

# %%
A_raw = np.load(f'{PROC}/sd_adj.npy')
A_fixed = fixed_adjacency(A_raw, 'row')
print('fixed adjacency D_out^-1 (A+I): row sums in [%.6f, %.6f] | nnz %d | self-loop weight min/max %.3f/%.3f' % (
    A_fixed.sum(1).min(), A_fixed.sum(1).max(), int((A_fixed > 0).sum()), A_fixed.diag().min(), A_fixed.diag().max()))
def make(graph):
    set_seed(SEED)                                                  # identical initialisation of the shared layers
    return STModel(graph, N_NODES, d=CFG['hidden_d'], t_in=T_IN, horizon=T_OUT, A_fixed=A_fixed, d_e=CFG['d_e'], top_k=CFG['top_k'], head_hidden=CFG['head_hidden'])
for g in ('none', 'fixed', 'adaptive', 'hybrid'):
    print('%-9s trainable parameters: %d' % (g, count_params(make(g))))
# estimated epoch time BEFORE training (40 optimisation steps timed, extrapolated to a full epoch + validation pass)
est = {}
for dev in sorted({DEVICE, 'cpu'}):
    mdl = make('hybrid').to(dev); opt = torch.optim.Adam(mdl.parameters(), 1e-3); it = iter(data['loaders']['train']); t0 = time.perf_counter(); n = 0
    for _ in range(40):
        xb, yb, mb = next(it); opt.zero_grad(); ((mdl(xb.to(dev)) - yb.to(dev)).abs() * mb.to(dev)).mean().backward(); opt.step(); n += 1
    if dev == 'cuda': torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / n; est[dev] = per * len(data['loaders']['train']) * 1.25
    print('estimated epoch time on %-4s: %.0f s (train %.0f ms/step x %d steps, +25%% for validation)' % (dev, est[dev], per * 1000, len(data['loaders']['train'])))
print('=> worst case %d epochs x %.0f s = %.1f min per model on %s' % (CFG['max_epochs'], est[DEVICE], CFG['max_epochs'] * est[DEVICE] / 60, DEVICE))
Q4a['epoch_time_estimate_sec'] = est

# %% [markdown]
# ## Q5(b) preliminaries — the adaptive adjacency (my own implementation, no PyG)
# $A_{adp}=\mathrm{Softmax}_{row}\big(\mathrm{ReLU}(E_1E_2^\top)\big)$ with trainable $E_1,E_2\in\mathbb R^{716\times10}$ (random $\mathcal N(0,0.5^2)$ init), see `models.AdaptiveAdjacency`.
#
# **Top-k (k = 20) — before or after the normalisation?** Selection happens **before** the softmax: for every row we keep the 20 largest raw scores $S_{ij}=(E_1E_2^\top)_{ij}$, apply the ReLU to the kept scores, and take the softmax **only over those 20 entries** (all other entries are exactly 0). So each row is a proper probability distribution over its 20 selected neighbours (rows sum to 1), exactly the same normalisation as the row-normalised fixed graph. (The alternative "softmax over all 716 entries, then keep top-k and renormalise" would give nearly the same ranking but the pre-softmax masking is cheaper and makes the kept weights independent of the 696 discarded entries.)
#
# **Gradient flow.** Index selection (`topk`) is piecewise constant in $(E_1,E_2)$ and has no gradient, but the *values* gathered at the selected indices are ordinary differentiable functions of $E_1,E_2$ and pass through ReLU → softmax → `scatter` → `Â Z`. Hence $\partial\mathcal L/\partial S_{ij}\neq0$ **only for the 20 selected entries of each row**; the other 696 entries per row get exactly zero gradient at that step. They can still enter the top-k later, because updating $E_1,E_2$ through the selected entries changes every score $S_{ij}$. Verified numerically below.

# %%
set_seed(SEED)
adp = AdaptiveAdjacency(N_NODES, CFG['d_e'], CFG['top_k'])
S = adp.E1 @ adp.E2.t(); S.retain_grad()                                   # raw scores, keep their gradient for the check
vals, idx = S.topk(adp.top_k, dim=1); w = torch.softmax(torch.relu(vals), 1); Aad = torch.zeros_like(S).scatter(1, idx, w)
Zr = torch.randn(N_NODES, 8); loss = ((Aad @ Zr) ** 2).sum(); loss.backward()
sel = torch.zeros_like(S, dtype=torch.bool).scatter(1, idx, True)
print('nonzero entries per row of A_adp:', int((Aad > 0).sum(1).min()), '-', int((Aad > 0).sum(1).max()), '| row sums', float(Aad.sum(1).min()), float(Aad.sum(1).max()))
print('dL/dS nonzero entries: %d (selected entries: %d) | nonzero gradient outside selection: %d' % (int((S.grad != 0).sum()), int(sel.sum()), int(((S.grad != 0) & ~sel).sum())))
assert ((S.grad != 0) & ~sel).sum() == 0 and torch.allclose(Aad.sum(1), torch.ones(N_NODES))
print('E1/E2 gradients: finite =', bool(torch.isfinite(adp.E1.grad).all() and torch.isfinite(adp.E2.grad).all()), '| fraction of E1 rows with non-zero gradient: %.3f' % float((adp.E1.grad.abs().sum(1) > 0).float().mean()))
Q5b_check = dict(nonzero_per_row=int(adp.top_k), grad_outside_selection=0, selected_entries=int(sel.sum()), grad_nonzero=int((S.grad != 0).sum()))

# %% [markdown]
# ## Training — all four models, one identical protocol (seed 42)
# Order: temporal-only → STGNN-fixed → STGNN-adaptive → STGNN-hybrid. After each run the best-validation checkpoint is restored and evaluated on validation and test; the test windows are used **only for this final report**, never for any training or selection decision. Wall-clock training time = `time.perf_counter()` around the whole epoch loop (train + per-epoch validation, i.e. including the early-stopping protocol cost) on the device given above.

# %%
MODELS = [('temporal_only', 'none', 'None'), ('stgnn_fixed', 'fixed', 'Fixed'), ('stgnn_adaptive', 'adaptive', 'Adaptive'), ('stgnn_hybrid', 'hybrid', 'Hybrid')]
RESULTS = {}
def run_one(name, graph, seed=SEED, save_pred=True, self_path=False):
    cfg = dict(CFG); cfg['seed'] = seed
    d_ = build_data(batch_size=cfg['batch_size'], seed=seed, verbose=False)
    set_seed(seed); mdl = STModel(graph, N_NODES, d=cfg['hidden_d'], t_in=T_IN, horizon=T_OUT, A_fixed=A_fixed, d_e=cfg['d_e'], top_k=cfg['top_k'], head_hidden=cfg['head_hidden'], self_path=self_path)
    info = train_model(mdl, d_, name, cfg, device=DEVICE, verbose=True)
    out = dict(name=name, graph=graph, seed=seed, self_path=self_path, params=count_params(mdl), device=DEVICE, **{k: info[k] for k in ('epochs_run', 'best_epoch', 'train_time_sec', 'best_val_mae')},
               history=info['history'], cfg=cfg)
    P, Y, M = predict(mdl, d_['loaders']['test'], d_['mean'], d_['std'], DEVICE)
    out['test'] = metrics_from(P, Y, M); out['val'] = evaluate(mdl, d_['loaders']['val'], d_['mean'], d_['std'], DEVICE)
    if graph == 'hybrid': out['alpha'] = float(mdl.alpha)
    if graph in ('adaptive', 'hybrid'):
        with torch.no_grad(): np.save(f'{CKPT}/A_adaptive_{name}_seed{seed}.npy', mdl.adp().cpu().numpy().astype(np.float32))
    if graph != 'none':
        with torch.no_grad(): np.save(f'{CKPT}/A_used_{name}_seed{seed}.npy', mdl.adjacency().cpu().numpy().astype(np.float32))
    if save_pred: np.save(f'{CKPT}/pred_test_{name}_seed{seed}.npy', P.numpy().astype(np.float32))
    save_json(out, f'{RES}/train_{name}_seed{seed}.json')
    print(f"==> {name}: epochs {out['epochs_run']} (best {out['best_epoch']}) | time {out['train_time_sec']:.1f}s | VAL MAE {out['val']['mae']:.3f} RMSE {out['val']['rmse']:.3f} | "
          f"TEST MAE {out['test']['mae']:.3f} RMSE {out['test']['rmse']:.3f}" + (f" | alpha {out['alpha']:.4f}" if 'alpha' in out else ''))
    return out

# %% [markdown]
# ### Q4(b)/(c): temporal-only baseline and Q5(a) fixed graph

# %%
RESULTS['temporal_only'] = run_one('temporal_only', 'none')

# %%
RESULTS['stgnn_fixed'] = run_one('stgnn_fixed', 'fixed')

# %% [markdown]
# ### Q5(b): learnable (adaptive) adjacency

# %%
RESULTS['stgnn_adaptive'] = run_one('stgnn_adaptive', 'adaptive')

# %% [markdown]
# ### Q5(c): hybrid adjacency, $A_{hybrid}=\alpha\hat A_{fixed}+(1-\alpha)A_{adaptive}$ with **learnable** $\alpha=\sigma(\theta)$, $\theta_0=0$ ($\alpha_0=0.5$)
# Both operands are row-stochastic (fixed: $D_{out}^{-1}(A+I)$; adaptive: row-softmax), so they are consistently normalised before mixing and the mixture is row-stochastic too.

# %%
RESULTS['stgnn_hybrid'] = run_one('stgnn_hybrid', 'hybrid')

# %% [markdown]
# ## Extension (clearly separate from the main table): graph layer **with a self path**
# The assignment's layer $H=\mathrm{ReLU}(\hat A Z W_g)$ gives a sensor *only* the weighted average of its neighbourhood: with the row-normalised fixed graph the sensor's own features carry on average only 17 % of the input weight (median 14 %; computed from the matrix in the Q6 notebook). The main table above uses exactly this formula. To test whether the graph helps once a sensor keeps its own signal, the same three graphs are also run with $H=\mathrm{ReLU}(\hat A Z W_g + Z W_s)$ (a separate self weight $W_s\in\mathbb R^{32\times32}$, the "root weight" of PyG's `GraphConv`). Everything else (encoder, head, data, seed, optimiser, early stopping) is identical, the temporal-only model is the same run as in the main table.

# %%
RESULTS_EXT = {}
for name, graph in [('stgnn_fixed_skip', 'fixed'), ('stgnn_adaptive_skip', 'adaptive'), ('stgnn_hybrid_skip', 'hybrid')]:
    RESULTS_EXT[name] = run_one(name, graph, self_path=True)

# %% [markdown]
# ## Q4(b) / Q5 / Q6(a) — validation and test results, original flow units (vehicles per 15 min)

# %%
rows = []
for name, graph, adj in MODELS:
    r = RESULTS[name]
    rows.append(dict(Model='Temporal-only' if graph == 'none' else 'STGNN', Adjacency=adj, params=r['params'], epochs=r['epochs_run'], best_epoch=r['best_epoch'],
                     val_MAE=r['val']['mae'], val_RMSE=r['val']['rmse'], test_MAE=r['test']['mae'], test_RMSE=r['test']['rmse'], train_time_s=r['train_time_sec']))
tab = pd.DataFrame(rows); print(tab.round(3).to_string(index=False))
hz = []
for name, graph, adj in MODELS:
    for split in ('val', 'test'):
        r = RESULTS[name][split]
        for h in range(4): hz.append(dict(model=name, split=split, horizon_min=15 * (h + 1), MAE=r['mae_h'][h], RMSE=r['rmse_h'][h]))
hz = pd.DataFrame(hz); print(hz.pivot_table(index=['split', 'model'], columns='horizon_min', values=['MAE', 'RMSE']).round(2).to_string())
tab.to_csv(f'{RES}/main_table_seed42.csv', index=False); hz.to_csv(f'{RES}/per_horizon_seed42.csv', index=False)
ext_rows = [dict(Model='Temporal-only (same run)', Adjacency='None', params=RESULTS['temporal_only']['params'], epochs=RESULTS['temporal_only']['epochs_run'], test_MAE=RESULTS['temporal_only']['test']['mae'], test_RMSE=RESULTS['temporal_only']['test']['rmse'], val_MAE=RESULTS['temporal_only']['val']['mae'], train_time_s=RESULTS['temporal_only']['train_time_sec'])]
for name, r in RESULTS_EXT.items(): ext_rows.append(dict(Model='STGNN + self path', Adjacency=r['graph'].capitalize(), params=r['params'], epochs=r['epochs_run'], test_MAE=r['test']['mae'], test_RMSE=r['test']['rmse'], val_MAE=r['val']['mae'], train_time_s=r['train_time_sec']))
ext_tab = pd.DataFrame(ext_rows); print('\nEXTENSION (self path):'); print(ext_tab.round(3).to_string(index=False)); ext_tab.to_csv(f'{RES}/extension_table_seed42.csv', index=False)
save_json(dict(Q4a=Q4a, Q5b_check=Q5b_check, historical_last=HL, table=tab.to_dict('records'), alpha_learned=RESULTS['stgnn_hybrid'].get('alpha')), f'{RES}/q4_q5_summary.json')

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for name, graph, adj in MODELS:
    h = RESULTS[name]['history']; lab = ('Temporal-only' if graph == 'none' else f'STGNN-{adj}')
    ax[0].plot([r['epoch'] for r in h], [r['val_mae'] for r in h], marker='o', ms=3, label=lab)
    ax[1].plot([r['epoch'] for r in h], [r['train_loss_norm'] * data['global_std'] for r in h], marker='o', ms=3, label=lab)
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('validation MAE (veh / 15 min)'); ax[0].set_title('Validation MAE per epoch'); ax[0].legend()
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('training MAE (veh / 15 min, running avg.)'); ax[1].set_title('Training loss per epoch (de-normalised)'); ax[1].legend()
fig.suptitle('Q4/Q5: training curves, seed 42'); fig.savefig(f'{FIG}/q4q5_training_curves.png', dpi=160); plt.show()

# %% [markdown]
# ## Q5(b)-4 — Visualising the learned adjacency (adaptive model) next to the fixed road graph

# %%
from matplotlib.colors import LogNorm
Aad_l = np.load(f'{CKPT}/A_adaptive_stgnn_adaptive_seed42.npy'); Afx = A_fixed.numpy(); Ahy = np.load(f'{CKPT}/A_used_stgnn_hybrid_seed42.npy')
cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('white'); cmap.set_under('lightgray')
fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
for a_, M_, ttl in zip(ax, (Afx, Aad_l, Ahy), ('Fixed  D_out^-1 (A+I)', 'Learned adaptive  softmax(top-20 ReLU(E1 E2^T))', 'Hybrid  alpha*A_fixed + (1-alpha)*A_adaptive')):
    im = a_.imshow(np.where(M_ > 0, M_, np.nan), cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=1.0), interpolation='nearest')
    a_.set_title(f'{ttl}\n({int((M_ > 0).sum()):,} non-zero entries)', fontsize=10); a_.set_xlabel('sensor j (column)'); a_.set_ylabel('sensor i (row)'); a_.grid(False)
    fig.colorbar(im, ax=a_, label='weight (log scale; zero entries white)')
fig.suptitle('Q5(b)-4: fixed vs learned vs hybrid adjacency (sensors in meta order, seed 42)'); fig.savefig(f'{FIG}/q5b_adjacency_fixed_vs_learned.png', dpi=160); plt.show()
print('learned adaptive adjacency: nnz/row = %.1f, max weight %.3f, mean max-row weight %.3f, diagonal mass = %.3f' % ((Aad_l > 0).sum(1).mean(), Aad_l.max(), Aad_l.max(1).mean(), np.diag(Aad_l).mean()))
print('hybrid learned alpha = %.4f (initial 0.5)' % RESULTS['stgnn_hybrid']['alpha'])
