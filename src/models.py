"""Q4/Q5 models: temporal Conv1d encoder + (optional) graph convolution + forecast head.

Shapes (B = batch, T = 8 input steps, N = 716 nodes, d = hidden dim, H = 4 horizons):
    x            (B, T, N, 1)
    encoder  ->  Z (B, N, d)                    Conv1d is applied on a (B*N, C, T) tensor, i.e. per sensor
    graph    ->  H = ReLU( A_hat @ Z @ W_g )    A_hat (N, N) multiplies the node axis, for every batch item
    head     ->  (B, N, H) -> permuted to Y_hat (B, H, N)   (the trailing feature axis F=1 is squeezed)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- adjacency helpers
def fixed_adjacency(A, mode='row'):
    """Fixed normalised adjacency from the raw LargeST matrix (diagonal self-weight removed first).

    mode='row' : D_out^{-1}(A + I)        (row-stochastic; correct choice for a DIRECTED graph)
    mode='sym' : D^{-1/2}(A + I)D^{-1/2}  (Kipf & Welling; only meaningful if A is symmetric)
    """
    A = np.array(A, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    At = A + np.eye(A.shape[0])
    if mode == 'row':
        out = At / At.sum(1, keepdims=True)
    elif mode == 'sym':
        d = At.sum(1)
        out = At / np.sqrt(d)[:, None] / np.sqrt(d)[None, :]
    else:
        raise ValueError(mode)
    return torch.tensor(out, dtype=torch.float32)


class AdaptiveAdjacency(nn.Module):
    """Learned graph  A_adp = Softmax_row( ReLU(E1 E2^T) )   -- own implementation (no PyG).

    * E1, E2 in R^{N x d_e} are trainable, random N(0, init_std^2) init, updated by back-prop only
      from the TRAINING loss (nothing about val/test touches them).
    * top_k=None  -> dense row-softmax over all N entries.
    * top_k=k     -> for every row keep the k largest raw scores S_ij=(E1E2^T)_ij, apply ReLU to the kept
      scores and take the softmax ONLY over the kept entries (so each row sums to 1 and the other N-k
      entries are exactly 0).  i.e. top-k selection comes BEFORE the softmax normalisation (we mask
      the logits, we do not renormalise afterwards).  Gradients: the index set is a piecewise-constant
      (non-differentiable) function of the parameters, but the VALUES gathered at those indices are
      differentiable, so dL/dE flows only through the selected k entries of each row; the N-k masked
      entries receive exactly zero gradient at this step.  (They can re-enter the top-k in later steps
      because the selected entries' gradients change E1, E2 and hence every score S_ij.)
    """

    def __init__(self, n_nodes, d_e=10, top_k=20, init_std=0.5):
        super().__init__()
        self.N, self.top_k = n_nodes, top_k
        self.E1 = nn.Parameter(torch.randn(n_nodes, d_e) * init_std)
        self.E2 = nn.Parameter(torch.randn(n_nodes, d_e) * init_std)

    def forward(self):
        S = self.E1 @ self.E2.t()                                   # (N, N) raw scores
        if self.top_k is None or self.top_k >= self.N:
            return F.softmax(F.relu(S), dim=1)
        vals, idx = S.topk(self.top_k, dim=1)                       # selection (non-differentiable indices)
        w = F.softmax(F.relu(vals), dim=1)                          # normalise over the kept entries only
        return torch.zeros_like(S).scatter(1, idx, w)               # scatter is differentiable w.r.t. w


# ----------------------------------------------------------------------------- model
class TemporalEncoder(nn.Module):
    """Two temporal Conv1d layers (kernel 3, 'same' padding) over the 8 time steps of every sensor,
    followed by a linear read-out over (channels x time) -> d features per sensor."""

    def __init__(self, d=32, t_in=8, k=3):
        super().__init__()
        self.c1 = nn.Conv1d(1, d, k, padding=k // 2)
        self.c2 = nn.Conv1d(d, d, k, padding=k // 2)
        self.proj = nn.Linear(d * t_in, d)

    def forward(self, x):                                           # x: (B, T, N, 1)
        B, T, N, _ = x.shape
        z = x.squeeze(-1).permute(0, 2, 1).reshape(B * N, 1, T)     # (B*N, 1, T): one series per sensor
        z = F.relu(self.c1(z))
        z = F.relu(self.c2(z))
        z = self.proj(z.flatten(1))                                 # (B*N, d)
        return z.view(B, N, -1)                                     # Z: (B, N, d)


class STModel(nn.Module):
    """graph in {'none','fixed','adaptive','hybrid'} -- identical encoder / head / W_g for all.
    self_path=False : H = ReLU(A_hat Z W_g)                (exactly the assignment's formula; used for the main table)
    self_path=True  : H = ReLU(A_hat Z W_g + Z W_s)        (documented extension: separate self weight, as PyG GraphConv)"""

    def __init__(self, graph='none', n_nodes=716, d=32, t_in=8, horizon=4, A_fixed=None, d_e=10, top_k=20,
                 head_hidden=64, self_path=False):
        super().__init__()
        assert graph in ('none', 'fixed', 'adaptive', 'hybrid')
        self.graph, self.self_path = graph, self_path
        if graph != 'none' and self_path:                           # EXTENSION (not the assignment formula): root/self weight
            self.Ws = nn.Linear(d, d, bias=False)
        self.enc = TemporalEncoder(d, t_in)
        if graph != 'none':
            self.Wg = nn.Linear(d, d)                               # W_g (+ bias)
        if graph in ('fixed', 'hybrid'):
            assert A_fixed is not None
            self.register_buffer('A_fixed', A_fixed.clone())        # buffer, NOT a parameter -> frozen
        if graph in ('adaptive', 'hybrid'):
            self.adp = AdaptiveAdjacency(n_nodes, d_e, top_k)
        if graph == 'hybrid':
            self.theta = nn.Parameter(torch.zeros(()))              # alpha = sigmoid(theta), init 0.5, learned
        self.head = nn.Sequential(nn.Linear(d, head_hidden), nn.ReLU(), nn.Linear(head_hidden, horizon))

    @property
    def alpha(self):
        return torch.sigmoid(self.theta) if self.graph == 'hybrid' else None

    def adjacency(self):
        if self.graph == 'fixed':
            return self.A_fixed
        if self.graph == 'adaptive':
            return self.adp()
        if self.graph == 'hybrid':       # both are row-stochastic (rows sum to 1) -> consistently normalised
            a = self.alpha
            return a * self.A_fixed + (1 - a) * self.adp()
        return None

    def forward(self, x):                                           # x: (B, T, N, 1) -> (B, H, N)
        Z = self.enc(x)                                             # (B, N, d)
        if self.graph == 'none':
            H = F.relu(Z)
        else:
            A = self.adjacency()                                    # (N, N)
            agg = self.Wg(torch.einsum('nm,bmd->bnd', A, Z))        # A_hat Z W_g, node axis, per batch item
            H = F.relu(agg + self.Ws(Z)) if self.self_path else F.relu(agg)   # literal spec: ReLU(A Z Wg); extension adds Z W_s
        return self.head(H).permute(0, 2, 1)                        # (B, N, H) -> (B, H, N)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
