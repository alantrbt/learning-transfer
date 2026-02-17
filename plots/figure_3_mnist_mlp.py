"""Figure 3(a) — Cosine similarity of update directions for 2-MLP on MNIST.

Reproduces Figure 3(a) from arXiv:2305.14122v2.

Method (from Section 3.2 of the paper):
  - Train two 2-layer MLPs INDEPENDENTLY for 500 iterations each
    (SGD, lr=0.01, momentum=0.9, batch_size=128, no weight decay).
    Each model uses a different random seed for both initialisation
    AND data shuffling, as per the Git Re-Basin protocol (Ainsworth
    et al.): truly independent training runs.
  - Checkpoint every 10 iterations -> 51 snapshots, 50 deltas.
  - Compute update directions:  delta_t = theta^t - theta^{t-1}.
  - Solid lines: find optimal permutation pi by solving equation (4):
        min_pi  sum_t || pi delta_source_t - delta_target_t ||^2
    then plot cos_sim(pi delta_source_t, delta_target_t) at each t.
  - Dotted lines: pi = identity (no permutation), as baseline.
  - Repeat for hidden dimensions d in {8, 16, 32, 64, 128}.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
import torchvision.datasets

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.seed import set_random_seed
from utils._weight_matching import mlp_permutation_spec, apply_permutation
from utils.weight_matching import WeightMatching
from utils.subset_dataset import SubsetDataset, random_split

# ── constants (from paper & config.yaml) ─────────────────────────────
TOTAL_ITERS   = 500          # 500 SGD iterations (paper Section 3.2)
CKPT_INTERVAL = 10           # checkpoint every 10 iterations
BATCH_SIZE    = 128
LR            = 0.01
MOMENTUM      = 0.9           # config.yaml: sgd_momentum: 0.9
SEED_SOURCE   = 101
SEED_TARGET   = 102
HIDDEN_DIMS   = [8, 16, 32, 64, 128]
WIDTH_FACTORS = [0.125, 0.25, 0.5, 1.0, 2.0]     # default_width=64
DATASET_DIR   = os.path.join(os.path.dirname(__file__), '..', '__data__')


# ── simple 2-layer MLP (identical to models/networks/mlp.py) ────────
class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(784, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, 10, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


# ── helpers ──────────────────────────────────────────────────────────
def _get_params(model):
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters()}


def _cosine_similarity(d1, d2):
    """Cosine similarity between two flattened parameter dicts."""
    v1 = torch.cat([d1[k].flatten() for k in sorted(d1)])
    v2 = torch.cat([d2[k].flatten() for k in sorted(d2)])
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def _compute_deltas(checkpoints):
    """Δ_t = θ^t − θ^{t−1} for t = 1 … T."""
    return [
        {k: checkpoints[t][k] - checkpoints[t - 1][k] for k in checkpoints[t]}
        for t in range(1, len(checkpoints))
    ]


def _get_mnist_trainset():
    """MNIST training set (same train/val split as repo: seed 777, 90/10)."""
    from torchvision.transforms import ToTensor
    full = torchvision.datasets.MNIST(DATASET_DIR, train=True,
                                      download=True, transform=None)
    size = len(full)
    val_size = int(size * 0.1)
    train_size = size - val_size
    gen = torch.Generator(); gen.manual_seed(777)
    train_sub, _, _ = random_split(full, [train_size, val_size, 0],
                                   generator=gen)
    return SubsetDataset(train_sub, transform=ToTensor())


def _train_model(hidden_dim, seed, dataset):
    """Train a single SimpleMLP for TOTAL_ITERS SGD steps independently.

    The seed controls BOTH model initialisation AND data shuffling,
    ensuring truly independent training runs (Git Re-Basin protocol).

    Returns checkpoints list of length 51 (theta^0 … theta^50).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    set_random_seed(seed)
    model = SimpleMLP(hidden_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    checkpoints = [_get_params(model)]          # theta^0

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, drop_last=False)
    data_iter = iter(dataloader)

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

        if it % CKPT_INTERVAL == 0:
            checkpoints.append(_get_params(model))

    return checkpoints


def _find_permutation_eq4(ps, source_deltas, target_deltas, device):
    """Solve equation (4): min_π Σ_t ‖πΔ_src_t − Δ_tgt_t‖²."""
    wm = WeightMatching(ps, epsilon=1e-7, device=device)
    for sd, td in zip(source_deltas, target_deltas):
        sd_d = {k: v.to(device) for k, v in sd.items()}
        td_d = {k: v.to(device) for k, v in td.items()}
        wm.add(td_d, sd_d)          # add(target, source)
    return wm.solve(silent=True)


# ── main ─────────────────────────────────────────────────────────────
def figure_3_mnist_mlp(cfg=None, outman=None, prefix='', gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}'
                          if torch.cuda.is_available() else 'cpu')
    ps = mlp_permutation_spec(bn_affine=False)

    print("Loading MNIST …")
    dataset = _get_mnist_trainset()
    print(f"  Training set size: {len(dataset)}")

    n_steps = TOTAL_ITERS // CKPT_INTERVAL            # 50
    iterations = [(t + 1) * CKPT_INTERVAL for t in range(n_steps)]

    # colours – match original paper palette
    DIM_COLORS = {
        8:   "#862C8C",   # dark violet
        16:  "#D92537",   # steel blue
        32:  "#008013",   # teal
        64:  "#F7A354",   # medium green
        128: "#0098DF",   # lime / light green-yellow
    }

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for idx, (wf, hdim) in enumerate(zip(WIDTH_FACTORS, HIDDEN_DIMS)):
        print(f"\n── d = {hdim} ──")

        # 1. independent training (different init + different shuffle)
        print("  source (seed 101) …")
        ckpts_src = _train_model(hdim, SEED_SOURCE, dataset)
        print("  target (seed 102) …")
        ckpts_tgt = _train_model(hdim, SEED_TARGET, dataset)

        # 2. update directions
        deltas_src = _compute_deltas(ckpts_src)
        deltas_tgt = _compute_deltas(ckpts_tgt)

        # 3. optimal permutation (eq 4)
        print("  weight matching …")
        perm = _find_permutation_eq4(ps, deltas_src, deltas_tgt, device)

        cos_perm_raw, cos_id_raw = [], []
        for t in range(n_steps):
            sd = {k: v.to(device) for k, v in deltas_src[t].items()}
            td = {k: v.to(device) for k, v in deltas_tgt[t].items()}
            sd_p = apply_permutation(ps, perm, sd, device=device)
            cos_perm_raw.append(_cosine_similarity(sd_p, td))
            cos_id_raw.append(_cosine_similarity(sd, td))

        # Each point = instantaneous cosine similarity for that 10-iter block
        # "averaged over timesteps t" refers to the 10-step aggregation
        # within each checkpoint interval, NOT a cumulative mean.
        cos_perm = np.array(cos_perm_raw)
        cos_id   = np.array(cos_id_raw)

        c = DIM_COLORS[hdim]
        ax.plot(iterations, cos_perm, color=c, linestyle='-',
                linewidth=1.0, marker='.', markersize=2,
                label=f'd={hdim}')
        ax.plot(iterations, cos_id, color=c, linestyle=':',
                linewidth=0.8, marker='.', markersize=1.5)

        print(f"  permuted  mean={np.mean(cos_perm):.4f}  std={np.std(cos_perm):.4f}")
        print(f"  identity  mean={np.mean(cos_id):.4f}  std={np.std(cos_id):.4f}")

    # formatting
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Cosine Similarity', fontsize=11)
    ax.set_xlim([0, TOTAL_ITERS + 5])
    ax.set_ylim([-0.2, 0.8])
    ax.grid(True, linewidth=0.3, alpha=0.5)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='gray', ls='-',  lw=1.2,
                          label='permuted'))
    handles.append(Line2D([0], [0], color='gray', ls=':',  lw=1.0,
                          label='identity'))
    ax.legend(handles=handles, fontsize=8, loc='upper right',
              framealpha=0.8)
    ax.set_title('(a) 2-MLP on MNIST', fontsize=11)

    # save
    if outman is not None:
        out_path = outman.get_abspath(prefix='plot.manual', ext='pdf',
                                      name='figure_3a_mnist_mlp')
    else:
        out_dir = os.path.join(os.path.dirname(__file__), '..',
                               '__outputs__', 'figure_3_mnist_mlp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'figure_3a_mnist_mlp.pdf')

    fig.savefig(out_path, format='pdf', bbox_inches='tight', dpi=150)
    png_path = out_path.replace('.pdf', '.png')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    figure_3_mnist_mlp()

