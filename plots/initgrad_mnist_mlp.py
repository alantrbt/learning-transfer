# plots/initgrad_mnist_mlp.py

import os
import copy
import math
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from utils.seed import set_random_seed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def _call_with_supported_kwargs(fn, **kwargs):
    """Call fn with only the kwargs it supports."""
    sig = inspect.signature(fn)
    kept = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**kept)

def _unwrap_learner(ret):
    """
    get_learner may return a tuple (learner, ...). We want the object that has .model.
    """
    if hasattr(ret, "model"):
        return ret
    if isinstance(ret, (list, tuple)):
        for x in ret:
            if hasattr(x, "model"):
                return x
    raise TypeError(f"Could not find a learner-like object with .model in return value of get_learner: {type(ret)}")


def _get_out_dir(cfg, outman, prefix):
    # Try a few common OutputManager APIs; fallback to cfg["output_dir"].
    for name in ["get_dir", "get_output_dir", "get_exp_dir", "get_prefix_dir"]:
        if hasattr(outman, name) and callable(getattr(outman, name)):
            try:
                return getattr(outman, name)(prefix)
            except TypeError:
                try:
                    return getattr(outman, name)()
                except Exception:
                    pass
    return cfg.get("output_dir", "__outputs__")

def _flatten_grads(grads_dict):
    vecs = []
    for k in sorted(grads_dict.keys()):
        g = grads_dict[k]
        if g is None:
            continue
        vecs.append(g.detach().reshape(-1).cpu())
    if not vecs:
        return torch.zeros(1)
    return torch.cat(vecs, dim=0)

def _cosine(a, b, eps=1e-12):
    a = a.float()
    b = b.float()
    return float((a @ b) / (a.norm() * b.norm() + eps))

def _infer_mlp_keys_from_state_dict(sd):
    """
    Heuristic: pick the two linear weight matrices by shape.
    For 2-layer MLP: W1 is (H, D), W2 is (C, H).
    """
    weight_keys = [k for k in sd.keys() if k.endswith("weight")]
    # collect (key, shape)
    items = [(k, tuple(sd[k].shape)) for k in weight_keys]
    # pick candidates where len(shape)==2
    items = [(k, s) for k, s in items if len(s) == 2]
    if len(items) < 2:
        raise RuntimeError(f"Could not find two linear weights in state_dict keys: {list(sd.keys())[:20]} ...")
    # sort by first dim descending (hidden usually largest)
    items_sorted = sorted(items, key=lambda x: x[1][0], reverse=True)

    # try every pair to satisfy W2 second dim equals W1 first dim
    for k1, s1 in items_sorted:
        for k2, s2 in items_sorted:
            if k1 == k2:
                continue
            H = s1[0]
            if s2[1] == H:  # (C, H)
                w1 = k1
                w2 = k2
                # biases: replace 'weight' by 'bias' if exists
                b1 = w1[:-6] + "bias"
                b2 = w2[:-6] + "bias"
                if b1 not in sd: b1 = None
                if b2 not in sd: b2 = None
                return w1, b1, w2, b2
    raise RuntimeError("Could not infer (W1,W2) pair for 2-layer MLP from state_dict shapes.")

def _hungarian_perm_from_weights(sd1, sd2):
    """
    Build permutation of hidden units to align sd1 to sd2.
    We use a feature vector per hidden unit that includes:
      - incoming weights + bias (row of W1, b1)
      - outgoing weights (column of W2)
    Then solve assignment maximizing dot-product similarity.
    """
    w1k, b1k, w2k, b2k = _infer_mlp_keys_from_state_dict(sd1)

    W1a = sd1[w1k].detach().cpu()  # (H, D)
    W1b = sd2[w1k].detach().cpu()
    H, D = W1a.shape

    b1a = sd1[b1k].detach().cpu() if b1k is not None else torch.zeros(H)
    b1b = sd2[b1k].detach().cpu() if b1k is not None else torch.zeros(H)

    W2a = sd1[w2k].detach().cpu()  # (C, H)
    W2b = sd2[w2k].detach().cpu()

    # features per hidden unit: [W1_row, b1, W2_col]
    Fa = torch.cat([W1a, b1a[:, None], W2a.t()], dim=1).numpy()  # (H, D+1+C)
    Fb = torch.cat([W1b, b1b[:, None], W2b.t()], dim=1).numpy()

    # similarity matrix S_ij = <Fa_i, Fb_j>
    S = Fa @ Fb.T
    # assignment wants min cost => use -S
    row_ind, col_ind = linear_sum_assignment(-S)
    # permutation p such that unit i in A maps to unit p[i] in B
    p = np.empty(H, dtype=np.int64)
    p[row_ind] = col_ind
    return p, (w1k, b1k, w2k, b2k)

def _apply_perm_to_grads(grads, p, keys):
    """
    Apply hidden permutation to grads of a 2-layer MLP:
      - grad W1: permute rows
      - grad b1: permute entries
      - grad W2: permute columns (because W2 is (C, H))
    """
    w1k, b1k, w2k, b2k = keys
    p_t = torch.as_tensor(p, dtype=torch.long, device=grads[w1k].device)

    g = dict(grads)  # shallow copy
    g[w1k] = g[w1k].index_select(0, p_t)  # rows
    if b1k is not None and b1k in g and g[b1k] is not None:
        g[b1k] = g[b1k].index_select(0, p_t)
    g[w2k] = g[w2k].index_select(1, p_t)  # cols
    # b2 unaffected
    return g

def _get_any_train_loader(learner, cfg):
    # 1) Try common attribute names
    candidates = [
        "train_loader", "train_dataloader", "train_dl", "trainloader",
        "train_data_loader", "dataloader_train", "loader_train"
    ]
    for name in candidates:
        if hasattr(learner, name):
            dl = getattr(learner, name)
            if dl is not None:
                return dl

    # 2) Try dict containers
    for name in ["loaders", "dataloaders", "data_loaders", "dl"]:
        if hasattr(learner, name):
            obj = getattr(learner, name)
            if isinstance(obj, dict):
                for k in ["train", "training", "tr"]:
                    if k in obj and obj[k] is not None:
                        return obj[k]

    # 3) Fallback: build DataLoader from train_dataset (present in this repo)
    if hasattr(learner, "train_dataset") and learner.train_dataset is not None:
        from torch.utils.data import DataLoader

        bs = int(cfg.get("batch_size", 128))
        num_workers = int(cfg.get("num_workers", 0))
        # On CPU, pin_memory not useful; keep False
        return DataLoader(
            learner.train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
        )

    raise AttributeError(
        f"Could not find a train dataloader on learner (type={type(learner)}). "
        f"Available attrs: {sorted([a for a in dir(learner) if 'train' in a.lower() or 'loader' in a.lower() or 'data' in a.lower()])}"
    )

def _get_loss_fn(learner):
    for name in ["criterion", "loss_fn", "loss", "get_loss"]:
        if hasattr(learner, name):
            obj = getattr(learner, name)
            if callable(obj) and name == "get_loss":
                return obj
            if obj is not None and not callable(obj):
                return obj
    return None

def _compute_grads_on_one_batch(learner, device, cfg):
    model = learner.model
    model.train()

    dl = _get_any_train_loader(learner, cfg)
    batch = next(iter(dl))
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    elif isinstance(batch, dict) and ("x" in batch and "y" in batch):
        x, y = batch["x"], batch["y"]
    else:
        raise RuntimeError(f"Unexpected batch format: {type(batch)}")

    x = x.to(device)
    y = y.to(device)

    model.zero_grad(set_to_none=True)

    out = model(x)

    loss_fn = _get_loss_fn(learner)
    if callable(loss_fn) and loss_fn.__name__ == "get_loss":
        loss = loss_fn(out, y)
    elif loss_fn is None:
        loss = torch.nn.CrossEntropyLoss()(out, y)
    else:
        loss = loss_fn(out, y)

    loss.backward()

    grads = {}
    for name, p in model.named_parameters():
        grads[name] = None if p.grad is None else p.grad.detach().clone()
    return grads, float(loss.detach().cpu())


def initgrad_mnist_mlp(cfg, outman, prefix, gpu_id):
    """
    Assumption (P) sanity check at initialization:
    cosine similarity between gradients of two independently initialized models,
    with and without a hidden-unit permutation (2-layer MLP case).
    """
    # CPU-friendly: force cuda off unless explicitly available and requested
    use_cuda = bool(cfg.get("use_cuda", False)) and torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")

    # We rely on the repo helper that constructs the learner (dataset, model, loader, etc.)
    from utils.learning_transfer import get_learner

    # seeds pair (from cfg, produced by exec_parallel grid)
    seeds = cfg.get("source_target_seeds", None)
    if seeds is None:
        raise RuntimeError("cfg['source_target_seeds'] is required for initgrad_mnist_mlp (e.g., [101,102]).")
    if len(seeds) != 2:
        raise RuntimeError(f"Expected two seeds in source_target_seeds, got: {seeds}")

    cfg1 = copy.deepcopy(cfg); cfg1["seed"] = seeds[0]
    cfg2 = copy.deepcopy(cfg); cfg2["seed"] = seeds[1]

    cfg1["epoch"] = 0
    cfg2["epoch"] = 0


    # Build learners at init (epoch_for_swap = -1 is typical; here we just want initial weights)
    # Use skip_train if supported; otherwise it will just construct and return.
    # get_learner signature in this repo: get_learner(exp_name, hparams, outman, ...)
    # We pass the same outman so outputs are consistent.
    hparams1 = cfg1
    hparams2 = cfg2
    
    set_random_seed(seeds[0])
    ret1 = _call_with_supported_kwargs(
        get_learner,
        exp_name="initgrad_mnist_mlp",
        hparams=hparams1,
        outman=outman,
        cfg=cfg1,
        gpu_id=gpu_id,
    )


    set_random_seed(seeds[1])
    ret2 = _call_with_supported_kwargs(
        get_learner,
        exp_name="initgrad_mnist_mlp",
        hparams=hparams2,
        outman=outman,
        cfg=cfg2,
        gpu_id=gpu_id,
    )

    learner1 = _unwrap_learner(ret1)
    learner2 = _unwrap_learner(ret2)

    learner1.model.to(device)
    learner2.model.to(device)

    # Compute grads on one batch (you can average over several batches later)
    grads1, loss1 = _compute_grads_on_one_batch(learner1, device, cfg1)
    grads2, loss2 = _compute_grads_on_one_batch(learner2, device, cfg2)


    # Build permutation from current weights (init)
    sd1 = learner1.model.state_dict()
    sd2 = learner2.model.state_dict()
    p, keys = _hungarian_perm_from_weights(sd1, sd2)

    # Baseline cosine (identity)
    v1 = _flatten_grads(grads1)
    v2 = _flatten_grads(grads2)
    cos_id = _cosine(v1, v2)

    # Permuted cosine
    grads1p = _apply_perm_to_grads(grads1, p, keys)
    v1p = _flatten_grads(grads1p)
    cos_perm = _cosine(v1p, v2)

    # Output
    out_dir = _get_out_dir(cfg, outman, prefix)
    os.makedirs(out_dir, exist_ok=True)

    # save csv
    csv_path = os.path.join(out_dir, f"{prefix}initgrad_mnist_mlp.csv")
    with open(csv_path, "w") as f:
        f.write("loss_seed1,loss_seed2,cos_identity,cos_permuted\n")
        f.write(f"{loss1},{loss2},{cos_id},{cos_perm}\n")

    # plot
    fig_path = os.path.join(out_dir, f"{prefix}initgrad_mnist_mlp.png")
    plt.figure()
    plt.bar(["identity", "permuted"], [cos_id, cos_perm])
    plt.ylim(0.0, 1.0)
    plt.title("MNIST 2-MLP: cosine similarity of init gradients")
    plt.ylabel("cosine similarity")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[initgrad_mnist_mlp] saved: {fig_path}")
    print(f"[initgrad_mnist_mlp] saved: {csv_path}")
