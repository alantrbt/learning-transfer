"""
SCRIPT: Visualisation des gradients initiaux (Assumption P - alignment des gradients)

Phases principales:
1. Charger les états initiaux (epoch-1) de deux modèles avec seeds différents
2. Extraire les gradients des deux modèles
3. Matcher les neurones cachés entre les deux modèles via Hungarian algorithm
4. Calculer la similarité cosinus des gradients (bruts vs permutés)
5. Tracer et afficher les résultats
"""

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

# ===================================================================
# PHASE 1: FONCTIONS UTILITAIRES
# ===================================================================


# --------------------------
# Helpers pour appels de fonction
# --------------------------

def _call_with_supported_kwargs(fn, **kwargs):
    """Appelle fn avec uniquement les kwargs qu'elle supporte"""
    sig = inspect.signature(fn)
    kept = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**kept)

def _unwrap_learner(ret):
    """Extrait le learner de la valeur retournée par get_learner"""
    if hasattr(ret, "model"):
        return ret
    if isinstance(ret, (list, tuple)):
        for x in ret:
            if hasattr(x, "model"):
                return x
    raise TypeError(f"Could not find a learner-like object with .model in return value of get_learner: {type(ret)}")

# --------------------------
# Helpers pour accès au répertoire
# --------------------------

def _get_out_dir(cfg, outman, prefix):
    """Récupère le répertoire de sortie de l'OutputManager"""
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

# --------------------------
# Extractions et calculs mathématiques
# --------------------------

def _flatten_grads(grads_dict):
    """Applatit un dictionnaire de gradients en vecteur unique"""
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
    """Calcule la similarité cosinus entre deux vecteurs"""
    a = a.float()
    b = b.float()
    return float((a @ b) / (a.norm() * b.norm() + eps))

# --------------------------
# Inférence de l'architecture MLP
# --------------------------

def _infer_mlp_keys_from_state_dict(sd):
    """Identifie W1 (H,D) et W2 (C,H) dans le state_dict du MLP
    
    Heuristique: cherche deux matrices de poids 2D où W2.shape[1] == W1.shape[0]
    """
    weight_keys = [k for k in sd.keys() if k.endswith("weight")]
    # Collecter les matrices 2D par (key, shape)
    items = [(k, tuple(sd[k].shape)) for k in weight_keys]
    # Filtrer pour garder seulement les matrices 2D
    items = [(k, s) for k, s in items if len(s) == 2]
    if len(items) < 2:
        raise RuntimeError(f"Could not find two linear weights in state_dict keys: {list(sd.keys())[:20]} ...")
    # Trier par première dimension décroissante (couche cachée usuellement plus grande)
    items_sorted = sorted(items, key=lambda x: x[1][0], reverse=True)

    # Chercher la paire (W1, W2) satisfaisant W2.shape[1] == W1.shape[0]
    for k1, s1 in items_sorted:
        for k2, s2 in items_sorted:
            if k1 == k2:
                continue
            H = s1[0]
            if s2[1] == H:  # (C, H)
                w1 = k1
                w2 = k2
                # Trouver les biais associés si ils existent
                b1 = w1[:-6] + "bias"
                b2 = w2[:-6] + "bias"
                if b1 not in sd: b1 = None
                if b2 not in sd: b2 = None
                return w1, b1, w2, b2
    raise RuntimeError("Could not infer (W1,W2) pair for 2-layer MLP from state_dict shapes.")

# --------------------------
# Matching Hungarian pour poids initiaux
# --------------------------

def _hungarian_perm_from_weights(sd1, sd2):
    """PHASE 2: Construit la permutation des neurones cachés
    
    Aligne sd1 vers sd2 en maximisant la similarité des features des neurones:
      - Features = [W1_row, b1, W2_col]
    Résout le problème d'assignement via Hungarian algorithm.
    """
    w1k, b1k, w2k, b2k = _infer_mlp_keys_from_state_dict(sd1)

    W1a = sd1[w1k].detach().cpu()  # (H, D)
    W1b = sd2[w1k].detach().cpu()
    H, D = W1a.shape

    b1a = sd1[b1k].detach().cpu() if b1k is not None else torch.zeros(H)
    b1b = sd2[b1k].detach().cpu() if b1k is not None else torch.zeros(H)

    W2a = sd1[w2k].detach().cpu()  # (C, H)
    W2b = sd2[w2k].detach().cpu()

    # Construire les features de chaque neurone: [W1_row, b1, W2_col]
    Fa = torch.cat([W1a, b1a[:, None], W2a.t()], dim=1).numpy()  # (H, D+1+C)
    Fb = torch.cat([W1b, b1b[:, None], W2b.t()], dim=1).numpy()

    # Matrice de similarité S_ij = <Fa_i, Fb_j>
    S = Fa @ Fb.T
    # Hungarian veut min cost => utiliser -S
    row_ind, col_ind = linear_sum_assignment(-S)
    # permutation p telle que neurone i dans A mapé vers neurone p[i] dans B
    p = np.empty(H, dtype=np.int64)
    p[row_ind] = col_ind
    return p, (w1k, b1k, w2k, b2k)

def _apply_perm_to_grads(grads, p, keys):
    """Applique la permutation π aux gradients d'un MLP 2-couches
    
    Permutation:
      - grad W1: permute les LIGNES
      - grad b1: permute les entrées
      - grad W2: permute les COLONNES (car W2 est (C, H))
      - grad b2: inchangé
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
    """Récupère la fonction de loss du learner"""
    for name in ["criterion", "loss_fn", "loss", "get_loss"]:
        if hasattr(learner, name):
            obj = getattr(learner, name)
            if callable(obj) and name == "get_loss":
                return obj
            if obj is not None and not callable(obj):
                return obj
    return None

def _compute_grads_on_one_batch(learner, device, cfg):
    """Calcule les gradients sur un batch d'entraînement
    
    Utilisé pour obtenir les gradients initiaux à epoch-1
    """
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

# ===================================================================
# PHASE 3: FONCTION PRINCIPALE
# ===================================================================

def initgrad_mnist_mlp(cfg, outman, prefix, gpu_id):
    """Test d'alignement des gradients initiaux (Assumption P)
    
    Calcule la similarité cosinus entre les gradients de deux modèles
    initialisés indépendamment, avec et sans permutation des neurones cachés
    (cas MLP 2-couches).
    
    Étapes:
    1. Créer deux modèles avec seeds différents (epoch-1)
    2. Calculer les gradients sur un batch
    3. Matcher les neurones et appliquer la permutation
    4. Calculer et tracer la similarité cosinus
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
