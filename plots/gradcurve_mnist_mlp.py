"""
SCRIPT: Courbes d'alignement des gradients (Figure 3a du papier)

Phases principales:
1. Charger checkpoints des 2 modèles (A, B) à différentes itérations
2. Calculer les deltas (changements de poids): ΔW = W(t) - W(t-1)
3. Matcher neurones cachés entre A et B via Hungarian algorithm
4. Calculer similarité cosinus des deltas (bruts vs permutés)
5. Tracer et afficher les courbes
"""

import matplotlib
matplotlib.use("Agg")

import copy
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from utils.learning_transfer import get_learner
from utils.seed import set_random_seed
from utils.output_manager import OutputManager

print("[gradcurve] SCRIPT PATH =", os.path.abspath(__file__))
print("[gradcurve] USING FILE =", __file__)

# ===================================================================
# PHASE 1: FONCTIONS UTILITAIRES MATHÉMATIQUES ET DE MATCHING
# ===================================================================

# --------------------------
# Calculs mathématiques
# --------------------------
def cos(u, v, eps=1e-12):
    """Calcule la similarité cosinus entre deux vecteurs u et v"""
    u = u.reshape(-1)
    v = v.reshape(-1)
    denom = (u.norm() * v.norm()).clamp_min(eps)
    return (u @ v / denom).item()

def moving_avg(arr, k=5):
    """Moyenne mobile: lisse les courbes pour visualiser la tendance"""
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0 or k <= 1:
        return arr
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        a = max(0, i - k + 1)
        out[i] = arr[a:i + 1].mean()
    return out

def flatten_deltas(model_t, model_prev):
    """Extrait les changements de poids: ΔW = W(t) - W(t-1) en vecteur aplati"""
    deltas = []
    for p_t, p_prev in zip(model_t.parameters(), model_prev.parameters()):
        deltas.append((p_t.detach() - p_prev.detach()).reshape(-1))
    return torch.cat(deltas)

# --------------------------
# Appariement des neurones via Hungarian algorithm
# --------------------------
def match_hidden_layer_on_delta(modelA_t, modelA_prev, modelB_t, modelB_prev):
    """PHASE 2: Trouver la permutation optimale des neurones cachés
    
    Résout le problème d'assignement en maximisant la similarité des ΔW1
    (poids d'entrée de la couche cachée).
    
    Retourne: permutation π telle que neurone A[i] ↔ neurone B[π[i]]
    """
    dW1_A = (modelA_t.linear1.weight.detach() - modelA_prev.linear1.weight.detach())  # (H, D)
    dW1_B = (modelB_t.linear1.weight.detach() - modelB_prev.linear1.weight.detach())  # (H, D)

    sim = torch.matmul(dW1_A, dW1_B.T)  # (H, H)
    cost = (-sim).cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    return col_ind

def permute_deltas_for_mlp(deltaA_flat, modelA_t, perm):
    """Applique la permutation π au vecteur delta de A
    
    Pour aligner les neurones de A avec ceux de B:
      - Permute les LIGNES de ΔW1 (poids entrée → caché)
      - Permute les COLONNES de ΔW2 (caché → sortie)
      - Laisse les biais inchangés
    
    Cela rend les vecteurs delta comparables après alignement.
    """
    H, D = modelA_t.linear1.weight.shape
    C, H2 = modelA_t.linear2.weight.shape
    assert H == H2

    has_b1 = (getattr(modelA_t.linear1, "bias", None) is not None)
    has_b2 = (getattr(modelA_t.linear2, "bias", None) is not None)

    n_w1 = H * D
    n_b1 = H if has_b1 else 0
    n_w2 = C * H
    n_b2 = C if has_b2 else 0
    total = n_w1 + n_b1 + n_w2 + n_b2

    if deltaA_flat.numel() != total:
        raise RuntimeError(f"[gradcurve] Taille delta inattendue: got {deltaA_flat.numel()} vs expected {total}")

    g = deltaA_flat
    idx = 0

    dW1 = g[idx:idx + n_w1].view(H, D); idx += n_w1
    if has_b1:
        db1 = g[idx:idx + n_b1].view(H); idx += n_b1
    else:
        db1 = None

    dW2 = g[idx:idx + n_w2].view(C, H); idx += n_w2
    if has_b2:
        db2 = g[idx:idx + n_b2].view(C); idx += n_b2
    else:
        db2 = None

    # perm: A[i] ↔ B[perm[i]] ; on veut réordonner A en ordre B => inv_perm
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    inv_perm_t = torch.from_numpy(inv_perm).to(g.device)

    dW1p = dW1[inv_perm_t, :]
    dW2p = dW2[:, inv_perm_t]

    parts = [dW1p.reshape(-1)]
    if db1 is not None:
        parts.append(db1[inv_perm_t].reshape(-1))
    parts.append(dW2p.reshape(-1))
    if db2 is not None:
        parts.append(db2.reshape(-1))

    return torch.cat(parts)

# ===================================================================
# PHASE 2: HELPERS DE CONFIGURATION ET CHARGEMENT
# ===================================================================

# --------------------------
# Helpers de configuration
# --------------------------
def ensure_full_cfg(cfg):
    """
    exec_parallel passe parfois seulement le sous-dico 'gradcurve_mnist_mlp'.
    On recharge le YAML complet pour récupérer __other_configs__ si nécessaire.
    """
    if "__other_configs__" in cfg:
        return cfg

    candidates = [
        cfg.get("__config_path__", None),
        cfg.get("config_path", None),
        "/workspace/config_synced_transfer.yaml",
        "config_synced_transfer.yaml",
    ]
    candidates = [c for c in candidates if c is not None]

    for path in candidates:
        if os.path.exists(path):
            full = yaml.safe_load(open(path, "r"))
            if "__other_configs__" not in full:
                full["__other_configs__"] = {k: v for k, v in full.items() if isinstance(v, dict)}
            merged = copy.deepcopy(full.get("gradcurve_mnist_mlp", {})) or copy.deepcopy(cfg)
            merged["__other_configs__"] = full["__other_configs__"]
            return merged

    return cfg

def normalize_seed_pairs(x):
    if isinstance(x, str):
        x = eval(x)

    if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, np.integer)) for v in x):
        return [list(x)]

    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)) and len(x[0]) == 2:
        return [list(p) for p in x]

    raise ValueError("[gradcurve] source_target_seeds doit être [a,b] ou [[a,b],[c,d],...]")

def make_outman_train(cfg, train_exp_name="mnist_mlp_sgd"):
    output_dir = cfg.get("output_dir", "__outputs__")
    # signature repo: OutputManager(output_dir, exp_name, prefix_hashing)
    omt = OutputManager(output_dir, train_exp_name, False)
    if hasattr(omt, "output_prefix_hashing"):
        omt.output_prefix_hashing = False
    if hasattr(omt, "prefix_hashing"):
        omt.prefix_hashing = False
    return omt

def job_prefix_from_hparams(lr, width_factor, seed):
    # IMPORTANT: doit matcher exactement les noms de fichiers générés par exec_parallel
    # ex: lr_0.01--width_factor_64--seed_201--
    # (OutputManager ajoute .{name}.pth automatiquement)
    wf = int(width_factor) if float(width_factor) == int(width_factor) else float(width_factor)
    return f"lr_{float(lr):g}--width_factor_{wf}--seed_{int(seed)}--"

def _ckpt_exists(outman_train, pref, ext="pth"):
    path = outman_train.get_abspath(prefix=pref, ext=ext)
    return os.path.exists(path)

# ===================================================================
# PHASE 3: CHARGEMENT DES MODÈLES DEPUIS CHECKPOINTS
# ===================================================================

# --------------------------
# Chargement checkpoint → modèle
# --------------------------
def _init_model_via_get_learner(cfg_base, outman_train, lr, width_factor, seed, gpu_id):
    """
    On construit le modèle via get_learner (qui connaît l'archi du repo),
    puis on remplace ses poids par un state_dict chargé depuis un checkpoint.
    """
    # Garder width_factor comme int si possible (pour matcher les noms de fichiers)
    wf = int(width_factor) if float(width_factor) == int(width_factor) else float(width_factor)
    hparams = {"lr": float(lr), "width_factor": wf, "seed": int(seed)}
    cfgX = copy.deepcopy(cfg_base)
    cfgX["seed"] = int(seed)
    cfgX.setdefault("data_parallel", False)
    set_random_seed(int(seed))

    # epoch=-1: on récupère un learner au bon format (même si on n'utilise pas le ckpt epoch-1)
    ret = get_learner(
        exp_name="mnist_mlp_sgd",
        cfg=cfgX,
        hparams=hparams,
        gpu_id=gpu_id,
        outman=outman_train,
        epoch=-1,
        skip_train=True,
        skip_test=True,
    )
    learner_epoch = ret[1]
    if learner_epoch is None:
        # fallback: parfois c'est ret[0]
        learner_epoch = ret[0]
    return learner_epoch.model

def load_model_from_ckpt_path(cfg_base, outman_train, lr, width_factor, seed, ckpt_prefix, gpu_id):
    """Charge un modèle depuis un checkpoint d'itération ou d'epoch
    
    ckpt_prefix = "iter120.lr_0.01--width_factor_16--seed_101--"
    """
    ckp = outman_train.load_checkpoint(prefix=ckpt_prefix, ext="pth")
    if not hasattr(ckp, "model_state_dict"):
        raise RuntimeError(f"[gradcurve] checkpoint {ckpt_prefix} n'a pas 'model_state_dict'")

    model = _init_model_via_get_learner(cfg_base, outman_train, lr, width_factor, seed, gpu_id)
    model.load_state_dict(ckp.model_state_dict)
    model.eval()
    return model

# ===================================================================
# PHASE 4: GÉNÉRATION DU GRAPHIQUE PRINCIPAL
# ===================================================================

def gradcurve_mnist_mlp(cfg, outman, prefix, gpu_id):
    """Génère la courbe d'alignement des gradients (Figure 3a)
    
    Étapes:
    1. Charger config et modèles
    2. Pour chaque width_factor et seed_pair:
       a. Charger les checkpoints à chaque itération
       b. Calculer les deltas de poids
       c. Matcher les neurones et calculer similarité cosinus
    3. Tracer les courbes (brute vs permutée)
    """
    device = torch.device("cpu")
    print("[gradcurve] device =", device)

    # --- Initialisation: charger la config complète ---
    cfg = ensure_full_cfg(cfg)

    # Forcer CPU (utile chez toi)
    cfg["use_cuda"] = False
    cfg["num_gpus"] = 0

    # --- Parser les hyperparamètres ---
    lr = float(cfg.get("lr", 0.01))

    # widths
    wf = cfg.get("width_factor", None)
    if wf is None:
        wf = cfg.get("hparams_grid", {}).get("width_factor", [64.0])
    if isinstance(wf, (list, tuple)):
        width_list = [float(w) for w in wf]
    else:
        width_list = [float(wf)]
    print("[gradcurve] width_list =", width_list)

    seed_pairs = normalize_seed_pairs(cfg["source_target_seeds"])
    print("[gradcurve] seed_pairs =", seed_pairs)

    outman_train = make_outman_train(cfg, "mnist_mlp_sgd")
    print("[gradcurve] outman_train -> mnist_mlp_sgd (hashing OFF)")

    # --- Déterminer l'axe X: itérations vs epochs ---
    # Mode iter (Figure 3a): compare_iterations fourni
    # Mode epoch: checkpoint_epochs fourni
    x_mode = None
    x_points = None

    if cfg.get("compare_iterations", None) is not None:
        x_points = cfg["compare_iterations"]
        if isinstance(x_points, str):
            x_points = eval(x_points)
        x_points = sorted(list(set(int(x) for x in x_points)))
        x_mode = "iter"
    elif cfg.get("checkpoint_iters", None) is not None:
        x_points = cfg["checkpoint_iters"]
        if isinstance(x_points, str):
            x_points = eval(x_points)
        x_points = sorted(list(set(int(x) for x in x_points)))
        x_mode = "iter"
    else:
        x_points = cfg.get("checkpoint_epochs", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        if isinstance(x_points, str):
            x_points = eval(x_points)
        x_points = [int(e) for e in x_points if int(e) >= 0]
        x_points = sorted(list(set(x_points)))
        x_mode = "epoch"

    if len(x_points) < 2:
        raise RuntimeError("[gradcurve] Besoin d'au moins 2 points (iters ou epochs) pour des deltas.")

    print(f"[gradcurve] mode={x_mode.upper()}, #points={len(x_points)}, min/max=({x_points[0]}, {x_points[-1]})")

    # === BOUCLE PRINCIPALE: tracer pour chaque width_factor ===
    plt.figure(figsize=(10, 6))

    for width_factor in width_list:
        print("\n[gradcurve] ===============================")
        print(f"[gradcurve] width_factor = {width_factor}")
        print("[gradcurve] ===============================")

        raw_curves = []      # Similarité cosinus SANS permutation (identity)
        perm_curves = []     # Similarité cosinus AVEC permutation

        for (seedA, seedB) in seed_pairs:
            seedA, seedB = int(seedA), int(seedB)
            print(f"[gradcurve] seeds {seedA} vs {seedB}")

            xs_used = []    # Itérations/epochs réellement traités (pas skippés)
            raw_vals = []   # Valeurs de similarité (brute)
            perm_vals = []  # Valeurs de similarité (permutée)

            prev_x = x_points[0]

            for x in x_points[1:]:
                jobA = job_prefix_from_hparams(lr, width_factor, seedA)
                jobB = job_prefix_from_hparams(lr, width_factor, seedB)

                if x_mode == "iter":
                    prefA_t = f"iter{int(x)}.{jobA}"
                    prefA_p = f"iter{int(prev_x)}.{jobA}"
                    prefB_t = f"iter{int(x)}.{jobB}"
                    prefB_p = f"iter{int(prev_x)}.{jobB}"
                else:
                    prefA_t = f"epoch{int(x)}.{jobA}"
                    prefA_p = f"epoch{int(prev_x)}.{jobA}"
                    prefB_t = f"epoch{int(x)}.{jobB}"
                    prefB_p = f"epoch{int(prev_x)}.{jobB}"

                # skip propre si ckpt manquants (évite ton crash epoch4 manquant)
                need = [prefA_t, prefA_p, prefB_t, prefB_p]
                if any(not _ckpt_exists(outman_train, p) for p in need):
                    print(f"[gradcurve]  skip (missing ckpt) x={x} prev={prev_x} seeds=({seedA},{seedB})")
                    prev_x = x
                    continue

                # --- Charger les 4 checkpoints (2 modèles × 2 itérations) ---
                mA_t = load_model_from_ckpt_path(cfg, outman_train, lr, width_factor, seedA, prefA_t, gpu_id)
                mA_p = load_model_from_ckpt_path(cfg, outman_train, lr, width_factor, seedA, prefA_p, gpu_id)
                mB_t = load_model_from_ckpt_path(cfg, outman_train, lr, width_factor, seedB, prefB_t, gpu_id)
                mB_p = load_model_from_ckpt_path(cfg, outman_train, lr, width_factor, seedB, prefB_p, gpu_id)

                # --- Étape 1: Extraire les changements de poids ΔW = W(t) - W(t-1) ---
                dA = flatten_deltas(mA_t, mA_p)
                dB = flatten_deltas(mB_t, mB_p)

                # --- Étape 2: Calculer similarité cosinus SANS permutation (baseline) ---
                c_raw = cos(dA, dB)

                # --- Étape 3: Trouver la meilleure permutation ---
                perm = match_hidden_layer_on_delta(mA_t, mA_p, mB_t, mB_p)
                dA_perm = permute_deltas_for_mlp(dA, mA_t, perm)
                # --- Étape 4: Calculer similarité cosinus AVEC permutation ---
                c_perm = cos(dA_perm, dB)

                xs_used.append(int(x))
                raw_vals.append(c_raw)
                perm_vals.append(c_perm)

                prev_x = x

            if len(xs_used) > 0:
                raw_curves.append((xs_used, raw_vals))
                perm_curves.append((xs_used, perm_vals))

        if len(raw_curves) == 0:
            print("[gradcurve] WARNING: aucune courbe utilisable pour width =", width_factor)
            continue

        # --- Fusion des seed_pairs: garde l'intersection des x disponibles ---
        # (sinon les courbes ne seraient pas comparables)
        common_x = set(raw_curves[0][0])
        for xs, _ in raw_curves[1:]:
            common_x = common_x.intersection(set(xs))
        common_x = sorted(list(common_x))

        if len(common_x) == 0:
            print("[gradcurve] WARNING: intersection vide pour width =", width_factor)
            continue

        def mean_on_common(curves):
            vals = []
            for x in common_x:
                tmp = []
                for xs, ys in curves:
                    j = xs.index(x)
                    tmp.append(ys[j])
                vals.append(float(np.mean(tmp)))
            return np.array(vals, dtype=float)

        raw_mean = mean_on_common(raw_curves)
        perm_mean = mean_on_common(perm_curves)

        # --- Lissage par moyenne mobile: filtre la tendance principale ---
        raw_sm = moving_avg(raw_mean, k=5)
        perm_sm = moving_avg(perm_mean, k=5)

        # --- Tracer les deux courbes: brute (identity) vs permutée ---
        colors = ["blue", "green", "red", "purple", "orange", "brown"]
        color_idx = list(width_list).index(width_factor)
        # perm: points petits, raw: lignes discontinues avec points de même taille
        plt.plot(common_x, perm_sm, marker="o", markersize=3, linewidth=2.0, color=colors[color_idx], label=f"{int(width_factor)} (permuted)")
        plt.plot(common_x, raw_sm, marker="o", markersize=3, linestyle=":", linewidth=2.0, color=colors[color_idx], alpha=0.5, label=f"{int(width_factor)} (identity)")

    plt.xlabel("Iteration" if x_mode == "iter" else "Epoch")
    plt.ylabel("Cosine similarity")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    # --- Finalisation: sauvegarder le graphique ---
    path = outman.get_abspath(prefix=f"{prefix}gradcurve_mnist_mlp", ext="png")
    plt.savefig(path, dpi=150)
    print("[gradcurve_mnist_mlp] ✓ Graphique sauvegardé:", path)

