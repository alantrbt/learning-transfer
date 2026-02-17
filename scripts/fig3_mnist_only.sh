#!/usr/bin/env bash
# Figure 3 - MNIST seulement (sans réentraînement)
# Utilise les modèles déjà entraînés et fait les synced transfers + plots

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
echo "=== Figure 3 - MNIST seulement (sans réentraînement) ==="
echo ""

# -- Étape 2 : Synced Transfer GMT MNIST --
echo "[2/7] Synced transfer GMT - MNIST/MLP..."
python3 exec_parallel.py synced_transfer synced_transfer_init-mnist_mlp --gpu_id "$GPU"

# -- Étape 5 : Synced Transfer Random Permutation MNIST --
echo ""
echo "[5/7] Synced transfer random perm - MNIST/MLP..."
python3 exec_parallel.py synced_transfer synced_transfer_init-mnist_mlp_random --gpu_id "$GPU"

# -- Plots --
echo ""
echo "[PLOT] Generation des PDF..."
python3 exec_parallel.py plot figure_3_mnist_mlp --gpu_id "$GPU"

echo ""
echo "=== Figure 3 MNIST terminee ==="
echo "PDFs :"
ls -la "__outputs__/figure_3_mnist_mlp/"*.pdf 2>/dev/null || echo "  (aucun PDF pour figure_3_mnist_mlp)"
