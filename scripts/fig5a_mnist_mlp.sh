#!/usr/bin/env bash
# Figure 5a - MNIST / MLP (Random Init to MNIST)
# Temps estime : ~10 min (GPU)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
echo "=== Figure 5a - MNIST / MLP (GPU $GPU) ==="

# Etape 1 : Entraînement (1 seed, 10 epoques)
echo "[1/3] Entraînement mnist_mlp_sgd..."
python3 exec_parallel.py train mnist_mlp_sgd --gpu_id "$GPU"

# Etape 2 : Transfert (4 methodes)
echo "[2/3] Transferts..."
python3 exec_parallel.py transfer transfer_init-mnist_mlp --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-mnist_mlp_fast --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-mnist_mlp_noperm --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-mnist_mlp_usetarget --gpu_id "$GPU"

# Etape 3 : Plot
echo "[3/3] Generation du PDF..."
python3 exec_parallel.py plot section_4_1_mnist_mlp --gpu_id "$GPU"

echo "=== Figure 5a terminee ==="
echo "PDF : __outputs__/section_4_1_mnist_mlp/"
ls -la __outputs__/section_4_1_mnist_mlp/*.pdf 2>/dev/null || echo "(aucun PDF trouvé)"
