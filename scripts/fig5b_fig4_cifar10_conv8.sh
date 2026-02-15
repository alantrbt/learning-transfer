#!/usr/bin/env bash
# Figures 5b + 4 - CIFAR-10 / Conv8
# Figure 5b : transfert principal (4 methodes)
# Figure 4  : ablations trajectoire lineaire vs reelle, scheduling uniforme vs cosinus
# Temps estime : ~5h (GPU) - domine par l'entraînement des 1 seeds
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
echo "=== Figures 5b + 4 - CIFAR-10 / Conv8 (GPU $GPU) ==="

# --- Entraînement (partage entre les deux figures) ---
echo "[1/4] Entraînement cifar10_conv8_sgd (1 seed, 60 époques)..."
python3 exec_parallel.py train cifar10_conv8_sgd --gpu_id "$GPU"

# --- Transferts Figure 5b ---
echo "[2/4] Transferts principaux (Figure 5b)..."
python3 exec_parallel.py transfer transfer_init-cifar10_conv8 --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-cifar10_conv8_fast --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-cifar10_conv8_noperm --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-cifar10_conv8_usetarget --gpu_id "$GPU"

# --- Transferts ablation Figure 4 ---
echo "[3/4] Transferts ablation (Figure 4)..."
python3 exec_parallel.py transfer transfer_init-cifar10_conv8_ablation_linear --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-cifar10_conv8_ablation_finegrained --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-cifar10_conv8_ablation_scheduling_uniform --gpu_id "$GPU"

# --- Plots ---
echo "[4/4] Génération des PDFs..."
python3 exec_parallel.py plot section_4_1_cifar10_conv8 --gpu_id "$GPU"
python3 exec_parallel.py plot section_3_3_real_vs_linear --gpu_id "$GPU"
python3 exec_parallel.py plot section_3_3_uniform_vs_cosine --gpu_id "$GPU"

echo "=== Figures 5b + 4 terminées ==="
for d in section_4_1_cifar10_conv8 section_3_3_real_vs_linear section_3_3_uniform_vs_cosine; do
    echo "--- $d ---"
    ls -la __outputs__/"$d"/*.pdf 2>/dev/null || echo "  (aucun PDF)"
done
