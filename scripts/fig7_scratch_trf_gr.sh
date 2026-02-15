#!/usr/bin/env bash
# Figure 7 - Scratch vs Transfer vs Ground-truth
# Compare l'accuracy d'un modele entraîne from scratch, par GMT transfer,
# et par fine-tuning (ground truth) sur Cars, CIFAR-100 et CUB.
#
# Prerequis :
#   - Entraînements ImageNet/CIFAR-10 pre-existants (fig 5a/5b)
#   - Transferts GMT termines (fig 5cde / fig 5f)
#   - Fine-tuning ground truth termine (fig 5b pour CIFAR, fig 5cde pour ImageNet)
#
# Usage : bash scripts/fig7_scratch_trf_gr.sh [GPU_ID]
# Temps estime : ~5h (Cars/CUB scratch) + plots
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
echo "=== Figure 7 - Scratch vs Transfer vs Ground-truth (GPU $GPU) ==="

# -- Étape 1 : Entraînement from scratch --
echo ""
echo "[1/4] Entraînement Cars from scratch (cars_resnet18_sgd)..."
python3 exec_parallel.py train cars_resnet18_sgd --gpu_id "$GPU"

echo "[2/4] Entraînement CUB from scratch (cub_resnet18_sgd)..."
python3 exec_parallel.py train cub_resnet18_sgd --gpu_id "$GPU"

echo "[3/4] Entraînement CIFAR-100 from scratch (cifar100-1_conv8_sgd)..."
python3 exec_parallel.py train cifar100-1_conv8_sgd --gpu_id "$GPU"

# -- Étape 2 : S'assurer que les transferts GMT existent --
echo ""
echo "[4/4] Verification des transferts GMT..."
echo "  Si les transferts ne sont pas faits, executez d'abord :"
echo "    bash scripts/fig5cde_6cd_imagenet.sh $GPU   (pour Cars et CUB)"
echo "    bash scripts/fig5f_6b_cifar100.sh $GPU       (pour CIFAR-100)"

# -- Étape 3 : Plots --
echo ""
echo "[PLOT] Generation des PDF..."
python3 exec_parallel.py plot figure_7_cars --gpu_id "$GPU"
python3 exec_parallel.py plot figure_7_cifar100 --gpu_id "$GPU"
python3 exec_parallel.py plot figure_7_cub --gpu_id "$GPU"

echo ""
echo "=== Figure 7 terminee ==="
echo "PDFs :"
for exp in figure_7_cars figure_7_cifar100 figure_7_cub; do
    ls -la "__outputs__/$exp/"*.pdf 2>/dev/null || echo "  (aucun PDF pour $exp)"
done
