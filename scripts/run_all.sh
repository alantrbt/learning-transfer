#!/usr/bin/env bash
# Script principal - Reproduction de toutes les figures
#
# Usage:
#   bash scripts/run_all.sh [GPU_ID]
#
# Chaque etape est independante : si une experience est deja
# terminee, la commande passe instantanement.
#
# Pour accelerer, lancer les entraînements sur plusieurs GPU :
#   terminal 1 : python3 exec_parallel.py train <exp> --gpu_id 0
#   terminal 2 : python3 exec_parallel.py train <exp> --gpu_id 1
# La coordination est automatique via des fichiers de lock.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}

echo "================================================"
echo "Reproduction des figures (GPU $GPU)"
echo "================================================"
echo ""

# --- Groupe 1 : MNIST (rapide, ~10 min) ---
echo "--- Groupe 1/4 : MNIST / MLP ---"
bash scripts/fig5a_mnist_mlp.sh "$GPU"

# --- Groupe 2 : CIFAR-10 (moyen, ~5h) ---
echo ""
echo "--- Groupe 2/4 : CIFAR-10 / Conv8 ---"
bash scripts/fig5b_fig4_cifar10_conv8.sh "$GPU"
bash scripts/fig6a_ft_cifar10.sh "$GPU"

# --- Groupe 3 : CIFAR-100 (moyen, ~3h, dépend du groupe 2) ---
echo ""
echo "--- Groupe 3/4 : CIFAR-10 to CIFAR-100 ---"
bash scripts/fig5f_6b_cifar100.sh "$GPU"

# # --- Groupe 4 : ImageNet (très long, plusieurs jours) ---
# echo ""
# echo "--- Groupe 4/7 : ImageNet ---"
# echo "ATTENTION : nécessite ImageNet (~150 GB), Cars et CUB."
# echo "Temps estimé : plusieurs jours."
# read -p "Lancer les expériences ImageNet ? [y/N] " -n 1 -r
# echo ""
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#     bash scripts/fig5cde_6cd_imagenet.sh "$GPU"
# else
#     echo "ImageNet ignoré. Lancez plus tard avec :"
#     echo "  bash scripts/fig5cde_6cd_imagenet.sh $GPU"
# fi

# --- Groupe 5 : Figure 3 - Synced Transfer ---
echo ""
echo "--- Groupe 5/7 : Figure 3 - Synced Transfer (partielle: MNIST + CIFAR-10) ---"
bash scripts/fig3_synced_transfer.sh "$GPU"

# --- Groupe 6 : Figure 7 - Scratch vs Transfer vs GT ---
echo ""
echo "--- Groupe 6/7 : Figure 7 - Scratch vs Transfer vs GT (partielle: CIFAR-100 uniquement) ---"
bash scripts/fig7_scratch_trf_gr.sh "$GPU"

# --- Groupe 7 : Figure 8 - ImageNet x0.1 width ---
echo ""
echo "--- Groupe 7/7 : Figure 8 - ImageNet x0.1 width ---"
echo "IGNORÉ — nécessite ImageNet (commenté dans le script)"
# bash scripts/fig8_imagenetx0_1.sh "$GPU"

# --- Bilan ---
echo ""
echo "================================================"
echo "BILAN"
echo "================================================"
echo ""
echo "PDFs générés :"
find __outputs__ -name "*.pdf" -newer scripts/run_all.sh 2>/dev/null | sort | while read -r f; do
    echo "  $f"
done
echo ""
count=$(find __outputs__ -name "*.pdf" 2>/dev/null | wc -l)
echo "Total : $count fichier(s) PDF dans __outputs__/"
