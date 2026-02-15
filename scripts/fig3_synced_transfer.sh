#!/usr/bin/env bash
# Figure 3 - Synced Transfer (GMT en ligne)
# Compare GMT vs Random Permutation sur MNIST, CIFAR-10 et ImageNet.
#
# Prerequis : les entraînements source/cible doivent deja etre termines :
#   - mnist_mlp_sgd (seeds 101, 102)
#   - cifar10_conv8_sgd (seeds 101, 102)
#   - imagenet_resnet18_sgd (seeds 101, 102)
#
# Usage : bash scripts/fig3_synced_transfer.sh [GPU_ID]
# Temps estime : ~30 min (MNIST) + ~2h (CIFAR-10) + ~24h (ImageNet)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
echo "=== Figure 3 - Synced Transfer (GPU $GPU) ==="

# -- Étape 1 : S'assurer que les modeles source/cible sont entraînes --
echo ""
echo "[1/7] Verification des entraînements source/cible..."

# MNIST
echo "  -> mnist_mlp_sgd..."
python3 exec_parallel.py train mnist_mlp_sgd --gpu_id "$GPU"

# CIFAR-10
echo "  -> cifar10_conv8_sgd..."
python3 exec_parallel.py train cifar10_conv8_sgd --gpu_id "$GPU"

# ImageNet - COMMENTÉ (trop long)
# echo "  -> imagenet_resnet18_sgd..."
# python3 exec_parallel.py train imagenet_resnet18_sgd --gpu_id "$GPU"

# -- Étape 2 : Synced Transfer GMT --
echo ""
echo "[2/7] Synced transfer GMT - MNIST/MLP..."
python3 exec_parallel.py synced_transfer synced_transfer_init-mnist_mlp --gpu_id "$GPU"

echo "[3/7] Synced transfer GMT - CIFAR-10/Conv8..."
python3 exec_parallel.py synced_transfer synced_transfer_init-cifar10_conv8 --gpu_id "$GPU"

# ImageNet synced transfer - COMMENTÉ (trop long)
# echo "[4/7] Synced transfer GMT - ImageNet/ResNet-18..."
# python3 exec_parallel.py synced_transfer synced_transfer_init-imagenet_resnet18 --gpu_id "$GPU"

# -- Étape 3 : Synced Transfer Random Permutation --
echo ""
echo "[5/7] Synced transfer random perm - MNIST/MLP..."
python3 exec_parallel.py synced_transfer synced_transfer_init-mnist_mlp_random --gpu_id "$GPU"

echo "[6/7] Synced transfer random perm - CIFAR-10/Conv8..."
python3 exec_parallel.py synced_transfer synced_transfer_init-cifar10_conv8_random --gpu_id "$GPU"

# ImageNet synced transfer random - COMMENTÉ (trop long)
# echo "[7/7] Synced transfer random perm - ImageNet/ResNet-18..."
# python3 exec_parallel.py synced_transfer synced_transfer_init-imagenet_resnet18_random --gpu_id "$GPU"

# -- Étape 4 : Plots --
echo ""
echo "[PLOT] Generation des PDF..."
python3 exec_parallel.py plot figure_3_mnist_mlp --gpu_id "$GPU"
python3 exec_parallel.py plot figure_3_cifar10_conv8 --gpu_id "$GPU"
# ImageNet plot - COMMENTÉ (trop long)
# python3 exec_parallel.py plot figure_3_imagenet_resnet18 --gpu_id "$GPU"

echo ""
echo "=== Figure 3 terminee ==="
echo "PDFs :"
for exp in figure_3_mnist_mlp figure_3_cifar10_conv8 figure_3_imagenet_resnet18; do
    ls -la "__outputs__/$exp/"*.pdf 2>/dev/null || echo "  (aucun PDF pour $exp)"
done
