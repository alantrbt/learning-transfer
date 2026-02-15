#!/usr/bin/env bash
# Figures 5f + 6b - CIFAR-10 -> CIFAR-100 / Conv8
# Figure 5f : transfert de trajectoire (section 4.1)
# Figure 6b : fine-tuning apres transfert (section 4.2)
# Pre-requis : cifar10_conv8_sgd entraîne (lance par fig5b_fig4_cifar10_conv8.sh)
# Temps estime : ~3h (GPU)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
echo "=== Figures 5f + 6b - CIFAR-10 -> CIFAR-100 (GPU $GPU) ==="

# Verifier le pre-requis
if [ ! -d "__outputs__/cifar10_conv8_sgd" ]; then
    echo "ERREUR : cifar10_conv8_sgd non entraîne. Lancez d'abord fig5b_fig4_cifar10_conv8.sh"
    exit 1
fi

# -- Fine-tuning sur CIFAR-100 --
echo "[1/3] Entraînement cifar10_to_cifar100-1_conv8_sgd..."
python3 exec_parallel.py train cifar10_to_cifar100-1_conv8_sgd --gpu_id "$GPU"

# -- Transferts (produisent transfer_results.json ET ft_results.json) --
echo "[2/3] Transferts..."
python3 exec_parallel.py transfer transfer_cifar10-cifar100_conv8 --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_cifar10-cifar100_conv8_fast --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_cifar10-cifar100_conv8_noperm --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_cifar10-cifar100_conv8_usetarget --gpu_id "$GPU"

# -- Plots --
echo "[3/3] Generation des PDFs..."
python3 exec_parallel.py plot section_4_1_cifar10_cifar100_conv8 --gpu_id "$GPU"
python3 exec_parallel.py plot section_4_2_cifar10_cifar100_conv8 --gpu_id "$GPU"

echo "=== Figures 5f + 6b terminees ==="
for d in section_4_1_cifar10_cifar100_conv8 section_4_2_cifar10_cifar100_conv8; do
    echo "-- $d --"
    ls -la __outputs__/"$d"/*.pdf __outputs__/"$d"/*.png 2>/dev/null || echo "  (aucune figure)"
done
