#!/usr/bin/env bash
# Figure 6a - Fine-tuning CIFAR-10 / Conv8 (Section 4.2)
# Pre-requis : transferts CIFAR-10/Conv8 termines (lances par fig5b_fig4_cifar10_conv8.sh)
# Les donnees ft_results.json sont produites lors du transfert -> il suffit de tracer.
# Temps estime : quelques secondes
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
echo "=== Figure 6a - Fine-tuning CIFAR-10/Conv8 (GPU $GPU) ==="

python3 exec_parallel.py plot section_4_2_cifar10_conv8 --gpu_id "$GPU"

echo "=== Figure 6a terminee ==="
ls -la __outputs__/section_4_2_cifar10_conv8/*.pdf 2>/dev/null || echo "(aucun PDF)"
