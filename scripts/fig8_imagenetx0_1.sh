#!/usr/bin/env bash
# Figure 8 - ImageNet x0.1 width (ResNet-18 etroit)
# ENTIEREMENT COMMENTEE - depend d'ImageNet (trop couteux)
#
# Usage : bash scripts/fig8_imagenetx0_1.sh [GPU_ID]
# Temps estimé : ~2 jours (entraînement ImageNet étroit) + ~12h (fine-tuning + transferts)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
echo "=== Figure 8 - ImageNet x0.1 width (GPU $GPU) ==="
echo ""
echo " COMMENTE - Cette figure necessite ImageNet (~150 GB)"
echo "    et ressources importantes pour entraîner ResNet-18 x0.1"
echo ""
echo "Pour l'activer, décommentez les étapes dans ce script :"
echo "  vim scripts/fig8_imagenetx0_1.sh"
echo ""
