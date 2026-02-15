#!/usr/bin/env bash
# Vérifie l'état de toutes les expériences sans rien lancer.
# Usage : bash scripts/check_status.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ok() { [ -d "__outputs__/$1" ] && [ "$(ls __outputs__/"$1"/*.pth 2>/dev/null | wc -l)" -gt 0 ] && echo "OK" || echo "--"; }
ok_json() { [ -d "__outputs__/$1" ] && ls __outputs__/"$1"/*transfer_results*.json &>/dev/null && echo "OK" || echo "--"; }
ok_pdf() { [ -d "__outputs__/$1" ] && ls __outputs__/"$1"/*.pdf &>/dev/null && echo "OK" || echo "--"; }
count() { ls __outputs__/"$1"/*.pth 2>/dev/null | wc -l; }

echo "====================================================================="
echo "ETAT DES EXPERIENCES"
echo "====================================================================="

echo ""
echo "--- Entraînements ---"
for exp in mnist_mlp_sgd cifar10_conv8_sgd cifar10_to_cifar100-1_conv8_sgd \
           cifar100-1_conv8_sgd \
           imagenet_resnet18_sgd imagenet_to_cars_resnet18_sgd imagenet_to_cub_resnet18_sgd \
           cars_resnet18_sgd cub_resnet18_sgd \
           imagenet_resnet18_x0_1_sgd \
           imagenet_x0_1_to_cars_resnet18_sgd imagenet_x0_1_to_cub_resnet18_sgd; do
    printf "  [%2s] %-45s (%s checkpoints)\n" "$(ok "$exp")" "$exp" "$(count "$exp" 2>/dev/null || echo 0)"
done

echo ""
echo "--- Transferts ---"
for exp in transfer_init-mnist_mlp transfer_init-mnist_mlp_fast \
           transfer_init-mnist_mlp_noperm transfer_init-mnist_mlp_usetarget \
           transfer_init-cifar10_conv8 transfer_init-cifar10_conv8_fast \
           transfer_init-cifar10_conv8_noperm transfer_init-cifar10_conv8_usetarget \
           transfer_cifar10-cifar100_conv8 transfer_cifar10-cifar100_conv8_fast \
           transfer_cifar10-cifar100_conv8_noperm transfer_cifar10-cifar100_conv8_usetarget \
           transfer_init-imagenet_resnet18 transfer_init-imagenet_resnet18_fast \
           transfer_imagenet-cars_resnet18 transfer_imagenet-cars_resnet18_fast \
           transfer_imagenet-cub_resnet18 transfer_imagenet-cub_resnet18_fast \
           transfer_imagenet_x0_1-cars_resnet18 transfer_imagenet_x0_1-cars_resnet18_fast \
           transfer_imagenet_x0_1-cars_resnet18_noperm transfer_imagenet_x0_1-cars_resnet18_usetarget \
           transfer_imagenet_x0_1-cub_resnet18 transfer_imagenet_x0_1-cub_resnet18_fast \
           transfer_imagenet_x0_1-cub_resnet18_noperm transfer_imagenet_x0_1-cub_resnet18_usetarget; do
    printf "  [%2s] %s\n" "$(ok_json "$exp")" "$exp"
done

echo ""
echo "--- Synced Transfer (Figure 3) ---"
ok_sync() { [ -d "__outputs__/$1" ] && ls __outputs__/"$1"/*synced_transfer_results*.pth &>/dev/null && echo "OK" || echo "--"; }
for exp in synced_transfer_init-mnist_mlp synced_transfer_init-mnist_mlp_random \
           synced_transfer_init-cifar10_conv8 synced_transfer_init-cifar10_conv8_random \
           synced_transfer_init-imagenet_resnet18 synced_transfer_init-imagenet_resnet18_random; do
    printf "  [%2s] %s\n" "$(ok_sync "$exp")" "$exp"
done

echo ""
echo "--- Figures (PDFs) ---"
for exp in section_4_1_mnist_mlp section_4_1_cifar10_conv8 \
           section_3_3_real_vs_linear section_3_3_uniform_vs_cosine \
           section_4_1_cifar10_cifar100_conv8 \
           section_4_2_cifar10_conv8 section_4_2_cifar10_cifar100_conv8 \
           section_4_1_imagenet_resnet18 \
           section_4_1_imagenet_cars_resnet18 section_4_2_imagenet_cars_resnet18 \
           section_4_1_imagenet_cub_resnet18 section_4_2_imagenet_cub_resnet18 \
           figure_3_mnist_mlp figure_3_cifar10_conv8 figure_3_imagenet_resnet18 \
           figure_7_cars figure_7_cifar100 figure_7_cub \
           figure_8_imagenetx0_1_cars figure_8_imagenetx0_1_cub; do
    printf "  [%2s] %s\n" "$(ok_pdf "$exp")" "$exp"
done

echo ""
echo "====================================================================="
echo "PDF trouves dans __outputs__/ :"
find __outputs__ -name "*.pdf" 2>/dev/null | sort | sed 's/^/  /'
echo "====================================================================="
