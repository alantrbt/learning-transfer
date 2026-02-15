#!/usr/bin/env bash
# Figures 5c, 5d, 5e, 6c, 6d - ImageNet / ResNet-18
# Couvre : Init->ImageNet, ImageNet->Cars, ImageNet->CUB
# Plus les figures de fine-tuning correspondantes
#
# ATTENTION : necessite les datasets suivants (~160 GB total) :
#   - ImageNet ILSVRC2012 dans __data__/imagenet/{train,val}/
#   - Stanford Cars       (telecharge automatiquement via pytorch_fgvc_dataset)
#   - CUB-200-2011        (telecharge automatiquement via pytorch_fgvc_dataset)
#
# Temps estime : plusieurs JOURS (1 seed × 100 epoques sur ImageNet)
# Recommande : lancer avec nohup ou dans tmux
#   nohup bash scripts/fig5cde_6cd_imagenet.sh 0 > logs/imagenet.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
echo "=== Figures ImageNet - ResNet-18 (GPU $GPU) ==="

# ==============================================
# Figure 5c : Random Init -> ImageNet
# ==============================================
echo ""
echo "== Figure 5c - Init->ImageNet =="

echo "[train] imagenet_resnet18_sgd"
python3 exec_parallel.py train imagenet_resnet18_sgd --gpu_id "$GPU"

echo "[transfer] 4 variantes"
python3 exec_parallel.py transfer transfer_init-imagenet_resnet18 --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-imagenet_resnet18_fast --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-imagenet_resnet18_noperm --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_init-imagenet_resnet18_usetarget --gpu_id "$GPU"

echo "[plot]"
python3 exec_parallel.py plot section_4_1_imagenet_resnet18 --gpu_id "$GPU"

# ==============================================
# Figure 5d + 6c : ImageNet -> Stanford Cars
# ==============================================
echo ""
echo "== Figure 5d + 6c - ImageNet->Cars =="

echo "[train] imagenet_to_cars_resnet18_sgd"
python3 exec_parallel.py train imagenet_to_cars_resnet18_sgd --gpu_id "$GPU"

echo "[transfer] 4 variantes"
python3 exec_parallel.py transfer transfer_imagenet-cars_resnet18 --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_imagenet-cars_resnet18_fast --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_imagenet-cars_resnet18_noperm --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_imagenet-cars_resnet18_usetarget --gpu_id "$GPU"

echo "[plot]"
python3 exec_parallel.py plot section_4_1_imagenet_cars_resnet18 --gpu_id "$GPU"
python3 exec_parallel.py plot section_4_2_imagenet_cars_resnet18 --gpu_id "$GPU"

# ==============================================
# Figure 5e + 6d : ImageNet -> CUB-200
# ==============================================
echo ""
echo "== Figure 5e + 6d - ImageNet->CUB =="

echo "[train] imagenet_to_cub_resnet18_sgd"
python3 exec_parallel.py train imagenet_to_cub_resnet18_sgd --gpu_id "$GPU"

echo "[transfer] 4 variantes"
python3 exec_parallel.py transfer transfer_imagenet-cub_resnet18 --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_imagenet-cub_resnet18_fast --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_imagenet-cub_resnet18_noperm --gpu_id "$GPU"
python3 exec_parallel.py transfer transfer_imagenet-cub_resnet18_usetarget --gpu_id "$GPU"

echo "[plot]"
python3 exec_parallel.py plot section_4_1_imagenet_cub_resnet18 --gpu_id "$GPU"
python3 exec_parallel.py plot section_4_2_imagenet_cub_resnet18 --gpu_id "$GPU"

# ==============================================
echo ""
echo "=== Toutes les figures ImageNet terminees ==="
for d in section_4_1_imagenet_resnet18 \
         section_4_1_imagenet_cars_resnet18 section_4_2_imagenet_cars_resnet18 \
         section_4_1_imagenet_cub_resnet18 section_4_2_imagenet_cub_resnet18; do
    echo "-- $d --"
    ls -la __outputs__/"$d"/*.pdf 2>/dev/null || echo "  (aucun PDF)"
done
