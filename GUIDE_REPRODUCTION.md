# Guide de reproduction — Transferring Learning Trajectories of Neural Networks

**Article** : Chijiwa et al. — [arXiv:2305.14122v2](https://arxiv.org/abs/2305.14122)  
**Repo** : `learning-transfer/`

---

## 1. Architecture du repo

```
exec_parallel.py          ← Point d'entrée unique
config.yaml               ← Toutes les expériences (train + transfer + plot)
commands/
  train.py                ← Entraîne un modèle, sauvegarde checkpoints
  transfer.py             ← GMT/FGMT/Naive/Oracle transfer + fine-tuning
  plot.py                 ← Charge les résultats JSON → génère PDF
plots/
  section_X_Y_*.py        ← Un script par figure de l'article
models/
  image_classification.py ← Wrapper d'entraînement
  networks/               ← MLP, Conv4/6/8, ResNet
utils/
  _weight_matching.py     ← PermutationSpecs + apply_permutation
  weight_matching.py      ← WeightMatching (accumulate + solve LAP)
  learning_transfer.py    ← get_transferred_params_faster() = cœur GMT/FGMT
  output_manager.py       ← Gère les chemins de sortie (JSON, checkpoints)
```

## 2. Prérequis

### Environnement Python
```bash
conda create -n lt python=3.9 -y
conda activate lt
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
pip install pyyaml pandas matplotlib scipy
```

### Datasets
| Dataset | Action requise |
|---------|---------------|
| MNIST | Automatique (`torchvision`) |
| CIFAR-10 | Automatique |
| CIFAR-100 | Automatique |
| ImageNet | Télécharger manuellement ILSVRC2012 → `__data__/imagenet/{train,val}/` |
| Stanford Cars | `torchvision.datasets.StanfordCars(download=True)` (URL parfois cassée) |
| CUB-200-2011 | Via `pytorch_fgvc_dataset/cub2011.py` |

### Préparation
```bash
cd learning-transfer
mkdir -p __outputs__ __data__ __sync__
```

## 3. Commande générale

```bash
python exec_parallel.py <command> <exp_name> [--gpu_id 0] [--config config.yaml]
```

- `<command>` : `train`, `transfer`, ou `plot`
- `<exp_name>` : clé définie dans `config.yaml`
- Le système itère automatiquement sur la grille `hparams_grid`

## 4. Pipeline par figure

### Figures 5a — MNIST / MLP (Random Init → MNIST)

**Temps estimé** : ~30 min sur GPU

```bash
# 1. Entraîner les modèles source/target (6 seeds)
python exec_parallel.py train mnist_mlp_sgd --gpu_id 0
# Relancer la commande jusqu'à ce que les 6 seeds soient complètes
# (le système détecte et skip les jobs déjà faits)

# 2. Transférer (4 variantes : GMT, FGMT, Naive, Oracle)
python exec_parallel.py transfer transfer_init-mnist_mlp --gpu_id 0
python exec_parallel.py transfer transfer_init-mnist_mlp_fast --gpu_id 0
python exec_parallel.py transfer transfer_init-mnist_mlp_noperm --gpu_id 0
python exec_parallel.py transfer transfer_init-mnist_mlp_usetarget --gpu_id 0

# 3. Générer la figure PDF
python exec_parallel.py plot section_4_1_mnist_mlp --gpu_id 0
```

### Figure 5b — CIFAR-10 / Conv8 (Random Init → CIFAR-10)

**Temps estimé** : ~2h sur GPU

```bash
python exec_parallel.py train cifar10_conv8_sgd --gpu_id 0

python exec_parallel.py transfer transfer_init-cifar10_conv8 --gpu_id 0
python exec_parallel.py transfer transfer_init-cifar10_conv8_fast --gpu_id 0
python exec_parallel.py transfer transfer_init-cifar10_conv8_noperm --gpu_id 0
python exec_parallel.py transfer transfer_init-cifar10_conv8_usetarget --gpu_id 0

python exec_parallel.py plot section_4_1_cifar10_conv8 --gpu_id 0
```

### Figure 5f — CIFAR-10 → CIFAR-100 / Conv8 (Pretrained)

```bash
# Pré-requis : cifar10_conv8_sgd déjà entraîné
python exec_parallel.py train cifar10_to_cifar100-1_conv8_sgd --gpu_id 0

python exec_parallel.py transfer transfer_cifar10-cifar100_conv8 --gpu_id 0
python exec_parallel.py transfer transfer_cifar10-cifar100_conv8_fast --gpu_id 0
python exec_parallel.py transfer transfer_cifar10-cifar100_conv8_noperm --gpu_id 0
python exec_parallel.py transfer transfer_cifar10-cifar100_conv8_usetarget --gpu_id 0

python exec_parallel.py plot section_4_1_cifar10_cifar100_conv8 --gpu_id 0
```

### Figure 5c — ImageNet / ResNet-18 (Random Init)

**Temps estimé** : ~3-7 jours sur 1 GPU A100

```bash
python exec_parallel.py train imagenet_resnet18_sgd --gpu_id 0

python exec_parallel.py transfer transfer_init-imagenet_resnet18 --gpu_id 0
python exec_parallel.py transfer transfer_init-imagenet_resnet18_fast --gpu_id 0
python exec_parallel.py transfer transfer_init-imagenet_resnet18_noperm --gpu_id 0
python exec_parallel.py transfer transfer_init-imagenet_resnet18_usetarget --gpu_id 0

python exec_parallel.py plot section_4_1_imagenet_resnet18 --gpu_id 0
```

### Figures 5d, 5e — ImageNet → Cars / CUB

```bash
# Cars
python exec_parallel.py train imagenet_to_cars_resnet18_sgd --gpu_id 0
python exec_parallel.py transfer transfer_imagenet-cars_resnet18 --gpu_id 0
python exec_parallel.py transfer transfer_imagenet-cars_resnet18_fast --gpu_id 0
python exec_parallel.py transfer transfer_imagenet-cars_resnet18_noperm --gpu_id 0
python exec_parallel.py transfer transfer_imagenet-cars_resnet18_usetarget --gpu_id 0
python exec_parallel.py plot section_4_1_imagenet_cars_resnet18 --gpu_id 0

# CUB
python exec_parallel.py train imagenet_to_cub_resnet18_sgd --gpu_id 0
python exec_parallel.py transfer transfer_imagenet-cub_resnet18 --gpu_id 0
python exec_parallel.py transfer transfer_imagenet-cub_resnet18_fast --gpu_id 0
python exec_parallel.py transfer transfer_imagenet-cub_resnet18_noperm --gpu_id 0
python exec_parallel.py transfer transfer_imagenet-cub_resnet18_usetarget --gpu_id 0
python exec_parallel.py plot section_4_1_imagenet_cub_resnet18 --gpu_id 0
```

### Figures 6a-d — Fine-tuning (Section 4.2)

Les résultats `ft_results.json` sont produits automatiquement par la commande `transfer`. Il suffit de lancer les plots :

```bash
python exec_parallel.py plot section_4_2_cifar10_conv8 --gpu_id 0
python exec_parallel.py plot section_4_2_cifar10_cifar100_conv8 --gpu_id 0
python exec_parallel.py plot section_4_2_imagenet_resnet18 --gpu_id 0
python exec_parallel.py plot section_4_2_imagenet_cars_resnet18 --gpu_id 0
python exec_parallel.py plot section_4_2_imagenet_cub_resnet18 --gpu_id 0
```

### Figures Section 3.3 — Ablations (linear vs actual, uniform vs cosine)

```bash
# Pré-requis : cifar10_conv8_sgd déjà entraîné
python exec_parallel.py transfer transfer_init-cifar10_conv8_ablation_scheduling_uniform --gpu_id 0
python exec_parallel.py transfer transfer_init-cifar10_conv8_ablation_linear --gpu_id 0
python exec_parallel.py transfer transfer_init-cifar10_conv8_ablation_linear_fast --gpu_id 0
python exec_parallel.py transfer transfer_init-cifar10_conv8_ablation_finegrained --gpu_id 0
python exec_parallel.py transfer transfer_init-cifar10_conv8_ablation_finegrained_fast --gpu_id 0

python exec_parallel.py plot section_3_3_real_vs_linear --gpu_id 0
python exec_parallel.py plot section_3_3_uniform_vs_cosine --gpu_id 0
```

### Figure 7 — Loss Landscape (Section 4.3)

```bash
python exec_parallel.py plot section_4_3_cifar10_conv8 --gpu_id 0
```
Ce script évalue des modèles interpolés en direct — GPU-intensif.

## 5. Comprendre le système de nommage (hash prefixes)

Le repo utilise `output_prefix_hashing: true` dans `config.yaml`. Chaque combinaison d'hyperparamètres est hashée en MD5 pour créer un préfixe de fichier unique. Les scripts de plot référencent ces hash en dur.

**Important** : les hashes sont déterministes (= MD5 de la chaîne des hyperparamètres). Si tu lances les expériences avec exactement les mêmes configs, les mêmes hashes seront générés et les plots fonctionneront.

## 6. Où sont les résultats

```
__outputs__/
  <exp_name>/
    <hash_prefix>.pth              ← Checkpoint du modèle
    <hash_prefix>.transfer_results.json  ← Accuracies pendant le transfert
    <hash_prefix>.ft_results.json  ← Accuracies pendant le fine-tuning
    <hash_prefix>.prefix           ← Mapping hash → hyperparamètres lisibles
```

## 7. Figures NON reproductibles par ce repo

| Figure | Raison |
|--------|--------|
| **Fig 1-2** | Diagrammes conceptuels (TikZ) |
| **Fig 3a-c** | Validation d'Assumption (P) — les PDFs sont dans `arXiv-2305.14122v2/resources/` mais le code de génération (entraîner 2 modèles en synchrone, mesurer la cosine similarity des gradients permutés) n'est **pas** dans le pipeline `config.yaml` standard. Il est dans `config_synced_transfer.yaml` qui utilise la commande `synced_transfer`. |
| **Fig 8** | Héritage de généralisation — les PDFs existent dans `resources/` mais pas de script de reproduction clair |

## 8. Résumé des temps de calcul

| Expérience | GPU estimé |
|-----------|------------|
| MNIST/MLP (tous transferts) | **~30 min** |
| CIFAR-10/Conv8 (training + transferts) | **~2-3h** |
| CIFAR-10→CIFAR-100/Conv8 | **~1-2h** (après CIFAR-10) |
| ImageNet/ResNet-18 (training 6 seeds) | **~3-7 jours** |
| ImageNet→Cars (training + transferts) | **~6-12h** (après ImageNet) |
| ImageNet→CUB (training + transferts) | **~6-12h** (après ImageNet) |
