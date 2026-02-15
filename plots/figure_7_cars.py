"""Figure 7 — Scratch vs Transfer vs Ground-truth: Stanford Cars / ResNet-18.

Compares final best val accuracy for:
  - Scratch: cars_resnet18_sgd (training from random init on Cars)
  - Transfer: transfer_imagenet-cars_resnet18 (GMT transfer from ImageNet)
  - Ground truth: imagenet_to_cars_resnet18_sgd (fine-tuning from ImageNet pre-trained)
"""
import glob
import os
import json
import torch
import matplotlib.pyplot as plt
from utils.output_manager import OutputManager


def _load_best_val_scores(output_dir, exp_name):
    """Load best_val_score from all dump checkpoints of a training experiment."""
    exp_dir = os.path.join(output_dir, exp_name)
    if not os.path.isdir(exp_dir):
        return []
    files = sorted(glob.glob(os.path.join(exp_dir, 'dump.*.pth')))
    scores = []
    for f in files:
        try:
            ckp = torch.load(f, map_location='cpu')
            if 'best_val_score' in ckp:
                scores.append(ckp['best_val_score'])
        except Exception:
            pass
    return scores


def _load_transfer_final_accs(output_dir, exp_name):
    """Load final val_accs from all transfer_results.json files."""
    exp_dir = os.path.join(output_dir, exp_name)
    if not os.path.isdir(exp_dir):
        return []
    files = sorted(glob.glob(os.path.join(exp_dir, '*.transfer_results.json')))
    accs = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if 'val_accs' in data and data['val_accs']:
            accs.append(max(data['val_accs']))
    return accs


def figure_7_cars(cfg, outman, prefix, gpu_id):
    output_dir = cfg['output_dir']

    scratch_scores = _load_best_val_scores(output_dir, 'cars_resnet18_sgd')
    transfer_accs = _load_transfer_final_accs(output_dir, 'transfer_imagenet-cars_resnet18')
    gt_scores = _load_best_val_scores(output_dir, 'imagenet_to_cars_resnet18_sgd')

    labels = ['Scratch', 'GMT Transfer', 'Fine-tuning\n(ground truth)']
    means = []
    errs = []
    import statistics as stats
    for name, values in [('Scratch', scratch_scores), ('Transfer', transfer_accs), ('GT', gt_scores)]:
        if values:
            means.append(stats.mean(values))
            errs.append(stats.stdev(values) if len(values) > 1 else 0)
            print(f"  {name}: {stats.mean(values):.4f} +/- {errs[-1]:.4f} (n={len(values)})")
        else:
            means.append(0)
            errs.append(0)
            print(f"  {name}: NO DATA")

    colors = ['C1', 'C0', 'C3']
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, means, yerr=errs, color=colors, capsize=5, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Best Val Accuracy')
    ax.set_title('Stanford Cars — ResNet-18')
    ax.grid(axis='y', color='lightgray')
    ax.set_axisbelow(True)

    for bar, m in zip(bars, means):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    name = 'cars.scratch_trf_gr'
    filepath = outman.get_abspath(prefix='plot.manual', ext='pdf', name=name)
    fig.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(filepath)
