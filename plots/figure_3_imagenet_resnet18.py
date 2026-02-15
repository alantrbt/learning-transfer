"""Figure 3 — Synced transfer comparison: ImageNet / ResNet-18."""
import glob
import os
import torch
import matplotlib.pyplot as plt
from utils.output_manager import OutputManager


def _load_synced_results(output_dir, exp_name):
    exp_dir = os.path.join(output_dir, exp_name)
    if not os.path.isdir(exp_dir):
        return []
    files = sorted(glob.glob(os.path.join(exp_dir, '*.synced_transfer_results.pth')))
    return [torch.load(f, map_location='cpu') for f in files]


def figure_3_imagenet_resnet18(cfg, outman, prefix, gpu_id):
    output_dir = cfg['output_dir']
    fig, ax = plt.subplots(figsize=(6, 4.8))

    gmt_results = _load_synced_results(output_dir, 'synced_transfer_init-imagenet_resnet18')
    if gmt_results:
        r = gmt_results[0]
        epochs = list(range(1, len(r['sc_val_accs']) + 1))
        ax.plot(epochs, r['sc_val_accs'], label='Source (trained)', color='C0',
                linestyle='--', linewidth=1.5, marker='o', markersize=2)
        ax.plot(epochs, r['tg_val_accs'], label='Target (trained)', color='C1',
                linestyle='--', linewidth=1.5, marker='s', markersize=2)
        ax.plot(epochs, r['tr_val_accs'], label='GMT (synced)', color='C2',
                linestyle='-', linewidth=2.0, marker='^', markersize=2)
    else:
        print(f"  WARNING: no results for synced_transfer_init-imagenet_resnet18")

    rnd_results = _load_synced_results(output_dir, 'synced_transfer_init-imagenet_resnet18_random')
    if rnd_results:
        r = rnd_results[0]
        epochs = list(range(1, len(r['tr_val_accs']) + 1))
        ax.plot(epochs, r['tr_val_accs'], label='Random Perm (synced)', color='C3',
                linestyle='-.', linewidth=1.5, marker='D', markersize=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy')
    ax.grid(color='lightgray')
    ax.legend()

    name = 'compare_sync_diff_imagenet_resnet18_500iters'
    filepath = outman.get_abspath(prefix='plot.manual', ext='pdf', name=name)
    fig.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(filepath)
