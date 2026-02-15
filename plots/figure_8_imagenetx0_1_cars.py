"""Figure 8 — Fine-tuning after transfer: ImageNet (x0.1 width) → Cars / ResNet-18.

Same structure as section_4_2 plots but for width_factor=0.1 experiments.
"""
import glob
import os
import matplotlib.pyplot as plt
import statistics as stats
from utils.output_manager import OutputManager


def _get_prefixes(output_dir, exp_name, json_name):
    pattern = os.path.join(output_dir, exp_name, f"*.{json_name}.json")
    matches = sorted(glob.glob(pattern))
    suffix = f".{json_name}.json"
    return [os.path.basename(p)[:-len(suffix)] for p in matches]


def _load_data(output_dir, json_name, exp_name, prefix_list,
               label='undef', color='undef', linestyle='-'):
    accs_list = []
    for prefix in prefix_list:
        outman = OutputManager(output_dir, exp_name, prefix_hashing=False)
        results = outman.load_json(json_name, prefix=prefix)
        accs_list.append(results['val_accs'])

    if not accs_list:
        return None

    tr = list(zip(*accs_list))
    means = [stats.mean(l) for l in tr]
    stdevs = [stats.stdev(l) for l in tr] if len(accs_list) > 1 else None

    return {
        'label': label, 'means': means, 'stdevs': stdevs,
        'color': color, 'linestyle': linestyle,
        'marker': 'o', 'markersize': 3.5, 'linewidth': 1.5, 'alpha': 0.1,
    }


def figure_8_imagenetx0_1_cars(cfg, outman, prefix, gpu_id):
    output_dir = cfg['output_dir']
    datas = []

    configs = [
        ('transfer_imagenet_x0_1-cars_resnet18_usetarget', 'Oracle Transfer', 'C3', '-.'),
        ('transfer_imagenet_x0_1-cars_resnet18_noperm', 'Naive Transfer', 'C1', '-'),
        ('transfer_imagenet_x0_1-cars_resnet18', 'GMT Transfer', 'C0', '-'),
        ('transfer_imagenet_x0_1-cars_resnet18_fast', 'FGMT Transfer', 'C2', '-'),
    ]

    for exp_name, label, color, ls in configs:
        prefixes = _get_prefixes(output_dir, exp_name, 'ft_results')
        if prefixes:
            data = _load_data(output_dir, 'ft_results', exp_name, prefixes,
                              label=label, color=color, linestyle=ls)
            if data:
                datas.append(data)
        else:
            print(f"  WARNING: no ft_results for {exp_name}")

    if not datas:
        print("  No data available for figure 8 (Cars). Run transfers first.")
        return

    xs = list(range(1, len(datas[0]['means']) + 1))
    name = 'section_4-2_imagenetx0_1-cars_resnet18'

    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy')
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.grid(color='lightgray')

    for data in datas:
        ys = data['means']
        ax.plot(xs, ys, label=data['label'], linewidth=data['linewidth'],
                linestyle=data['linestyle'], marker=data['marker'],
                markersize=data['markersize'], color=data['color'])
        # DISABLED: Shaded areas with single seed (no variance)
        # if data['stdevs']:
        #     devs = data['stdevs']
        #     ax.fill_between(xs, [y - s for y, s in zip(ys, devs)],
        #                     [y + s for y, s in zip(ys, devs)],
        #                     alpha=data['alpha'], color=data['color'])

    ax.legend()
    filepath = outman.get_abspath(prefix='plot.finetuning', ext='pdf', name=name)
    fig.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(filepath)
