import os
import json
import matplotlib.pyplot as plt
from utils.output_manager import OutputManager

def section_4_1_mnist_mlp_figure5a(cfg, outman, prefix, gpu_id):
    """
    Generate Figure 5(a) from Chijiwa et al. - MNIST 2-MLP Transfer Learning
    Shows 4 methods compared over T=5 trajectory timesteps
    """
    
    # Manually specify the correct files with T=5 checkpoint_epochs=[-1,14] num_splits=5
    data_sources = [
        {
            'exp_name': 'transfer_init-mnist_mlp_usetarget',
            'prefix': 'e22ed292c129b3a1877fcfd53a6b195c',
            'label': 'Oracle Transfer',
            'color': 'C3',
            'linestyle': '-.',
        },
        {
            'exp_name': 'transfer_init-mnist_mlp_noperm',
            'prefix': 'e22ed292c129b3a1877fcfd53a6b195c',
            'label': 'Naive Transfer',
            'color': 'C1',
            'linestyle': '-',
        },
        {
            'exp_name': 'transfer_init-mnist_mlp',
            'prefix': '9ca3be88be1de8b579ab3df146702456',
            'label': 'GMT Transfer',
            'color': 'C0',
            'linestyle': '-',
        },
        {
            'exp_name': 'transfer_init-mnist_mlp_fast',
            'prefix': '8f9380bfd90f4119fa9c59d989335134',
            'label': 'FGMT Transfer',
            'color': 'C2',
            'linestyle': '-',
        },
    ]
    
    datas = []
    for source in data_sources:
        json_path = os.path.join(
            cfg['output_dir'],
            source['exp_name'],
            f"{source['prefix']}.transfer_results.json"
        )
        
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
            
            val_accs = results.get('val_accs', [])
            
            # Validate that we have exactly 5 points (T=5)
            if len(val_accs) != 5:
                print(f"WARNING: {source['label']} has {len(val_accs)} points, expected 5")
            
            data = {
                'label': source['label'],
                'means': val_accs,
                'stdevs': [0.0] * len(val_accs),  # Single seed, no variance
                'color': source['color'],
                'alpha': 0.1,
                'marker': 'o',
                'markersize': 3.5,
                'linewidth': 1.5,
                'linestyle': source['linestyle'],
            }
            datas.append(data)
            print(f"Loaded {source['label']}: {val_accs}")
            
        except FileNotFoundError as e:
            print(f"ERROR: Could not find {json_path}")
            raise
    
    # ===== Meta Data =====
    xs = list(range(1, len(datas[0]['means']) + 1))
    name = 'section_4-1_init-mnist_mlp'
    # ================

    # Validate all have same length
    for d in datas:
        assert len(datas[0]['means']) == len(d['means']), \
            f"Mismatch: {datas[0]['label']} has {len(datas[0]['means'])} points, {d['label']} has {len(d['means'])} points"

    filepath = outman.get_abspath(prefix='plot.manual', ext='pdf', name=name)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.set_xlabel('Trajectory Timestep')
    ax.set_ylabel('Val Accuracy')

    x_min = xs[0] - 0.5
    x_max = xs[-1] + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(10, 100)
    ax.grid(color='lightgray')

    for data in datas:
        ys = data['means']
        color = data['color']
        label = data['label']
        marker = data['marker']
        markersize = data['markersize']
        linewidth = data['linewidth']
        linestyle = data['linestyle']
        
        ax.plot(xs, ys,
                label=label, linewidth=linewidth, linestyle=linestyle,
                marker=marker, markersize=markersize,
                color=color)

    ax.legend(loc='lower right')

    fig.savefig(filepath, format="pdf", bbox_inches='tight')
    print(f"Saved PDF to: {filepath}")
