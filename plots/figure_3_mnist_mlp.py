"""Figure 3(a) — Cosine similarity during synced transfer over different hidden layer widths.

Loads checkpoints from synced_transfer_init-mnist_mlp (GMT with optimized permutation)
and synced_transfer_init-mnist_mlp_identity (identity permutation baseline).
Plots cosine similarity for 5 hidden layer dimensions: 8, 16, 32, 64, 128 neurons.
- Solid lines: GMT (optimized permutation π)
- Dotted lines: Identity (π = I, no permutation)
"""
import glob
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.output_manager import OutputManager


def _build_width_to_file_map(output_dir, exp_name):
    """Build mapping from width_factor to checkpoint file."""
    exp_dir = os.path.join(output_dir, exp_name)
    if not os.path.isdir(exp_dir):
        return {}
    
    width_to_file = {}
    files = sorted(glob.glob(os.path.join(exp_dir, '*.synced_transfer_results.pth')))
    
    for f in files:
        # Get hash from filename
        hash_prefix = os.path.basename(f).split('.')[0]
        
        # Find corresponding prefix file
        prefix_files = glob.glob(os.path.join(exp_dir, f'{hash_prefix}.*mlp*.prefix'))
        if prefix_files:
            try:
                with open(prefix_files[0]) as pf:
                    prefix = pf.read().strip()
                
                # Extract width_factor from prefix
                if 'width_factor_' in prefix:
                    width_str = prefix.split('width_factor_')[1].split('--')[0]
                    width_factor = float(width_str)
                    width_to_file[width_factor] = f
            except Exception as e:
                print(f"Error processing {prefix_files[0]}: {e}")
    
    return width_to_file


def figure_3_mnist_mlp(cfg, outman, prefix, gpu_id):
    output_dir = cfg['output_dir']
    
    # Width factors corresponding to hidden layer sizes: 8, 16, 32, 64, 128
    # mlp.default_width = 64, so width_factor * 64 = hidden size
    width_factors = [0.125, 0.25, 0.5, 1.0, 2.0]
    hidden_sizes = [8, 16, 32, 64, 128]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    
    # Build mappings
    gmt_files = _build_width_to_file_map(output_dir, 'synced_transfer_init-mnist_mlp')
    id_files = _build_width_to_file_map(output_dir, 'synced_transfer_init-mnist_mlp_identity')
    
    if not gmt_files or not id_files:
        print(f"ERROR: Could not find files. GMT: {len(gmt_files)}, ID: {len(id_files)}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot for each dimension
    for width_factor, hidden_size, color in zip(width_factors, hidden_sizes, colors):
        # GMT (optimized permutation) - solid line
        if width_factor in gmt_files:
            try:
                gmt_results = torch.load(gmt_files[width_factor], map_location='cpu')
                if 'cosine_similarities' in gmt_results and gmt_results['cosine_similarities']:
                    iterations = [x['iteration'] for x in gmt_results['cosine_similarities']]
                    cosines = [x['cosine_similarity'] for x in gmt_results['cosine_similarities']]
                    ax.plot(iterations, cosines, label=f'GMT (d={hidden_size})', 
                            color=color, linestyle='-', linewidth=2.0, marker='o', markersize=5)
            except Exception as e:
                print(f"Error loading GMT file for width_factor {width_factor}: {e}")
        
        # Identity (baseline) - dotted line
        if width_factor in id_files:
            try:
                id_results = torch.load(id_files[width_factor], map_location='cpu')
                if 'cosine_similarities' in id_results and id_results['cosine_similarities']:
                    iterations = [x['iteration'] for x in id_results['cosine_similarities']]
                    cosines = [x['cosine_similarity'] for x in id_results['cosine_similarities']]
                    ax.plot(iterations, cosines, label=f'Identity (d={hidden_size})', 
                            color=color, linestyle=':', linewidth=2.0, marker='s', markersize=5)
            except Exception as e:
                print(f"Error loading ID file for width_factor {width_factor}: {e}")
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_ylim([-0.05, 1.05])
    ax.grid(color='lightgray', alpha=0.5, linestyle='--')
    if ax.get_legend_handles_labels()[0]:  # Only add legend if there are items
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=2)
    ax.set_title('Figure 3(a): Cosine Similarity of Gradient Matching Transfer\nAcross Hidden Layer Dimensions', 
                 fontsize=13, fontweight='bold')
    
    name = 'figure_3_mnist_mlp_cosine_all_dims'
    filepath = outman.get_abspath(prefix='plot.manual', ext='pdf', name=name)
    fig.savefig(filepath, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(filepath)



