#!/usr/bin/env python3
"""
Figure 9(a) - MNIST 2-MLP Real Trajectories
Charge les résultats de transfer déjà exécutées
"""

import os
import json
import glob
import matplotlib.pyplot as plt

OUTPUT_DIR = '__outputs__'

def load_transfer_results(exp_name):
    """Charger les résultats de transfert pour une expérience"""
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    
    if not os.path.exists(exp_dir):
        return None
    
    # Chercher TOUS les fichiers transfer_results.json
    json_files = glob.glob(os.path.join(exp_dir, '*transfer_results.json'))
    
    if not json_files:
        return None
    
    # Prendre le dernier fichier (plus récent)
    json_path = sorted(json_files)[-1]
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        val_accs = data.get('val_accs', [])
        print(f"✓ Loaded {exp_name}")
        print(f"    Timesteps: {len(val_accs)}")
        print(f"    Range: {min(val_accs):.2f}% → {max(val_accs):.2f}%")
        print(f"    Final: {val_accs[-1]:.2f}%")
        return val_accs
    
    except Exception as e:
        print(f" Error reading {json_path}: {e}")
        return None

def main():
    print("="*70)
    print("Figure 9(a): MNIST 2-MLP Transfer with Real Trajectories")
    print("="*70 + "\n")
    
    # Charger les résultats des 4 méthodes finegrained
    results = {}
    
    experiments = [
        ('transfer_init-mnist_mlp_finegrained_usetarget', 'Oracle Transfer'),
        ('transfer_init-mnist_mlp_finegrained_noperm', 'Naive Transfer'),
        ('transfer_init-mnist_mlp_finegrained', 'GMT Transfer'),
        ('transfer_init-mnist_mlp_finegrained_fast', 'FGMT Transfer'),
    ]
    
    for exp_name, method_name in experiments:
        val_accs = load_transfer_results(exp_name)
        if val_accs:
            results[method_name] = val_accs
    
    print()
    
    if not results:
        print("ERROR: No results found!")
        return 1
    
    # Générer la figure
    fig, ax = plt.subplots(figsize=(11, 8))
    
    colors = {
        'Oracle Transfer': 'C3',
        'Naive Transfer': 'C1',
        'GMT Transfer': 'C0',
        'FGMT Transfer': 'C2',
    }
    
    linestyles = {
        'Oracle Transfer': '-.',
        'Naive Transfer': '-',
        'GMT Transfer': '-',
        'FGMT Transfer': '-',
    }
    
    # Plot les courbes en ordre préféré
    plot_order = ['Oracle Transfer', 'Naive Transfer', 'GMT Transfer', 'FGMT Transfer']
    
    for method in plot_order:
        if method in results:
            val_accs = results[method]
            xs = list(range(1, len(val_accs) + 1))
            
            ax.plot(xs, val_accs,
                    label=method,
                    color=colors[method],
                    linestyle=linestyles[method],
                    marker='o',
                    markersize=5,
                    linewidth=2)
    
    # Configuration des axes
    ax.set_xlabel('Trajectory Timestep', fontsize=13)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=13)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95, edgecolor='black')
    ax.set_title('Figure 9(a): MNIST 2-MLP Transfer with Real Trajectories',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Sauvegarder la figure
    output_pdf = os.path.join(OUTPUT_DIR, 'figure_9a_mnist_mlp_real_trajectories.pdf')
    
    fig.tight_layout()
    fig.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    
    print("="*70)
    print(f" Figure saved to: {output_pdf}")
    print("="*70 + "\n")

    
    plt.close(fig)
    
    return 0

if __name__ == '__main__':
    try:
        exit(main())
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
