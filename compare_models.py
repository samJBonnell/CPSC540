import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_comparison(file_paths, labels=None, objective_names=None, 
                           obj_indices=(0, 2), negate_objectives=(False, True),
                           figsize=(10, 6)):
    """
    Plot multiple Pareto fronts from saved .npy files with flexible objective selection
    
    Parameters:
    -----------
    file_paths : list of str
        List of paths to optimization_results_F.npy files
        Example: ['results/opt_F_cnn.npy', 'results/opt_F_mlp.npy']
    labels : list of str, optional
        Labels for each model. If None, uses filename-based labels
    objective_names : list of str, optional
        Names for objectives [obj1_name, obj2_name]
    obj_indices : tuple of int, optional
        Indices of objectives to plot (x_axis, y_axis). Default: (0, 2)
    negate_objectives : tuple of bool, optional
        Whether to negate each objective for plotting. Default: (False, True)
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.figure(figsize=figsize)
    
    # Default labels if not provided - use filename without path and extension
    if labels is None:
        labels = []
        for fp in file_paths:
            name = fp.split('/')[-1].replace('_F.npy', '')
            labels.append(name)
    
    # Colors and markers
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Extract objective indices
    x_idx, y_idx = obj_indices
    x_negate, y_negate = negate_objectives
    
    # Load and plot each Pareto front
    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        try:
            F = np.load(file_path)
            
            # Extract and optionally negate objectives
            x_vals = -F[:, x_idx] if x_negate else F[:, x_idx]
            y_vals = -F[:, y_idx] if y_negate else F[:, y_idx]
            
            # Plot scatter
            plt.scatter(x_vals, y_vals, 
                       label=label, 
                       alpha=0.7, 
                       s=60, 
                       color=colors[idx],
                       marker=markers[idx % len(markers)],
                       edgecolors='black',
                       linewidth=0.5)
            
            # Connect points to show the front
            idx_sort = np.argsort(x_vals)
            plt.plot(x_vals[idx_sort], y_vals[idx_sort], 
                    '--', 
                    alpha=0.3, 
                    linewidth=1.5,
                    color=colors[idx])
        except FileNotFoundError:
            print(f"Warning: Could not find file {file_path}")
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
    
    # Labels
    if objective_names:
        plt.xlabel(objective_names[0], fontsize=12)
        plt.ylabel(objective_names[1], fontsize=12)
    else:
        x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
        y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
    
    plt.title('Pareto Front Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return plt.gcf()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    objective_names = ['Mass (kg)', 'Stability (delMN)']    
    file_paths_multi = [
        # 'opt_results/mlp_2_16_LeakyReLU_0_05_F.npy',
        # 'opt_results/mlp_3_16_LeakyReLU_0_05_F.npy',
        'opt_results/mlp_4_16_LeakyReLU_0_05_UC2_F.npy',
        # 'opt_results/cnn_32_64__32__20_F.npy',
        # 'opt_results/cnn_32_64__32__20_constrained_F.npy',
        # 'opt_results/cnn_32_64__32__40_constrained_F.npy',
        # 'opt_results/cnn_32_64_128__64__40_constrained_F.npy',
        # 'opt_results/cnn_32_64_128__64__20_UC2_F.npy',
    ]
    
    # labels = [
    #     # 'MLP (2 layers)',
    #     # 'MLP (3 layers)',
    #     # 'MLP (4 layers)',
    #     # 'CNN',
    #     'CNN (20)'
    # ]
    
    fig = plot_pareto_comparison(
        file_paths_multi,
        # labels=labels,
        objective_names=objective_names,
        obj_indices=(0, 2),
        negate_objectives=(False, True)
    )
    
    # plt.savefig('opt_results/pareto_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example 3: Different objective combination (e.g., objectives 0 vs 1)
    fig = plot_pareto_comparison(
        file_paths_multi,
        # labels=labels,
        objective_names=['Mass (kg)', 'Buckling Strength (MN)'],
        obj_indices=(0, 1),
        negate_objectives=(False, True)
    )
    
    # plt.savefig('opt_results/pareto_comparison_obj_0_1.png', dpi=300, bbox_inches='tight')
    plt.show()