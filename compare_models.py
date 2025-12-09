import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_comparison(file_paths, labels=None, objective_names=None, figsize=(10, 6)):
    """
    Plot multiple Pareto fronts from saved .npy files
    
    Parameters:
    -----------
    file_paths : list of str
        List of paths to optimization_results_F.npy files
        Example: ['results/opt_F_cnn.npy', 'results/opt_F_mlp.npy']
    labels : list of str, optional
        Labels for each model. If None, uses 'Model 1', 'Model 2', etc.
    objective_names : list of str, optional
        Names for objectives [obj1_name, obj2_name]
    """
    plt.figure(figsize=figsize)
    
    # Default labels if not provided
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(file_paths))]
    
    # Colors and markers
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Load and plot each Pareto front
    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        F = np.load(file_path)
        
        # Plot scatter
        plt.scatter(F[:, 0], -F[:, 1], 
                   label=label, 
                   alpha=0.7, 
                   s=60, 
                   color=colors[idx],
                   marker=markers[idx % len(markers)],
                   edgecolors='black',
                   linewidth=0.5)
        
        # Connect points to show the front
        idx_sort = np.argsort(F[:, 0])
        plt.plot(F[idx_sort, 0], -F[idx_sort, 1], 
                '--', 
                alpha=0.3, 
                linewidth=1.5,
                color=colors[idx])
    
    # Labels
    if objective_names:
        plt.xlabel(objective_names[0], fontsize=12)
        plt.ylabel(objective_names[1], fontsize=12)
    else:
        plt.xlabel('Objective ', fontsize=12)
        plt.ylabel('Objective 2', fontsize=12)
    
    plt.title('Pareto Front Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    return plt.gcf()

# Usage Example:
file_paths = [
    # 'opt_results/mlp_0_1000_F.npy',
    # 'opt_results/mlp_1_1000_F.npy',
    # 'opt_results/mlp_2_10000_F.npy',
    # 'opt_results/cnn_0_200_F.npy',
    # 'opt_results/cnn_1_200_F.npy',
    # 'opt_results/cnn_2_200_F.npy',
    # 'opt_results/mlp_0_200_F.npy',
    # 'opt_results/mlp_1_200_F.npy',
    # 'opt_results/mlp_2_200_F.npy',
    # 'opt_results/mlp_3_2000_F.npy',
    'opt_results/mlp_0_2000_F.npy',
    'opt_results/mlp_3_20000_F.npy',
    'opt_results/mlp_4_20000_F.npy',
]

objective_names = ['Mass (kg)', 'Buckling Strength (MN)']

fig = plot_pareto_comparison(file_paths, file_paths, objective_names)
plt.savefig('opt_results/pareto_comparison.png', dpi=300, bbox_inches='tight')
plt.show()