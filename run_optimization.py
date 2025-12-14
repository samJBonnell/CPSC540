# Python Imports
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Pymoo Imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.config import Config
Config.warnings['not_compiled'] = False

# Personal Imports
from optimizer import Optimizer
from data_loader import data_loader, create_cnn_matrix

# Personal Definitions
from us_lib.models.cnn import EncoderToVector
from us_lib.models.mlp import MLP

# Global variables for model and normalizers
model = None
X_normalizer = None
y_normalizer = None
model_type = None
convolution_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_prediction(x):
        global model, X_normalizer, y_normalizer, model_type, convolution_size, device
        
        x_copy = np.array(x)
        x_copy[0] = int(x_copy[0])
        
        # For CNN, create the convolution matrix
        if model_type == "cnn":
            x_input = create_cnn_matrix(x_copy, convolution_size, convolution_size)
        else:
            x_input = x_copy

        # Normalize input
        if X_normalizer is not None:
            x_normalized = X_normalizer.normalize(x_input)
        else:
            x_normalized = x_input
        
        x_normalized = torch.from_numpy(x_normalized).float().to(device)
        
        # Get predictions from surrogate model
        with torch.no_grad():
            predictions = model(x_normalized)
            
            # Denormalize predictions
            if y_normalizer is not None:
                predictions = y_normalizer.denormalize(predictions.unsqueeze(0).cpu().numpy())

        return predictions

class U1(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=6,
                        n_obj=2,
                        n_ieq_constr=0,
                        xl=np.array([1.0, 0.001, 0.001, 0.001, 0.015, 0.010]),
                        xu=np.array([8.0, 0.100, 0.100, 0.100, 0.750, 0.500])
                        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        predictions = model_prediction(x)
        
        # Mass
        mass = 4130 * (x[0]*(x[4]*x[2] + x[5]*x[3]) + x[1]*3) * 3

        # First eigen_mode
        eigen_buckling = predictions[0][0] if len(predictions) > 0 else 1.0
        
        # Objective 1: Minimize mass
        f1 = mass
        
        # Objective 2: Maximize buckling strength
        f2 = -eigen_buckling
        
        out["F"] = [f1, f2]

class U2(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=6,
                        n_obj=3,
                        n_ieq_constr=0,
                        xl=np.array([1.0, 0.001, 0.001, 0.001, 0.015, 0.010]),
                        xu=np.array([8.0, 0.100, 0.100, 0.100, 0.750, 0.500])
                        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        predictions = model_prediction(x)
        
        # Mass
        mass = 4130 * (x[0]*(x[4]*x[2] + x[5]*x[3]) + x[1]*3) * 3
        # mass = 4130 * (x[3]*x[1] + x[4]*x[2] + x[0]*3) * 3

        # First eigen_mode
        eigen_buckling = predictions[0][0] if len(predictions) > 0 else 1.0
        
        # Objective 1: Minimize mass
        f1 = mass
        
        # Objective 2: Maximize buckling strength
        f2 = -eigen_buckling

        # Objective 3: Maximize difference between first and second eigenmode
        f3 = -abs(predictions[0][0] - predictions[0][1]) if len(predictions[0]) > 1 else 0
        
        out["F"] = [f1, f2, f3]

class C1(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=6,
                        n_obj=2,
                        n_ieq_constr=1,
                        xl=np.array([1.0, 0.001, 0.001, 0.001, 0.015, 0.010]),
                        xu=np.array([8.0, 0.100, 0.100, 0.100, 0.750, 0.500])
                        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        predictions = model_prediction(x)
        
        # Mass
        mass = 4130 * (x[0]*(x[4]*x[2] + x[5]*x[3]) + x[1]*3) * 3
        # mass = 4130 * (x[3]*x[1] + x[4]*x[2] + x[0]*3) * 3

        # First eigen_mode
        eigen_buckling = predictions[0][0] if len(predictions) > 0 else 1.0
        
        # Objective 1: Minimize mass
        f1 = mass
        
        # Objective 2: Maximize buckling strength
        f2 = -eigen_buckling

        # Objective 3: Maximize difference between first and second eigenmode
        f3 = -abs(predictions[0][0] - predictions[0][1]) if len(predictions[0]) > 1 else 0
        g1 = 250 - abs(f3)
        
        out["F"] = [f1, f2]
        out["G"] = [g1]

class C2(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=6,
                        n_obj=3,
                        n_ieq_constr=2,
                        xl=np.array([1.0, 0.001, 0.001, 0.001, 0.015, 0.010]),
                        xu=np.array([8.0, 0.100, 0.100, 0.100, 0.750, 0.500])
                        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        predictions = model_prediction(x)
        
        # Mass
        mass = 4130 * (x[0]*(x[4]*x[2] + x[5]*x[3]) + x[1]*3) * 3
        # mass = 4130 * (x[3]*x[1] + x[4]*x[2] + x[0]*3) * 3

        # First eigen_mode
        eigen_buckling = predictions[0][0] if len(predictions) > 0 else 1.0
        
        # Objective 1: Minimize mass
        f1 = mass
        
        # Objective 2: Maximize buckling strength
        f2 = -eigen_buckling

        # Objective 3: Maximize difference between first and second eigenmode
        f3 = -abs(predictions[0][0] - predictions[0][1]) if len(predictions[0]) > 1 else 0

        g1 = 1499 - abs(f2)
        g2 = abs(f2) - 1501

        out["F"] = [f1, f2, f3]
        out["G"] = [g1, g2]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optimize using ML surrogate model')
    
    parser.add_argument('--model_name', type=str, default='',
                        help='The name of the saved model (default: )')
    
    parser.add_argument('--population', type=int, default=100,
                        help='Population size for NSGA-II (default: 100)')
    
    parser.add_argument('--offspring', type=int, default=50,
                        help='Number of offspring per generation (default: 50)')
    
    parser.add_argument('--generations', type=int, default=1000,
                        help='Number of generations (default: 1000)')
    
    parser.add_argument('--sbx_prob', type=float, default=0.9,
                        help='SBX crossover probability (default: 0.9)')
    
    parser.add_argument('--sbx_eta', type=float, default=15,
                        help='SBX distribution index (default: 15)')
    
    parser.add_argument('--mutation_pm', type=float, default=20,
                        help='Polynomial mutation eta (default: 20)')
    
    parser.add_argument('--output_name', type=str, default='optimization_results',
                        help='Output file name for results (default: optimization_results)')
    
    parser.add_argument('--plot', action='store_true',
                    help='Generate plots of Pareto front and solution space')
    
    parser.add_argument('--conv_size', type=int, default=20,
                    help='Size of the input convolution (default: 20)')
    
    parser.add_argument('--problem', type=str, default='U1',
                    help='Optimization problem (default: U1)')


    return parser.parse_args()


def main():
    global model, X_normalizer, y_normalizer, model_type
    
    # Parse command line arguments
    args = parse_args()
    
    print(f"Loading model: {args.model_name}")
    var_names = ['num_longitudinal', 't_panel', 't_longitudinal_web', 't_longitudinal_flange', 'h_longitudinal_web', 'w_longitudinal_flange']

    # Load the appropriate model
    if "cnn" in args.model_name:
        model_type = "cnn"
        convolution_size = args.conv_size
        model, X_normalizer, y_normalizer = EncoderToVector.load(f'models/{args.model_name}.pth')
        print(f"Loaded CNN model with convolution size {convolution_size}x{convolution_size}")
    elif "mlp" in args.model_name:
        model_type = "mlp"
        model, X_normalizer, y_normalizer = MLP.load(f'models/{args.model_name}.pth')
        print("Loaded MLP model")
    else:
        raise ValueError(f"{args.model_name} is not a viable model. Must contain 'cnn' or 'mlp'.")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize the optimization algorithm
    nsga2_alg = NSGA2(
        pop_size=args.population,
        n_offsprings=args.offspring,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=args.sbx_prob, eta=args.sbx_eta),
        mutation=PM(eta=args.mutation_pm),
        eliminate_duplicates=True
    )
    
    # Create the problem instance
    # obj_func = U1()
    problem = args.problem
    if problem == 'U1':
        obj_func = U1()
    if problem == 'U2':
        obj_func = U2()
    if problem == 'C1':
        obj_func = C1()
    if problem == 'C2':
        obj_func = C2()
    
    optimizer = Optimizer(algorithm=nsga2_alg, objective=obj_func)

    termination = get_termination("n_gen", args.generations)
    optimizer.termination = termination
    
    print(f"\nStarting optimization:")
    print(f"  Population: {args.population}")
    print(f"  Offspring: {args.offspring}")
    print(f"  Generations: {args.generations}")
    
    # Run the optimizer using the algorithm and objective function defined
    optimizer.run()
    
    # Get results from optimizer
    res, _ = optimizer.results()

      # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nNumber of Pareto optimal solutions: {len(res.X)}")
    print(f"\nObjective space (F):")
    print(f"  Mass range: [{res.F[:, 0].min():.6f}, {res.F[:, 0].max():.6f}]")
    print(f"  Buckling (neg): [{res.F[:, 1].min():.6f}, {res.F[:, 1].max():.6f}]")
    
    # Save results
    np.save(f'opt_results/{args.output_name}_X.npy', res.X)
    np.save(f'opt_results/{args.output_name}_F.npy', res.F)
    
    print(f"\nResults saved to:")
    print(f"  {args.output_name}_X.npy (design variables)")
    print(f"  {args.output_name}_F.npy (objective values)")
    
    # Print top 5 solutions
    print("\nTop 5 solutions by mass:")
    sorted_indices = np.argsort(res.F[:, 0])[:5]
    for i, idx in enumerate(sorted_indices, 1):
        print(f"\n  Solution {i}:")
        print(f"    Design vars: {res.X[idx]}")
        print(f"    Mass: {res.F[idx, 0]:.6f}")
        print(f"    Buckling: {-res.F[idx, 1]:.6f}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        plot_pareto_front(res.F, args.output_name)
        plot_solution_space(res.X, res.F, args.output_name, var_names=var_names)
        print(f"Plots saved as {args.output_name}_pareto.png and {args.output_name}_solutions.png")

def plot_pareto_front(F, output_name):
    """
    Plot the Pareto front in objective space
    
    Args:
        F: Objective values (N x 2 array)
        output_name: Base name for output file
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Pareto front
    ax.scatter(F[:, 0], -F[:, 1], c='blue', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Sort by mass for connecting line
    sorted_indices = np.argsort(F[:, 0])
    ax.plot(F[sorted_indices, 0], -F[sorted_indices, 1], 'r--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Mass (kg)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Eigen Buckling Load', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Front: Mass vs Buckling Strength', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for extreme points
    min_mass_idx = np.argmin(F[:, 0])
    max_buckling_idx = np.argmax(-F[:, 1])
    
    ax.annotate('Min Mass', xy=(F[min_mass_idx, 0], -F[min_mass_idx, 1]),
                xytext=(10, -10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.annotate('Max Buckling', xy=(F[max_buckling_idx, 0], -F[max_buckling_idx, 1]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(f'opt_results/{output_name}_pareto.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_solution_space(X, F, output_name, var_names=None):
    """
    Plot the solution space showing design variables (works for any N variables)
    
    Args:
        X: Design variables (N x D array) where D is number of design variables
        F: Objective values (N x 2 array) for coloring [mass, buckling]
        output_name: Base name for output file
        var_names: Optional list of variable names. If None, uses generic names
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    n_samples, n_vars = X.shape
    
    # Generate variable names if not provided
    if var_names is None:
        var_names = [f'Var_{i+1}' for i in range(n_vars)]
    elif len(var_names) != n_vars:
        print(f"Warning: Expected {n_vars} variable names, got {len(var_names)}. Using generic names.")
        var_names = [f'Var_{i+1}' for i in range(n_vars)]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Design variables colored by mass
    ax1 = fig.add_subplot(2, 3, 1)
    for i in range(n_vars):
        scatter = ax1.scatter(range(len(X)), X[:, i], c=F[:, 0], s=30, alpha=0.6, 
                             cmap='viridis', label=var_names[i])
    ax1.set_xlabel('Solution Index', fontweight='bold')
    ax1.set_ylabel('Design Variable Value', fontweight='bold')
    ax1.set_title('Design Variables (colored by mass)', fontweight='bold')
    # Only show legend if not too many variables
    if n_vars <= 10:
        ax1.legend(loc='best', fontsize=8, ncol=2 if n_vars > 5 else 1)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Mass (kg)', fontweight='bold')
    
    # 2. Design variables colored by buckling
    ax2 = fig.add_subplot(2, 3, 2)
    for i in range(n_vars):
        scatter = ax2.scatter(range(len(X)), X[:, i], c=-F[:, 1], s=30, alpha=0.6, 
                             cmap='plasma', label=var_names[i])
    ax2.set_xlabel('Solution Index', fontweight='bold')
    ax2.set_ylabel('Design Variable Value', fontweight='bold')
    ax2.set_title('Design Variables (colored by buckling)', fontweight='bold')
    if n_vars <= 10:
        ax2.legend(loc='best', fontsize=8, ncol=2 if n_vars > 5 else 1)
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter, ax=ax2)
    cbar2.set_label('Buckling Load', fontweight='bold')
    
    # 3. Parallel coordinates plot
    ax3 = fig.add_subplot(2, 3, 3)
    # Normalize design variables to [0, 1] for better visualization
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
    for i in range(len(X_norm)):
        ax3.plot(range(n_vars), X_norm[i], alpha=0.3, linewidth=0.5)
    ax3.set_xticks(range(n_vars))
    ax3.set_xticklabels(var_names, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Normalized Value', fontweight='bold')
    ax3.set_title('Parallel Coordinates Plot', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 3D scatter: First 3 design variables (if we have at least 3)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    if n_vars >= 3:
        scatter4 = ax4.scatter(X[:, 0], X[:, 1], X[:, 2], c=F[:, 0], s=50, 
                              alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)
        ax4.set_xlabel(var_names[0], fontweight='bold', fontsize=9)
        ax4.set_ylabel(var_names[1], fontweight='bold', fontsize=9)
        ax4.set_zlabel(var_names[2], fontweight='bold', fontsize=9)
        ax4.set_title('3D Design Space (First 3 Variables)', fontweight='bold')
        cbar4 = plt.colorbar(scatter4, ax=ax4, pad=0.1, shrink=0.8)
        cbar4.set_label('Mass (kg)', fontweight='bold')
    elif n_vars == 2:
        scatter4 = ax4.scatter(X[:, 0], X[:, 1], F[:, 0], c=F[:, 0], s=50,
                              alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)
        ax4.set_xlabel(var_names[0], fontweight='bold', fontsize=9)
        ax4.set_ylabel(var_names[1], fontweight='bold', fontsize=9)
        ax4.set_zlabel('Mass (kg)', fontweight='bold', fontsize=9)
        ax4.set_title('3D: Variables + Mass', fontweight='bold')
        cbar4 = plt.colorbar(scatter4, ax=ax4, pad=0.1, shrink=0.8)
        cbar4.set_label('Mass (kg)', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 0.5, 'Need at least 2 variables\nfor 3D plot',
                ha='center', va='center', fontsize=12)
        ax4.set_title('3D Plot (Insufficient Variables)', fontweight='bold')
    
    # 5. Box plots for each design variable
    ax5 = fig.add_subplot(2, 3, 5)
    box_data = [X[:, i] for i in range(n_vars)]
    bp = ax5.boxplot(box_data, tick_labels=var_names, patch_artist=True)
    
    # Use color cycling for many variables
    colors = plt.cm.Set3(np.linspace(0, 1, max(n_vars, 12)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax5.set_ylabel('Value', fontweight='bold')
    ax5.set_title('Distribution of Design Variables', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    # Adjust label size for many variables
    if n_vars > 10:
        ax5.tick_params(axis='x', labelsize=7)
    
    # 6. Correlation heatmap between design variables and objectives
    ax6 = fig.add_subplot(2, 3, 6)
    # Combine X and F for correlation
    combined = np.hstack([X, F])
    corr_matrix = np.corrcoef(combined.T)
    im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    labels = var_names + ['Mass', 'Buckling (neg)']
    ax6.set_xticks(range(n_vars + 2))
    ax6.set_yticks(range(n_vars + 2))
    
    # Adjust font size based on number of variables
    fontsize = max(6, min(9, 100 // (n_vars + 2)))
    ax6.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize)
    ax6.set_yticklabels(labels, fontsize=fontsize)
    ax6.set_title('Correlation Matrix', fontweight='bold')
    
    # Add correlation values (only if not too many variables)
    if n_vars <= 8:
        text_fontsize = max(5, min(7, 80 // (n_vars + 2)))
        for i in range(n_vars + 2):
            for j in range(n_vars + 2):
                text = ax6.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", 
                              fontsize=text_fontsize)
    
    cbar6 = plt.colorbar(im, ax=ax6)
    cbar6.set_label('Correlation', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'opt_results/{output_name}_solutions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Solution space plot saved to {output_name}_solutions.png")
    print(f"  - Number of design variables: {n_vars}")
    print(f"  - Number of solutions: {n_samples}")

if __name__ == '__main__':
    main()