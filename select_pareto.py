import numpy as np
import matplotlib.pyplot as plt

def interactive_pareto_selection(X_file, F_file, obj_indices=(0, 2), 
                                 negate_objectives=(False, True)):
    """
    Click on points to see their index and design variables
    
    Parameters:
    -----------
    X_file : str
        Path to X.npy file
    F_file : str
        Path to F.npy file
    obj_indices : tuple
        Which objectives to plot (x_axis, y_axis)
    negate_objectives : tuple
        Whether to negate each objective
    """
    X = np.load(X_file)
    F = np.load(F_file)
    
    x_idx, y_idx = obj_indices
    x_negate, y_negate = negate_objectives
    
    x_vals = -F[:, x_idx] if x_negate else F[:, x_idx]
    y_vals = -F[:, y_idx] if y_negate else F[:, y_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all points
    scatter = ax.scatter(x_vals, y_vals, c=F[:, 0], 
                        cmap='viridis', s=100, picker=True,
                        edgecolors='black', linewidth=0.5, alpha=0.7)
    
    ax.set_xlabel('Mass (kg)', fontsize=12)
    ax.set_ylabel('Stability (MN)', fontsize=12)
    ax.set_title('Click on a point to see its index and design variables', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mass (kg)', fontsize=10)
    
    # Text box to show selected point info
    textbox = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                     fontsize=9, family='monospace')
    
    # Store selected point marker
    selected_point = [None]
    
    def on_pick(event):
        ind = event.ind[0]  # Get index of clicked point
        
        # Remove previous selection marker
        if selected_point[0] is not None:
            selected_point[0].remove()
        
        # Add new selection marker
        selected_point[0] = ax.scatter(x_vals[ind], y_vals[ind], 
                                      s=200, c='red', marker='*', 
                                      edgecolors='black', linewidth=2,
                                      zorder=10)
        
        design = X[ind]
        objectives = F[ind]
        
        # Format text for display
        info_text = f"""INDEX: {ind}
        
OBJECTIVES:
  Mass:      {objectives[0]:8.2f} kg
  Buckling:  {-objectives[1]:8.2f} MN
  Stability: {-objectives[2]:8.2f} MN

DESIGN VARIABLES:
  x[0]: {design[0]:.4f}
  x[1]: {design[1]:.4f}
  x[2]: {design[2]:.4f}
  x[3]: {design[3]:.4f}
  x[4]: {design[4]:.4f}
  x[5]: {design[5]:.4f}"""
        
        textbox.set_text(info_text)
        
        # Also print to console
        print("\n" + "=" * 70)
        print(f"SELECTED POINT INDEX: {ind}")
        print("=" * 70)
        print(f"Objectives:")
        print(f"  Mass:      {objectives[0]:.2f} kg")
        print(f"  Buckling:  {-objectives[1]:.2f} MN")
        print(f"  Stability: {-objectives[2]:.2f} MN")
        print(f"\nDesign Variables:")
        print(f"  x[0] (Length):      {design[0]:.4f}")
        print(f"  x[1] (Thickness_1): {design[1]:.4f}")
        print(f"  x[2] (Thickness_2): {design[2]:.4f}")
        print(f"  x[3] (Thickness_3): {design[3]:.4f}")
        print(f"  x[4] (Width_1):     {design[4]:.4f}")
        print(f"  x[5] (Width_2):     {design[5]:.4f}")
        print(f"\nCopy this to use the design:")
        print(f"design_{ind} = np.array({list(design)})")
        
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()


# Usage
interactive_pareto_selection(
    'opt_results/mlp_C2_X.npy',
    'opt_results/mlp_C2_F.npy',
    obj_indices=(0, 2),
    negate_objectives=(False, True)
)