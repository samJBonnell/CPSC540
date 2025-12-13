# # # # import matplotlib.pyplot as plt
# # # # import numpy as np


# # # # def plot_pareto_comparison(file_paths, labels=None, objective_names=None, 
# # # #                            obj_indices=(0, 2), negate_objectives=(False, True),
# # # #                            figsize=(10, 6)):
# # # #     """
# # # #     Plot multiple Pareto fronts from saved .npy files with flexible objective selection
    
# # # #     Parameters:
# # # #     -----------
# # # #     file_paths : list of str
# # # #         List of paths to optimization_results_F.npy files
# # # #         Example: ['results/opt_F_cnn.npy', 'results/opt_F_mlp.npy']
# # # #     labels : list of str, optional
# # # #         Labels for each model. If None, uses filename-based labels
# # # #     objective_names : list of str, optional
# # # #         Names for objectives [obj1_name, obj2_name]
# # # #     obj_indices : tuple of int, optional
# # # #         Indices of objectives to plot (x_axis, y_axis). Default: (0, 2)
# # # #     negate_objectives : tuple of bool, optional
# # # #         Whether to negate each objective for plotting. Default: (False, True)
# # # #     figsize : tuple, optional
# # # #         Figure size (width, height)
    
# # # #     Returns:
# # # #     --------
# # # #     matplotlib.figure.Figure
# # # #         The created figure object
# # # #     """
# # # #     plt.figure(figsize=figsize)
    
# # # #     # Default labels if not provided - use filename without path and extension
# # # #     if labels is None:
# # # #         labels = []
# # # #         for fp in file_paths:
# # # #             name = fp.split('/')[-1].replace('_F.npy', '')
# # # #             labels.append(name)
    
# # # #     # Colors and markers
# # # #     colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
# # # #     markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
# # # #     # Extract objective indices
# # # #     x_idx, y_idx = obj_indices
# # # #     x_negate, y_negate = negate_objectives
    
# # # #     # Load and plot each Pareto front
# # # #     for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
# # # #         try:
# # # #             F = np.load(file_path)
            
# # # #             # Extract and optionally negate objectives
# # # #             x_vals = -F[:, x_idx] if x_negate else F[:, x_idx]
# # # #             y_vals = -F[:, y_idx] if y_negate else F[:, y_idx]
            
# # # #             # Plot scatter
# # # #             plt.scatter(x_vals, y_vals, 
# # # #                        label=label, 
# # # #                        alpha=0.7, 
# # # #                        s=60, 
# # # #                        color=colors[idx],
# # # #                        marker=markers[idx % len(markers)],
# # # #                        edgecolors='black',
# # # #                        linewidth=0.5)
            
# # # #             # Connect points to show the front
# # # #             idx_sort = np.argsort(x_vals)
# # # #             plt.plot(x_vals[idx_sort], y_vals[idx_sort], 
# # # #                     '--', 
# # # #                     alpha=0.3, 
# # # #                     linewidth=1.5,
# # # #                     color=colors[idx])
# # # #         except FileNotFoundError:
# # # #             print(f"Warning: Could not find file {file_path}")
# # # #         except Exception as e:
# # # #             print(f"Warning: Error loading {file_path}: {e}")
    
# # # #     # Labels
# # # #     if objective_names:
# # # #         plt.xlabel(objective_names[0], fontsize=12)
# # # #         plt.ylabel(objective_names[1], fontsize=12)
# # # #     else:
# # # #         x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
# # # #         y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
# # # #         plt.xlabel(x_label, fontsize=12)
# # # #         plt.ylabel(y_label, fontsize=12)
    
# # # #     plt.title('Pareto Front Comparison', fontsize=14, fontweight='bold')
# # # #     plt.legend(fontsize=11, loc='best')
# # # #     plt.grid(True, alpha=0.3, linestyle='--')
# # # #     plt.tight_layout()
    
# # # #     return plt.gcf()


# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # def is_pareto_efficient(costs):
# # #     """
# # #     Find the Pareto-efficient points
    
# # #     Parameters:
# # #     -----------
# # #     costs : numpy.ndarray
# # #         An (n_points, n_costs) array where we want to minimize all costs
        
# # #     Returns:
# # #     --------
# # #     pareto_mask : numpy.ndarray
# # #         Boolean array indicating which points are Pareto efficient
# # #     """
# # #     is_efficient = np.ones(costs.shape[0], dtype=bool)
# # #     for i, c in enumerate(costs):
# # #         if is_efficient[i]:
# # #             # Remove points that are dominated by point i
# # #             is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
# # #             is_efficient[i] = True
# # #     return is_efficient


# # # def plot_pareto_all_points(file_paths, labels=None, objective_names=None, 
# # #                            obj_indices=(0, 2), negate_objectives=(False, True),
# # #                            figsize=(10, 6)):
# # #     """
# # #     Plot all evaluated points from multiple optimization runs
    
# # #     Parameters:
# # #     -----------
# # #     file_paths : list of str
# # #         List of paths to optimization_results_F.npy files
# # #     labels : list of str, optional
# # #         Labels for each model. If None, uses filename-based labels
# # #     objective_names : list of str, optional
# # #         Names for objectives [obj1_name, obj2_name]
# # #     obj_indices : tuple of int, optional
# # #         Indices of objectives to plot (x_axis, y_axis). Default: (0, 2)
# # #     negate_objectives : tuple of bool, optional
# # #         Whether to negate each objective for plotting. Default: (False, True)
# # #     figsize : tuple, optional
# # #         Figure size (width, height)
        
# # #     Returns:
# # #     --------
# # #     matplotlib.figure.Figure
# # #         The created figure object
# # #     """
# # #     plt.figure(figsize=figsize)
    
# # #     if labels is None:
# # #         labels = []
# # #         for fp in file_paths:
# # #             name = fp.split('/')[-1].replace('_F.npy', '')
# # #             labels.append(name)
    
# # #     colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
# # #     markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
# # #     x_idx, y_idx = obj_indices
# # #     x_negate, y_negate = negate_objectives
    
# # #     for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
# # #         try:
# # #             F = np.load(file_path)
# # #             x_vals = -F[:, x_idx] if x_negate else F[:, x_idx]
# # #             y_vals = -F[:, y_idx] if y_negate else F[:, y_idx]
            
# # #             plt.scatter(x_vals, y_vals, 
# # #                        label=label, 
# # #                        alpha=0.7, 
# # #                        s=60, 
# # #                        color=colors[idx],
# # #                        marker=markers[idx % len(markers)],
# # #                        edgecolors='black',
# # #                        linewidth=0.5)
            
# # #             idx_sort = np.argsort(x_vals)
# # #             plt.plot(x_vals[idx_sort], y_vals[idx_sort], 
# # #                     '--', 
# # #                     alpha=0.3, 
# # #                     linewidth=1.5,
# # #                     color=colors[idx])
                    
# # #         except FileNotFoundError:
# # #             print(f"Warning: Could not find file {file_path}")
# # #         except Exception as e:
# # #             print(f"Warning: Error loading {file_path}: {e}")
    
# # #     if objective_names:
# # #         plt.xlabel(objective_names[0], fontsize=12)
# # #         plt.ylabel(objective_names[1], fontsize=12)
# # #     else:
# # #         x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
# # #         y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
# # #         plt.xlabel(x_label, fontsize=12)
# # #         plt.ylabel(y_label, fontsize=12)
    
# # #     plt.title('All Evaluated Points', fontsize=14, fontweight='bold')
# # #     plt.legend(fontsize=11, loc='best')
# # #     plt.grid(True, alpha=0.3, linestyle='--')
# # #     plt.tight_layout()
# # #     return plt.gcf()


# # # def plot_pareto_front_only(file_paths, labels=None, objective_names=None, 
# # #                            obj_indices=(0, 2), negate_objectives=(False, True),
# # #                            figsize=(10, 6)):
# # #     """
# # #     Plot only the Pareto-efficient points from multiple optimization runs
    
# # #     Parameters:
# # #     -----------
# # #     file_paths : list of str
# # #         List of paths to optimization_results_F.npy files
# # #     labels : list of str, optional
# # #         Labels for each model. If None, uses filename-based labels
# # #     objective_names : list of str, optional
# # #         Names for objectives [obj1_name, obj2_name]
# # #     obj_indices : tuple of int, optional
# # #         Indices of objectives to plot (x_axis, y_axis). Default: (0, 2)
# # #     negate_objectives : tuple of bool, optional
# # #         Whether to negate each objective for plotting. Default: (False, True)
# # #     figsize : tuple, optional
# # #         Figure size (width, height)
        
# # #     Returns:
# # #     --------
# # #     matplotlib.figure.Figure
# # #         The created figure object
# # #     """
# # #     plt.figure(figsize=figsize)
    
# # #     if labels is None:
# # #         labels = []
# # #         for fp in file_paths:
# # #             name = fp.split('/')[-1].replace('_F.npy', '')
# # #             labels.append(name)
    
# # #     colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
# # #     markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
# # #     x_idx, y_idx = obj_indices
# # #     x_negate, y_negate = negate_objectives
    
# # #     for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
# # #         try:
# # #             F = np.load(file_path)
            
# # #             # Extract objectives for Pareto filtering (in minimization form)
# # #             obj1 = -F[:, x_idx] if x_negate else F[:, x_idx]
# # #             obj2 = -F[:, y_idx] if y_negate else F[:, y_idx]
            
# # #             # Create cost matrix for Pareto filtering (we want to minimize both)
# # #             costs = np.column_stack([obj1, obj2])
            
# # #             # Find Pareto-efficient points
# # #             pareto_mask = is_pareto_efficient(costs)
            
# # #             # Extract Pareto front
# # #             x_vals = obj1[pareto_mask]
# # #             y_vals = obj2[pareto_mask]
            
# # #             # Sort by x-axis for proper line connection
# # #             sort_idx = np.argsort(x_vals)
# # #             x_sorted = x_vals[sort_idx]
# # #             y_sorted = y_vals[sort_idx]
            
# # #             # Plot Pareto front
# # #             plt.scatter(x_sorted, y_sorted, 
# # #                        label=f'{label} ({len(x_vals)} points)', 
# # #                        alpha=0.8, 
# # #                        s=80, 
# # #                        color=colors[idx],
# # #                        marker=markers[idx % len(markers)],
# # #                        edgecolors='black',
# # #                        linewidth=0.7,
# # #                        zorder=3)
            
# # #             # Connect Pareto points
# # #             plt.plot(x_sorted, y_sorted, 
# # #                     '-', 
# # #                     alpha=0.5, 
# # #                     linewidth=2,
# # #                     color=colors[idx],
# # #                     zorder=2)
                    
# # #         except FileNotFoundError:
# # #             print(f"Warning: Could not find file {file_path}")
# # #         except Exception as e:
# # #             print(f"Warning: Error loading {file_path}: {e}")
    
# # #     if objective_names:
# # #         plt.xlabel(objective_names[0], fontsize=12)
# # #         plt.ylabel(objective_names[1], fontsize=12)
# # #     else:
# # #         x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
# # #         y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
# # #         plt.xlabel(x_label, fontsize=12)
# # #         plt.ylabel(y_label, fontsize=12)
    
# # #     plt.title('Pareto Front Comparison', fontsize=14, fontweight='bold')
# # #     plt.legend(fontsize=11, loc='best')
# # #     plt.grid(True, alpha=0.3, linestyle='--')
# # #     plt.tight_layout()
# # #     return plt.gcf()


# # # Example usage:
# # # fig1 = plot_pareto_all_points(
# # #     ['results/opt_F_cnn.npy', 'results/opt_F_mlp.npy'],
# # #     labels=['CNN', 'MLP'],
# # #     objective_names=['Accuracy', 'Model Size'],
# # #     obj_indices=(0, 2),
# # #     negate_objectives=(False, True)
# # # )
# # #
# # # fig2 = plot_pareto_front_only(
# # #     ['results/opt_F_cnn.npy', 'results/opt_F_mlp.npy'],
# # #     labels=['CNN', 'MLP'],
# # #     objective_names=['Accuracy', 'Model Size'],
# # #     obj_indices=(0, 2),
# # #     negate_objectives=(False, True)
# # # )

# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.spatial import ConvexHull

# # def is_pareto_efficient(costs):
# #     """
# #     Find the Pareto-efficient points
    
# #     Parameters:
# #     -----------
# #     costs : numpy.ndarray
# #         An (n_points, n_costs) array where we want to minimize all costs
        
# #     Returns:
# #     --------
# #     pareto_mask : numpy.ndarray
# #         Boolean array indicating which points are Pareto efficient
# #     """
# #     is_efficient = np.ones(costs.shape[0], dtype=bool)
# #     for i, c in enumerate(costs):
# #         if is_efficient[i]:
# #             # Remove points that are dominated by point i
# #             is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
# #             is_efficient[i] = True
# #     return is_efficient


# # def plot_pareto_all_points(file_paths, labels=None, objective_names=None, 
# #                            obj_indices=(0, 2), negate_objectives=(False, True),
# #                            figsize=(10, 6)):
# #     """
# #     Plot all evaluated points from multiple optimization runs
    
# #     Parameters:
# #     -----------
# #     file_paths : list of str
# #         List of paths to optimization_results_F.npy files
# #     labels : list of str, optional
# #         Labels for each model. If None, uses filename-based labels
# #     objective_names : list of str, optional
# #         Names for objectives [obj1_name, obj2_name]
# #     obj_indices : tuple of int, optional
# #         Indices of objectives to plot (x_axis, y_axis). Default: (0, 2)
# #     negate_objectives : tuple of bool, optional
# #         Whether to negate each objective for plotting. Default: (False, True)
# #     figsize : tuple, optional
# #         Figure size (width, height)
        
# #     Returns:
# #     --------
# #     matplotlib.figure.Figure
# #         The created figure object
# #     """
# #     plt.figure(figsize=figsize)
    
# #     if labels is None:
# #         labels = []
# #         for fp in file_paths:
# #             name = fp.split('/')[-1].replace('_F.npy', '')
# #             labels.append(name)
    
# #     colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
# #     markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
# #     x_idx, y_idx = obj_indices
# #     x_negate, y_negate = negate_objectives
    
# #     for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
# #         try:
# #             F = np.load(file_path)
# #             x_vals = -F[:, x_idx] if x_negate else F[:, x_idx]
# #             y_vals = -F[:, y_idx] if y_negate else F[:, y_idx]
            
# #             plt.scatter(x_vals, y_vals, 
# #                        label=label, 
# #                        alpha=0.7, 
# #                        s=60, 
# #                        color=colors[idx],
# #                        marker=markers[idx % len(markers)],
# #                        edgecolors='black',
# #                        linewidth=0.5)
            
# #             idx_sort = np.argsort(x_vals)
# #             plt.plot(x_vals[idx_sort], y_vals[idx_sort], 
# #                     '--', 
# #                     alpha=0.3, 
# #                     linewidth=1.5,
# #                     color=colors[idx])
                    
# #         except FileNotFoundError:
# #             print(f"Warning: Could not find file {file_path}")
# #         except Exception as e:
# #             print(f"Warning: Error loading {file_path}: {e}")
    
# #     if objective_names:
# #         plt.xlabel(objective_names[0], fontsize=12)
# #         plt.ylabel(objective_names[1], fontsize=12)
# #     else:
# #         x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
# #         y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
# #         plt.xlabel(x_label, fontsize=12)
# #         plt.ylabel(y_label, fontsize=12)
    
# #     plt.title('All Evaluated Points', fontsize=14, fontweight='bold')
# #     plt.legend(fontsize=11, loc='best')
# #     plt.grid(True, alpha=0.3, linestyle='--')
# #     plt.tight_layout()
# #     return plt.gcf()


# # def get_pareto_front_hull(x_vals, y_vals):
# #     """
# #     Get the points forming the Pareto front hull (non-convex, actual front)
# #     Extracts the lower-left boundary of points
    
# #     Parameters:
# #     -----------
# #     x_vals : numpy.ndarray
# #         X coordinates (minimizing - want smaller values)
# #     y_vals : numpy.ndarray
# #         Y coordinates (minimizing - want smaller values)
        
# #     Returns:
# #     --------
# #     x_front : numpy.ndarray
# #         X coordinates of front points
# #     y_front : numpy.ndarray
# #         Y coordinates of front points
# #     """
# #     # Combine and sort by x coordinate
# #     points = np.column_stack([x_vals, y_vals])
# #     sorted_idx = np.argsort(points[:, 0])
# #     sorted_points = points[sorted_idx]
    
# #     # Extract lower envelope (Pareto front)
# #     front_points = [sorted_points[0]]  # Start with leftmost point
    
# #     for point in sorted_points[1:]:
# #         # Add point if it has a lower y-value than the current front
# #         if point[1] < front_points[-1][1]:
# #             front_points.append(point)
    
# #     front_array = np.array(front_points)
# #     return front_array[:, 0], front_array[:, 1]


# # def plot_pareto_front_only(file_paths, labels=None, objective_names=None, 
# #                            obj_indices=(0, 2), negate_objectives=(False, True),
# #                            figsize=(10, 6), show_hull=True):
# #     """
# #     Plot only the Pareto-efficient points from multiple optimization runs
    
# #     Parameters:
# #     -----------
# #     file_paths : list of str
# #         List of paths to optimization_results_F.npy files
# #     labels : list of str, optional
# #         Labels for each model. If None, uses filename-based labels
# #     objective_names : list of str, optional
# #         Names for objectives [obj1_name, obj2_name]
# #     obj_indices : tuple of int, optional
# #         Indices of objectives to plot (x_axis, y_axis). Default: (0, 2)
# #     negate_objectives : tuple of bool, optional
# #         Whether to negate each objective for plotting. Default: (False, True)
# #     figsize : tuple, optional
# #         Figure size (width, height)
# #     show_hull : bool, optional
# #         If True, connects points with lines to show the front. Default: True
        
# #     Returns:
# #     --------
# #     matplotlib.figure.Figure
# #         The created figure object
# #     """
# #     plt.figure(figsize=figsize)
    
# #     if labels is None:
# #         labels = []
# #         for fp in file_paths:
# #             name = fp.split('/')[-1].replace('_F.npy', '')
# #             labels.append(name)
    
# #     colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
# #     markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
# #     x_idx, y_idx = obj_indices
# #     x_negate, y_negate = negate_objectives
    
# #     for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
# #         try:
# #             F = np.load(file_path)
            
# #             # Extract objectives for Pareto filtering (in minimization form)
# #             obj1 = -F[:, x_idx] if x_negate else F[:, x_idx]
# #             obj2 = -F[:, y_idx] if y_negate else F[:, y_idx]
            
# #             # Get Pareto front using the hull function
# #             x_sorted, y_sorted = get_pareto_front_hull(obj1, obj2)
            
# #             # Plot Pareto front
# #             plt.scatter(x_sorted, y_sorted, 
# #                        label=f'{label} ({len(x_sorted)} points)', 
# #                        alpha=0.8, 
# #                        s=80, 
# #                        color=colors[idx],
# #                        marker=markers[idx % len(markers)],
# #                        edgecolors='black',
# #                        linewidth=0.7,
# #                        zorder=3)
            
# #             # Connect Pareto points if requested
# #             if show_hull:
# #                 plt.plot(x_sorted, y_sorted, 
# #                         '-', 
# #                         alpha=0.5, 
# #                         linewidth=2,
# #                         color=colors[idx],
# #                         zorder=2)
                    
# #         except FileNotFoundError:
# #             print(f"Warning: Could not find file {file_path}")
# #         except Exception as e:
# #             print(f"Warning: Error loading {file_path}: {e}")
    
# #     if objective_names:
# #         plt.xlabel(objective_names[0], fontsize=12)
# #         plt.ylabel(objective_names[1], fontsize=12)
# #     else:
# #         x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
# #         y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
# #         plt.xlabel(x_label, fontsize=12)
# #         plt.ylabel(y_label, fontsize=12)
    
# #     plt.title('Pareto Front Comparison', fontsize=14, fontweight='bold')
# #     plt.legend(fontsize=11, loc='best')
# #     plt.grid(True, alpha=0.3, linestyle='--')
# #     plt.tight_layout()
# #     return plt.gcf()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import ConvexHull

# def is_pareto_efficient(costs):
#     """
#     Find the Pareto-efficient points
    
#     Parameters:
#     -----------
#     costs : numpy.ndarray
#         An (n_points, n_costs) array where we want to minimize all costs
        
#     Returns:
#     --------
#     pareto_mask : numpy.ndarray
#         Boolean array indicating which points are Pareto efficient
#     """
#     is_efficient = np.ones(costs.shape[0], dtype=bool)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             # Remove points that are dominated by point i
#             is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
#             is_efficient[i] = True
#     return is_efficient


# def plot_pareto_all_points(file_paths, labels=None, objective_names=None, 
#                            obj_indices=(0, 2), negate_objectives=(False, True),
#                            figsize=(10, 6)):
#     """
#     Plot all evaluated points from multiple optimization runs
    
#     Parameters:
#     -----------
#     file_paths : list of str
#         List of paths to optimization_results_F.npy files
#     labels : list of str, optional
#         Labels for each model. If None, uses filename-based labels
#     objective_names : list of str, optional
#         Names for objectives [obj1_name, obj2_name]
#     obj_indices : tuple of int, optional
#         Indices of objectives to plot (x_axis, y_axis). Default: (0, 2)
#     negate_objectives : tuple of bool, optional
#         Whether to negate each objective for plotting. Default: (False, True)
#     figsize : tuple, optional
#         Figure size (width, height)
        
#     Returns:
#     --------
#     matplotlib.figure.Figure
#         The created figure object
#     """
#     plt.figure(figsize=figsize)
    
#     if labels is None:
#         labels = []
#         for fp in file_paths:
#             name = fp.split('/')[-1].replace('_F.npy', '')
#             labels.append(name)
    
#     colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
#     markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
#     x_idx, y_idx = obj_indices
#     x_negate, y_negate = negate_objectives
    
#     for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
#         try:
#             F = np.load(file_path)
#             x_vals = -F[:, x_idx] if x_negate else F[:, x_idx]
#             y_vals = -F[:, y_idx] if y_negate else F[:, y_idx]
            
#             plt.scatter(x_vals, y_vals, 
#                        label=label, 
#                        alpha=0.7, 
#                        s=60, 
#                        color=colors[idx],
#                        marker=markers[idx % len(markers)],
#                        edgecolors='black',
#                        linewidth=0.5)
            
#             idx_sort = np.argsort(x_vals)
#             plt.plot(x_vals[idx_sort], y_vals[idx_sort], 
#                     '--', 
#                     alpha=0.3, 
#                     linewidth=1.5,
#                     color=colors[idx])
                    
#         except FileNotFoundError:
#             print(f"Warning: Could not find file {file_path}")
#         except Exception as e:
#             print(f"Warning: Error loading {file_path}: {e}")
    
#     if objective_names:
#         plt.xlabel(objective_names[0], fontsize=12)
#         plt.ylabel(objective_names[1], fontsize=12)
#     else:
#         x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
#         y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
#         plt.xlabel(x_label, fontsize=12)
#         plt.ylabel(y_label, fontsize=12)
    
#     plt.title('All Evaluated Points', fontsize=14, fontweight='bold')
#     plt.legend(fontsize=11, loc='best')
#     plt.grid(True, alpha=0.3, linestyle='--')
#     plt.tight_layout()
#     return plt.gcf()


# def get_pareto_front_hull(x_vals, y_vals):
#     """
#     Get the points forming the lower-left boundary of the point cloud
#     This extracts the Pareto front for minimization objectives
    
#     Parameters:
#     -----------
#     x_vals : numpy.ndarray
#         X coordinates (after any negation has been applied)
#     y_vals : numpy.ndarray
#         Y coordinates (after any negation has been applied)
        
#     Returns:
#     --------
#     x_front : numpy.ndarray
#         X coordinates of front points
#     y_front : numpy.ndarray
#         Y coordinates of front points
#     """
#     # Combine and sort by x coordinate (ascending)
#     points = np.column_stack([x_vals, y_vals])
#     sorted_idx = np.argsort(points[:, 0])
#     sorted_points = points[sorted_idx]
    
#     # Extract lower envelope (Pareto front for minimization)
#     front_points = [sorted_points[0]]  # Start with leftmost point
    
#     for point in sorted_points[1:]:
#         # Add point if it has a lower y-value than the current front
#         if point[1] < front_points[-1][1]:
#             front_points.append(point)
    
#     front_array = np.array(front_points)
#     return front_array[:, 0], front_array[:, 1]


# def get_boundary_hull(x_vals, y_vals):
#     """
#     Get all extreme boundary points of the point cloud using convex hull
    
#     Parameters:
#     -----------
#     x_vals : numpy.ndarray
#         X coordinates
#     y_vals : numpy.ndarray
#         Y coordinates
        
#     Returns:
#     --------
#     x_boundary : numpy.ndarray
#         X coordinates of boundary points (ordered around perimeter)
#     y_boundary : numpy.ndarray
#         Y coordinates of boundary points (ordered around perimeter)
#     """
#     # Combine points
#     points = np.column_stack([x_vals, y_vals])
    
#     # Compute convex hull
#     hull = ConvexHull(points)
    
#     # Extract hull points in order
#     hull_points = points[hull.vertices]
    
#     # Sort by angle from centroid to ensure proper ordering
#     centroid = np.mean(hull_points, axis=0)
#     angles = np.arctan2(hull_points[:, 1] - centroid[1], 
#                         hull_points[:, 0] - centroid[0])
#     sorted_idx = np.argsort(angles)
    
#     hull_sorted = hull_points[sorted_idx]
    
#     return hull_sorted[:, 0], hull_sorted[:, 1]


# def plot_pareto_front_only(file_paths, labels=None, objective_names=None, 
#                            obj_indices=(0, 2), negate_objectives=(False, True),
#                            figsize=(10, 6), show_hull=True, use_convex_hull=False):
#     """
#     Plot only the Pareto-efficient points from multiple optimization runs
    
#     Parameters:
#     -----------
#     file_paths : list of str
#         List of paths to optimization_results_F.npy files
#     labels : list of str, optional
#         Labels for each model. If None, uses filename-based labels
#     objective_names : list of str, optional
#         Names for objectives [obj1_name, obj2_name]
#     obj_indices : tuple of int, optional
#         Indices of objectives to plot (x_axis, y_axis). Default: (0, 2)
#     negate_objectives : tuple of bool, optional
#         Whether to negate each objective for plotting. Default: (False, True)
#     figsize : tuple, optional
#         Figure size (width, height)
#     show_hull : bool, optional
#         If True, connects points with lines to show the front. Default: True
#     use_convex_hull : bool, optional
#         If True, extracts the convex hull boundary (all extreme points).
#         If False, extracts only the Pareto front (lower-left boundary). Default: False
        
#     Returns:
#     --------
#     matplotlib.figure.Figure
#         The created figure object
#     """
#     plt.figure(figsize=figsize)
    
#     if labels is None:
#         labels = []
#         for fp in file_paths:
#             name = fp.split('/')[-1].replace('_F.npy', '')
#             labels.append(name)
    
#     colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
#     markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
#     x_idx, y_idx = obj_indices
#     x_negate, y_negate = negate_objectives
    
#     for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
#         try:
#             F = np.load(file_path)
            
#             # Extract objectives (apply negation for display)
#             obj1 = -F[:, x_idx] if x_negate else F[:, x_idx]
#             obj2 = -F[:, y_idx] if y_negate else F[:, y_idx]
            
#             # Get boundary points
#             if use_convex_hull:
#                 x_sorted, y_sorted = get_boundary_hull(obj1, obj2)
#             else:
#                 x_sorted, y_sorted = get_pareto_front_hull(obj1, obj2)
            
#             # Plot Pareto front
#             plt.scatter(x_sorted, y_sorted, 
#                        label=f'{label} ({len(x_sorted)} points)', 
#                        alpha=0.8, 
#                        s=80, 
#                        color=colors[idx],
#                        marker=markers[idx % len(markers)],
#                        edgecolors='black',
#                        linewidth=0.7,
#                        zorder=3)
            
#             # Connect boundary/front points if requested
#             if show_hull:
#                 # For convex hull, close the loop
#                 if use_convex_hull:
#                     x_plot = np.append(x_sorted, x_sorted[0])
#                     y_plot = np.append(y_sorted, y_sorted[0])
#                     plt.plot(x_plot, y_plot, 
#                             '-', 
#                             alpha=0.5, 
#                             linewidth=2,
#                             color=colors[idx],
#                             zorder=2)
#                 else:
#                     plt.plot(x_sorted, y_sorted, 
#                             '-', 
#                             alpha=0.5, 
#                             linewidth=2,
#                             color=colors[idx],
#                             zorder=2)
                    
#         except FileNotFoundError:
#             print(f"Warning: Could not find file {file_path}")
#         except Exception as e:
#             print(f"Warning: Error loading {file_path}: {e}")
    
#     if objective_names:
#         plt.xlabel(objective_names[0], fontsize=12)
#         plt.ylabel(objective_names[1], fontsize=12)
#     else:
#         x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
#         y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
#         plt.xlabel(x_label, fontsize=12)
#         plt.ylabel(y_label, fontsize=12)
    
#     title = 'Boundary Points' if use_convex_hull else 'Pareto Front Comparison'
#     plt.title(title, fontsize=14, fontweight='bold')
#     plt.legend(fontsize=11, loc='best')
#     plt.grid(True, alpha=0.3, linestyle='--')
#     plt.tight_layout()
#     return plt.gcf()


# # Example usage:
# # For Pareto front (lower-left boundary only):
# # fig = plot_pareto_front_only(
# #     ['results/opt_F_cnn.npy'],
# #     labels=['MLP'],
# #     objective_names=['Mass', 'Stability'],
# #     obj_indices=(0, 2),
# #     negate_objectives=(False, True),
# #     use_convex_hull=False
# # )
# #
# # For all boundary points (convex hull):
# # fig = plot_pareto_front_only(
# #     ['results/opt_F_cnn.npy'],
# #     labels=['MLP'],
# #     objective_names=['Mass', 'Stability'],
# #     obj_indices=(0, 2),
# #     negate_objectives=(False, True),
# #     use_convex_hull=True
# # )

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points
    
    Parameters:
    -----------
    costs : numpy.ndarray
        An (n_points, n_costs) array where we want to minimize all costs
        
    Returns:
    --------
    pareto_mask : numpy.ndarray
        Boolean array indicating which points are Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Remove points that are dominated by point i
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def plot_pareto_all_points(file_paths, labels=None, objective_names=None, 
                           obj_indices=(0, 2), negate_objectives=(False, True),
                           figsize=(10, 6)):
    """
    Plot all evaluated points from multiple optimization runs
    
    Parameters:
    -----------
    file_paths : list of str
        List of paths to optimization_results_F.npy files
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
    
    if labels is None:
        labels = []
        for fp in file_paths:
            name = fp.split('/')[-1].replace('_F.npy', '')
            labels.append(name)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    x_idx, y_idx = obj_indices
    x_negate, y_negate = negate_objectives
    
    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        try:
            F = np.load(file_path)
            x_vals = -F[:, x_idx] if x_negate else F[:, x_idx]
            y_vals = -F[:, y_idx] if y_negate else F[:, y_idx]
            
            plt.scatter(x_vals, y_vals, 
                       label=label, 
                       alpha=0.7, 
                       s=60, 
                       color=colors[idx],
                       marker=markers[idx % len(markers)],
                       edgecolors='black',
                       linewidth=0.5)
            
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
    
    if objective_names:
        plt.xlabel(objective_names[0], fontsize=12)
        plt.ylabel(objective_names[1], fontsize=12)
    else:
        x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
        y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
    
    plt.title('All Evaluated Points', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return plt.gcf()


def get_pareto_front_hull(x_vals, y_vals):
    """
    Get the points forming the lower-left boundary of the point cloud
    This extracts the Pareto front for minimization objectives
    
    Parameters:
    -----------
    x_vals : numpy.ndarray
        X coordinates (after any negation has been applied)
    y_vals : numpy.ndarray
        Y coordinates (after any negation has been applied)
        
    Returns:
    --------
    x_front : numpy.ndarray
        X coordinates of front points
    y_front : numpy.ndarray
        Y coordinates of front points
    """
    # Combine and sort by x coordinate (ascending)
    points = np.column_stack([x_vals, y_vals])
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]
    
    # Extract lower envelope (Pareto front for minimization)
    front_points = [sorted_points[0]]  # Start with leftmost point
    
    for point in sorted_points[1:]:
        # Add point if it has a lower y-value than the current front
        if point[1] < front_points[-1][1]:
            front_points.append(point)
    
    front_array = np.array(front_points)
    return front_array[:, 0], front_array[:, 1]


def get_boundary_hull(x_vals, y_vals):
    """
    Get all extreme boundary points of the point cloud using convex hull
    
    Parameters:
    -----------
    x_vals : numpy.ndarray
        X coordinates
    y_vals : numpy.ndarray
        Y coordinates
        
    Returns:
    --------
    x_boundary : numpy.ndarray
        X coordinates of boundary points (ordered around perimeter)
    y_boundary : numpy.ndarray
        Y coordinates of boundary points (ordered around perimeter)
    """
    # Combine points
    points = np.column_stack([x_vals, y_vals])
    
    # Compute convex hull
    hull = ConvexHull(points)
    
    # Extract hull points in order
    hull_points = points[hull.vertices]
    
    # Sort by angle from centroid to ensure proper ordering
    centroid = np.mean(hull_points, axis=0)
    angles = np.arctan2(hull_points[:, 1] - centroid[1], 
                        hull_points[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    
    hull_sorted = hull_points[sorted_idx]
    
    return hull_sorted[:, 0], hull_sorted[:, 1]


def get_concave_hull(x_vals, y_vals, k=1):
    """
    Get the concave (non-convex) boundary of points using k-nearest neighbors
    This follows the actual shape including concave regions
    
    Parameters:
    -----------
    x_vals : numpy.ndarray
        X coordinates
    y_vals : numpy.ndarray
        Y coordinates
    k : int, optional
        Number of nearest neighbors to consider. Lower = tighter fit, higher = smoother.
        Default: 10
        
    Returns:
    --------
    x_boundary : numpy.ndarray
        X coordinates of boundary points (ordered around perimeter)
    y_boundary : numpy.ndarray
        Y coordinates of boundary points (ordered around perimeter)
    """
    from scipy.spatial import distance_matrix
    
    points = np.column_stack([x_vals, y_vals])
    
    # Start with the point with minimum x (leftmost)
    current_idx = np.argmin(points[:, 0])
    boundary_indices = [current_idx]
    visited = {current_idx}
    
    while True:
        current_point = points[current_idx]
        
        # Calculate distances to all other points
        distances = np.sqrt(np.sum((points - current_point)**2, axis=1))
        
        # Get k nearest neighbors (excluding visited points)
        nearest_indices = np.argsort(distances)
        
        # Find the next unvisited neighbor
        next_idx = None
        for idx in nearest_indices[1:k+10]:  # Look at k+10 nearest
            if idx not in visited:
                next_idx = idx
                break
        
        # If we've come back to start or can't find next point, we're done
        if next_idx is None or next_idx == boundary_indices[0]:
            break
        
        # Move to next point
        boundary_indices.append(next_idx)
        visited.add(next_idx)
        current_idx = next_idx
        
        # Safety check to avoid infinite loops
        if len(boundary_indices) > len(points):
            break
    
    boundary_points = points[boundary_indices]
    return boundary_points[:, 0], boundary_points[:, 1]


def plot_pareto_front_only(file_paths, labels=None, objective_names=None, 
                           obj_indices=(0, 2), negate_objectives=(False, True),
                           figsize=(10, 6), show_hull=True, hull_type='pareto', k=10):
    """
    Plot only the Pareto-efficient points from multiple optimization runs
    
    Parameters:
    -----------
    file_paths : list of str
        List of paths to optimization_results_F.npy files
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
    show_hull : bool, optional
        If True, connects points with lines to show the front. Default: True
    hull_type : str, optional
        Type of boundary to extract:
        - 'pareto': Pareto front only (lower-left boundary)
        - 'convex': Convex hull (all extreme points, straight edges)
        - 'concave': Concave hull (follows actual curved boundary)
        Default: 'pareto'
    k : int, optional
        For concave hull: number of nearest neighbors (lower=tighter, higher=smoother).
        Default: 10
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    plt.figure(figsize=figsize)
    
    if labels is None:
        labels = []
        for fp in file_paths:
            name = fp.split('/')[-1].replace('_F.npy', '')
            labels.append(name)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    x_idx, y_idx = obj_indices
    x_negate, y_negate = negate_objectives
    
    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        try:
            F = np.load(file_path)
            
            # Extract objectives (apply negation for display)
            obj1 = -F[:, x_idx] if x_negate else F[:, x_idx]
            obj2 = -F[:, y_idx] if y_negate else F[:, y_idx]
            
            # Get boundary points based on hull_type
            if hull_type == 'convex':
                x_sorted, y_sorted = get_boundary_hull(obj1, obj2)
            elif hull_type == 'concave':
                x_sorted, y_sorted = get_concave_hull(obj1, obj2, k=k)
            else:  # 'pareto'
                x_sorted, y_sorted = get_pareto_front_hull(obj1, obj2)
            
            # Plot Pareto front
            plt.scatter(x_sorted, y_sorted, 
                       label=f'{label} ({len(x_sorted)} points)', 
                       alpha=0.8, 
                       s=80, 
                       color=colors[idx],
                       marker=markers[idx % len(markers)],
                       edgecolors='black',
                       linewidth=0.7,
                       zorder=3)
            
            # Connect boundary/front points if requested
            if show_hull:
                # For convex/concave hull, close the loop
                if hull_type in ['convex', 'concave']:
                    x_plot = np.append(x_sorted, x_sorted[0])
                    y_plot = np.append(y_sorted, y_sorted[0])
                    plt.plot(x_plot, y_plot, 
                            '-', 
                            alpha=0.5, 
                            linewidth=2,
                            color=colors[idx],
                            zorder=2)
                else:
                    plt.plot(x_sorted, y_sorted, 
                            '-', 
                            alpha=0.5, 
                            linewidth=2,
                            color=colors[idx],
                            zorder=2)
                    
        except FileNotFoundError:
            print(f"Warning: Could not find file {file_path}")
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
    
    if objective_names:
        plt.xlabel(objective_names[0], fontsize=12)
        plt.ylabel(objective_names[1], fontsize=12)
    else:
        x_label = f'Objective {x_idx}' + (' (negated)' if x_negate else '')
        y_label = f'Objective {y_idx}' + (' (negated)' if y_negate else '')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
    
    titles = {
        'pareto': 'Pareto Front Comparison',
        'convex': 'Convex Hull Boundary',
        'concave': 'Concave Hull Boundary'
    }
    title = titles.get(hull_type, 'Boundary Points')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return plt.gcf()

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
    
    fig = plot_pareto_front_only(
        file_paths_multi,
        # labels=labels,
        objective_names=objective_names,
        obj_indices=(0, 2),
        negate_objectives=(False, True),
        hull_type='concave'
    )
    
    # plt.savefig('opt_results/pareto_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig = plot_pareto_all_points(
        file_paths_multi,
        # labels=labels,
        objective_names=objective_names,
        obj_indices=(0, 2),
        negate_objectives=(False, True),
    )
    plt.show()

    
    # Example 3: Different objective combination (e.g., objectives 0 vs 1)
    fig = plot_pareto_front_only(
        file_paths_multi,
        # labels=labels,
        objective_names=['Mass (kg)', 'Buckling Strength (MN)'],
        obj_indices=(0, 1),
        negate_objectives=(False, True),
    )
    
    # plt.savefig('opt_results/pareto_comparison_obj_0_1.png', dpi=300, bbox_inches='tight')
    plt.show()