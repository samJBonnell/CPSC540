import numpy as np
import matplotlib.pyplot as plt
import argparse


def get_pareto_front(x_vals, y_vals, y_negate=False):
    """Extract Pareto front points and original indices."""
    points = np.column_stack([x_vals, y_vals])
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]

    front_points = [sorted_points[0]]
    front_indices = [sorted_idx[0]]

    for i in range(1, len(sorted_points)):
        if y_negate:
            if sorted_points[i, 1] > front_points[-1][1]:
                front_points.append(sorted_points[i])
                front_indices.append(sorted_idx[i])
        else:
            if sorted_points[i, 1] < front_points[-1][1]:
                front_points.append(sorted_points[i])
                front_indices.append(sorted_idx[i])

    front_points = np.array(front_points)
    return front_points[:, 0], front_points[:, 1], np.array(front_indices)


def plot_points(
    file_paths,
    labels,
    objective_names,
    obj_indices=(0, 1),
    negate_objectives=(False, True),
    pareto_only=True,
    interactive=False,
    figsize=(10, 6),
):
    """
    Plot multiple datasets with optional interactive point inspection.
    """

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']

    x_idx, y_idx = obj_indices
    x_negate, y_negate = negate_objectives

    datasets = []
    scatters = []

    # ---------------------------------------------------------
    # Load + plot datasets
    # ---------------------------------------------------------
    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        F = np.load(file_path)
        X = None
        if interactive:
            X = np.load(file_path.replace('_F.npy', '_X.npy'))

        x_vals = -F[:, x_idx] if x_negate else F[:, x_idx]
        y_vals = -F[:, y_idx] if y_negate else F[:, y_idx]

        if pareto_only:
            x_plot, y_plot, indices = get_pareto_front(x_vals, y_vals, y_negate)
        else:
            x_plot, y_plot = x_vals, y_vals
            indices = np.arange(len(x_vals))

        sc = ax.scatter(
            x_plot,
            y_plot,
            label=f"{label} ({len(x_plot)})",
            s=80 if pareto_only else 60,
            alpha=0.8,
            color=colors[i],
            marker=markers[i % len(markers)],
            edgecolors="black",
            linewidth=0.7,
            picker=True if interactive else False,
            zorder=3,
        )

        if pareto_only:
            ax.plot(x_plot, y_plot, color=colors[i], linewidth=2, alpha=0.5)

        # Attach metadata
        sc._dataset_id = i
        sc._indices = indices

        datasets.append(
            {
                "label": label,
                "X": X,
                "F": F,
                "x_vals": x_vals,
                "y_vals": y_vals,
            }
        )

        scatters.append(sc)

    # ---------------------------------------------------------
    # Labels & formatting
    # ---------------------------------------------------------
    ax.set_xlabel(objective_names[0], fontsize=12)
    ax.set_ylabel(objective_names[1], fontsize=12)
    ax.set_title("Pareto Front" if pareto_only else "All Points",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    # ---------------------------------------------------------
    # Interactive picker
    # ---------------------------------------------------------
    if interactive:
        textbox = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
            fontsize=9,
            family="monospace",
        )

        selected_point = [None]

        def on_pick(event):
            sc = event.artist
            dataset_id = sc._dataset_id
            plot_idx = event.ind[0]
            orig_idx = sc._indices[plot_idx]

            dataset = datasets[dataset_id]
            X = dataset["X"]
            F = dataset["F"]
            label = dataset["label"]

            # Remove old marker
            if selected_point[0] is not None:
                selected_point[0].remove()

            selected_point[0] = ax.scatter(
                dataset["x_vals"][orig_idx],
                dataset["y_vals"][orig_idx],
                s=200,
                c="red",
                marker="*",
                edgecolors="black",
                linewidth=2,
                zorder=10,
            )

            obj = F[orig_idx]
            design = X[orig_idx] if X is not None else None

            info = (
                f"MODEL: {label}\n"
                f"INDEX: {orig_idx}\n\n"
                f"OBJECTIVES:\n"
                f"  Mass:     {obj[0]:8.2f} kg\n"
                f"  Buckling: {-obj[1]:8.2f} MN\n"
            )

            if F.shape[1] > 2:
                info += f"  Stability:{-obj[2]:8.2f} MN\n"

            textbox.set_text(info)

            print("\n" + "=" * 70)
            print(f"MODEL: {label}")
            print(f"INDEX: {orig_idx}")
            print(f"Mass: {obj[0]:.2f} kg")
            print(f"Buckling: {-obj[1]:.2f} MN")
            if F.shape[1] > 2:
                print(f"Stability: {-obj[2]:.2f} MN")
            if design is not None:
                print(f"Design vector:\n{design}")
                print(f"Extract with: X_{label}[{orig_idx}]")

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", on_pick)

    plt.tight_layout()
    plt.show()


def plot_problem(problem, interactive=False):
    problems = {
        "U1": {
            "files": ["opt_results/mlp_U1_F.npy", "opt_results/cnn_U1_F.npy"],
            "plots": [
                (0, 1, ["Mass (kg)", "Buckling Strength (MN)"]),
            ],
        },
        "U2": {
            "files": ["opt_results/mlp_U2_F.npy", "opt_results/cnn_U2_F.npy"],
            "plots": [
                (0, 1, ["Mass (kg)", "Buckling Strength (MN)"]),
                (0, 2, ["Mass (kg)", r"Stability ($\Delta$MN)"]),
            ],
        },
        "C1": {
            "files": ["opt_results/mlp_C1_F.npy", "opt_results/cnn_C1_F.npy"],
            "plots": [
                (0, 1, ["Mass (kg)", "Buckling Strength (MN)"]),
            ],
        },
        "C2": {
            "files": ["opt_results/mlp_C2_F.npy", "opt_results/cnn_C2_F.npy"],
            "plots": [
                (0, 1, ["Mass (kg)", "Buckling Strength (MN)"]),
                (0, 2, ["Mass (kg)", r"Stability ($\Delta$MN)"]),
            ],
        },
    }

    if problem not in problems:
        raise ValueError(f"Unknown problem: {problem}")

    config = problems[problem]
    labels = ["MLP", "CNN"]

    for x_idx, y_idx, names in config["plots"]:
        plot_points(
            config["files"],
            labels,
            objective_names=names,
            obj_indices=(x_idx, y_idx),
            negate_objectives=(False, True),
            pareto_only=True,
            interactive=interactive,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="U1", choices=["U1", "U2", "C1", "C2"])
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    plot_problem(args.problem, args.interactive)

if __name__ == "__main__":
    main()