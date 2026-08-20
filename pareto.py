import argparse
import math
import os
import sys
import yaml
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_FILE_TITLE = (
    "chess_models_pareto"  # Static base name used for the saved PNG file
)
OUTPUT_DIR = "/home/raph/leela/plots"


def compute_pareto_front(models):
    """Computes Pareto optimal points (maximizing both NPS and Policy)."""
    sorted_models = sorted(
        models, key=lambda m: (m["nps"], m["policy"]), reverse=True
    )

    pareto_front = []
    max_policy = -float("inf")

    for model in sorted_models:
        if model["policy"] > max_policy:
            pareto_front.append(model)
            max_policy = model["policy"]

    return sorted(pareto_front, key=lambda m: m["nps"])


def generate_pareto_plot(yaml_file_path, file_title=DEFAULT_FILE_TITLE):
    # 1. Load YAML data
    if not os.path.exists(yaml_file_path):
        print(f"Error: File '{yaml_file_path}' not found.")
        sys.exit(1)

    with open(yaml_file_path, "r") as f:
        data = yaml.safe_load(f)

    models = []
    for name, metrics in data.items():
        models.append(
            {
                "name": str(name),
                "policy": float(metrics["policy"]),
                "nps": float(metrics["nps"]),
            }
        )

    if not models:
        print("Error: No valid model data found in YAML.")
        sys.exit(1)

    # 2. Compute Pareto Front
    pareto_models = compute_pareto_front(models)

    # 3. Setup Plot Aesthetics
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

    # 4. Plot Points (each model gets a unique color, standard circular marker)
    cmap = (
        plt.cm.tab10(range(len(models)))
        if len(models) <= 10
        else plt.cm.tab20(range(len(models)))
    )

    for i, model in enumerate(models):
        ax.scatter(
            model["nps"],
            model["policy"],
            color=cmap[i % len(cmap)],
            s=75,
            marker="o",
            edgecolors="black",
            linewidth=0.8,
            zorder=4,
        )

    # 5. Position Model Names Close to Points (with localized collision staggering)
    nps_vals = [m["nps"] for m in models]
    pol_vals = [m["policy"] for m in models]
    nps_min, nps_max = min(nps_vals), max(nps_vals)
    pol_min, pol_max = min(pol_vals), max(pol_vals)
    nps_range = (nps_max - nps_min) or 1.0
    pol_range = (pol_max - pol_min) or 1.0

    placed_points = []
    for model in models:
        norm_x = (model["nps"] - nps_min) / nps_range
        norm_y = (model["policy"] - pol_min) / pol_range

        cluster_index = 0
        for px, py in placed_points:
            if math.hypot(norm_x - px, norm_y - py) < 0.04:
                cluster_index += 1

        placed_points.append((norm_x, norm_y))

        if cluster_index == 0:
            dx, dy = 7, 4
        elif cluster_index % 2 == 1:
            dx, dy = 7, 4 + (cluster_index // 2 + 1) * 12
        else:
            dx, dy = 7, 4 - (cluster_index // 2) * 13

        ax.annotate(
            model["name"],
            xy=(model["nps"], model["policy"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color="#111111",
            fontweight="medium",
            zorder=5,
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)
            ],
        )

    # 6. Apply Margins and Capture Absolute Axes Limits
    ax.margins(x=0.06, y=0.08)
    ax.autoscale_view()
    x_min_lim, x_max_lim = ax.get_xlim()
    y_min_lim, y_max_lim = ax.get_ylim()

    # 7. Plot Pareto Frontier & Dominated Region Flushed to the Axes
    pareto_x = [m["nps"] for m in pareto_models]
    pareto_y = [m["policy"] for m in pareto_models]

    # Extend boundary all the way to the left axis edge (x_min_lim)
    extended_fill_x = [x_min_lim] + pareto_x
    extended_fill_y = [pareto_y[0]] + pareto_y

    # Translucent region spanning completely down to the bottom axis (y_min_lim)
    ax.fill_between(
        extended_fill_x,
        extended_fill_y,
        y2=y_min_lim,
        color="#d62728",
        alpha=0.12,
        zorder=1,
        label="Dominated Region",
    )

    # Dashed Pareto Frontier line
    ax.plot(
        pareto_x,
        pareto_y,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        zorder=2,
        label="Pareto Frontier",
    )

    # Lock limits to prevent fill_between from artificially expanding bounds
    ax.set_xlim(x_min_lim, x_max_lim)
    ax.set_ylim(y_min_lim, y_max_lim)

    # 8. Axes & Legend Formatting
    ax.set_xlabel(
        "Nodes per second (batch size = 32)", fontsize=11, labelpad=10
    )
    ax.set_ylabel("Policy Accuracy (%)", fontsize=11, labelpad=10)
    ax.yaxis.set_major_formatter("{x:.2f}%")

    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=10,
    )

    plt.tight_layout()

    # 9. Save Plot (Deterministic Static Filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_title = "".join(
        c for c in file_title if c.isalnum() or c in ("-", "_")
    ).strip("_")
    output_path = os.path.join(OUTPUT_DIR, f"{clean_title}.png")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Plot successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Pareto front for chess models."
    )
    parser.add_argument(
        "--yaml_file",
        "-y",
        type=str,
        required=True,
        help="Path to the metrics YAML file",
    )
    parser.add_argument(
        "--file_title",
        "-t",
        type=str,
        default=DEFAULT_FILE_TITLE,
        help="Static base name for the output file (defaults to 'chess_models_pareto')",
    )
    args = parser.parse_args()

    generate_pareto_plot(args.yaml_file, file_title=args.file_title)