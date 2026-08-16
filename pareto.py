import argparse
import datetime
import os
import sys
import yaml
import matplotlib.pyplot as plt


def compute_pareto_front(models):
    """Computes Pareto optimal points (maximizing both NPS and Policy).

    Returns models for which no other model has both strictly greater or equal
    NPS and Policy accuracy.
    """
    # Sort models by NPS descending, then Policy descending
    sorted_models = sorted(
        models, key=lambda m: (m["nps"], m["policy"]), reverse=True
    )

    pareto_front = []
    max_policy = -float("inf")

    for model in sorted_models:
        # A point is Pareto optimal if its policy is strictly higher than
        # any model seen so far with higher or equal NPS.
        if model["policy"] > max_policy:
            pareto_front.append(model)
            max_policy = model["policy"]

    # Re-sort Pareto front by NPS ascending for drawing the frontier line
    return sorted(pareto_front, key=lambda m: m["nps"])


def generate_pareto_plot(yaml_file_path):
    # 1. Load YAML data
    if not os.path.exists(yaml_file_path):
        print(f"Error: File '{yaml_file_path}' not found.")
        sys.exit(1)

    with open(yaml_file_path, "r") as f:
        data = yaml.safe_load(f)

    models = []
    for name, metrics in data.items():
        models.append(
            {"name": name, "policy": metrics["policy"], "nps": metrics["nps"]}
        )

    if not models:
        print("Error: No valid model data found in YAML.")
        sys.exit(1)

    # 2. Compute Pareto Front
    pareto_models = compute_pareto_front(models)
    pareto_names = {m["name"] for m in pareto_models}

    # 3. Setup Plot Aesthetics
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Generate a distinct color palette for models
    cmap = plt.cm.get_cmap("tab20", len(models))

    # 4. Plot Non-Pareto and Pareto Points
    for i, model in enumerate(models):
        is_pareto = model["name"] in pareto_names
        color = cmap(i)

        ax.scatter(
            model["nps"],
            model["policy"],
            color=color,
            s=120 if is_pareto else 60,
            marker="*" if is_pareto else "o",
            edgecolors="black" if is_pareto else "none",
            linewidth=1.2 if is_pareto else 0,
            zorder=4 if is_pareto else 3,
            label=f"{model['name']} ('Pareto')" if is_pareto else model["name"],
        )

    # 5. Plot Pareto Front Line (Step-wise to illustrate efficiency boundary)
    pareto_x = [m["nps"] for m in pareto_models]
    pareto_y = [m["policy"] for m in pareto_models]

    ax.plot(
        pareto_x,
        pareto_y,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        zorder=2,
        label="Pareto Frontier",
    )

    # 6. Formatting Labels, Title, and Axes
    ax.set_title(
        "Chess Models: Policy Accuracy vs. Inference Speed",
        fontsize=14,
        pad=15,
        weight="bold",
    )
    ax.set_xlabel("Nodes per second (batch size = 32)", fontsize=11, labelpad=10)
    ax.set_ylabel("Policy Accuracy (%)", fontsize=11, labelpad=10)

    # Format Y-axis to show percentages cleanly
    ax.yaxis.set_major_formatter("{x:.1f}%")

    # Put legend outside the plot area on the right to preserve readability
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        fontsize=9,
        title="Models",
        title_fontsize="10",
    )

    plt.tight_layout()

    # 7. Save Output with Unique Filename
    output_dir = "/home/raph/leela/plots"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"pareto_front_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Plot successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Pareto front for chess models."
    )
    parser.add_argument(
        "--yaml_file", "-y", type=str, help="Path to the metrics YAML file"
    )
    args = parser.parse_args()

    generate_pareto_plot(args.yaml_file)