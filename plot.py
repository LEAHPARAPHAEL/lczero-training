import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_log_file(filepath):
    """
    Reads a single log file and extracts test steps and metrics.
    Returns a dictionary of dictionaries: data[metric_name][step_number] = value
    """
    step_pattern = re.compile(r"Test step\s+(\d+)\s*:")
    metric_pattern = re.compile(r">\s*(.+?)\s*=\s*([0-9\.eE\+\-]+)")

    data = defaultdict(dict)
    current_step = None

    if not os.path.exists(filepath):
        print(f"Warning: Placeholder or missing file skipped -> {filepath}")
        return data

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            step_match = step_pattern.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                continue

            if current_step is not None:
                metric_match = metric_pattern.search(line)
                if metric_match:
                    metric_name = metric_match.group(1).strip()
                    try:
                        value = float(metric_match.group(2))
                        data[metric_name][current_step] = value
                    except ValueError:
                        pass 

    return data

def plot_training_metrics(model_paths, metrics_to_plot=None, output_path=None):
    """
    Fetches data for all models and plots the requested metrics.
    Saves the plots to output_path if provided, otherwise displays them.
    """
    all_model_data = {}
    all_found_metrics = set()

    # 1. Parse all data
    for path in model_paths:
        model_name = os.path.splitext(os.path.basename(path))[0]
        model_data = parse_log_file(path)
        
        if model_data: 
            all_model_data[model_name] = model_data
            all_found_metrics.update(model_data.keys())

    if not all_model_data:
        print("No valid log files were found. Please check your model_paths.")
        return

    # 2. Determine which metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = sorted(list(all_found_metrics))

    # 3. Handle Output Directory
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # 4. Generate one plot per metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        metric_has_data = False

        for model_name, model_data in all_model_data.items():
            if metric in model_data:
                steps = sorted(model_data[metric].keys())
                values = [model_data[metric][step] for step in steps]
                
                plt.plot(steps, values, linestyle='-', linewidth=2, label=model_name)
                metric_has_data = True

        if metric_has_data:
            plt.title(f"Comparison: {metric}", fontsize=14, fontweight='bold')
            plt.xlabel("Training Step", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc="best")
            plt.tight_layout()
            
            if output_path:
                # Sanitize the metric name for the file system (e.g., "Policy Loss" -> "Policy_Loss.png")
                safe_metric_name = metric.replace(" ", "_").replace("/", "_").replace("\\", "_")
                save_path = os.path.join(output_path, f"{safe_metric_name}.png")
                
                plt.savefig(save_path, dpi=300)
                print(f"Saved: {save_path}")
                plt.close() # Close figure to free memory and avoid pop-up spam
            else:
                plt.show() # Only show pop-up if no output path was specified
        else:
            plt.close() 
            print(f"Skipped plotting '{metric}' (no data found in given logs).")

if __name__ == "__main__":
    # --- CONFIGURATION PARAMETERS ---
    
    # 1. List of paths to your models' log files
    MODELS_LOG_PATHS = [
        "/home/raph/leela/logs/Tx8.txt",
        "/home/raph/leela/logs/Mx2-Tx8.txt",
        "/home/raph/leela/logs/Mx4-Tx6.txt",
        "/home/raph/leela/logs/Mx6-Tx4.txt",
        "/home/raph/leela/logs/Mx8-Tx2.txt",
        "/home/raph/leela/logs/Mx10.txt",
    ]

    # 2. List of metrics to plot (Set to None to plot ALL found metrics)
    TARGET_METRICS = ["Policy Loss", "Policy Accuracy", "MSE Loss", "Moves Left Loss"]

    # 3. Directory where plots will be saved
    # If set to None, plots will open in pop-up windows instead.
    OUTPUT_PLOT_DIR = "/home/raph/leela/plots/"

    # --- EXECUTION ---
    plot_training_metrics(
        model_paths=MODELS_LOG_PATHS, 
        metrics_to_plot=TARGET_METRICS,
        output_path=OUTPUT_PLOT_DIR
    )