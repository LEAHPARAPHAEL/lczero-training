#!/usr/bin/env python3
import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def format_latex_str(text: str) -> str:
    """Escape LaTeX special characters unless it is already a LaTeX math expression."""
    if text.startswith("$") and text.endswith("$"):
        return text
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def colorize_score(score: float) -> str:
    """Format a score with LaTeX bolding and color coding."""
    score_str = f"{score:.1f}"
    if score < 40.0:
        color = "scoreRed"
    elif score < 50.0:
        color = "scoreOrange"
    elif score < 60.0:
        color = "scoreBlue"
    else:
        color = "scoreGreen"
    return f"\\textcolor{{{color}}}{{\\textbf{{{score_str}}}}}"


def calculate_elo_diff(score_pct: float) -> str:
    """
    Calculate estimated Elo difference from a win percentage (0 to 100)
    using the standard logistic Elo formula:
    Delta Elo = 400 * log10(P / (1 - P))
    """
    p = score_pct / 100.0
    if p <= 0.001:
        return "-800.0"
    if p >= 0.999:
        return "+800.0"
    
    elo = 400.0 * math.log10(p / (1.0 - p))
    sign = "+" if elo > 0 else ""
    return f"{sign}{elo:.1f}"


def find_and_parse_match(config_dir: Path, model_a: str, model_b: str) -> Optional[Tuple[float, float]]:
    """
    Look for match config file between model_a and model_b in config_dir.
    Returns (score_a, score_b) normalized out of 100.
    """
    possible_names = [
        f"{model_a}_VS_{model_b}.json",
        f"{model_a}_vs_{model_b}.json",
        f"{model_b}_VS_{model_a}.json",
        f"{model_b}_vs_{model_a}.json",
        f"{model_a}_VS_{model_b}",
        f"{model_b}_VS_{model_a}",
    ]

    target_file = None
    for name in possible_names:
        candidate = config_dir / name
        if candidate.exists():
            target_file = candidate
            break

    # Fallback: case-insensitive scan of the folder
    if not target_file:
        for file in config_dir.glob("*.json"):
            stem = file.stem.lower()
            if (f"{model_a.lower()}_vs_{model_b.lower()}" == stem or 
                f"{model_b.lower()}_vs_{model_a.lower()}" == stem):
                target_file = file
                break

    if not target_file:
        return None

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = data.get("stats", {})
    if not stats:
        return None

    # Retrieve match key (e.g., "sparse vs dense" or "dense vs sparse")
    stat_key = next(iter(stats.keys()))
    match_data = stats[stat_key]

    wins = match_data.get("wins", 0)
    losses = match_data.get("losses", 0)
    draws = match_data.get("draws", 0)
    total_games = wins + losses + draws

    if total_games == 0:
        return (50.0, 50.0)

    # Determine which engine corresponds to the first part of the stats key
    first_engine_in_key = stat_key.split(" vs ")[0].strip()
    
    # Calculate score for the first engine in stats
    score_first = (wins + 0.5 * draws) / total_games * 100.0
    score_second = 100.0 - score_first

    if model_a.lower() == first_engine_in_key.lower():
        return (score_first, score_second)
    else:
        return (score_second, score_first)


def build_tournament_matrix(models: List[str], config_dir: Path):
    baseline_model = models[0]
    n = len(models)
    
    # Pairwise matrix: matrix[A][B] = score of A against B
    matrix: Dict[str, Dict[str, Optional[float]]] = {m: {other: None for other in models} for m in models}

    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = models[i], models[j]
            result = find_and_parse_match(config_dir, m1, m2)
            if result:
                matrix[m1][m2], matrix[m2][m1] = result
            else:
                print(f"[Warning] No match summary found between '{m1}' and '{m2}'.", file=sys.stderr)
                matrix[m1][m2] = 50.0
                matrix[m2][m1] = 50.0

    # Calculate average scores across all opponents
    avg_scores: Dict[str, float] = {}
    for m in models:
        scores = [matrix[m][other] for other in models if other != m and matrix[m][other] is not None]
        avg_scores[m] = sum(scores) / len(scores) if scores else 50.0

    # Sort models by average score descending
    ranked_models = sorted(models, key=lambda m: avg_scores[m], reverse=True)

    return ranked_models, baseline_model, matrix, avg_scores


def generate_latex_table(models: List[str], config_dir: Path, caption: str, label: str) -> str:
    ranked_models, baseline_model, matrix, avg_scores = build_tournament_matrix(models, config_dir)
    
    num_cols = 4 + len(ranked_models)
    col_align = "c" * num_cols

    lines = []
    lines.append(r"\definecolor{scoreRed}{HTML}{C0392B}    % Muted Red")
    lines.append(r"\definecolor{scoreOrange}{HTML}{D4AC0D} % Muted Orange")
    lines.append(r"\definecolor{scoreBlue}{HTML}{2980B9}   % Muted Blue")
    lines.append(r"\definecolor{scoreGreen}{HTML}{27AE60}  % Muted Green")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    lines.append(r"\toprule")

    # Header Row
    headers = [r"rank", r"model", r"Elo", r"score"]
    for m in ranked_models:
        headers.append(f"\\rotatebox{{90}}{{{format_latex_str(m)}}}")
    lines.append("  " + " & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    # Table Body
    for rank, m in enumerate(ranked_models, start=1):
        row = []
        row.append(str(rank))
        row.append(format_latex_str(m))

        # Elo Diff against baseline model
        if m == baseline_model:
            row.append(r"\textbf{+0.0}")
        else:
            score_vs_base = matrix[m][baseline_model]
            if score_vs_base is not None:
                elo_diff = calculate_elo_diff(score_vs_base)
                row.append(f"\\textbf{{{elo_diff}}}")
            else:
                row.append("-")

        # Average Score
        row.append(colorize_score(avg_scores[m]))

        # Head-to-Head columns
        for opp in ranked_models:
            if m == opp:
                row.append("-")
            else:
                s = matrix[m][opp]
                row.append(colorize_score(s) if s is not None else "-")

        lines.append("  " + " & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate a LaTeX tournament summary table from fastchess configs.")
    parser.add_argument("models", nargs="+", help="List of model names. The FIRST model is used as the Elo baseline.")
    parser.add_argument("-c", "--config-dir", default="./configs", help="Directory containing the fastchess JSON summaries.")
    parser.add_argument("-o", "--output", default=None, help="Optional output .tex file path.")
    parser.add_argument("--caption", default=r"\textbf{Tournament results between models.}", help="Table caption.")
    parser.add_argument("--label", default="tab:tournament_results", help="Table label.")

    args = parser.parse_args()
    config_dir = Path(args.config_dir)

    if not config_dir.exists():
        print(f"Error: Config directory '{config_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    latex_code = generate_latex_table(args.models, config_dir, args.caption, args.label)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(latex_code + "\n")
        print(f"LaTeX table saved to {args.output}")
    else:
        print(latex_code)


if __name__ == "__main__":
    main()