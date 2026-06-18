import json
import os
import sys

def escape_latex(text):
    """Escapes underscores and other characters for LaTeX layout safety."""
    return text.replace('_', r'\_').replace('%', r'\%')

def get_color_macro(score):
    """Determines the color tag based on your performance score thresholds."""
    if score >= 60.0:
        return "scoreGreen"
    elif score >= 50.0:
        return "scoreBlue"
    elif score >= 40.0:
        return "scoreOrange"
    else:
        return "scoreRed"

def parse_fastchess_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 1. Collect all unique engine names from the file metadata or stats keys
    engines = [e['name'] for e in data.get('engines', [])]
    stats = data.get('stats', {})
    
    if not engines:
        engines_set = set()
        for key in stats.keys():
            parts = key.split(' vs ')
            if len(parts) == 2:
                engines_set.update(parts)
        engines = list(engines_set)

    # 2. Build the bidirectional round-robin matchup matrix
    matrix = {e: {} for e in engines}
    
    for e1 in engines:
        for e2 in engines:
            if e1 == e2:
                continue
            
            # Check standard combination "A vs B"
            key = f"{e1} vs {e2}"
            if key in stats:
                w = stats[key]['wins']
                l = stats[key]['losses']
                d = stats[key]['draws']
                total = w + l + d
                if total > 0:
                    matrix[e1][e2] = ((w + 0.5 * d) / total) * 100
            else:
                # Check inverse combination "B vs A" 
                key_rev = f"{e2} vs {e1}"
                if key_rev in stats:
                    w = stats[key_rev]['wins']
                    l = stats[key_rev]['losses']
                    d = stats[key_rev]['draws']
                    total = w + l + d
                    if total > 0:
                        # Wins for E2 are Losses for E1, and vice versa
                        matrix[e1][e2] = ((l + 0.5 * d) / total) * 100

    # 3. Calculate macro-average score across all encountered opponents
    avg_scores = {}
    for e in engines:
        scores = list(matrix[e].values())
        avg_scores[e] = sum(scores) / len(scores) if scores else 0.0
        
    # 4. Sort engines based on descending average scores
    sorted_engines = sorted(engines, key=lambda x: avg_scores[x], reverse=True)

    # 5. Extract configuration parameters to construct a descriptive caption
    rounds = data.get('rounds', '?')
    games_per_round = data.get('games', 1)
    try:
        total_games = int(rounds) * int(games_per_round)
    except ValueError:
        total_games = '?'

    tc_desc = "Simulations"
    if 'engines' in data and len(data['engines']) > 0:
        limit = data['engines'][0].get('limit', {})
        if 'nodes' in limit and limit['nodes'] > 0:
            tc_desc = f"{limit['nodes']} nodes per move"
        elif 'tc' in limit and limit['tc'].get('time', 0) > 0:
            tc = limit['tc']
            tc_desc = f"time control {tc['time']/1000:g}s + {tc.get('increment', 0)}ms"

    caption_str = f"\\caption{{\\textbf{{Tournament with {tc_desc} and {total_games} games per match}}}}"

    # =========================================================================
    # 6. Generate LaTeX String Output
    # =========================================================================
    num_opponents = len(sorted_engines)
    latex = []
    latex.append("% Ensure \\usepackage{xcolor} and \\usepackage{float} are in your preamble")
    latex.append("\\definecolor{scoreRed}{HTML}{C0392B}    % Muted Red")
    latex.append("\\definecolor{scoreOrange}{HTML}{D4AC0D} % Muted Orange")
    latex.append("\\definecolor{scoreBlue}{HTML}{2980B9}   % Muted Blue")
    latex.append("\\definecolor{scoreGreen}{HTML}{27AE60}  % Muted Green\n")
    
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append(caption_str)
    latex.append(f"\\label{{tab:arena_{os.path.basename(json_path).split('.')[0]}}}")
    latex.append("\\resizebox{\\textwidth}{!}{%")
    latex.append(f"\\begin{{tabular}}{{l c *{{{num_opponents}}}{{c}}}}")
    latex.append("\\toprule")
    
    # Header Row Construction
    header_cols = [f"\\rotatebox{{45}}{{{escape_latex(e)}}}" for e in sorted_engines]
    latex.append(f"  & \\rotatebox{{45}}{{Score}} & " + " & ".join(header_cols) + " \\\\")
    latex.append("\\midrule")
    
    # Body Row Construction
    for e1 in sorted_engines:
        row_str = f"{escape_latex(e1)}"
        
        # Format Average Score Column
        avg = avg_scores[e1]
        row_str += f" & \\textcolor{{{get_color_macro(avg)}}}{{\\textbf{{{avg:.1f}}}}}"
        
        # Matchup Data Points
        for e2 in sorted_engines:
            if e1 == e2:
                row_str += " & -"
            else:
                score = matrix[e1].get(e2)
                if score is None:
                    row_str += " & -"
                else:
                    row_str += f" & \\textcolor{{{get_color_macro(score)}}}{{{score:.1f}}}"
                    
        row_str += " \\\\"
        latex.append(row_str)
        
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}%")
    latex.append("}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_table.py <path_to_fastchess_json>")
        sys.exit(1)
        
    json_input = sys.argv[1]
    if not os.path.exists(json_input):
        print(f"Error: File '{json_input}' not found.")
        sys.exit(1)
        
    print(parse_fastchess_json(json_input))