import re

def beautify_latex_table(
    latex_code: str,
    caption_name: str = "Table Caption",
    midrule_lines: list = None,
    colored_rows: list = None,
) -> str:
    """
    Beautifies LaTeX code from DataFrame.to_latex() with added table environment,
    midrules, row coloring, and bold headers. Fixes the header formatting bug.

    Parameters:
    - latex_code (str): LaTeX string from DataFrame.to_latex()
    - caption_name (str): Caption text for the table
    - midrule_lines (list): List of row indices (0-based) after which to insert \midrule
    - colored_rows (list): List of tuples (row_index, color_name) for applying \rowcolor

    Returns:
    - str: Formatted LaTeX table string
    """
    midrule_lines = midrule_lines or []
    colored_rows = colored_rows or []

    lines = latex_code.strip().split("\n")

    # Extract header index
    header_idx = next(i for i, line in enumerate(lines) if '&' in line and '\\\\' in line)

    # Bold headers correctly (exclude \\ from bolding)
    header_line = lines[header_idx]
    match = re.match(r"(.*?\\\\)", header_line.strip())
    if match:
        content = match.group(1).replace('\\\\', '')
        headers = [f"\\textbf{{{col.strip()}}}" for col in content.split("&")]
        lines[header_idx] = " & ".join(headers) + " \\\\"

    # Insert midrules and row colors (process from bottom to top to avoid index shifts)
    data_start = header_idx + 2  # skip \midrule after header

    for row_idx, color in sorted(colored_rows, key=lambda x: -x[0]):
        insert_idx = data_start + row_idx
        lines.insert(insert_idx, f"\\rowcolor{{{color}}}")

    for mid_idx in sorted(midrule_lines, reverse=True):
        insert_idx = data_start + mid_idx
        lines.insert(insert_idx, "\\midrule")

    # Wrap with table environment
    table_env = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        *lines,
        f"\\caption{{{caption_name}}}",
        "\\end{table}"
    ]

    return "\n".join(table_env)
import re

def beautify_latex_table_resizable(
    latex_code: str,
    caption_name: str = "Table Caption",
    midrule_lines: list = None,
    colored_rows: list = None,
    group_insertions: list = None,
    resize_to_textwidth: bool = True,
) -> str:
    """
    Beautifies LaTeX code from DataFrame.to_latex() with enhancements:
    - Adds table environment, caption, and bold headers
    - Supports \midrule insertion with priority before group headers
    - Supports \rowcolor for specific rows
    - Supports \multicolumn-based subsection labels with background
    - Optionally wraps content in \resizebox{\textwidth}{!}{...}

    Parameters:
    - latex_code (str): LaTeX string from DataFrame.to_latex()
    - caption_name (str): Caption text for the table
    - midrule_lines (list): Row indices after which to insert \midrule
    - colored_rows (list): List of tuples (row_idx, color_name) to color rows
    - group_insertions (list): List of (row_idx, color, label) for section headers
    - resize_to_textwidth (bool): Wrap with \resizebox if True

    Returns:
    - str: Fully formatted LaTeX table
    """
    midrule_lines = midrule_lines or []
    colored_rows = colored_rows or []
    group_insertions = group_insertions or []

    lines = latex_code.strip().split("\n")

    # Find and bold the header row
    header_idx = next(i for i, line in enumerate(lines) if '&' in line and '\\\\' in line)
    match = re.match(r"(.*?)(\\\\)", lines[header_idx].strip())
    if match:
        content = match.group(1)
        headers = [f"\\textbf{{{col.strip()}}}" for col in content.split("&")]
        lines[header_idx] = " & ".join(headers) + " \\\\"

    data_start = header_idx + 2  # after header and its \midrule

    # Combine all insertions with priority: midrule < rowcolor < group
    insert_actions = []

    for idx in midrule_lines:
        insert_actions.append((idx, "midrule", None))
    for idx, color in colored_rows:
        insert_actions.append((idx, "rowcolor", color))
    for idx, color, label in group_insertions:
        insert_actions.append((idx, "group", (color, label)))

    priority = {"midrule": 0, "rowcolor": 1, "group": 2}

    for idx, kind, val in sorted(insert_actions, key=lambda x: (x[0], priority[x[1]]), reverse=True):
        insert_at = data_start + idx
        if kind == "midrule":
            lines.insert(insert_at, "\\midrule")
        elif kind == "rowcolor":
            lines.insert(insert_at, f"\\rowcolor{{{val}}}")
        elif kind == "group":
            color, label = val
            n_cols = lines[header_idx].count("&") + 1
            group_line = f"\\multicolumn{{{n_cols}}}{{l}}{{\\cellcolor{{{color}}} \\textit{{{label}}}}} \\\\"
            lines.insert(insert_at, group_line)

    # Wrap with \resizebox if needed
    if resize_to_textwidth:
        lines = ["\\resizebox{\\textwidth}{!}{%", *lines, "}"]

    # Wrap in table environment
    return "\n".join([
        "\\begin{table}[h!]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        *lines,
        f"\\caption{{{caption_name}}}",
        "\\end{table}"
    ])
