"""
Reads every elo_ratings_summary_*.docx in the current folder, merges the per-k
ELO tables across files, and produces two Leiden-baseline-relative figures:

  - elo_progression_avg_by_k_vs_baseline.png  (single panel, average across 4 metrics)
  - elo_progression_by_k_vs_baseline.png      (2x2 panel, one per metric, SHARED y-axis)

The 2x2 panel uses a single shared y-axis range across all four metrics so that
cross-metric magnitude comparisons are visually honest -- a +400 line in
Directness sits at the same vertical height as a +400 line in Diversity.

Usage:
    python plot_elo_vs_baseline_from_docx.py

Baseline: Leiden ('c0'). To change, edit BASELINE_METHOD below.
"""

import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from docx import Document

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASELINE_METHOD = "c0"   # 'c0' = Leiden. Change to 'nac2', 'spectral', etc. if needed.

label_map = {
    "c0": "Leiden", "spectral": "Spectral", "score": "SCORE",
    "nac1": "NAC1", "nac2": "NAC2",
}
color_map = {
    "c0": "#1f77b4", "spectral": "#ff7f0e", "score": "#2ca02c",
    "nac1": "#d62728", "nac2": "#9467bd",
}
methods_order = ["c0", "spectral", "score", "nac1", "nac2"]
metrics = ["Comprehensiveness", "Diversity", "Empowerment", "Directness"]

DOCX_LABEL_TO_CODE = {
    "leiden": "c0",
    "spectral": "spectral",
    "score": "score",
    "nac1": "nac1",
    "nac2": "nac2",
}

K_HEADER_RE = re.compile(r"ELO Ratings Summary\s*-\s*K\s*=\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Docx parsing
# ---------------------------------------------------------------------------
def _is_elo_header_row(row_texts: List[str]) -> bool:
    if not row_texts:
        return False
    first = row_texts[0].strip().lower()
    if first != "method":
        return False
    if len(row_texts) >= 2 and row_texts[1].strip().lower().startswith("comprehensiv"):
        return True
    return False


def _resolve_metric_column(header_text: str) -> str:
    h = header_text.strip().lower()
    if h.startswith("comprehensiv"):
        return "Comprehensiveness"
    if h.startswith("divers"):
        return "Diversity"
    if h.startswith("empower"):
        return "Empowerment"
    if h.startswith("direct"):
        return "Directness"
    return ""


def parse_elo_from_docx(path: str) -> Dict[int, Dict[str, Dict[str, float]]]:
    doc = Document(path)
    result: Dict[int, Dict[str, Dict[str, float]]] = {}
    current_k: int = None
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}", 1)[-1]
        if tag == "p":
            text = "".join(t.text or "" for t in child.iter() if t.tag.endswith("}t"))
            m = K_HEADER_RE.search(text)
            if m:
                current_k = int(m.group(1))
        elif tag == "tbl":
            rows = []
            for tr in child.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"):
                row = []
                for tc in tr.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"):
                    cell_text = "".join(
                        t.text or "" for t in tc.iter() if t.tag.endswith("}t")
                    )
                    row.append(cell_text)
                rows.append(row)
            if not rows or not _is_elo_header_row(rows[0]):
                continue
            if current_k is None:
                print(f"  [warn] ELO-looking table in {os.path.basename(path)} "
                      f"has no preceding 'K=...' header — skipping.")
                continue
            header = [h.strip() for h in rows[0]]
            metric_cols: List[Tuple[int, str]] = []
            for col_idx, h in enumerate(header):
                metric_name = _resolve_metric_column(h)
                if metric_name:
                    metric_cols.append((col_idx, metric_name))
            k_table: Dict[str, Dict[str, float]] = {m: {} for m in metrics}
            for data_row in rows[1:]:
                if not data_row:
                    continue
                method_label = data_row[0].strip().lower()
                method_code = DOCX_LABEL_TO_CODE.get(method_label)
                if method_code is None:
                    continue
                for col_idx, metric_name in metric_cols:
                    if col_idx >= len(data_row):
                        continue
                    raw = data_row[col_idx].strip()
                    try:
                        k_table[metric_name][method_code] = float(raw)
                    except ValueError:
                        pass
            if any(k_table[m] for m in metrics):
                result[current_k] = k_table
            current_k = None
    return result


def collect_all_elo(folder: str = ".") -> Dict[int, Dict[str, Dict[str, float]]]:
    pattern = os.path.join(folder, "elo_ratings_summary_*.docx")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No elo_ratings_summary_*.docx files found in {os.path.abspath(folder)}."
        )
    merged: Dict[int, Dict[str, Dict[str, float]]] = {}
    print(f"Found {len(files)} docx file(s):")
    for path in files:
        parsed = parse_elo_from_docx(path)
        ks_found = sorted(parsed.keys())
        print(f"  - {os.path.basename(path)}: k = {ks_found}")
        for k, table in parsed.items():
            if k in merged:
                print(f"    [warn] k={k} already loaded from an earlier file; "
                      f"overwriting with values from {os.path.basename(path)}.")
            merged[k] = table
    if not merged:
        raise RuntimeError("No ELO tables parsed from any docx file.")
    print(f"Merged ELO tables for k = {sorted(merged.keys())}")
    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_avg_vs_baseline(all_elo: Dict[int, Dict[str, Dict[str, float]]],
                         out_path: str) -> None:
    k_values = sorted(all_elo.keys())
    baseline_avg = {}
    for k in k_values:
        rs = [all_elo[k][m][BASELINE_METHOD]
              for m in metrics
              if BASELINE_METHOD in all_elo[k].get(m, {})]
        if rs:
            baseline_avg[k] = sum(rs) / len(rs)

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in methods_order:
        xs, ys = [], []
        for k in k_values:
            rs = [all_elo[k][m][method]
                  for m in metrics
                  if method in all_elo[k].get(m, {})]
            if rs and k in baseline_avg:
                xs.append(k)
                ys.append(sum(rs) / len(rs) - baseline_avg[k])
        if not ys:
            continue
        is_baseline = (method == BASELINE_METHOD)
        ax.plot(
            xs, ys,
            marker='o',
            linewidth=3.5 if is_baseline else 2.5,
            linestyle='--' if is_baseline else '-',
            markersize=11,
            label=f"{label_map[method]} (baseline)" if is_baseline else label_map[method],
            color=color_map[method],
        )
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_title(
        f"Average ELO Relative to {label_map[BASELINE_METHOD]} Baseline",
        fontsize=17, fontweight='bold',
    )
    ax.set_xlabel("Number of Communities", fontsize=14)
    ax.set_ylabel(f"Δ ELO vs {label_map[BASELINE_METHOD]} (4-metric avg)", fontsize=14)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_per_metric_vs_baseline(all_elo: Dict[int, Dict[str, Dict[str, float]]],
                                out_path: str) -> None:
    """2x2 per-metric panel with a SHARED y-axis across all four panels."""
    k_values = sorted(all_elo.keys())

    # First pass: compute all delta values so we can derive a shared y-range.
    per_metric_lines: Dict[str, Dict[str, Tuple[List[int], List[float]]]] = {m: {} for m in metrics}
    all_ys: List[float] = []
    for metric in metrics:
        baseline_per_k = {
            k: all_elo[k][metric][BASELINE_METHOD]
            for k in k_values
            if BASELINE_METHOD in all_elo[k].get(metric, {})
        }
        for method in methods_order:
            xs, ys = [], []
            for k in k_values:
                if (method in all_elo[k].get(metric, {})) and (k in baseline_per_k):
                    xs.append(k)
                    ys.append(all_elo[k][metric][method] - baseline_per_k[k])
            if ys:
                per_metric_lines[metric][method] = (xs, ys)
                all_ys.extend(ys)

    # Pad the global y-range by 8% so lines don't touch the panel edges.
    if all_ys:
        y_lo, y_hi = min(all_ys), max(all_ys)
        pad = max(20.0, 0.08 * (y_hi - y_lo))
        y_lim = (y_lo - pad, y_hi + pad)
    else:
        y_lim = None

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    for idx, metric in enumerate(metrics):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        for method in methods_order:
            if method not in per_metric_lines[metric]:
                continue
            xs, ys = per_metric_lines[metric][method]
            is_baseline = (method == BASELINE_METHOD)
            ax.plot(
                xs, ys,
                marker='o',
                linewidth=3.5 if is_baseline else 2.5,
                linestyle='--' if is_baseline else '-',
                markersize=10,
                label=f"{label_map[method]} (baseline)" if is_baseline else label_map[method],
                color=color_map[method],
            )
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_title(metric, fontweight='bold', fontsize=18)
        ax.set_xlabel("Number of Communities", fontsize=14)
        # Only put y-label on the left column for a cleaner look.
        if col == 0:
            ax.set_ylabel(f"Δ ELO vs {label_map[BASELINE_METHOD]}", fontsize=14)
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3)
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        if idx == 0:
            ax.legend(loc='best', fontsize=12)
    fig.suptitle(
        f"ELO by Metric Relative to {label_map[BASELINE_METHOD]} Baseline",
        fontsize=20, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "Output")
    os.makedirs(output_dir, exist_ok=True)
    all_elo = collect_all_elo(output_dir)
    plot_avg_vs_baseline(all_elo, os.path.join(output_dir, "elo_progression_avg_by_k_vs_baseline.png"))
    plot_per_metric_vs_baseline(all_elo, os.path.join(output_dir, "elo_progression_by_k_vs_baseline.png"))
    print("Done.")
