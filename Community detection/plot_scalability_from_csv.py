"""
Standalone scalability plot. Reads scalability_timings.csv and produces a
2-panel chart (linear axis + log-log axis) of community-detection time vs N.

Does not re-run the timing experiment. Run any time the CSV updates.

Usage:
    python plot_scalability_from_csv.py
"""

import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt

label_map = {"c0": "Leiden", "spectral": "Spectral", "score": "SCORE",
             "nac1": "NAC1", "nac2": "NAC2"}
color_map = {"c0": "#1f77b4", "spectral": "#ff7f0e", "score": "#2ca02c",
             "nac1": "#d62728", "nac2": "#9467bd"}
method_order = ["c0", "spectral", "score", "nac1", "nac2"]


def load_timings(csv_path: str):
    data = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["algorithm"]].append((int(row["N"]), float(row["seconds"])))
    for alg in data:
        data[alg].sort()
    return data


def fit_loglog_exponent(points):
    """Return alpha such that t ~= c * N^alpha (least-squares on log-log)."""
    xs = [math.log(N) for N, t in points if t > 0]
    ys = [math.log(t) for N, t in points if t > 0]
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "Output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "scalability_timings.csv")
    data = load_timings(csv_path)

    # Print empirical scaling exponents (handy to paste into the paper).
    print(f"{'Method':<10} {'Empirical scaling':<18} {'t@max N (s)':<12}")
    print("-" * 42)
    for alg in method_order:
        if alg not in data:
            continue
        alpha = fit_loglog_exponent(data[alg])
        n_max, t_max = data[alg][-1]
        print(f"{label_map[alg]:<10} O(N^{alpha:.2f}){'':<6} {t_max:>8.2f}  (N={n_max})")

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    for alg in method_order:
        if alg not in data:
            continue
        Ns = [p[0] for p in data[alg]]
        ts = [p[1] for p in data[alg]]
        axes[0].plot(Ns, ts, marker="o", linewidth=3.0, markersize=11,
                     label=label_map[alg], color=color_map[alg])
        axes[1].plot(Ns, ts, marker="o", linewidth=3.0, markersize=11,
                     label=label_map[alg], color=color_map[alg])

    axes[0].set_xlabel("Number of Entities (N)", fontsize=17)
    axes[0].set_ylabel("Community Detection Time (s)", fontsize=17)
    axes[0].set_title("Linear Scale", fontsize=19, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=14)
    axes[0].legend(loc="best", fontsize=14)

    axes[1].set_xlabel("Number of Entities (N, log)", fontsize=17)
    axes[1].set_ylabel("Community Detection Time (s, log)", fontsize=17)
    axes[1].set_title("Log–Log Scale", fontsize=19, fontweight="bold")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].tick_params(labelsize=14)

    fig.suptitle("Community Detection Scalability vs Graph Size",
                 fontsize=22, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "scalability_curve_new.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
