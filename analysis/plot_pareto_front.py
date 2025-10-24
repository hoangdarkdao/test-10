import json
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from utils import read_score_from_path, find_pareto_front_from_scores
from pathlib import Path


def compare_pareto_from_algorithms(file_dict: dict[str, list[str]], show_global=True):
    """
    Compare Pareto fronts from multiple algorithms, showing both local and global PF analysis.
    """

    plt.figure(figsize=(12, 10))
    algo_fronts = {}
    all_scores = []
    algo_labels = []

    print("\n======================")
    print("🔍 LOCAL PARETO ANALYSIS")
    print("======================")

    # === Step 1: Find local Pareto fronts for each algorithm ===
    for algo, file_list in file_dict.items():
        algo_scores = []

        for file_path in file_list:
            scores = read_score_from_path(file_path)
            if len(scores) == 0:
                continue
            algo_scores.extend(scores)

        if len(algo_scores) == 0:
            print(f"⚠️  Warning: No valid scores found for {algo}")
            continue

        algo_scores = np.array(algo_scores, dtype=float)
        pareto_front = find_pareto_front_from_scores(algo_scores)
        algo_fronts[algo] = pareto_front

        # Print detailed analysis
        print(f"\n📊 Algorithm: {algo}")
        print(f"  Total solutions collected: {len(algo_scores)}")
        print(f"  Pareto-optimal points found: {len(pareto_front)}")
        print("  Pareto front points (objective_1, objective_2):")
        for i, point in enumerate(pareto_front):
            print(f"    {i+1:2d}: [{point[0]:.6f}, {point[1]:.6f}]")

        # Plot local Pareto front
        plt.scatter(
            pareto_front[:, 0],
            pareto_front[:, 1],
            label=f"{algo} (local PF)",
            s=150,
            alpha=1
        )

        # Keep for global PF computation
        all_scores.append(algo_scores)
        algo_labels.extend([algo] * len(algo_scores))

    if not all_scores:
        raise ValueError("❌ No valid data found across algorithms.")

    # === Step 2: Global Pareto front ===
    if show_global:
        print("\n======================")
        print("🌍 GLOBAL PARETO ANALYSIS")
        print("======================")

        all_scores = np.vstack(all_scores)
        nd_idx = NonDominatedSorting().do(all_scores, only_non_dominated_front=True)
        global_pf = all_scores[nd_idx]

        # Count contributions from each algorithm
        contribution_count = {algo: 0 for algo in file_dict.keys()}
        contribution_points = {algo: [] for algo in file_dict.keys()}

        for point in global_pf:
            for score, algo in zip(all_scores, algo_labels):
                if np.allclose(score, point, atol=1e-8):
                    contribution_count[algo] += 1
                    contribution_points[algo].append(point)
                    break

        print(f"  Total global Pareto-optimal points: {len(global_pf)}")
        print("  Global Pareto front points:")
        for i, point in enumerate(global_pf):
            print(f"    {i+1:2d}: [{point[0]:.6f}, {point[1]:.6f}]")

        print("\n  🌟 Contribution Summary:")
        for algo, count in contribution_count.items():
            print(f"    {algo:<15}: {count:2d} points")
            if count > 0:
                for p in contribution_points[algo]:
                    print(f"       ↳ [{p[0]:.6f}, {p[1]:.6f}]")

        # Plot global Pareto front
        plt.scatter(
            global_pf[:, 0],
            global_pf[:, 1],
            color="black",
            s=120,
            marker="*",
            label="Global Pareto Front",
            edgecolor="white",
            linewidth=1.2
        )

    # === Step 3: Plot settings ===
    plt.xlim(-1.06, -0.80)
    plt.ylim(-0.1, 0.8)
    plt.xlabel("Hypervolume (negative, minimize)")
    plt.ylabel("Runtime (positive, minimize)")
    plt.title("Pareto Front Comparison Across Algorithms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


