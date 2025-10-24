from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from utils import read_score_from_path



def calculate_igd_from_path(json_path: str, true_pf_approx: np.ndarray, max_eval=300, step=10) -> dict:
    """
    Calculate IGD progression for one run (one JSON file).
    """
    F_hist = read_score_from_path(json_path)
    F_hist = np.array(F_hist)
    if len(F_hist) == 0:
        return {}

    target_evals = list(range(0, max_eval + 1, step))
    igd_at_targets = {}

    metric = IGD(true_pf_approx, zero_to_one=True)

    for target in target_evals:
        archive = F_hist[:target + 1]
        if len(archive) == 0:
            continue

        nd_idx = NonDominatedSorting().do(np.array(archive), only_non_dominated_front=True)
        P = np.array(archive)[nd_idx]
        igd_at_targets[target] = metric.do(P)

    return igd_at_targets


def aggregate_igd_curves(json_paths: list[str], true_pf_approx: np.ndarray, max_eval=300, step=10):
    """
    Aggregate IGD curves across multiple runs of the same algorithm.
    """
    all_curves = []

    for path in json_paths:
        igd_at_targets = calculate_igd_from_path(path, true_pf_approx, max_eval, step)
        if len(igd_at_targets) > 0:
            all_curves.append(igd_at_targets)

    if not all_curves:
        return [], [], []

    eval_points = sorted(set().union(*[curve.keys() for curve in all_curves]))
    all_arrays = []

    for curve in all_curves:
        arr = [curve.get(ev, np.nan) for ev in eval_points]
        all_arrays.append(arr)

    all_arrays = np.array(all_arrays)  # shape: (n_runs, n_eval_points)

    mean_curve = np.nanmean(all_arrays, axis=0)
    std_curve = np.nanstd(all_arrays, axis=0)

    return eval_points, mean_curve, std_curve


def compare_igd_curves_multi(algorithms: dict, true_pf_approx: np.ndarray, max_eval=300, step=10, print_detail=True):
    """
    Compare IGD progression across multiple algorithms.
    """
    plt.figure(figsize=(7, 5))

    final_igd_summary = {}

    for label, paths in algorithms.items():
        evals, mean_curve, std_curve = aggregate_igd_curves(paths, true_pf_approx, max_eval, step)
        if len(evals) == 0:
            print(f"⚠️ No valid data for {label}")
            continue

        plt.plot(evals, mean_curve, marker="o", label=label)
        plt.fill_between(evals, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

        # Save and print final IGD value
        final_igd = mean_curve[-1]
        final_std = std_curve[-1]
        final_igd_summary[label] = (final_igd, final_std)

        if print_detail:
            print(f"\n✅ Final IGD ({label}): {final_igd:.4f} ± {final_std:.4f}")

    # Optional ranking
    if print_detail and final_igd_summary:
        print("\n🏁 IGD Ranking (lower is better):")
        ranked = sorted(final_igd_summary.items(), key=lambda x: x[1][0])
        for rank, (algo, (igd, std)) in enumerate(ranked, 1):
            print(f"  {rank}. {algo:<20} → IGD = {igd:.4f} ± {std:.4f}")

    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    plt.ylim(0, 1)
    plt.title("IGD Comparison (Mean ± Std across runs)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
