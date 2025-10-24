import numpy as np
import matplotlib.pyplot as plt
import json
from utils import read_score_from_path
from pymoo.indicators.hv import Hypervolume



def calculate_hv_progression(algorithms, batch_size=10, visualize=True, max_samples=300, print_detail=True):
    """
    Calculate and optionally visualize Hypervolume (HV) progression for multiple algorithms.

    Parameters:
    - algorithms: dict[str, list[str]] mapping algorithm name → list of file paths
    - batch_size: step size for HV computation
    - visualize: whether to show the HV curve
    - max_samples: limit on number of points read from each file
    - print_detail: whether to print final HV summary
    """

    # --- Step 1: Collect all Pareto points globally ---
    all_F_global = []
    for algo, files in algorithms.items():
        for file_path in files:
            F = read_score_from_path(file_path)
            if len(F) == 0:
                continue
            F = F[:max_samples]
            all_F_global.append(F)

    if not all_F_global:
        raise ValueError("No valid data found in any file.")

    all_F_global = np.vstack(all_F_global)
    z_ideal = all_F_global.min(axis=0)
    z_nadir = all_F_global.max(axis=0)
    print(f"\n🌍 Global Ideal: {z_ideal}, Nadir: {z_nadir}")

    # Reference point (slightly worse than nadir)
    ref_point = [1.1, 1.1]

    if visualize:
        plt.figure(figsize=(8, 5))

    # --- Step 2: Compute HV for each algorithm ---
    for algo, files in algorithms.items():
        hv_runs = []

        for file_path in files:
            F = read_score_from_path(file_path)
            if len(F) == 0:
                continue

            F = np.array(F[:max_samples], dtype=float)

            metric = Hypervolume(
                ref_point=ref_point,
                norm_ref_point=False,
                zero_to_one=True,
                ideal=z_ideal,
                nadir=z_nadir
            )

            hv_values = []
            for end in range(batch_size, len(F) + 1, batch_size):
                F_subset = F[:end]
                hv = metric(F_subset)
                hv_values.append(hv)

            hv_runs.append(hv_values)

        if not hv_runs:
            continue

        # Align runs to same length
        max_len = max(len(run) for run in hv_runs)
        hv_array = np.full((len(hv_runs), max_len), np.nan)
        for i, run in enumerate(hv_runs):
            hv_array[i, :len(run)] = run

        mean_hv = np.nanmean(hv_array, axis=0)
        std_hv = np.nanstd(hv_array, axis=0)
        batches = np.arange(1, max_len + 1) * batch_size

        # --- Step 3: Print only the final HV value ---
        if print_detail:
            print(f"\n✅ Final HV ({algo}): {mean_hv[-1]:.4f} ± {std_hv[-1]:.4f}")

        # --- Step 4: Visualization ---
        if visualize:
            plt.plot(batches, mean_hv, marker='o', label=algo)
            plt.fill_between(batches, mean_hv - std_hv, mean_hv + std_hv, alpha=0.3)

    if visualize:
        plt.xlabel(f"Number of Samples (batch size = {batch_size})")
        plt.ylabel("Hypervolume (HV)")
        plt.title("HV Progression per Algorithm (Mean ± Std)")
        plt.grid(True, linestyle="--", alpha=1)
        plt.legend()
        plt.tight_layout()
        plt.show()

