import numpy as np
import matplotlib.pyplot as plt
from utils import read_population_scores_from_folder

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pymoo.indicators.hv import Hypervolume

def calculate_hv_per_generation_multiple(folder_paths_grouped, labels=None, visualize=True, max_gen=50):
    """
    Calculates mean & std HV per generation for multiple algorithms (each may have multiple runs),
    using global z_ideal and z_nadir across ALL runs/algorithms for fair comparison.
    
    Parameters:
    - folder_paths_grouped: list of lists, each sublist contains folders for one algorithm's runs
    - labels: list of algorithm labels
    - max_gen: maximum number of generations to plot
    """
    # ---------------------------------------------------
    # Pass 1: Collect all objective values across ALL runs
    # ---------------------------------------------------
    all_F_global = []

    for folders in folder_paths_grouped:
        for folder_path in folders:
            F_hist = read_population_scores_from_folder(folder_path)
            F_hist = [np.array(gen, dtype=float) for gen in F_hist]

            if len(F_hist) == 0:
                continue
            all_F_global.append(np.vstack(F_hist))

    if not all_F_global:
        raise ValueError("No valid data found in any folder.")

    all_F_global = np.vstack(all_F_global)
    z_ideal = all_F_global.min(axis=0)
    z_nadir = all_F_global.max(axis=0)
    print(f"Z_ideal: {z_ideal}, z_nadir: {z_nadir}")
    # Reference point (usually (1,1,...,1) after normalization)
    M = all_F_global.shape[1]
    ref_point = [1.1, 1.1]

    # ---------------------------------------------------
    # Pass 2: Compute HV for each run using global bounds
    # ---------------------------------------------------
    plt.figure(figsize=(8, 5))

    for idx, folders in enumerate(folder_paths_grouped):
        hv_runs = []

        for folder_path in folders:
            F_hist = read_population_scores_from_folder(folder_path)
            F_hist = [np.array(gen, dtype=float) for gen in F_hist]

            if len(F_hist) == 0:
                continue

            metric = Hypervolume(
                ref_point=ref_point,
                norm_ref_point=False,
                zero_to_one=True,
                ideal=z_ideal,
                nadir=z_nadir
            )

            hv_values = [metric(np.atleast_2d(gen)) for gen in F_hist[:max_gen]]
            hv_runs.append(hv_values)

        if not hv_runs:
            continue
        print(f"Length of hv runs: {len(hv_runs)}")
        # Align runs to same length
        min_len = min(len(run) for run in hv_runs)
        print(f"Min_len: {min_len}")
        hv_runs = [run[:min_len] for run in hv_runs]

        hv_array = np.array(hv_runs)   # shape: (n_runs, n_gens)
        mean_hv = hv_array.mean(axis=0)
        std_hv = hv_array.std(axis=0)

        # Label
        label_name = labels[idx] if labels and idx < len(labels) else f"Algo {idx+1}"

        # Plot mean ± std
        gens = np.arange(1, len(mean_hv) + 1)
        plt.plot(gens, mean_hv, marker='o', linestyle='-', label=label_name)
        plt.fill_between(gens, mean_hv - std_hv, mean_hv + std_hv, alpha=0.2)

    if visualize:
        plt.xlabel("Iterations")
        plt.ylabel("HV")
        plt.title("HV Progression per Generation (Mean ± Std, Global Normalization)")
        plt.grid(True)
        plt.legend()
        plt.xlim(1, max_gen)
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    folders_grouped = [
        [
           "logs/meoh/tsp_semo/20250929_133056/population"
        
        ],
        [
           "logs/momcts/tsp_semo/codestral_v1/population",
           "logs/momcts/tsp_semo/codestral_v2/population",
           "logs/momcts/tsp_semo/codestral_v3/population",
        ],
        [
           "logs/nsga2/tsp_semo/codestral_300_v1/population",
           "logs/nsga2/tsp_semo/codestral_300_v2/population",
           "logs/nsga2/tsp_semo/codestral_300_v3/population"
        ],
        [
            "logs/mpage/tsp_semo/codestral_300_v1/population",
            "logs/mpage/tsp_semo/codestral_300_v2/population",
        ]
         
    ]
    labels = ["MEOH", "MOMCTS", "NSGA2", "MPaGe"]
    calculate_hv_per_generation_multiple(folders_grouped, labels)