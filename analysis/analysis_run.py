from HV_func_eval import calculate_hv_progression
from IGD_func_eval import compare_igd_curves_multi
from utils import calculate_true_pareto_front, read_json
from plot_pareto_front import compare_pareto_from_algorithms


def run_analysis(metric="pareto", problem="tsp_semo"):
    """
    Run analysis for a given metric and problem.

    Parameters
    ----------
    metric : str
        One of {"hv", "igd", "pareto"}.
    problem : str
        One of {"tsp_semo", "bi_kp"}.
    """

    problem_dict = read_json("analysis/analysis_problem.json")
    algorithms = problem_dict[problem]

    if metric == "hv":
        calculate_hv_progression(algorithms, batch_size=10)

    elif metric == "igd":
        true_pf_approx = calculate_true_pareto_front([
            f"logs/momcts/{problem}",
            f"logs/meoh/{problem}",
            f"logs/nsga2/{problem}",
            f"logs/mpage/{problem}"
        ])
        compare_igd_curves_multi(algorithms, true_pf_approx)

    elif metric == "pareto":
        compare_pareto_from_algorithms(algorithms)

    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    run_analysis(metric="pareto", problem="bi_cvrp")
