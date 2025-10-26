from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_kp.get_instance import GetData
from llm4ad.task.optimization.bi_kp.template import template_program, task_description
from pymoo.indicators.hv import Hypervolume
import random
import time

__all__ = ['BIKPEvaluation']


PENALTY = 1e10  # large positive penalty for infeasible (since we minimize)

def knapsack_value(solution: np.ndarray, weight_lst: np.ndarray, value1_lst: np.ndarray, value2_lst: np.ndarray, capacity: float):
    # ensure it's an array of ints 0/1
    sol = np.asarray(solution).astype(int)
    if len(sol) != len(weight_lst):
        return PENALTY, PENALTY
    # check 0/1
    if not np.all(np.isin(sol, [0, 1])):
        return PENALTY, PENALTY
    total_weight = np.sum(sol * weight_lst)
    if total_weight > capacity:
        return PENALTY, PENALTY  # infeasible -> large positive penalty (bad)
    total_val1 = np.sum(sol * value1_lst)
    total_val2 = np.sum(sol * value2_lst)
    # we negate values because we want minimization (more negative = better)
    return -float(total_val1), -float(total_val2)


def dominates(a, b):
    """True if a dominates b (minimization)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def random_solution(weight_lst, capacity, problem_size):
    # Generate a permutation of the problem size and then select a subset of items in order with a probability of 0.5 till reaching the capacity
    # This is a simple random solution generator for the knapsack problem
    sol = list(range(problem_size))
    random.shuffle(sol)
    selected_items = []
    total_weight = 0
    for item in sol:
        if total_weight + weight_lst[item] <= capacity:
            selected_items.append(item)
            total_weight += weight_lst[item]
    return np.array([1 if i in selected_items else 0 for i in range(problem_size)])


def evaluate(instance_data, n_instance, problem_size, ref_point, capacity, eva: callable):
    obj_1 = np.ones(n_instance) * PENALTY  # initialize with large (bad) values
    obj_2 = np.ones(n_instance) * PENALTY
    all_objs = []
    Archives = []
    final_list = []

    for idx, (weight_lst, value1_lst, value2_lst) in enumerate(instance_data):
        # init random seed if you want reproducibility
        s_list = [random_solution(weight_lst, capacity, problem_size) for _ in range(20)]
        Archive = []
        # build initial archive, compute value once per solution
        for s_ in s_list:
            f = knapsack_value(s_, weight_lst, value1_lst, value2_lst, capacity)
            if f[0] < PENALTY:  # feasible
                Archive.append((np.asarray(s_, dtype=int), (f[0], f[1])))

        # main loop: generate candidates via eva
        for _ in range(8000):
            # ensure eva returns a 0/1 array of correct length
            s_prime = np.array(eva(Archive, weight_lst, value1_lst, value2_lst, capacity))
            if s_prime.shape[0] != problem_size:
                # invalid candidate, skip or raise
                # skip for robustness
                continue
            f_s_prime = knapsack_value(s_prime, weight_lst, value1_lst, value2_lst, capacity)

            # skip infeasible (penalized)
            if f_s_prime[0] >= PENALTY:
                # infeasible -> skip
                continue

            # check if any existing dominates the new one (minimization)
            if any(dominates(f_a, f_s_prime) for _, f_a in Archive):
                continue  # there exists an archive member that dominates s_prime

            # remove archive members that are dominated by s_prime
            Archive = [(a, f_a) for a, f_a in Archive if not dominates(f_s_prime, f_a)]
            Archive.append((np.asarray(s_prime, dtype=int), (f_s_prime[0], f_s_prime[1])))

        objs = np.array([obj for _, obj in Archive]) if len(Archive) > 0 else np.empty((0,2))
        Archives.append(objs)
        all_objs.append(objs)

    for n_ins, objs in enumerate(all_objs):
        if objs.size > 0:
            obj_1[n_ins] = np.min(objs[:, 0])  # remember: more negative is better
            obj_2[n_ins] = np.min(objs[:, 1])
        final_list.append(objs.tolist())

    return np.mean(obj_1), np.mean(obj_2)


class BIKPEvaluation(Evaluation):
    """Evaluator for the Bi-objective Knapsack Problem (BI-KP) using a custom algorithm."""

    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=90
        )
        self.n_instance = 8
        self.problem_size = 200
        getData = GetData(self.n_instance, self.problem_size)
        self._datasets, self.cap = getData.generate_instances() 
        self.ref_point = np.array([1.1, 1.1]) 

    def evaluate_program(self, program_str: str, callable_func: callable):
        return evaluate(self._datasets, self.n_instance, self.problem_size, self.ref_point, self.cap, callable_func)
    
import numpy as np
from typing import List, Tuple
import random
import json
import multiprocessing
import os
import warnings
warnings.filterwarnings("ignore")

def run_exec_and_eval(code_str, result_queue):
    try:
        local_vars = {}
        exec(code_str, globals(), local_vars)
        select_neighbor_func = local_vars["select_neighbor"]
        tsp = BIKPEvaluation()
        cst1, cst2 = tsp.evaluate_program('_', select_neighbor_func)
        result_queue.put([cst1, cst2])
    except Exception as e:
        result_queue.put(f"Error: {e}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    with open(f"Illustration/Bi KP 50/EoH/population_0/pop_1.json", "r") as f:
        data = json.load(f)
    for k in range(len(data)):
        if k == 9:
            for _ in range(1):
                select_neighbor_code = data[k]["function"]
                result_queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=run_exec_and_eval, args=(select_neighbor_code, result_queue))
                p.start()
                p.join(timeout=3600)
                if p.is_alive():
                    print(f"Timeout on code {k+1}, skipping.")
                    p.terminate()
                    p.join()
                    data[k]["score"] = data[k-1]["score"]
                    continue
                result = result_queue.get()
                if isinstance(result, str) and result.startswith("Error"):
                    print(f"Error on code {k+1}: {result}")
                    continue
                data[k]["score"] = result
                print(f"Evaluating with code {k+1}...", result)
