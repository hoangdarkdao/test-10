import json
import numpy as np
import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Any
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def read_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def _filter_items_by_seen_orders(items: List[Dict[str, Any]], seen_orders: set, order_key: str = "sample_order"):
    """
    Return (filtered_items, removed_count). Items whose sample_order is already in seen_orders are skipped.
    If an item has no sample_order, it is kept.
    """
    out = []
    removed = 0
    for item in items:
        so = item.get(order_key)
        if so is None:
            out.append(item)
            continue
        try:
            key = int(so)
        except Exception:
            key = so
        if key in seen_orders:
            removed += 1
            continue
        seen_orders.add(key)
        out.append(item)
    return out, removed

def _dedupe_items_in_list(items: List[Dict[str, Any]], order_key: str = "sample_order"):
    """
    Dedupe within a single list: keep the first occurrence and remove later duplicates.
    Returns (filtered_items, removed_count).
    """
    seen = set()
    out = []
    removed = 0
    for item in items:
        so = item.get(order_key)
        if so is None:
            out.append(item)
            continue
        try:
            key = int(so)
        except Exception:
            key = so
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(item)
    return out, removed

def read_score_from_path(json_data: str) -> List[List[float]]:
    """
    Read a single JSON file (list of items) and return list of [score0, score1].
    If the file contains duplicate sample_order values, later duplicates are removed (keeps first).
    """
    skip_item_num = 0
    with open(json_data, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {json_data}")

    filtered, removed = _dedupe_items_in_list(data)
    
    scores: List[List[float]] = []
    for item in filtered:
        if 'score' in item and isinstance(item['score'], list) and len(item['score']) == 2:
            try:
                s0 = float(item['score'][0])
                s1 = float(item['score'][1])
                scores.append([s0, s1])
            except Exception:
                skip_item_num += 1
                print("Warning: Skipping item due to non-numeric 'score'.")
        else:
            skip_item_num += 1
            print("Warning: Skipping item due to invalid or missing 'score'.")
    return scores

def find_pareto_front_from_scores(scores: list[list[float, float]]):
    F_hist_np = np.array(scores)
    nd_indices = NonDominatedSorting().do(F_hist_np, only_non_dominated_front=True)
    true_pf_approx = F_hist_np[nd_indices]

    return true_pf_approx


def read_population_scores_from_folder(folder_path: str) -> list[list[float, float]]:
    '''
    Args:
        mark = 0: the score is negative
        mark = 1: objective is positive
    '''
    mark = 0
    files = glob.glob(os.path.join(folder_path, "pop_*.json"))
    if len(files) == 0:
        mark = 1
        files = glob.glob(os.path.join(folder_path, "population_generation_*.json"))
        
    files.sort(key=lambda x: int(re.search(r"(\d+)", os.path.basename(x)).group()))
    data_list = []

    for file_path in files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = None  # or {}
            data_list.append({
                "filename": os.path.basename(file_path),
                "content": data
            })

    F_list = []
    for data in data_list:
        F = []
        for x in data["content"]:
            if mark == 0:
                obj, runtime = x["score"]
            else:
                obj, runtime = x["score"]
            F.append([obj, runtime])
        F_list.append(F)

    return F_list

def calculate_true_pareto_front(folder_list: list[str]) -> np.ndarray:
    full_scores = []
    file_path_name = ["samples_1~200.json", "samples_0~200.json", "samples_1~300.json", "samples_0~300.json"]
    for folder in folder_list:
        folder_path = Path(folder)
        for name in file_path_name:
            for file_path in folder_path.rglob(name):  # recursive search
                try:
                    print(f"Get from file path: {file_path}")
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    scores = [item.get("score") for item in data if item.get("score") is not None]
                    scores = [[x for x in pair] for pair in scores]
                    full_scores.extend(scores)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    F_hist_np = np.array(full_scores)
    true_nd_indices = NonDominatedSorting().do(F_hist_np, only_non_dominated_front=True)
    true_pf_approx = F_hist_np[true_nd_indices]  # get true Pareto front
    
    return true_pf_approx