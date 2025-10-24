from __future__ import annotations
import random
import copy
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import math
from copy import deepcopy
import numpy as np
from typing import List, Tuple, Any, Optional  # Ensure Optional is imported

class MCTSNode:
    def __init__(self, algorithm, code, obj: List[float], individual=None, depth=0, is_root=False, parent=None, visit=0, raw_info=None):
        self.algorithm = algorithm
        self.code: str = code
        self.parent: MCTSNode = parent
        self.individual = individual
        self.depth: int = depth
        self.rewards_collected: List[List[float]] = []
        self.children: List[MCTSNode] = []  # list of MCTSNode class
        self.children_info: List[dict] = []  # Raw info dictionaries of children, often used for prompting LLMs
        self.visits: int = visit
        self.subtree: List[MCTSNode] = []
        self.raw_info: List[MCTSNode] = raw_info
        self.reward_vector: List[float] = np.array(obj)  

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.algorithm}, visits={self.visits})"


class MCTS:
    def __init__(self, root_answer: Any, num_objectives: int, exploration_constant_0: float = 0.1, alpha: float = 0.5):
        self.exploration_constant_0 = exploration_constant_0  # Parameter for UCB
        self.num_objectives = num_objectives
        self.max_depth = 10
        self.epsilon = 1e-10
        self.alpha = alpha # used for progressive widening
        self.root = MCTSNode(algorithm=root_answer, code=root_answer, obj=[0.0] * num_objectives,
                             is_root=True)
        self.global_pareto_front: List[List[float, float]] = []
        self.rewards = []
        self.selected_nodes: List[MCTSNode] = []
        self.rank_list = []
        
    @staticmethod
    def dominates(reward_a: List[float], reward_b: List[float]) -> bool: 
        '''
        Args:
            Minimization problem
        '''
        if reward_a is None or reward_b is None:
            return False
        
        is_strictly_better_on_at_least_one = False
        for i in range(len(reward_a)):
            if reward_a[i] > reward_b[i]:  
                return False 
            if reward_a[i] < reward_b[i]:  
                is_strictly_better_on_at_least_one = True
        return is_strictly_better_on_at_least_one

    @staticmethod
    def is_non_dominated(rewards: List[List[float]], new_reward: List[float]) -> bool:
       
        for r in rewards:
            if MCTS.dominates(r, new_reward):
                return False
        return True
    
    def update_pareto_front(self, new_reward: List[float]) -> List[List[float]]:
        """
        Updates the global Pareto front with a new reward vector,
        maintaining only non-dominated solutions.
        """
        # If new_reward is dominated by the current front, return unchanged
        if not self.is_non_dominated(self.global_pareto_front, new_reward):
            print(f"Dominated solution, pareto front keep the same, pareto front is: {self.global_pareto_front}")
            return self.global_pareto_front

        # Otherwise, add new_reward and prune dominated ones
        updated_front = [r for r in self.global_pareto_front if not self.dominates(new_reward, r)]
        updated_front.append(new_reward)

        # Update the global archive in place
        self.global_pareto_front = updated_front
        
        print(f"Updated pareto front: {self.global_pareto_front}")
        return self.global_pareto_front

    def backpropagate(self, node: MCTSNode, reward_vector: List[float]):
        
        current_node = node
        while current_node:
            current_node.visits += 1
            current_node.rewards_collected.append(reward_vector)
            current_node = current_node.parent

    def _calculate_hypervolume(self, front: List[List[float, float]]) -> float: 
        
        front_array = np.array(front) # [NHV, runtime]
        print(f"Current Pareto Front to calculate HV is: {front_array}")        
        if not front:
            return 0.0
        
        z_ideal = np.array([-1.5, 0]) # lower bound of [NHV, runtime]
        z_nadir = np.array([0, 20]) # upper bound of [NHV, runtime]
        
        print(f"Z_ideal: {z_ideal}, Z_nadir: {z_nadir}")
                
        metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                        norm_ref_point=False,
                        zero_to_one=True, # tell to normalize all points to [0, 1]
                        ideal=z_ideal,
                        nadir=z_nadir)
        
        hv = metric(front_array)
        print(f"Final HV indicator for current front: {hv}")
        return hv

    def _calculate_penalty(self, reward_vector: List[float],
                          pareto_front: List[List[float]],
                          reference_point: List[float]) -> float:
        '''
        Args:
            High level idea: calculate the distance from a dominated solution (reward_vector) to pareto front
        '''
        for p in pareto_front:
            if np.array_equal(p, reward_vector):
                return 0.0

        sorted_front = sorted(pareto_front, key=lambda x: x[0], reverse=True)
        
        r = np.array(reward_vector)
        z = np.array(reference_point)
        line_dir = r - z
        
        # Find the intersection of this line with the Pareto front's envelope
        for i in range(len(sorted_front) - 1):
            p1 = np.array(sorted_front[i])
            p2 = np.array(sorted_front[i+1])
            
            # Define the line segment between two consecutive Pareto points
            front_dir = p2 - p1
            
            denom = line_dir[0] * front_dir[1] - line_dir[1] * front_dir[0] # if denom = 0, that mean 2 lines are parallel
        
            if abs(denom) > 1e-9: # Avoid division by zero
                t = ((p1[0] - z[0]) * front_dir[1] - (p1[1] - z[1]) * front_dir[0]) / denom
                u = -((p1[0] - z[0]) * line_dir[1] - (p1[1] - z[1]) * line_dir[0]) / denom
                
                if 0 <= t and 0 <= u <= 1:
                    projection_point = p1 + u * front_dir
                    penalty = np.linalg.norm(r - projection_point)
                    print(f"penalty score is: {penalty}")
                    return float(penalty)
        return 0.0 # Return default values

    def _calculate_multi_objective_ucb(self, child: MCTSNode, parent_visits: int) -> List[float, float]:
        
        avg_reward = []
        for i in range(self.num_objectives):
            avg = sum(r[i] for r in child.rewards_collected) / child.visits if child.visits > 0 else 0.0
            avg_reward.append(avg)
            
        print(f"Avg_reward for dim before normalization: {avg_reward}")
        
        exploration_term = self.exploration_constant_0 * math.sqrt(
            math.log(parent_visits + 1) / (child.visits + self.epsilon)
        )
        print(f"Exploration term: {exploration_term}")
        ucb_vector = [obj - exploration_term for obj in avg_reward]
        
        print(f"Final UCB Vector: {ucb_vector}")
        return ucb_vector 
    
    
    def best_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        if not node.children:
            return None

        best_child = None
        best_weighted_sum = float('inf')  # Initialize to infinity for minimization

        print(f"\nEvaluating {len(node.children)} children for node with {node.visits} visits")
        
        # Collect UCB vectors and corresponding children
        children_info = []
        for i, child in enumerate(node.children):
            print(f"\n--- Child {i+1} ---")
            print(f"Visits: {child.visits}")
            if child.visits == 0:
                return child  # Prioritize unvisited nodes for exploration
            
            # Calculate UCB vector for the child
            r_sa = self._calculate_multi_objective_ucb(child, node.visits)
            print(f"UCB vector (r_sa): {r_sa}")
            children_info.append({"index": i, "child": child, "r_sa": r_sa})

        # Perform non-dominated sorting on UCB vectors
        F = np.array([info["r_sa"] for info in children_info])
        nds = NonDominatedSorting()
        fronts = nds.do(F)
        
        # Select the best child from the first non-dominated front
        print(f"Non-dominated fronts: {fronts}")
        first_front_indices = fronts[0]  # Get indices of the first front
        if len(first_front_indices) > 0:
            # Calculate average for each child in the first front
            for idx in first_front_indices:
                child_info = children_info[idx]
                r_sa = child_info["r_sa"]
                weighted_sum = (r_sa[0] + r_sa[1]) / 2  # Thay đổi: lấy trung bình của 2 objs (cost1 và cost2 thô)
                print(f"Child {idx+1} average ( (ucb_1 + ucb_2) / 2 ): {weighted_sum:.4f}")
                
                # Update best child if weighted sum is smaller
                if weighted_sum < best_weighted_sum:
                    best_weighted_sum = weighted_sum
                    best_child = child_info["child"]
                    print(f"→ Child {idx+1} is the new best candidate with average {best_weighted_sum:.4f}")
        
        # If no non-dominated children, select from all children using average
        if best_child is None:
            for child_info in children_info:
                r_sa = child_info["r_sa"]
                weighted_sum = (r_sa[0] + r_sa[1]) / 2  # Thay đổi: lấy trung bình của 2 objs
                print(f"Child {child_info['index']+1} average ( (ucb_1 + ucb_2) / 2 ): {weighted_sum:.4f}")
                
                if weighted_sum < best_weighted_sum:
                    best_weighted_sum = weighted_sum
                    best_child = child_info["child"]
                    print(f"→ Child {child_info['index']+1} is the new best candidate with average {best_weighted_sum:.4f}")
        
        print(f"\nFinal selection → best child with average = {best_weighted_sum:.4f}")
        return best_child