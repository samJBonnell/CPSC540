# Python Imports
from pkgutil import get_data
import numpy as np
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime
# Pymoo Imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
# Remove a non-compilation warning 
from pymoo.config import Config
Config.warnings['not_compiled'] = False
import numpy as np

class Optimizer:
    """
    Global class to handle the optimization of the ML models created using Torch
    Supports both single-objective and multi-objective optimization
    """
    def __init__(self, algorithm=None, objective: ElementwiseProblem = None):
        self.algorithm = algorithm
        self.objective = objective
        self.best = []
        self.is_multi_objective = objective.n_obj > 1 if objective is not None else False
        
    def set_parameters(self, **kwargs):
        allowed = {'population', 'sbx_prob', 'sbx_eta', 'mutation_pm', 'stopping_crit', 'n_gen', 'evals'}
        for key, value in kwargs.items():
            if key in allowed:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        # Define the stopping criterion of the optimizer
        if hasattr(self, 'stopping_crit'):
            if self.stopping_crit == 'n_gen':
                self.termination = get_termination("n_gen", self.n_gen)
            elif self.stopping_crit == 'evals':
                self.termination = get_termination("evals", self.evals)
    
    def run(self):
        """ Optimize the objective function and optimize using the specified algorithm """
        self.algorithm.setup(self.objective, termination=self.termination, seed=None, verbose=False)
        pbar = tqdm(desc="Optimization Progress")
        
        # Run the algorithm, printing the generations of each trial
        while self.algorithm.has_next():
            self.algorithm.next()
            current_objective = self.algorithm.pop.get("F")
            
            if self.is_multi_objective:
                # For multi-objective: track the best solution by first objective (e.g., mass)
                # You can change this to track by a different objective or use a weighted sum
                index_min = np.argmin(current_objective[:, 0])
                self.best.append(self.algorithm.pop.get("X")[index_min])
                pbar.set_postfix({"Best Obj 1": f"{current_objective[index_min, 0]:.4f}"})
            else:
                # For single-objective: find the minimum
                index_min = min(range(len(current_objective)), key=current_objective.__getitem__)
                self.best.append(self.algorithm.pop.get("X")[index_min])
                pbar.set_postfix({"Best": f"{current_objective[index_min]:.4f}"})

            pbar.update(1)
        
        pbar.close()
        self.res = self.algorithm.result()
    
    def results(self, num: int = 0):
        """
        Returns the optimization results and the best solution from a specific generation
        
        Args:
            num: Generation number (default: 0 for first generation)
        
        Returns:
            res: Full optimization result object containing Pareto front (multi-obj) or best solution (single-obj)
            best: Best individual from the specified generation
        """
        return self.res, self.best[num]
    
    def get_pareto_front(self):
        """
        Get the Pareto front solutions (only valid for multi-objective optimization)
        
        Returns:
            X: Design variables of Pareto optimal solutions
            F: Objective values of Pareto optimal solutions
        """
        if not self.is_multi_objective:
            raise ValueError("Pareto front only exists for multi-objective optimization")
        return self.res.X, self.res.F
    
    def get_best_by_objective(self, obj_index=0):
        """
        Get the best solution according to a specific objective
        
        Args:
            obj_index: Index of the objective to minimize (default: 0)
        
        Returns:
            best_x: Design variables of the best solution
            best_f: Objective values of the best solution
        """
        if self.is_multi_objective:
            index_min = np.argmin(self.res.F[:, obj_index])
            return self.res.X[index_min], self.res.F[index_min]
        else:
            index_min = np.argmin(self.res.F)
            return self.res.X[index_min], self.res.F[index_min]