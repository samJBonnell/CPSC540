# Python Imports
from pkgutil import get_data
import numpy as np
import time
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

# Generic Imports
import os
import string
from venv import create
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
np.set_printoptions(linewidth=200)
from datetime import datetime

import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from torchinfo import summary

# ML Imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from scipy.linalg import svd

from data_loader import data_loader, create_cnn_matrix

# Personal Definitions
from us_lib.models.mlp import MLP, weighted_mse_loss
from us_lib.models.cnn import EncoderBlock, Bridge, EncoderToVector
from us_lib.data.normalization import NormalizationHandler
from us_lib.data.reader import load_records
from us_lib.models.pod import training_data_constructor, plot_field
from us_lib.data.parsing import extract_attributes
from us_lib.data.samples import iterate_folds, load_folds

class Optimizater:
    """
    Global class to handle the optimization of the ML models created using Torch
    """

    def __init__(self, algorithm = None, objective : ElementwiseProblem = None):
        self.algorithm = algorithm
        self.objective = objective
        self.best = []

    def set_parameters(self, **kwargs):
        allowed = {'population', 'sbx_prob', 'sbx_eta', 'mutation_pm', 'stopping_crit', 'g_gen', 'evals'}
        
        for key, value in kwargs.items():
            if key in allowed:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
            
        # Define the stopping criterion of the optimizer
        if self.stopping_crit == 'n_gen':
            self.termination = get_termination("n_gen", self.n_gen)
        elif self.stopping_crit == 'evals':
            self.termination = get_termination("evals", self.evals)
            
    def optimize(self):
        """ Optimize the objective function and optimize using the specifed algorithm """
        self.algorithm.setup(self.objective, termination=self.termination, seed=None, verbose=False)
        # Run the algorithm, printing the generations of each trial
        while self.algorithm.has_next():
            self.algorithm.next()

            current_objective = self.algorithm.pop.get("F")
            index_min = min(range(len(current_objective)), key=current_objective.__getitem__)
            self.best.append(self.algorithm.pop.get("X")[index_min])

        self.res = self.algorithm.result()
            
    
            
# variables = FEM.load_extern('initial.csv') # Load an initial set of variables from an external source. The geometric information will be overwritten but mesh sizes, etc. will be preserved through the process.
# variables = np.array(variables)
# stiffened_panel = FEM.FiniteElementModel("StiffenedPanel.py", input_path, output_path, input_keys, output_keys)

# Optimization Parameters
Generations = 100
Population = 20
Offspring = 8

# NSGA-II Parameters
SBX_prob = 0.7
SBX_eta = 40
Mutation_PM = 140

class StiffenedPanelOptimizationCS3(ElementwiseProblem):
    # Give access to meshsize and pressure for simulation
    global Pressure
    global PlateMesh
    global TransverseStiffenerMesh
    global TransverseFlangeMesh
    global LongitudinalStiffenerMesh
    global LongitudinalFlangeMesh
    global PanelWidth
    global PanelLength

    def __init__(self):
        super().__init__(n_var=11,
                         n_obj=1,
                         n_ieq_constr=2,
                         xl = np.array([2, 2, 0.005, 0.05, 0.005, 0.005, 0.010, 0.010, 0.005, 0.005, 0.010]),
                         xu = np.array([9, 14, 0.075, 1.00, 0.075, 0.050, 0.750, 0.950, 0.075, 0.075, 0.500]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Create 'template' integer values for stiffener numbers
        x[0] = round(x[0], 0)
        x[1] = round(x[1], 0)

        if x[7] >= x[3]:
            x[7] = x[3] - 0.0001

        variables[0:11] = x

        data = stiffened_panel.evaluate(variables) # Evalute the objective function
        stress = data["output"][0]
        mass = data["output"][1]

        # The objective function is to minimze the mass of the panel
        f1 = mass
        
        # Constraints are defined by max stress & geometric tolerancing due to assignment in ABAQUS
        g1 = stress - 1.75e+8
        g2 = x[7] - x[3]

        out["F"] = [f1]
        out["G"] = [g1, g2]

        stiffened_panel.save("temp\\history.csv")

problem = StiffenedPanelOptimizationCS3()

algorithm = NSGA2(
    pop_size=Population,
    n_offsprings=Offspring,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=SBX_prob, eta=SBX_eta),
    mutation=PM(eta=Mutation_PM),
    eliminate_duplicates=True
)

print("\n--- Started Analysis ---\n")

best = []
termination = get_termination("n_gen", Generations)
algorithm.setup(problem, termination=termination, seed=None, verbose=False)

# Run the algorithm, printing the generations of each trial
while algorithm.has_next():
    algorithm.next()

    tempList = algorithm.pop.get("F")
    index_min = min(range(len(tempList)), key=tempList.__getitem__)
    best.append(algorithm.pop.get("X")[index_min])

    print(f"Generation: {algorithm.n_gen - 1}")
    writeGens = open("temp\\generations.csv", 'a')
    writeGens.write(str(algorithm.n_gen - 1) + "," + str(algorithm.evaluator.n_eval) + "\n")
    writeGens.close()

res = algorithm.result()

print("\n\n!!! - Program executed in: %s hours" % round((time.time() - start_time)/3600, 3))

writeResults = open("temp\\results.csv", 'w')
for gen in best:
    writeResults.write(str(gen) + "\n")
writeResults.close()

print("\n--- Analysis Completed ---\n")

if __name__ == '__main__':
    main()