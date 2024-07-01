# Application of Bayseian method using pymc modulus
"""
This file tried to reproduce the process of Bayesian Analysis using MCMC.

"""

import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the style of the figure background
az.style.use("arviz-darkgrid")

# Loading dataset using pandas
dir = r'C:\Users\user\Desktop\Lehigh\pre-phd\BayesianStatistics_HawjengChiou\Ch01'
obs_data = pd.read_csv(os.path.join(dir,'data1.csv'),header=None)
print(obs_data)
