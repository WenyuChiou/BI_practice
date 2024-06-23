# Third example of Bayesian Inference
# By Wenyu Chiou 

""" 
The algorithm used in this example is Metropolis-Hastings, which is ofthen used on Bayesian Inference to determine
whether the new values (forecastes) should be accepted or reserve the current value for calculating posteriors.

Note: MH algorithm is not an optimization method.


link: https://bayesiancomputationbook.com/markdown/chp_01.html
    
    """
    
from scipy import stats
import numpy as np


# definition of posterior distribution given unknown prior distribution and likelihood function
def post(theta, T, alpha = 1, beta = 1):
    if (0 <= theta <= 1):
        prior = stats.beta(alpha, beta).pdf(theta)
        like = stats.bernoulli(theta).pmf(T).prod()
        prob = prior * like
    else:
        prob = -np.inf
    return prob
        

