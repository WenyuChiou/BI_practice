# Example of Bayesian method 
# By Wenyu Chiou 

"""The scholar intends to explore the influences of the education level and gender on salary levels

#Step 1: Inspection of Attributes of Variable
    Variable: 
        Gender(g): The boys and girls are denoted to 1 and 0.
        Education years(ey): The unit is years and the variable is continuous.
        Salary Levels (sl): 

"""
import numpy as np # type: ignore
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import os 
import random

#Step 2: Framing statistic model

def model(g,ey,coeff):  
    """ 
    Model: salary = Beta0 + Beta1 * gender + Beta2 * education year
    
    Input: 
        g,ey = np.array(g), np.array(ey) # gender and eductaion year [yr]
    
    Parameter:
    """
    sl = np.zeros(len(g))       # forecast salary 
    #---regression model---
    # Coefficient of model 
    Beta0 = coeff['B0']
    Beta1 = coeff['B1']
    Beta2 = coeff['B2']

    for ii in range(len(g)):
        sl[ii] = Beta0 + Beta1*g[ii] + Beta2*ey[ii]
        
    #Plot the histogram of estimated salary levels
    
    # fig, ax1 = plt.subplots(figsize=(10, 5))
   
    # ax1.hist( sl,  
    #       bins=np.arange(sl.min(), sl.max()+1000, 1000), 
    #       rwidth=0.5 )
    
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Probability Density')
    
    #Calcualted the mean and standard deviation of estimated salary levels
    mu, std = norm.fit(sl)
       
    return sl, std, mu

dir = r'C:\Users\user\Desktop\Lehigh\pre-phd\BayesianStatistics_HawjengChiou\Ch01'


# g = [random.randint(1,40) for _ in range(100)] #random seeds for gender data
# ey = [random.randint(0,1) for _ in range(100)] #random seeds for education years

# Observed data
obs_data = pd.read_csv(os.path.join(dir,'data1.csv'),header=None)
g, ey = obs_data.iloc[:,1], obs_data.iloc[:,3]
sl = obs_data.iloc[:,4]

model_data, std1, mu1 = model(g,ey,{'B0':28,'B1':8.4,'B2':3.5})

#Step 2: Specification and Inspection of parameter (coefficient) in priors

# Define acceptance ratio
def lnlike(obs, model, yerr):

    Lnlike = 0
    for i in range(len(obs)):
        Lnlike += -1/2*((obs - model)/yerr)**2  
    
    return Lnlike

def lnprior(coeff):
    """ Check the coefficient within the priors"""
    Beta0 = coeff['B0']
    Beta1 = coeff['B1']
    Beta2 = coeff['B2']
    
    if Beta0 > 0: 
        return 0.0
    else:
        return -np.inf

def lnprob(coeff,obs, model, yerr):
    lp = lnprior(coeff)
    if lp != 0.0:
        return -np.inf
    return lp + lnlike(coeff, obs, model, yerr) #recall if lp not -inf, its 0, so this just returns likelihood








    





    










