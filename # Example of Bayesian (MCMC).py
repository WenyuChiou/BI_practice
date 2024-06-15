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
import emcee
import os 
import random

#Step 2: Framing statistic model

def model(coeff):  
    """ 
    Model: salary = Beta0 + Beta1 * gender + Beta2 * education year 
    
    Input: 
        g,ey = np.array(g), np.array(ey) # gender and eductaion year [yr]
    
    Parameter:
    """
    dir = r'C:\Users\user\Desktop\Lehigh\pre-phd\BayesianStatistics_HawjengChiou\Ch01'
    
    obs_data = pd.read_csv(os.path.join(dir,'data1.csv'),header=None)
    g, ey = obs_data.iloc[:,1], obs_data.iloc[:,3]
    sl = np.zeros(len(g))       # forecast salary
    
    #---regression model---
    # Coefficient of model 
    Beta0, Beta1, Beta2 = coeff

    for ii in range(len(g)):
        sl[ii] = Beta0 + Beta1*g[ii] + Beta2*ey[ii]
        
    #Plot the histogram of estimated salary levels
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
   
    ax1.hist( sl,  
          bins=np.arange(sl.min(), sl.max()+1000, 1000), 
          rwidth=0.5 )
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Probability Density')
    plt.show()
    
    #Calcualted the mean and standard deviation of estimated salary levels
    mu, std = norm.fit(sl)
       
    return sl

dir = r'C:\Users\user\Desktop\Lehigh\pre-phd\BayesianStatistics_HawjengChiou\Ch01'


# g = [random.randint(1,40) for _ in range(100)] #random seeds for gender data
# ey = [random.randint(0,1) for _ in range(100)] #random seeds for education years

# Observed data
obs_data = pd.read_csv(os.path.join(dir,'data1.csv'),header=None)
sl = obs_data.iloc[:,4]


#Step 2: Specification and Inspection of parameter (coefficient) in priors

# Define acceptance ratio
def lnlike(theta, y, yerr):
    return -0.5 * np.sum(((y - model(theta))/yerr) ** 2)

def lnprior(coeff):
    """ Check the coefficient within the priors"""
    Beta0, Beta1, Beta2 = coeff
    
    if Beta0 > 0: 
        return 0.0
    else:
        return -np.inf

def lnprob(theta, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, y, yerr)

# Step 3: Implementation of estimating the parameter

#Initnial Data and Setting

Terr = 0.05*np.mean(sl)
data = (sl, Terr)
nwalkers = 128
niter = 500
initial = np.array([28,8.4,3.5])
ndim = len(initial)
p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]

def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state

# sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)








    





    










