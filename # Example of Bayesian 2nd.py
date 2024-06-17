# Second example of Bayesian method 
# By Wenyu Chiou 

import numpy as np # type: ignore
from scipy.stats import norm
import scipy.stats as sts
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import os 

def read_data(dir:str,file_name:str):
    """Loading gender and education years data from excel files
    Input: 
        dir: file direction
        file_name: file_name (i.e. data.xlsx)
    Output:
        g: geneder (girl:0 , boy:,1; format: dataframe)
        ey: education year (unit: year; formate: dataframe) 
    """
    
    obs_data = pd.read_csv(os.path.join(dir,file_name),header=None)
    g, ey, obs_sl = obs_data.iloc[:,1], obs_data.iloc[:,3], obs_data.iloc[:,4]
    
    return g,ey,sorted(obs_sl)



def model(coeff,dir,file_name):  
    """ 
    Model: salary = Beta0 + Beta1 * gender + Beta2 * education year 
    
    Input: 
        g,ey = np.array(g), np.array(ey) # gender and eductaion year [yr]
        sl: Salary levels
    
    Parameter:
    """
    
    

    g, ey, sl = read_data(dir,file_name=file_name)
    
    sl = np.zeros(len(g))       # forecast salary
    
    #---regression model---
    # Coefficient of model 
    Beta0, Beta1, Beta2 = coeff
    

    for ii in range(len(g)):
        sl[ii] = Beta0 + Beta1*g[ii] + Beta2*ey[ii]
        
    #Plot the histogram of estimated salary levels
    
    # fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # ax1.plot(sl)
   
    # ax1.hist( sl,  
    #       bins=np.arange(sl.min(), sl.max()+1000, 1000), 
    #       rwidth=0.5 )
    
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Probability Density')
    # plt.show()
    
    #Calcualted the mean and standard deviation of estimated salary levels
    mu, std = norm.fit(sl)
       
    return sorted(sl)

def model_prior(x,mu,std):
    N_dist = norm.pdf(x,mu,std)
    N_dist = N_dist/N_dist.sum() #Normalized 
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, N_dist)
    ax1.set_title('PDF of prior')
    plt.show()
    
    return N_dist

def likelihood_func(datum, mu, std):
    likelihood_out = sts.gamma.pdf(datum, mu, std) #Note that mu here is an array of values, so the output is also
    
    # fig, ax1 = plt.subplots(figsize=(10, 5))
    # ax1.plot(datum, likelihood_out)
    # ax1.set_title('PDF of likelihood')
    #plt.show()
    
    return likelihood_out/likelihood_out.sum()

def posterior(prior,likihood, obs_data):
    
    unnormalzied_posterior = prior * likihood # 
    p_data = sc.integrate.trapezoid(unnormalzied_posterior, obs_data)
    posterior = unnormalzied_posterior/p_data
    
    return posterior





dir = r'C:\Users\user\Desktop\Lehigh\pre-phd\BayesianStatistics_HawjengChiou\Ch01'
filename= 'data1.csv'


# 1st Bayesian update
model_data = model([5,8,8],dir,filename)
_,_,true_data = read_data(dir,filename)




