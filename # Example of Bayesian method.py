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
import statistics 

#Step 2: Framing statistic model

def stat_model(g,ey):
    g,ey = np.array(g), np.array(ey) # gender and eductaion year [yr]
    sl = np.zeros(len(g))       # forecast salary 

    #---regression model---
    # Coefficient of model 
    Beta0 = 1 
    Beta1 = 2
    Beta2 = 3

    for ii in range(len(g)):
        sl[ii] = Beta0 + Beta1*g[ii] + Beta2*ey[ii]
        
    std = statistics.variance(sl)
    
    print(f'stdev: {std}')
    return sl, std

test ,st = stat_model([1,1,1,1,0],[20,10,15,18,17])






    





    










