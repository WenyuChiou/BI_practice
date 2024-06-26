# Third example of Bayesian Inference
# By Wenyu Chiou 

""" 
The algorithm used in this example is Metropolis-Hastings, which is ofthen used on Bayesian Inference to determine
whether the new values (forecastes) should be accepted or reserve the current value for calculating posteriors.

Note: MH algorithm is not an optimization method.

The MH algorithm is defined as:
    Step 1: Initialize the value of the parameter X at x(i)
    Step 2: Use a proposal distribution q(x(i+1)|(x(i))) to generate a new value from the old one x(i)
    Step 3: Compute the probability of accepting the new values as:
        P(x(i+1)|x(i)) = min(1, P(x(i+1)|q(x(i)|x(i+1))/p(x(i)q(x(i+1)|x(i)))))
    Step 4: if Pa(x(i+1)|x(i)) > R where R ~ U(0,1), save the values, otherwise save the old one.
    Step 5: Iterate 2 to 4 until a sufficiently large sample of values has been generated.

link: https://bayesiancomputationbook.com/markdown/chp_01.html
    
    """
    
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

# definition of posterior distribution given unknown prior distribution and likelihood function
# P(x(i+1)|q(x(i))) can be computed by Bayeisan method (Posterior)
def post(theta, data, alpha = 1, beta = 1):
    if (0 <= theta <= 1):
        prior = stats.beta(alpha, beta).pdf(theta)
        like = stats.bernoulli(theta).pmf(data).prod()

        prob = prior * like
    else:
        prob = -np.inf
    return prob

# Generate random data
data = stats.bernoulli(0.7).rvs(20)

#MH algorithm
n_iter = 1000
can_sd = 0.05
alpha = beta = 1
theta = 0.5
trace = {"theta":np.zeros(n_iter)}
p2 = post(theta, data, alpha, beta)

for iter in range(n_iter):
    theta_can = stats.norm(theta, can_sd).rvs(1) #Generate a proposal distribution by sampling from normal distriubtion
    p1 = post(theta_can, data, alpha, beta) #Evaluate the posterior at the new generated values (theta_can)
    pa = p1/p2 #Computing the probability of acceptance
    
    if pa > stats.norm(0,1).rvs(1): # if pa is less than the random sample from normal distribution, the state tends to be stable
        theta = theta_can
    else:
        p2 = p1
        
    trace["theta"][iter] = theta

_, axes = plt.subplots(1,2, sharey=True)
axes[0].plot(trace['theta'], '0.5')
axes[0].set_ylabel('θ', rotation=0, labelpad=15)
axes[1].hist(trace['theta'], color='0.5', orientation="horizontal", density=True)
axes[1].set_xticks([])
plt.show()
    
# Declare a model in PyMC
# if __name__ == '__main__':
#     with pm.Model() as model:
#         # Specify the prior distribution of unknown parameter
#         θ = pm.Beta("θ", alpha=1, beta=1)

#         # Specify the likelihood distribution and condition on the observed data
#         y_obs = pm.Binomial("y_obs", n=1, p=θ, observed=data)

#         # Sample from the posterior distribution
#         idata = pm.sample(1000, return_inferencedata=True)


