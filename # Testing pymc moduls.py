# Testing pymc moduls 
"""
This example is provided by guideline of pcmc and for Poesterior predicitive check (PPCs),
which is an useful way to validate a model. 

PPCs is used to assess the degree to which data generated from the model
deviate from the true distribution (or model).
    
link: https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/posterior_predictive.html
"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import xarray as xr

from scipy.special import expit as logistic


print(f"Running on PyMC v{pm.__version__}")

az.style.use("arviz-darkgrid")

# Construct a generator with given the numbers of the seeds
RANDOM_SEED = 58
rng = np.random.default_rng(RANDOM_SEED)

# Define the process of standardization: (x - mu)/std
def standardize(series):
    """Standardize a pandas series"""
    return (series - series.mean()) / series.std()

"""
Generate random data (predictor )from a normal distribution and calculate true means
with a simple linear regression model

Assume true mean can be estimated from linear regression model by given by predictor

"""
N = 100

# mu = a + b*x; predicto
true_a, true_b, predictor = 0.5, 3.0, rng.normal(loc=2, scale=6, size=N)
true_mu = true_a + true_b * predictor
true_sd = 2.0

#Generate the random data from a normal distribution with true mean 
outcome = rng.normal(loc=true_mu, scale=true_sd, size=N)
print(outcome)

f"{predictor.mean():.2f}, {predictor.std():.2f}, {outcome.mean():.2f}, {outcome.std():.2f}"

predictor_scaled = standardize(predictor)
outcome_scaled = standardize(outcome)

f"{predictor_scaled.mean():.2f}, {predictor_scaled.std():.2f}, {outcome_scaled.mean():.2f}, {outcome_scaled.std():.2f}"

# Produce a weakly informative prior (mu: 0, std:0.5 or 1)

with pm.Model() as model_1:
    a = pm.Normal("a", 0.0, 0.5)
    b = pm.Normal("b", 0.0, 1.0)

    mu = a + b * predictor_scaled
    sigma = pm.Exponential("sigma", 1.0)

    pm.Normal("obs", mu=mu, sigma=sigma, observed=outcome_scaled)
    idata = pm.sample_prior_predictive(draws=50, random_seed=rng)

_, ax = plt.subplots()

x = xr.DataArray(np.linspace(-2, 2, 50), dims=["plot_dim"])
prior = idata.prior
y = prior["a"] + prior["b"] * x

ax.plot(x, y.stack(sample=("chain", "draw")), c="k", alpha=0.4)

ax.set_xlabel("Predictor (stdz)")
ax.set_ylabel("Mean Outcome (stdz)")
ax.set_title("Prior predictive checks -- Flat priors");


if __name__ == '__main__':
    with model_1:
        idata.extend(pm.sample(1000, tune=2000, random_seed=rng))

    az.plot_trace(idata)
    plt.show()





