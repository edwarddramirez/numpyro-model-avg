import numpy as np # for numerical operations
import numpyro.distributions as dist
from scipy.stats import chi2

# poisson interval
def poisson_interval(k, alpha=0.32): 
    """ 
    Uses chi2 to get the poisson interval.
    """
    a = alpha
    low, high = (chi2.ppf(a/2, 2*k) / 2, chi2.ppf(1-a/2, 2*k + 2) / 2)
    if k == 0: 
        low = 0.0
    return k - low, high - k

# poisson likelihood function
def log_like_poisson(mu, data):
    return dist.discrete.Poisson(mu).log_prob(data)