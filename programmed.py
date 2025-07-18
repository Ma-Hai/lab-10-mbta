# IMPORTS
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha":0.25,
})

def plot_ecdf(data, ax=None,**kw):
    ec = ECDF(data)
    ax=ax or plt.gca()
    ax.step(ec.x,ec.y,where='post',**kw)
    ax.set_xlabel("Wait (min)")
    ax.set_ylabel("F_hat(x)")
    return ax

def aic(logL, k):
    return 2*k - 2*logL

def mean_from_params(dist_name, params):
    if dist_name == "gamma":
        a, loc, scale = params # a = shape (k)
        return loc + a*scale # Gamma mean = loc + k*theta
    elif dist_name == "weibull_min":
        c, loc, scale = params # c = shape
    # Weibull mean uses the gamma function:
        return loc + scale*st.gamma(1 + 1/c)
    elif dist_name == "lognorm":
        s, loc, scale = params # s = sigma (log-scale SD); scale = exp(mu)
        # mean = loc + scale * exp( sigma^2 / 2 )
        return loc + scale * np.exp(s**2 / 2)
    else:
        return np.nan # unknown model

#LOAD DATA
df = pd.read_csv("mbta_wait_times.csv")
pre = df.loc[df.period=="PRE","wait_min"].to_numpy()
post = df.loc[df.period=="POST","wait_min"].to_numpy()

print(f"PRE n={pre.size}, mean = {pre.mean():.2f} min")
print(f"POST n={post.size}, mean = {post.mean():.2f} min")


#Q 1.1: How many total waits were recorded in each period
## For the PRE, 600 waits were collected. For POST, 650 were collected
#Q 1.2: What are the raw sample means?
## The raw mean of PRE is 3.85, the raw mean of POST is 4.84

# 22 GRID
fig, axs = plt.subplots(2,2,figsize=(9,6),sharex='col')
#TOP ROW
axs[0,0].hist(pre,bins='auto',density=True,alpha=.65, edgecolor='k')