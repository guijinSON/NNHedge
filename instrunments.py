import math 
import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(S0=100, r=0.01,sigma=0.2, T=1, ts=30, n_path=10000):
    dt = T/ts
    paths = np.zeros((ts+1,n_path))
    paths[0] = S0
    for t in range(1,ts+1):
        paths[t] = paths[t-1] * np.exp((r-0.5*(sigma**2)) + sigma * math.sqrt(dt) * np.random.standard_normal(n_path))
    return paths.T
