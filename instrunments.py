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

def d1(S0, K, dt, r, sigma):
    return (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * dt) / (sigma * np.sqrt(dt))

def d2(S0, K, dt, r, sigma):
    return d1(S0, K, dt, r, sigma) - sigma * np.sqrt(dt)

def price_put_BS(S0, K, dt, r, sigma):
    return (stats.norm.cdf(-d2(S0, K, dt, r, sigma)) * K * np.exp(-r * dt) - 
                           stats.norm.cdf(-d1(S0, K, T, r, sigma)) * S0)

def price_call_BS(S0, K, dt, r, sigma):
    return (stats.norm.cdf(d1(S0, K, dt, r, sigma)) * S0 - 
            stats.norm.cdf(d2(S0, K, dt, r, sigma)) * K * np.exp(-r * dt))

def delta_put_BS(S0, K, dt, r, sigma):
    return -stats.norm.cdf(-d1(S0, K, dt, r, sigma))

def delta_call_BS(S0, K, dt, r, sigma):
    return stats.norm.cdf(d1(S0, K, dt, r, sigma));
