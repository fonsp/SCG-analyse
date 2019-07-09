#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import globals


def ewma_vectorized(data, alpha):
    # Not suited for large inputs!
    alpha_rev = 1-alpha

    scale = 1/alpha
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha**(r+1)
    pw0 = alpha_rev*alpha**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def ewmavar_vectorized(data, avg, alpha):
    avg_shift = np.roll(avg, 1)
    avg_shift[0] = 0
    var = (avg_shift-data)**2
    var[0] = 0
    ewmavar = ewma_vectorized(var, alpha)
    return ewmavar


def ewma2(x, a=0.8, b=0.2):
    mu = ewma_vectorized(x, a)
    var = ewmavar_vectorized(x, mu, 1-b)
    return mu, var


def cusum(x, threshold, reset_drift=True, show=True):
    """
    Implementation of the classical cusum scheme. If reset_drift is set to True, drift will be estimated after an alarm.
    :param x: array like
    :param threshold: threshold for alarm
    :param reset_drift: reset drift after alarm or not
    :param show: plot result
    :return: returns
    """

    x = np.atleast_1d(x).astype('float64')
    g = np.zeros(x.size)
    drift = 0.0
    ta = np.array([], dtype=int)
    tastart = np.array([0])
    for i in range(1, x.size):
        g[i] = g[i-1] + x[i] - drift
        if g[i] < 0:
            g[i] = 0
            tastart = np.append(tastart, i)
        if g[i] > threshold:
            ta = np.append(ta, i)
            if reset_drift:
                drift = (x[i] - x[tastart[-1]]) / (ta[-1] - tastart[-1])
                print('Drift reset to: ' + str(drift))
            g[i] = 0
            tastart = np.append(tastart, i)
    plot_if_requested(g, show, x)
    return ta, tastart, g


def linear(x, threshold, show=True):
    x = np.atleast_1d(x).astype('float64')
    g = np.zeros(x.size)
    g[0] = x[0]
    drift = 0.0
    ta = np.array([], dtype=int)
    tastart = np.array([0])
    for i in range(1, x.size):
        g[i] = g[i-1] + drift
        if abs(x[i] - g[i]) > threshold:
            ta = np.append(ta, i)
            g[i] = x[i]
            drift = (x[i] - x[tastart[-1]]) / (ta[-1] - tastart[-1])
            print('Drift reset to: ' + str(drift))
            tastart = np.append(tastart, i)
    plot_if_requested(g, show, x)
    return ta, tastart, g


def plot_if_requested(g, show, x):
    if show:
        plt.plot(x)
        plt.plot(g)
        plt.show()


def validate_alarms(x, y, mu, ta, threshold=5):
    # Only give alarm if pd charge is over 1 nC:
    ta_new = ta[x[ta] > threshold]

    # Allow a few days startup period:
    # TODO: refine this
    ta_new = ta_new[ta_new > 2]
    ta_new = ta_new[ta_new < x.size - 1]

    # Only give alarm if current and next discharges > 10
    temp1 = ta_new[y[ta_new+1] > 10]
    temp2 = ta_new[y[ta_new] > 10]
    ta_new = np.unique(np.concatenate((temp1, temp2), 0))

    # Only give alarm if next measurment is above ewma or # > 70
    temp1 = ta_new[x[ta_new+1] > mu[ta_new+1]]
    temp2 = ta_new[y[ta_new] > 70]
    temp3 = ta_new[y[ta_new+1] > 70]
    ta_new = np.unique(np.concatenate((temp1, temp2, temp3), 0))

    return ta_new


def ewma(x, a=0.8, b=0.2, k=2):
    # Solution: just use pandas rolling ewma en std.
    """
    Exponentially weighted moving average based alarm
    :param x: Pandas series
    :param a: decay parameter for sequence. Higher value means less weight on most recent observations.
    :param b: decay parameter for variance. Higher value gives less weight on most recent observations.
    :param k: number of standard deviations for alarm
    :return:
    """
    # Convert series x to numpy series of floats
    x = np.atleast_1d(x).astype('float64')

    mu = np.zeros(x.size)
    var = np.zeros(x.size)
    ta = np.array([], dtype=int)
    mu[0] = x[0]
    var[0] = 0.0
    for i in range(1, x.size):
        # TODO: vectorize this loop
        mu[i] = a*mu[i-1] + (1-a)*x[i]
        var[i] = (1-b)*var[i-1] + b*(x[i]-mu[i-1])**2
        if x[i] - mu[i-1] > k*np.sqrt(var[i]):
            ta = np.append(ta, i)
    return ta, mu, var


def plot(x, a=0.8, b=0.2, k=2, title=None, method='ewma'):
    ta, mu, var = ewma(x, a, b, k)
    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.autofmt_xdate()
    ax.plot(x, color='orange', label='signal')
    mu = pd.Series(mu, index=x.index)
    var = pd.Series(var, index=x.index)
    ax.plot(mu, label='ewma')
    ax.fill_between(x.index, mu-k*np.sqrt(var), mu+k*np.sqrt(var), alpha=0.4)
    ax.plot(x.index[ta], x[x.index[ta]], 'o', mfc='r', mec='r', mew=1, ms=5, label='alarm')
    ax.legend()
    plt.show()


def main():
    """
    The main function which gets called when this module is ran as a script.
    This main function show some examples of the different alarms.
    """
    rng = pd.date_range('1/1/2011', periods=300, freq='H')
    x = np.random.randn(300)/5
    x[100:200] += np.arange(0, 4, 4/100)
    ta, mu, var = ewma(x, 0.8)
    ts = pd.Series(mu, index=rng)
    plot(ts, 0.8, 0.1, 2, 'test', 'ewma')
    

if __name__ == '__main__':
    main()



