#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

#import plotly.offline as py
#import plotly.graph_objs as go

from scipy import stats
from sklearn.cluster import DBSCAN 
from sklearn.neighbors import KernelDensity
import sys
sys.path.append('../..')
import clusterizer
import globals, circuit as cc


THRESHOLD_LEVELS = [0, 1, 5, 10, 50, 100, 500, 1000, 5000]


def get_colors(n):
    colordict = {-1: 'gray'}
    cm = matplotlib.cm.get_cmap('rainbow', n)
    for i in range(n):
        colordict[i] = matplotlib.colors.to_hex(cm(i))
    return colordict


def get_heatmap_frame(circuit):
    frame = circuit.pd.dropna().copy()
    frame['Date/time (UTC)'] = frame['Date/time (UTC)'].dt.round('1d')
    frame['percbin'] = (100 * frame['Location in meters (m)'] / circuit.circuitlength).astype(int)
    grframe = frame.groupby(['Date/time (UTC)', 'percbin'])['Charge (picocoulomb)'].count()
    grframe = grframe.unstack()
    grframefull = pd.DataFrame(index=grframe.index, columns=np.arange(0, 100, 1))
    return grframefull.fillna(grframe)


def offtime(circuit):
    """
    Return minutes without value
    :param circuit:
    :return:
    """
    #TODO: implement this
    pass


def check_normality():
    pass
    #
    # unique = charge.unique()
    # unique = np.sort(unique)
    #
    # start = unique[0]
    # bins = [start]
    # for x in unique:
    #     if (x - start) > 0.5:
    #         bins.append((x+start)/2)
    #         start = x
    # bins.append(unique[-1])
    #
    # digitized = np.digitize(charge, bins)
    # binned_charge = np.histogram(charge, bins)[0]
    #
    # # Distances are bucketed in the data. TODO: manuale define buckets to perform normal fit.
    # mu, std = stats.norm.fit(charge)
    # print(stats.kstest(charge, 'norm', args=(mu, std)))
    # charge = charge - charge.mean()
    #
    # r = stats.norm.rvs(loc=mu, scale=std, size=len(charge))
    # data2 = go.Histogram(x=r)
    # py.plot([data, data2])


def get_precision(circuit, distance, noiselevel=50):
    charge = circuit.pd.dropna()['Location in meters (m)']
    lbound = distance - 5
    rbound = distance + 5

    # Check if charge around distance is higher than threshold level:
    if charge[(charge > lbound) & (charge < rbound)].count() < noiselevel:
        return None

    # Get lower and upperbounds in iterative steps of 5m.
    # Only works for isolated distance!
    while charge[(charge < rbound) & (charge > (rbound - 5))].count() > noiselevel:
        rbound += 5
    while charge[(charge > lbound) & (charge < (lbound + 5))].count() > noiselevel:
        lbound -= 5
    print('Bounds: ', lbound, rbound)
    charge = charge[(charge < rbound) & (charge > lbound)]

    # Plot histogram
    plt.hist(charge)
    plt.show()

    # Precision is approximated by 2 std. Assuming normality and cl = 2.5%.
    std = np.std(charge)
    return 2 * std / circuit.circuitlength


def plot_noise_data(circuit):
    hmframe = get_heatmap_frame(circuit)
    data = [go.Heatmap(visible=False, z=hmframe[hmframe > level].values,
                       x=np.linspace(0, circuit.circuitlength, 100),
                       y=hmframe.index,
                       name='level = ' + str(level)) for level in THRESHOLD_LEVELS]
    data[0]['visible'] = True

    steps = []
    for i in range(len(data)):
        step = dict(
            method='restyle',
            label=str(THRESHOLD_LEVELS[i]),
            args=['visible', [False] * len(data)],
        )
        step['args'][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(steps=steps,
                    pad={'t': 100},
                    currentvalue={'prefix': 'Drempelwaarde #ontladingen: '})]

    layout = dict(title='Ontladingsdichtheid',
                  sliders=sliders,
                  xaxis=dict(title='Locatie (m)'))

    fig = dict(data=data, layout=layout)
    filename = str(globals.plotdir / 'noise-slider.html')
    py.plot(fig, filename=filename)


def get_noise_level(circuit, rel=False):
    """
    Returns dict of noise level (absolute counts or relative counts) above threshold value
    :param circuit:
    :param rel: True is relative counts, False is absolute counts
    :return: dictionary
    """
    if circuit.pd is None:
        noisedict = {'PD > ' + str(level): np.nan for level in THRESHOLD_LEVELS}
    elif rel:
        hmframe = get_heatmap_frame(circuit)
        dim = hmframe.shape[0] * hmframe.shape[1]
        noisedict = {'PD > ' + str(level): (hmframe > level).values.sum() / dim for level in THRESHOLD_LEVELS}
    else:
        hmframe = get_heatmap_frame(circuit)
        noisedict = {'PD > ' + str(level): (hmframe > level).values.sum() for level in THRESHOLD_LEVELS}
    noisedict['circuitnr'] = circuit.circuitnr
    return noisedict


def plot_clusterdensity(clusterframe):
    colordict = get_colors(len(clusterframe['label'].unique()))
    charge_plot = np.linspace(-50, 150, 5000)[:, np.newaxis]
    chargedata = []
    for label in clusterframe['label'].unique():
        charge = clusterframe[clusterframe['label'] == label]['Charge (picocoulomb)'] / 1000
        charge = charge.values.reshape(-1, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(charge)
        log_dens = kde.score_samples(charge_plot)
        chargedata.append(plt.scatter(x=charge_plot[:, 0],
                                     y=np.exp(log_dens),
                                     
                                     )
                          )
    layout = dict(title='Verdeling van ontladingsgroottes clusters',
                  xaxis=dict(title='Ontladingsgrootte (nC)'),
                  yaxis=dict(title='Aantal (%)', range=[0, 0.8]))
    fig = dict(data=chargedata, layout=layout)
    filename = str(globals.plotdir / 'chargedistribution.html')
    plt.plot()


def plot_clusters(circuit, min_samples=500, threshold=0):
    clusterframe = find_clusters(circuit, min_samples, threshold)
    colordict = get_colors(len(clusterframe['label'].unique()))
    clusterdata = []
    for label in clusterframe['label'].unique():
        labelframe = clusterframe[clusterframe['label'] == label]
        clusterdata.append(plt.scatter(x=labelframe['Location in meters (m)'],
                                        y=labelframe['Date/time (UTC)'],
                                      
                                        )
                           )
    layout = dict(showlegend=True,
                  title='Gevonden clusters met DBSCAN',
                  xaxis=dict(title='Locatie (m)'))
    fig = dict(data=clusterdata, layout=layout)
    filename = str(globals.plotdir / 'clusters.html')
    plt.plot()
    return clusterframe


def find_clusters(circuit, min_samples=500, threshold=0):
    eps = circuit.circuitlength / 100

    # Empty frame to return
    emptyframe = pd.DataFrame()
    emptyframe['label'] = np.NAN
    emptyframe['Location (m)'] = np.NAN
    emptyframe['count'] = np.NAN
    emptyframe['Date/time (UTC)'] = np.NAN
    emptyframe['Location in meters (m)'] = np.NAN
    emptyframe['Charge (picocoulomb)'] = np.NAN
    emptyframe['Timedelta (#)'] = np.NAN

    if circuit.pd is None:
        return emptyframe

    clusterframe = circuit.pd.dropna().copy()

    if clusterframe.empty:
        return emptyframe

    # Process data
    clusterframe['Date/time (UTC)'] = clusterframe['Date/time (UTC)'].dt.round('1d')
    #clusterframe['Location (%)'] = np.round((100 * clusterframe['Location in meters (m)'] / circuit.circuitlength)).astype(int)
    clusterframe['Location (m)'] = np.round((clusterframe['Location in meters (m)'])).astype(int)

    # Convert datetime to timedelta integer
    xmin = clusterframe['Date/time (UTC)'].min()
    clusterframe['Timedelta (#)'] = clusterframe['Date/time (UTC)'].apply(lambda x: (x - xmin).days)

    # To speed up DBSCAN we group equal gridpoints together and use counts as weights.
    X = clusterframe[['Location (m)', 'Timedelta (#)']]
    X = X.groupby(['Location (m)', 'Timedelta (#)'])['Location (m)'].count().reset_index(name='count')
    X = X[X['count'] > threshold]
    if X.shape[0] > 0:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X[['Location (m)', 'Timedelta (#)']], sample_weight=X['count'])
        X['label'] = db.labels_
        n_clusters_ = len(set(X['label'])) - (1 if -1 in X['label'] else 0)
    else:
        X['label'] = -1
        n_clusters_ = 0
    print('Found: ', n_clusters_, ' clusters')

    # Add labels to the original ungrouped dataset and plot, datapoints below threshold are labeled as noise
    clusterframe = pd.merge(clusterframe, X, on=['Location (m)', 'Timedelta (#)'], how='left')
    clusterframe['label'] = clusterframe['label'].fillna(-1)
    return clusterframe

# The following statistical functions are necessary to prevent pandas FutureWarning: using a dict with renaming
# is deprecated

def q75(x):
    return x.quantile(0.75)


def q90(x):
    return x.quantile(0.90)


def q95(x):
    return x.quantile(0.95)


def q99(x):
    return x.quantile(0.99)


def q999(x):
    return x.quantile(0.999)


def q9995(x):
    return x.quantile(0.9995)


def q9999(x):
    return x.quantile(0.9999)


def report_clusters(circuit):
    clusterframe = find_clusters(circuit)

    aggdict = {
        'Date/time (UTC)': ['min', 'max'],
        'Charge (picocoulomb)': ['sum',
                                 'min',
                                 'max',
                                 'mean',
                                 'median',
                                 'std',
                                 q75,
                                 q90,
                                 q95,
                                 q99,
                                 q999,
                                 q9995,
                                 q9999],
        'Location in meters (m)': ['min',
                                   'max',
                                   'mean',
                                   'median',
                                   'std'],
        'count': ['sum']
    }
    results = clusterframe.groupby('label').agg(aggdict)
    results.columns = ["_".join(x) for x in results.columns.ravel()]
    results['circuitnr'] = circuit.circuitnr
    return results.reset_index()


def analyse_cc(circuitnr, plot=True, min_samples=500):
    """
    Analyse circuit. Steps: build, judge noise level
    :param circuitnr:
    :return:
    """
    # initialize circuit
    circuit = cc.load(circuitnr)

    # analyse noise level
    if plot:
        plot_noise_data(circuit)
    noise_level = get_noise_level(circuit)
    print('Noise levels: ', noise_level)

    # plot densities
    clusterframe = plot_clusters(circuit, min_samples=min_samples)
    if plot:
        plot_clusterdensity(clusterframe)
    return clusterframe


if __name__ == '__main__':
    print('Ready')
    #circuit = analyse_cc(2368)
    #circuit = cc.load(2063)
    #levels = get_noise_level(circuit)
    #distance = 2236
    #prec = get_precision(circuit, distance)
    #analyse_cc(2063)
    # TODO: 2057
    # 1107: complete noise


