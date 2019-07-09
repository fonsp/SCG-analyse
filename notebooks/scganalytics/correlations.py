#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import plotly.offline as py
import plotly.graph_objs as go


from scganalytics import circuit as cc, globals


def standardize(series):
    """Returns standardized series"""
    return (series - series.mean()) / series.std()


def get_correlation(series1, series2, threshold=10):
    """Return correlation coefficient between two series if #observations is above threshold for both"""
    if np.count_nonzero(series1) > threshold and np.count_nonzero(series2) > threshold:

        return np.corrcoef(series1, series2)[1, 0]
    else:
        return 0


def get_correlationseries(dist1, dist2, bandwidth=0.01, relative=True, timeres='6h', startdate=None, enddate=None):
    charge1 = circuit.get_charge(dist1, bandwidth, relative, timeres, startdate=startdate, enddate=enddate)
    charge2 = circuit.get_charge(dist2, bandwidth, relative, timeres, startdate=startdate, enddate=enddate)
    #correlation = np.corrcoef(charge1, charge2)[1, 0]
    correlation = get_correlation(charge1, charge2)
    return correlation


def get_correlation_frame(circuit, startdate=None, enddate=None, timeres='3h', bandwidth=0.01, relative=True, how='sum'):
    lengths = np.arange(0, circuit.circuitlength, circuit.circuitlength / 200)
    corrframe = pd.DataFrame()
    for x in lengths:
        corrframe[x] = circuit.get_charge(x, bandwidth, relative, timeres, startdate=startdate, enddate=enddate, how=how)
    return corrframe


def get_correlation_matrix(corframe, startdate=None, enddate=None, threshold=10):
    if startdate is not None:
        corframe = corframe[(corframe.index > startdate)]
    if enddate is not None:
        corframe = corframe[corframe.index < enddate]

    for col in corframe.columns:
        if np.count_nonzero(corframe[col]) < threshold:
            corframe[col].values[:] = 0
    return corframe


def plot_correlations(circuit, location, bandwidth=0.01, timeres='6h', startdate=None, enddate=None, how='count'):
    contcharge = circuit.get_charge(location=location, bandwidth=bandwidth, timeres=timeres, how=how)
    contcharge = contcharge[contcharge != 0]
    ptime = circuit.propagation.set_index('Date/time (UTC)', drop=True).resample(timeres).mean()

    contpd = go.Scattergl(
        name='Chargesum',
        x=contcharge.index,
        y=standardize(contcharge),
        mode='lines',
        marker=dict(size=4,
                    color="#6897BB"),
    )
    proptime = go.Scattergl(
        name='Propagation time (ns)',
        x=ptime.index,
        y=standardize(ptime['Propagation time (ns)']),
        mode='lines',
        yaxis='y2'
    )
    layout = go.Layout(
        title='Discharges vs. Propagation time',
        yaxis=dict(title='Charge (nC)'),
        yaxis2=dict(title='Propagation time (ns)',
                    overlaying='y',
                    side='right'),
        showlegend=True)
    fig = dict(data=[contpd, proptime], layout=layout)
    py.plot(fig)
    return [contpd, proptime], layout


def plot_pd_proptime(circuit, location, bandwidth=0.01, timeres='6h', startdate=None, enddate=None, how='count'):
    # TODO: implement startdate, enddate

    charge = circuit.get_charge(location, 0.02, True) / 1000
    contcharge = circuit.get_charge(location=location, bandwidth=bandwidth, timeres=timeres, how=how)
    charge = charge[charge != 0]
    contcharge = contcharge[contcharge != 0]
    ptime = circuit.propagation.set_index('Date/time (UTC)', drop=True)

    scatter = go.Scattergl(
        name='PD charges (nC)',
        x=charge.index,
        y=charge,
        mode='markers',
        marker=dict(size=4,
                    color="#6897BB")
    )
    contpd = go.Scattergl(
        name='Chargesum',
        x=contcharge.index,
        y=contcharge,
        mode='lines',
        marker=dict(size=4,
                    color="#7F1F7D"),
    )
    proptime = go.Scattergl(
        name='Propagation time (ns)',
        x=ptime.index,
        y=ptime['Propagation time (ns)'],
        mode='lines',
        yaxis='y2'
    )
    layout = go.Layout(
        title='Discharges vs. Propagation time',
        yaxis=dict(title='Charge (nC)'),
        yaxis2=dict(title='Propagation time (ns)',
                    overlaying='y',
                    side='right'),
        showlegend=True)
    fig = dict(data=[scatter, proptime, contpd], layout=layout)
    py.plot(fig)
    return [charge], layout


def plot_ptimecorr_data(circuit, timeres='12h', timewindow=14):
    corframe = get_correlation_frame(circuit, timeres=timeres, how='count')
    ptime = circuit.propagation.set_index('Date/time (UTC)', drop=True).resample(timeres).mean()
    corframe = pd.merge(corframe, ptime, how='inner', left_index=True, right_index=True)
    corframe = corframe.replace(0, np.nan)
    rollingframe = corframe.rolling(window=timewindow, min_periods=10).corr(corframe[corframe.columns[-1]])
    rollingframe = rollingframe.drop('Propagation time (ns)', 1)
    rollingframe = rollingframe.clip(-1, 1)
    rollingframe = rollingframe.replace(1, 0)
    #rollingframe = rollingframe[abs(rollingframe) > 0.4]

    data = [go.Heatmap(
        z=rollingframe,
        x=rollingframe.columns,
        y=rollingframe.index,
        zmin=-1,
        zmax=1
    )]
    layout = dict(title='Correlatie PD-Propagatietijd Hittekaart',
                  xaxis=dict(title='Locatie (m)',
                             range=[0, circuit.circuitlength]),
                  yaxis=dict(title='Datum',
                             range=[0, circuit.circuitlength]))

    fig = dict(data=data, layout=layout)
    filename = str(globals.plotdir / 'proptimecorr.html')
    py.plot(fig, filename=filename)
    return


def demo_ptimecorr():
    timeres = '12h'
    circuit = cc.load(2145)
    charge = circuit.get_charge(942, bandwidth=0.01, relative=True, timeres=timeres, how='count')
    charge.name = 'charge'
    ptime = circuit.propagation.set_index('Date/time (UTC)', drop=True).resample(timeres).mean()
    frame = pd.concat([charge, ptime], axis=1, join='inner')
    startdate = pd.datetime(2018, 3, 7)
    enddate = pd.datetime(2018, 3, 14)
    frame = frame[frame.index < enddate]
    frame = frame[frame.index > startdate]
    plt.plot(standardize(frame['charge']))
    plt.plot(standardize(frame['Propagation time (ns)']))
    plt.show()
    print(get_correlation(frame['charge'], frame['Propagation time (ns)']))


def plot_distcorr_data(circuit, frequency='14d'):
    corframe = get_correlation_frame(circuit)
    daterange = pd.date_range(corframe.index[0], corframe.index[-1], freq=frequency)
    data = [go.Heatmap(visible=False, z=get_correlation_matrix(corframe, date, date + pd.DateOffset(days=5)).corr().fillna(0).values,
                       zmin=-1,
                       zmax=1,
                       x=corframe.columns,
                       y=corframe.columns,
                       name='level = ' + str(date)) for date in daterange]
    data[0]['visible'] = True

    steps = []
    for i in range(len(data)):
        step = dict(
            method='restyle',
            label=str(daterange[i]),
            args=['visible', [False] * len(data)],
        )
        step['args'][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(steps=steps,
                    pad={'t': 100},
                    currentvalue={'prefix': 'Gekozen periode (startdatum): '})]

    layout = dict(title='Correlaties per 14 dagen',
                  sliders=sliders,
                  xaxis=dict(title='Locatie (m)',
                             range=[0, circuit.circuitlength]),
                  yaxis=dict(title='Locatie (m)',
                             range=[0, circuit.circuitlength]))

    fig = dict(data=data, layout=layout)
    filename = str(globals.plotdir / 'noise-slider.html')
    py.plot(fig, filename=filename)
    return data


if __name__ == '__main__':
    #circuit = cc.load(4091)
    circuit = cc.load(2145)
    xvalues = np.arange(0, circuit.circuitlength, circuit.circuitlength/100)
    #plot_pd_proptime(circuit, location=942)
    plot_correlations(circuit, location=942)
    #corr = [get_correlationseries(870, x, enddate=pd.datetime(2019, 4, 3)) for x in xvalues]
    #plt.plot(xvalues, corr)

    #plot_distcorr_data(cc.load(4091))

    #plot_correlation_data(cc.load(2145))

    #plot_correlation_data(cc.load(2368))






