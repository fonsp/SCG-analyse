#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


from scganalytics import globals, datadump as dd

# REMARK: selecting clusters with low std in location is essential in classifying level warning vs. noise.

FEATURELIST = ['Charge (picocoulomb)_median',
               'Charge (picocoulomb)_mean',
               'Charge (picocoulomb)_std',
               'Charge (picocoulomb)_min',
               'Charge (picocoulomb)_max',
               'Charge (picocoulomb)_q90',
               'Charge (picocoulomb)_q99',
               'Charge (picocoulomb)_q999',
               'Location in meters (m)_std']
LABELS = ['c-level']


def get_data():
    """
    Returns the report table from the datadump class.
    :return:
    """
    dump = dd.DataDump()
    dump.read()
    report = dd.combine_data(dump)

    # Select resin and oil
    #report = report[report['nearest-jointtype'].isin(['Joint (oil)', 'Joint (resin)'])]

    # Remove some old noisy circuits
    report = report[report['circuitnr'] != 226]
    return report


def get_SOT():
    dump = dd.DataDump()
    dump.read()
    successen = pd.read_excel(globals.successendir / '20190415_SCG_Dashboard_april_half.xlsx',
                              sheet_name='Data resultaten en successen',
                              converters={'Plaatsings-ID': str, 'Datum event': pd.to_datetime})
    successen = successen[(successen['Type resultaat'] == 'PD') & (successen['Kwalificatie resultaat'] == 'Succes')]
    successen['circuitnr'] = successen['Plaatsings-ID'].apply(lambda x: dump.get_ccnumber_from_id(x))
    successen = successen.sort_values('circuitnr')
    return successen


def merge_report_SOT(report):
    # Outer join with circuitnr reports in sot list with sot list.
    sot = get_SOT()
    report = report[report.circuitnr.isin(sot.circuitnr)]

    sot = sot[sot['Component verzonden'] != 'Nee']
    merged = sot.merge(report, on='circuitnr', how='outer')

    merged['Date/time (UTC)_min'] = pd.to_datetime(merged['Date/time (UTC)_min'])
    merged['Date/time (UTC)_max'] = pd.to_datetime(merged['Date/time (UTC)_max'])

    # Select rows where sot location falls within cluster
    merged['sot_loc_in_cluster'] = (merged['Gemelde afstand [m]'] > merged['Location in meters (m)_min']) & (merged['Gemelde afstand [m]'] < merged['Location in meters (m)_max'])
    merged['sot_time_in_cluster'] = (merged['Datum event'] < merged['Date/time (UTC)_max']) & (merged['Datum event'] > merged['Date/time (UTC)_min'])

    # Circuit 2870 valt uit voorgaande selectie. Betreft kabeldom oliemof met vocht in kabel. Locatie mof niet in cluster.

    merged['nearestjoint'] = merged['nearest-jointtype']

    temp = merged[['circuitnr',
                   'Volgnummer',
                   'Type component (input SOT / Nestor)',
                   'nearest-jointtype',
                   'nearestjoint',
                   'Gemelde afstand [m]',
                   'Location in meters (m)_median',
                   'Date/time (UTC)_min',
                   'Date/time (UTC)_max']]

    # Volgnummer 115 is dubbeling van 135 (circuit 2491)
    merged = merged[merged['Volgnummer'] != 115.0]

    # Volgnummer 131 is dubbeling van 159 (circuit 2718)
    merged = merged[merged['Volgnummer'] != 131.0]

    # Circuit 2063 heeft cluster waar een warmtekrimpmof vervangen is door een koudekrimp.
    merged['nearest-jointtype'][merged['Volgnummer'] == 90.0] = 'Joint (resin)'
    merged['nearest-jointtype'][merged['Volgnummer'] == 156.0] = 'Joint (heat shrink)'
    merged['nearest-jointtype'][merged['Volgnummer'] == 155.0] = 'Joint (heat shrink)'
    merged['nearest-jointtype'][merged['Volgnummer'] == 133.0] = 'Joint (heat shrink)'
    merged['nearest-jointtype'][merged['Volgnummer'] == 135.0] = 'Joint (heat shrink)'
    merged['nearest-jointtype'][merged['Volgnummer'] == 159.0] = 'Joint (heat shrink)'
    merged['nearest-jointtype'][merged['Volgnummer'] == 158.0] = 'Joint (heat shrink)'
    merged['nearest-jointtype'][merged['Volgnummer'] == 136.0] = 'Joint (heat shrink)'

    # Op deze locatie zet eerst een nekaldietmof die eruit is gehaald.
    merged['nearestjoint'][(merged['Date/time (UTC)_min'] == pd.datetime(2017, 11, 30)) & (merged['circuitnr'] == 2368)] = 'Joint (resin)'

    # TODO Andere clusters kunnen eigenlijk weg.
    merged['nearestjoint'][(merged['Date/time (UTC)_min'] == pd.datetime(2018, 8, 3)) & (merged['circuitnr'] == 2491)] = 'Joint (heat shrink)'
    return merged


def wrongjoints(data):
    ccjointtuples = list(zip(data['circuitnr'].values, data['nearest-jointlocation'].values))
    wrongjoints = [(3166, 30.0),  # Resin, waarschijnlijk vocht in kabel op plek van resin mof
                   (1508, 2811.83),  # Resin, typisch hoog ontladingspatroon -> reflecties
                   (1957, 81.0),  # Resin, typisch hoog ontladingspatroon -> reflecties
                   (1963, 4869.0),  # Resin, typisch hoog ontladingspatroon -> reflecties
                   #(1351, 1916.31),  # Duidelijk reflectiepatroonover hele lengte, maar geen ontlading. 2016.
                   #(1333, 600.94),  # Alleen ontladingen op deze locatie en iets eromheen, onregelmatig.
                   #(1293, 105.17),  # Reflecties tussen twee MSR, 2016
                   (1357, 1226.1),  # Resin, oliemof licht dichterbij centrum cluster.
                   ]
    wjlist = [x in wrongjoints for x in ccjointtuples]
    wjlist = [not x for x in wjlist]
    data['wrongjoints'] = wjlist
    return data


def get_features(data):
    # Clean data
    data = data[FEATURELIST+LABELS]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    features = data[FEATURELIST]
    labels = data[LABELS]
    labels = labels.values.ravel()
    return features, labels


def plot_clusters(filtered):
    # TODO 1674 wrong joint: should be oil nearest, not resin.
    clusterdata = []
    for label in filtered['nearest-jointtype'].unique():
        # labelframe = filtered[filtered['nearest-jointtype'] == label]
        labelframe = filtered[filtered['nearest-jointtype'] == label]
        textlabel = labelframe['circuitnr'].astype(str) + ' (' + labelframe['nearest-jointlocation'].astype(str) + 'm)'
        clusterdata.append(go.Scattergl(x=labelframe['Charge (picocoulomb)_median'],
                                        y=labelframe['Charge (picocoulomb)_q9999'],
                                        name=str(label),
                                        mode='markers',
                                        text=textlabel))
    layout = dict(showlegend=True,
                  title='Clusters van ontladingen dichtbij moffen',
                  xaxis=dict(title='Mediaan ontladingen (pC)'),
                  yaxis=dict(title='Dichtheid (per da)'))
    fig = dict(data=clusterdata, layout=layout)
    py.plot(fig)


def plot_clusterhist(filtered):
    clusterdata = []
    for label in filtered['nearest-jointtype'].unique():
        labelframe = filtered[filtered['nearest-jointtype'] == label]
        clusterdata.append(go.Histogram(x=labelframe['Charge (picocoulomb)_mean'],
                                        xbins=dict(
                                            start=0,
                                            end=50000.0,
                                            size=1000
                                        ),
                                        name=str(label)))
    layout = go.Layout(showlegend=True)
    fig = dict(data=clusterdata, layout=layout)
    filename = str(globals.plotdir / 'allclusters.html')
    py.plot(fig, filename=filename)


if __name__=='__main__':
    report = get_data()
    report = report[report['Location in meters (m)_std'] < 50]
    sotreport = merge_report_SOT(report)
    features, labels = get_features(report)
    scaled_data = preprocessing.scale(features)

    lr = LogisticRegression(solver='liblinear', C=0.9)
    lr.fit(scaled_data, labels)
    proba = lr.predict_proba(scaled_data)
    score = lr.score(features, labels)

