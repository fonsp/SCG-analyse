#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd


from scganalytics import globals, circuit as cc, datadump as dd


def recent_clusters(dump):
    report = dd.combine_data(dump)

    recent_clusters = report[pd.to_datetime(report['Date/time (UTC)_max']) > pd.datetime(2019, 5, 1)]
    return recent_clusters


def remove_rmus(report):
    normu = report[report['nearest-rmudistance'] > 10]
    jointrmu_delta = abs(normu['nearest-jointdistance'] - normu['nearest-rmudistance'])
    return normu[jointrmu_delta > 0.01]


def remove_noiseclusters(report):
    report = report[report['cluster-count'] > 1]
    return report[report['label'] > -1]


def remove_noise(report):
    report = report[report['Charge (picocoulomb)_mean'] > 500]
    return report[report['PD > 50'] > 0]


def validate_clusterreport(report):
    report = remove_noiseclusters(report)


    # TODO: make this a function that outputs Joint TRUE or FALSE.
    report = report[(report['nearest-jointdistrel'] < 0.02) | (report['nearest-jointdistance'] < 25)]

    # filtered = filtered[filtered['Location (m)_std'] < 50]
    return report


if __name__ == '__main__':
    dump = dd.DataDump()
    dump.read()
    recent = recent_clusters(dump)
    level2 = recent[recent['c-level2'] == True]
