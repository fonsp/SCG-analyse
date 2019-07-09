#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Instructions for adding new data to datadump:
# 1. Extract DNV GL zip file to folder
# 2. Run convert_to_csv to folder
# 3. Copy folder to datadump-server folder
# 4. Run cache_circuits method from DataDump object with overwwrite=True.
# TODO: write script for the above steps

# TODO: clean up the datadump module. It contains too many functions.

from pathlib import Path

import datetime
import pandas as pd
import pickle
#import plotly.offline as py
#import plotly.graph_objs as go
import numpy as np


import globals, circuit as cc


def load(datadumpname):
    """
    Load a pickled datadump from cache
    :param datadumpname:
    :return: datadump object
    """
    dumpcachepath = globals.cache / (datadumpname + '.pickle')
    dp = None
    if dumpcachepath.is_file():
        with open(dumpcachepath, 'rb') as f:
            dp = pickle.load(f)
    return dp


def save(datadump):
    """
    Save a datadump object to pickle
    :param datadump:
    """
    with open(globals.cache / (datadump.name + '.pickle'), 'wb') as f:
        pickle.dump(datadump, f, pickle.HIGHEST_PROTOCOL)


def convert_to_csv(datadump):
    """
    Converts the .xlsx files of a DataDump instance to .csv
    :param datadump: instance of a DataDump
    :return: None
    """
    for file in (globals.datadir / 'datadumps' / datadump).glob('**/*.xlsx'):
        if not file.with_suffix('.csv').is_file():
            try:
                data_xls = pd.read_excel(file)
                data_xls.to_csv(file.with_suffix('.csv'), sep=';', index=False)
            except Exception as e:
                print('Conversion to csv failed for: ', file, 'Exception: ', e)


def get_alarms(datadump):
    df = datadump.circuits_df['circuitnr'].copy()
    alarms = []
    for circuitnr in datadump.circuits_df['circuitnr']:
        circuit = cc.load(circuitnr)
        print('Current circuit: ', circuit.circuitnr)
        alarmcount = circuit.get_alarms().sum()
        alarms.append(alarmcount)
    df['alarmcount'] = alarms
    df.to_csv(globals.datadir / 'alarms.csv', sep=';')
    return df


class DataDump:
    """
    Class that holds all relevant information of a datadump directory.
    """

    def __init__(self, dumpname=globals.datadumpdir.stem):
        self.name = dumpname
        self.date = None
        self.dir = Path(globals.datadumpsdir / dumpname)
        self.circuitdict = None  # list of all selected circuit objects in the datadump
        self.circuits_df = None  # dataframe of data of selected circuits
        self.components = None
        self.date = None
        self.joints = None
        self.is_read = False

    def read(self):
        """
        Initialises all circuits in the SCG datadump and processes the resulting foldernames.
        Returns a dataframe with all circuits, or a selection
        """
        if not self.is_read:
            print('Reading datadump from directory: ', self.dir)
            initialised = 0
            circuitnrs = []
            circuitids = []
            circuitstartlocs = []
            circuitendlocs = []
            dates = []
            circuitdict = {}
            for subdir in self.dir.iterdir():
                circuit = cc.Circuit(subdir.name)
                circuitdict[circuit.circuitnr] = circuit
                circuitnrs.append(circuit.circuitnr)
                dates.append(circuit.date)
                circuitids.append(circuit.circuitid)
                circuitstartlocs.append(circuit.startlocation)
                circuitendlocs.append(circuit.endlocation)
                if circuit.init:
                    initialised += 1
            dumpdict = {
                'circuitnr': circuitnrs,
                'date': dates,
                'circuitid': circuitids,
                'circuitstartloc': circuitstartlocs,
                'circuitendloc': circuitendlocs,
            }
            ccframe = pd.DataFrame.from_dict(dumpdict)
            grouped = ccframe.groupby('circuitnr', as_index=False)

            self.circuits_df = grouped.agg({'date': [min, max],
                                            'circuitid': [max],
                                            'circuitstartloc': [max],
                                            'circuitendloc': [max]})
            self.circuits_df.columns = ['circuitnr',
                                        'startdate',
                                        'enddate',
                                        'circuitid',
                                        'circuitstartloc',
                                        'circuitendloc']
            self.circuits_df['circuitnr'] = pd.to_numeric(self.circuits_df['circuitnr'])
            self.date = self.circuits_df['enddate'].max()
            self.circuitdict = circuitdict
            self.is_read = True

    def check_files(self):
        """
        Function that checks if circuits contain necessary files
        :return:
        """
        missingcircuits = []
        for subdir in self.dir.iterdir():
            if not (subdir / 'pd.csv').is_file():
                missingcircuits.append(subdir.stem)
        return missingcircuits

    def get_ccnumber_from_id(self, idstring):
        self.circuits_df['circuitid'] = self.circuits_df['circuitid'].astype(str)
        filtered = self.circuits_df[self.circuits_df['circuitid'].str.contains(idstring)]
        if filtered.shape[0] == 1:
            return filtered['circuitnr'].iloc[0]
        else:
            return np.nan

    def cache_circuits(self, overwrite=False):
        total = len(self.circuits_df['circuitnr'])
        for index, circuitnr in enumerate(self.circuits_df['circuitnr']):
            print('Caching circuit: ', index+1, 'out of ', total)
            circuit = cc.MergedCircuit(circuitnr)
            if not overwrite:
                if not (globals.circuitcache / (str(circuit.circuitnr) + '.pickle')).is_file():
                    circuit.build()
                    cc.save(circuit)
                    del circuit
            else:
                circuit.build()
                cc.save(circuit)
                del circuit

    def analyse_joints(self, circuitselection, writefile=False, bandwidth=0.01):
        # Get component dataframe of circuit selection
        componentlist = get_components(circuitselection)

        # Create list of warningdataframes for circuit selection. Bandwidth set to 0.03.
        warninglist = [cc.get_warnings(circuit, 0.03) for circuit in circuitselection]
        warninglist = pd.concat(warninglist, ignore_index=True)

        # Create list of specific features for circuit selection
        rmu_featurelist = [cc.get_rmu_features(circuit) for circuit in circuitselection]
        rmu_featurelist = pd.concat(rmu_featurelist, ignore_index=True)

        # Merge Components, warnings and specific features
        joints = pd.concat([componentlist, warninglist, rmu_featurelist], axis=1, join='inner')

        # Select the joints from this dataframe
        joints = select_joints(joints)

        # Calculate charges for the joints and concatenate basic features
        charges = get_joints_charges(circuitselection, bandwidth)
        joints = pd.concat([joints, calculate_basic_features(charges)], axis=1, join='inner')

        self.joints = joints


def select_circuits(datadump, year, month):
    return [circuit for circuit in datadump.circuits if circuit.date.year == year and circuit.date.month == month]


def select_joints(cableconfig):
    cableconfig_joints = cableconfig.drop(['Length (m)', 'Component name'], axis=1)
    return cableconfig_joints.loc[cableconfig_joints['Component type'].str.startswith('Joint')].reset_index(drop=True)


def build_selection(circuitselection):
    for circuit in circuitselection:
        circuit.build()


def get_joints_charges(circuitselection, bandwidth):
    """
    Helper function to get the charges for the joints of the selected circuits using the given bandwidth
    :param circuitselection: selection of circuits
    :param bandwidth:
    :return: list of charges, list of joints
    """
    chargelist = []
    for circuit in circuitselection:
        if circuit.built:
            joints = select_joints(circuit.cableconfig)
            charge = joints['Cumulative length (m)'].apply(lambda x: circuit.get_charge(x, bandwidth, True).values)
            chargelist.append(charge)
    return pd.concat(chargelist, ignore_index=True)


def get_components(circuitselection):
    build_selection(circuitselection)
    complist = []
    for circuit in circuitselection:
        if circuit.built:
            comp = circuit.cableconfig.copy()

            # Add circuitnr, date and circuitlength to dataframe
            comp['circuitnr'] = circuit.circuitnr
            comp['date'] = circuit.date
            comp['circuitlength'] = circuit.circuitlength

            complist.append(comp)
    return pd.concat(complist, ignore_index=True)


def calculate_basic_features(charges):
    """
    :param charges: list of timeseries
    :return: dataframe of features for the charges
    """
    print('Calculating basic features...')
    features = {
        'pdmin (nC)': charges.apply(np.min) / 1000,
        'pdmax (nC)': charges.apply(np.max) / 1000,
        'pdcount': charges.apply(np.count_nonzero),
        'pdmean (nC)': charges.apply(np.mean) / 1000,
        'pdmedian (nC)': charges.apply(np.median) / 1000,
        'pdq90 (nC)': charges.apply(lambda x: np.percentile(x, 90)) / 1000,
        'pdq95 (nC)': charges.apply(lambda x: np.percentile(x, 95)) / 1000,
        'pdq99 (nC)': charges.apply(lambda x: np.percentile(x, 99)) / 1000,
        'pdq99.9 (nC)': charges.apply(lambda x: np.percentile(x, 99.9)) / 1000,
        'pdq99.95 (nC)': charges.apply(lambda x: np.percentile(x, 99.95)) / 1000,
        'pdq99.99 (nC)': charges.apply(lambda x: np.percentile(x, 99.99)) / 1000,
        'pddensity (nC/min)': charges.apply(lambda x: np.sum(x) / (1000 * len(x))),
        'countdensity (#/min)': charges.apply(lambda x: np.count_nonzero(x) / len(x))
    }
    return pd.DataFrame(features)


def get_all_components():
    datadump = DataDump(globals.datadumpdir)
    datadump.read()
    componentlist = []
    for circuitnr in datadump.circuits_df:
        circuit = cc.MergedCircuit(circuitnr)
        circuit.build_cableconfig()
        cableconfig = circuit.cableconfig
        if cableconfig is not None:
            cableconfig['circuitnr'] = circuit.circuitnr
        componentlist.append(cableconfig)
    return pd.concat(componentlist)


#def get_noise_levels(dump):
#    """
#    Function that returns number of buckets (day x perc circuitlength) with PD > level.
#    :param dump:
#    :return:
#    """
#    clusterlist = []
#    for circuitnr in dump.circuits_df['circuitnr']:
#        print(circuitnr)
#        circuit = cc.load(circuitnr)
#        clusterlist.append(cca.get_noise_level(circuit))
#    noiseframe = pd.DataFrame(clusterlist)
#    noiseframe.to_csv(globals.datadir / 'circuits-noise.csv', sep=';', index=False)
#    return pd.DataFrame(noiseframe)


def convert_warningarray_to_boolframe(warningarrays):
    """
    Converts array with array of warning levels to DataFrame with dict of Bools
    :param warningarray:
    :return:
    """
    level1 = []
    level2 = []
    level3 = []
    noise = []
    for array in warningarrays:
        # level1.append('1' in array.astype(str))
        # level2.append('2' in array.astype(str))
        # level3.append('3' in array.astype(str))
        # noise.append('N' in array.astype(str))
        level1.append('1' in array)
        level2.append('2' in array)
        level3.append('3' in array)
        noise.append('N' in array)
    warningframe = pd.DataFrame()
    warningframe['level1'] = level1
    warningframe['level2'] = level2
    warningframe['level3'] = level3
    warningframe['noise'] = noise
    return warningframe


def get_warnings(dump):
    """
    Dataframe describing whether circuits have had level 1, 2, or 3 warnings
    :param dump:
    :return:
    """
    circuitwarnings = []
    circuitlengths = []
    for circuitnr in dump.circuits_df['circuitnr']:
        circuit = cc.load(circuitnr)
        print(circuitnr)
        circuitlengths.append(circuit.circuitlength)
        if circuit.warning is not None:
            circuitwarnings.append(circuit.warning['SCG warning level (1 to 3 or Noise)'].unique())
        else:
            circuitwarnings.append([np.nan])
    warningframe = convert_warningarray_to_boolframe(circuitwarnings)
    warningframe['circuitnr'] = dump.circuits_df['circuitnr']
    warningframe['circuitlength'] = circuitlengths
    warningframe.to_csv(globals.datadir / 'circuits-warnings.csv', sep=';', index=False)
    return warningframe


def get_nearest_joints(circuit, location):
    joints = select_joints(circuit.cableconfig)
    joints['distances'] = abs(joints['Cumulative length (m)'] - location)
    joints['locations'] = joints['Cumulative length (m)']
    joints = joints.sort_values(by='distances').reset_index(drop=True)
    return joints['Component type'], joints['locations'], joints['distances']


def add_nearest_comps(frame):
    compframe = pd.DataFrame()
    dist = []
    jointlist = []
    jointloclist = []
    jointdistlist = []
    secjointlist = []
    secjointloclist = []
    secjointdistlist = []
    for index, row in frame.iterrows():
        circuit = cc.load(row['circuitnr'])
        location = row['Location in meters (m)_median']
        dist.append(cc.get_nearest_rmu_dist(circuit, location))

        nearestjoints, nearestjointlocs, nearestjointdist = get_nearest_joints(circuit, location)
        if len(nearestjoints) > 1:
            jointlist.append(nearestjoints[0])
            jointloclist.append(nearestjointlocs[0])
            jointdistlist.append(nearestjointdist[0])
            secjointlist.append(nearestjoints[1])
            secjointloclist.append(nearestjointlocs[1])
            secjointdistlist.append(nearestjointdist[1])
        elif len(nearestjoints) == 1:
            jointlist.append(nearestjoints[0])
            jointloclist.append(nearestjointlocs[0])
            jointdistlist.append(nearestjointdist[0])
            secjointlist.append(np.nan)
            secjointloclist.append(np.nan)
            secjointdistlist.append(np.nan)
        else:
            jointlist.append(np.nan)
            jointloclist.append(np.nan)
            jointdistlist.append(np.nan)
            secjointlist.append(np.nan)
            secjointloclist.append(np.nan)
            secjointdistlist.append(np.nan)

    compframe['nearest-rmudistance'] = dist
    compframe['nearest-jointtype'] = jointlist
    compframe['nearest-jointlocation'] = jointloclist
    compframe['nearest-jointdistance'] = jointdistlist
    compframe['secondnearest-jointtype'] = secjointlist
    compframe['secondnearest-jointlocation'] = secjointloclist
    compframe['secondnearest-jointdistance'] = secjointdistlist
    return compframe


#def get_clusters(dump):
#    clusterlist = []
#    rmulist = []
#    for circuitnr in dump.circuits_df['circuitnr']:
#        print(circuitnr)
#        circuit = cc.load(circuitnr)
#        clusterreport = cca.report_clusters(circuit)
#        clusterreport = pd.concat([clusterreport, add_nearest_comps(clusterreport)], axis=1)
#        clusterlist.append(clusterreport)
#    clusterframe = pd.concat(clusterlist)
#    clusterframe.to_csv(globals.datadir / 'circuits-clusters.csv', sep=';', index=False)
#    return clusterframe


#def combine_data(dump):
#    """
#    This function combines all data of a datadump into one report
#    :param dump:
#    :return:
#    """
#    # Read circuits from datadump
#    report = dump.circuits_df
#
#    # Read noise
#    if (globals.datadir / 'circuits-noise.csv').is_file():
#        print('Noise levels found, reading file')
#        noiselevels = pd.read_csv(globals.datadir / 'circuits-noise.csv', sep=';')
#    else:
#        noiselevels = get_noise_levels(dump)
#
#    # Read warnings
#    if (globals.datadir / 'circuits-warnings.csv').is_file():
#        print('Warning file found, reading file')
#        warnings = pd.read_csv(globals.datadir / 'circuits-warnings.csv', sep=';')
#    else:
#        warnings = get_warnings(dump)
#    warnings['warning present'] = (warnings['level1'] | warnings['level2'] | warnings['level3'] | warnings['noise'])
#
#    # Read clusters
#    if (globals.datadir / 'circuits-clusters.csv').is_file():
#        print('Clusters file found, reading file')
#        clusters = pd.read_csv(globals.datadir / 'circuits-clusters.csv', sep=';')
#    else:
#        clusters = get_clusters(dump)
#
#    report = report.merge(noiselevels, on='circuitnr')
#    report = report.merge(warnings, on='circuitnr')
#    report = report.merge(clusters, on='circuitnr', how='outer')
#
#    grouped = report.groupby('circuitnr')['label'].count()
#    grouped.name = 'cluster-count'
#    grouped = grouped.reset_index()
#    report = report.merge(grouped, on='circuitnr')
#
#    # Add warnings to separate clusters
#    if (globals.datadir / 'cluster-warnings.csv').is_file():
#        print('Clusters warnings file found, appending file')
#        clusterwarnings = pd.read_csv(globals.datadir / 'cluster-warnings.csv', sep=';')
#        report = pd.concat([report, clusterwarnings], axis=1)
#    else:
#        report = add_cluster_warnings(report)
#
#    # Add additional features
#    report['warning present'] = (report['level1'] | report['level2'] | report['level3'] | report['noise'])
#    report['nearest-jointdistrel'] = report['nearest-jointdistance'] / report['circuitlength']
#    report['nearest-rmudistrel'] = report['nearest-rmudistance'] / report['circuitlength']
#    report['location skew'] = report['Location in meters (m)_mean'] - report['Location in meters (m)_median']
#    report['cluster-timedelta'] = (pd.to_datetime(report['Date/time (UTC)_max']) - pd.to_datetime(report['Date/time (UTC)_min'])).dt.days
#    report['cluster-locationdelta'] = report['Location in meters (m)_max'] - report['Location in meters (m)_min']
#    report['cluster-density'] = report['count_sum'] / (report['cluster-timedelta'])
#    report.to_csv(globals.datadir / 'circuits-report.csv', sep=';', index=False)
#    return report
#
#
#def add_cluster_warnings(report):
#    """
#    Adds warnings to separate clusters. Slow and ugly, but fit for purpose.
#    :param report:
#    :return:
#    """
#    warninglist = []
#    for index, row in report.iterrows():
#        circuit = cc.load(row['circuitnr'])
#        print(circuit.circuitnr)
#        warnings = cc.group_warnings(circuit.warning)
#        if warnings is not None:
#            warnings = warnings[warnings['Location in meters (m)'] > row['Location in meters (m)_min']]
#            warnings = warnings[warnings['Location in meters (m)'] < row['Location in meters (m)_max']]
#            warnings = warnings[warnings['Start Date/time (UTC)'] > row['Date/time (UTC)_min']]
#            warnings = warnings[warnings['End Date/time (UTC)'] < row['Date/time (UTC)_max']]
#
#        if warnings is None:
#            warninglist.append(np.array([], dtype=object))
#        else:
#            warninglist.append(warnings['SCG warning level (1 to 3 or Noise)'].unique())
#
#    warningframe = convert_warningarray_to_boolframe(warninglist)
#    warningframe['c-level'] = (warningframe['level1'] | warningframe['level2'] | warningframe['level3'])
#    warningframe.columns = ['c-level1', 'c-level2', 'c-level3', 'c-noise', 'c-level']
#
#    warningframe.to_csv(globals.datadir / 'cluster-warnings.csv', sep=';', index=False)
#    return pd.concat([report, warningframe], axis=1)
#
#
#def plot_correlation(report, relative=False):
#    #report = report[report['label'] != -1]
#    if relative:
#        std = 100 * report['Location in meters (m)_std'] / report['circuitlength']
#        title = 'Standard deviation (% of circuitlength)'
#    else:
#        std = report['Location in meters (m)_std']
#        title = 'Standard deviation (m)'
#    length = report['circuitlength']
#    clusterdata = [go.Scattergl(x=std, y=length, mode='markers', marker=dict(size=3))]
#    layout = dict(showlegend=True,
#                  title='Scatterplot',
#                  xaxis=dict(title=title),
#                  yaxis=dict(title='Circuitlength (m)'))
#    fig = dict(data=clusterdata, layout=layout)
#    filename = str(globals.plotdir / 'clusters.html')
#    py.plot(fig, filename=filename)


if __name__ == '__main__':
    dump = DataDump()
    dump.read()
    # report = combine_data(dump)
    # test = add_cluster_warnings(report)
    # filtered = validate_clusterreport(report)


    # highdens = filtered[filtered['cluster-density'] > 1000]
    # highmedianresin = filtered[filtered['Charge (picocoulomb)_median'] > 10000]
    # highmedianresin = highmedianresin[highmedianresin['nearest-jointtype'] == 'Joint (resin)']
    # highmedianresin = highmedianresin.set_index('circuitnr')
    # oil = filtered[filtered['nearest-jointtype'] == 'Joint (oil)']
    # oil = oil.sort_values(by=['cluster-density'])
    # resin = filtered[filtered['nearest-jointtype'] == 'Joint (resin)']
    #
    # # hoog minimum: nauwelijks resin, alleen de bekende foute gevallen
    # highmin = filtered[filtered['Charge (picocoulomb)_min'] > 1000]
    # highmin = highmin.set_index('circuitnr')


