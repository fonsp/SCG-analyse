#!/usr/bin/env python
# -*- coding: utf-8 -*-

import calendar
import pickle
import re
import sys
import timeit

from datetime import date

import numpy as np
import pandas as pd

import globals, alarms as al

# The following line is required on Windows systems until this bug has been solved:
# https://github.com/pandas-dev/pandas/issues/15086
# sys._enablelegacywindowsfsencoding()


def save(circuit):
    """
    Save a circuit object to pickle
    :param circuit: circuit object to pickle
    """
    with open(globals.circuitcache / (str(circuit.circuitnr) + '.pickle'), 'wb') as f:
        pickle.dump(circuit, f, pickle.HIGHEST_PROTOCOL)


def load(circuitnr):
    """
    Load a pickled circuit from cache
    :param circuitnr: circuitnr to load
    :return: datadump object
    """
    circuitcachepath = globals.circuitcache / (str(circuitnr) + '.pickle')
    circuit = None
    if circuitcachepath.is_file():
        with open(circuitcachepath, 'rb') as f:
            circuit = pickle.load(f)
    return circuit


def check_data(dataframe):
    """
    Function that checks if a circuit dataframe contains data and can be used.
    :param dataframe:
    :return: True if dataframe contains data, False otherwise
    """
    if dataframe.iloc[0, 0] == 'No data available':
        return False
    else:
        return True


def read_foldername(folder):
    """
    Read a folder name and return data contained therein. Assumes that foldername starts with circuitnr,
    year and month. If not, None is returned.
    :param folder: foldername that contains SCG circuit data
    :return: sixtuple of circuitnr, year, month, circuitid, startlocation, endlocation
    """
    match = re.search(r'(\d+)\s-\s(\d\d\d\d)-(\d\d)\s-\s(.+).*\s-\s(.+)\s--\s(.+)', folder)
    if match is not None:
        foldertuple = match.groups()
    else:
        match = re.search(r'(\d+)\s-\s(\d\d\d\d)-(\d\d)\s-\s()(.*)\s--\s(.+)', folder)
        if match is not None:
            foldertuple = match.groups()
        else:
            match = re.search(r'(\d+)\s-\s(\d\d\d\d)-(\d\d)()()()', folder)
            if match is not None:
                foldertuple = match.groups()
            else:
                foldertuple = None
    return foldertuple


def get_warnings(circuit, bandwidth=0.03):
    """
    Adds the warnings to a cableconfig dataframe
    :param circuit: instance of a circuit
    :param bandwidth: percentage of circuitlength used to assign warnings to components.
    :return:
    """
    if (circuit.warning is None) or (circuit.cableconfig is None):
        return
    length = circuit.cableconfig.shape[0]
    warning1 = np.zeros(length, dtype=bool)
    warning2 = np.zeros(length, dtype=bool)
    warning3 = np.zeros(length, dtype=bool)
    noise = np.zeros(length, dtype=bool)
    threshold = bandwidth * circuit.cableconfig['Cumulative length (m)'].max()
    for index, row in circuit.warning.iterrows():
        closecomponents = abs(circuit.cableconfig['Cumulative length (m)'] - row['Location in meters (m)']) < threshold
        warningit = np.where(closecomponents, row['SCG warning level (1 to 3 or Noise)'], None)
        noise = np.logical_or(warningit == 'N', noise)*1.0
        warning1 = np.logical_or(warningit == '1', warning1)*1.0
        warning2 = np.logical_or(warningit == '2', warning2)*1.0
        warning3 = np.logical_or(warningit == '3', warning3)*1.0
    warnings = {
        'DNVG GL Warning 1': warning1,
        'DNVG GL Warning 2': warning2,
        'DNV GL Warning 3': warning3,
        'DNV GL Noise': noise
    }
    return pd.DataFrame(warnings)


def group_warnings(dfwarnings):
    """
    Function that combines all the warnings of one type for a specific distance. Assumes that one warning type for a
    specific distance is not interupted.
    :param dfwarnings:
    :return: groupedwarnings dataframe
    """
    if dfwarnings is None:
        return None
    dfwarnings['Start Date/time (UTC)'] = pd.to_datetime(dfwarnings['Start Date/time (UTC)'])
    dfwarnings['End Date/time (UTC)'] = pd.to_datetime(dfwarnings['End Date/time (UTC)'])
    dfwarnings['SCG warning level (1 to 3 or Noise)'] = dfwarnings['SCG warning level (1 to 3 or Noise)'].astype(str)
    groupedwarnings = dfwarnings.groupby(['Location in meters (m)', 'SCG warning level (1 to 3 or Noise)'],
                                         as_index=False)
    return groupedwarnings.agg({'Start Date/time (UTC)': min, 'End Date/time (UTC)': max})


def get_nearest_rmu_dist(circuit, location):
    # TODO: fix this to use str.startwith as in datadump select_joints
    if not circuit.built:
        return
    indices = circuit.cableconfig['Component type'].isin(['RMU',
                                                          'MV terminations',
                                                          'Termination (unknown)',
                                                          'Termination (bitumen)',
                                                          'Termination (resin)',
                                                          'Termination (grease)(liquid if heated, solid after cooling)',
                                                          'Termination (oil)',
                                                          'Termination (polymer stress cone)',
                                                          'Termination (heat shrink)',
                                                          'Termination (cold shrink)',
                                                          'Termination (polymer, unknown)'
                                                          ])
    rmus = abs(circuit.cableconfig['Cumulative length (m)'][indices].unique() - location)
    try:
        return np.min(rmus)
    except Exception as e:
        return np.nan


def get_rmu_features(circuit):
    """
    Adds the distance in meters and in percentage of circuitlength to the nearest RMU column to a cableconfig dataframe
    :return:
    """
    if not circuit.built:
        return
    dist_to_nearest_rmu_list = []
    nearest_rmu_list = []
    rmucorrslist = []
    rmudistances = circuit.cableconfig['Cumulative length (m)'][
        circuit.cableconfig['Component type'].isin(['RMU',
                                                    'MV terminations',
                                                    'Termination (unknown)',
                                                    'Termination (bitumen)',
                                                    'Termination (resin)',
                                                    'Termination (grease)(liquid if heated, solid after cooling)',
                                                    'Termination (oil)',
                                                    'Termination (polymer stress cone)',
                                                    'Termination (heat shrink)',
                                                    'Termination (cold shrink)',
                                                    'Termination (polymer, unknown)'
                                                    ])]
    for index, row in circuit.cableconfig.iterrows():
        dist_rmu = abs(rmudistances - row['Cumulative length (m)'])
        dist_nearest_rmu = min(dist_rmu)
        nearest_rmu = rmudistances[dist_rmu.idxmin()]
        charge_nearest_rmu = circuit.get_charge(nearest_rmu, 0.01, True, '6h')
        charge = circuit.get_charge(row['Cumulative length (m)'], 0.01, True, '6h')
        rmucorr = np.corrcoef(charge_nearest_rmu, charge)[1, 0]
        nearest_rmu_list.append(nearest_rmu)
        dist_to_nearest_rmu_list.append(dist_nearest_rmu)
        rmucorrslist.append(rmucorr)
    rmu_features = {
        'Nearest RMU (m)': nearest_rmu_list,
        'Distance to nearest RMU (m)': dist_to_nearest_rmu_list,
        'Distance to nearest RMU (% of circuit)': dist_to_nearest_rmu_list / circuit.circuitlength,
        'Corr coeff nearest RMU': rmucorrslist
    }
    return pd.DataFrame(rmu_features)


class Circuit:
    """
    Class that holds all relevant information related to a single SCG circuit (pd, propagation time, components).
    """

    def __init__(self, circuitdir):
        """
        Create a circuit from a directoy name coming from the DNV GL datadump
        Example: '1910 - 2018-08 - [201096] - MSR S V GENTSTEEG -- MSR ACHTEROMSTRAAT (OS Weesp veld 147)'
        Relevant information is extracted from this directory name on initialization using a regular expression.
        """
        self.circuitdir = circuitdir
        self.init = False
        self.built = False
        self.pd = None
        self.warning = None
        self.propagation = None
        self.cableconfig = None
        self.circuitlength = None
        

        try:
            (self.circuitnr,
             year,
             month,
             self.circuitid,
             self.startlocation,
             self.endlocation) = read_foldername(circuitdir)
            self.date = date(int(year), int(month), 1)
            self.init = True
        except Exception as e:
            print('Circuit initialization failed: ', circuitdir, e)

    def build(self):
        """
        Read the relevant datafiles for the circuit. Either in .csv or in xlsx format
        """
        if self.built:
            return

        try:
            self.build_pd()
            self.build_warning()
            self.build_propagation()
            self.build_cableconfig()
            self.built = True
        except Exception as e:
            print('Circuit build failed: ', e, ' Circuit number:  ', self.circuitnr)

    def build_cableconfig(self):
        self.cableconfig = pd.read_csv(globals.datadumpdir / self.circuitdir / 'CableConfiguration.csv',
                                       sep=';')
        self.cableconfig['Cumulative length (m)'] = self.cableconfig['Cumulative length (m)'].fillna(method='ffill')
        self.circuitlength = self.cableconfig['Cumulative length (m)'].max()
        # If cableconfig start with NA, this is missed by previous line and means start of circuit, fill with 0.
        self.cableconfig['Cumulative length (m)'] = self.cableconfig['Cumulative length (m)'].fillna(0)

    def build_propagation(self):
        self.propagation = pd.read_csv(globals.datadumpdir / self.circuitdir / 'PropagationTime.csv',
                                       sep=';')
        if check_data(self.propagation):
            self.propagation['Date/time (UTC)'] = pd.to_datetime(self.propagation['Date/time (UTC)'],
                                                                 format='%Y-%m-%d %H:%M:%S')

    def build_pd(self):
        self.pd = pd.read_csv(globals.datadumpdir / self.circuitdir / 'PD.csv',
                              parse_dates=[0], sep=';')
        

    def build_warning(self):
        self.warning = pd.read_csv(globals.datadumpdir / self.circuitdir / 'SCGWarning.csv', sep=';')
        self.warning['SCG warning level (1 to 3 or Noise)'] = self.warning['SCG warning level (1 to 3 or Noise)'].astype(str)

        # If enddate for warning level is missing, then the warning is active.
        # So we impute the max date in the dataset.
        if check_data(self.warning):
            if (self.pd is None) or not check_data(self.pd):
                last_day = calendar.monthrange(self.date.year, self.date.month)[1]
                self.warning = self.warning.fillna(date(self.date.year, self.date.month, last_day))
            else:
                self.warning = self.warning.fillna(self.pd['Date/time (UTC)'].max())

    def get_charge(self,
                   location,
                   bandwidth=0.01,
                   relative=True,
                   timeres='1min',
                   startdate=None,
                   enddate=None,
                   how='sum'):
        """
        Returns charge timeseries with pd within distance +/- bandwidth
        If relative is true: bandwidth in percentage. If false: bandwidth in m.
        Warning, self.pd and self.circuitlength need to be filled.
        :param location: location in meters
        :param bandwidth: bandwidth in meters or in percentage (if relative is True).
        :param timeres: time resolution, 1 min is the minimum available in the data.
        :param relative: if relative is true, get charge around percentage bandwidth around distance.
        :param startdate: startdate
        :param enddate: enddate
        :param how: sum charges or count charges in given timeres.
        :return: charge np array
        """
        if self.pd is None:
            return None
        pdframe = self.pd.copy()
        pdframe = pdframe.dropna()
        if not check_data(pdframe):
            return pd.Series(np.full(1, np.nan))
        if startdate is not None:
            pdframe = pdframe[pdframe['Date/time (UTC)'] >= startdate]
        if enddate is not None:
            pdframe = pdframe[pdframe['Date/time (UTC)'] <= enddate]

        if relative:
            charge = np.where(abs(location - pdframe['Location in meters (m)']) / self.circuitlength <=
                              bandwidth, pdframe['Charge (picocoulomb)'], 0)
        else:
            charge = np.where(abs(pdframe['Location in meters (m)']-location) <= bandwidth,
                              pdframe['Charge (picocoulomb)'], 0)

        if how == 'sum':
            if timeres == '1min':
                return pd.Series(charge, index=pdframe['Date/time (UTC)'])
            else:
                return pd.Series(charge, index=pdframe['Date/time (UTC)']).resample(timeres).sum()
        elif how == 'count':
            return pd.Series(charge, index=pdframe['Date/time (UTC)']).resample(timeres).apply(np.count_nonzero)

    def get_ewma_data(self, location, a=0.8, b=0.2, k=2, bandwidth=0.01, relative=True, timeres='1d'):
        # Get charge
        x = self.get_charge(location, bandwidth, relative, timeres)/1000

        # Get count
        xmin = self.get_charge(location, bandwidth, True)
        y = xmin.astype(bool).resample(timeres).sum()

        ta, mu, var = al.ewma(x, a, b, k)
        ta = al.validate_alarms(x, y, mu, ta)
        bound = k*np.sqrt(var)
        mu = pd.Series(mu, index=x.index)
        bound = pd.Series(bound, index=x.index)
        alarms = pd.Series(x[x.index[ta]], index=x.index[ta])
        return x, mu, bound, alarms

    def get_alarms(self):
        """Get a pd series of number of alarms indexed by location"""
        alarmlist = []
        locations = np.arange(0, self.circuitlength, self.circuitlength/100)
        for location in locations:
            # Get charge
            x = self.get_charge(location=location, timeres='1d') / 1000

            # Get count
            xmin = self.get_charge(location=location)
            y = xmin.astype(bool).resample('1d').sum()

            ta, mu, var = al.ewma(x)
            ta = al.validate_alarms(x, y, mu, ta)
            alarmcount = ta.size
            alarmlist.append(alarmcount)
        return pd.Series(alarmlist, index=locations)


class MergedCircuit(Circuit):
    """
    Class that combines data from all occurences of a single circuit in a datadump.
    """

    def __init__(self, circuitnr):
        """
        :param circuitnr:
        """
        self.circuitnr = circuitnr
        self.circuitid = None
        self.startlocation = None
        self.endlocation = None

        self.init = False
        self.built = False

        self.pd = None
        self.warning = None
        self.propagation = None
        self.cableconfig = None
        self.circuitlength = None
        self.circuits = []

        try:
            for subdir in globals.datadumpdir.iterdir():
                if subdir.name.startswith(str(circuitnr) + ' -'):
                    self.circuits.append(Circuit(subdir.name))
            self.circuitid = self.circuits[-1].circuitid
            self.startlocation = self.circuits[-1].startlocation
            self.endlocation = self.circuits[-1].endlocation
            self.init = True
        except Exception as e:
            print('Circuit initialization failed: ', circuitnr, e)

    def build_warning(self):
        warninglist = []
        for circ in self.circuits:
            try:
                circ.build_warning()
                if check_data(circ.warning):
                    warninglist.append(circ.warning)
            except FileNotFoundError:
                continue
        if warninglist:
            self.warning = pd.concat(warninglist)

    def build_pd(self):
        pdlist = []
        for circ in self.circuits:
            circ.build_pd()
            if check_data(circ.pd):
                pdlist.append(circ.pd)
        if pdlist:
            self.pd = pd.concat(pdlist)

    def build_propagation(self):
        propagationlist = []
        for circ in self.circuits:
            circ.build_propagation()
            if check_data(circ.propagation):
                propagationlist.append(circ.propagation)
        if propagationlist:
            self.propagation = pd.concat(propagationlist)

    def build_cableconfig(self):
        try:
            circuit = self.circuits[-1]
            circuit.build_cableconfig()
            self.cableconfig = circuit.cableconfig
            self.circuitlength = self.cableconfig['Cumulative length (m)'].max()
        except FileNotFoundError:
            pass

    def build(self, warning_bandwidth=0.03):
        if self.built:
            return
        try:
            self.build_pd()
            self.build_warning()
            self.build_propagation()
            self.build_cableconfig()
            self.built = True
        except Exception as e:
            print('Circuit build failed: ', e, ' Circuit number:  ', self.circuitnr)

    def check_cableconfig(self):
        """
        Function that checks if the cableconfiguration is the same for all months.
        :return: True if cableconfig is same for all months. False otherwise.
        """
        for circuit in self.circuits:
            circuit.build_cableconfig()
            if not self.circuits[0].cableconfig.equals(circuit.cableconfig):
                return False
        return True

    def check_cableconfig2(self):
        for circuit in self.circuits:
            circuit.build_cableconfig()
        for index, circuit in enumerate(self.circuits):
            pass


# TODO: 3097 changed cableconfig completely, so circuitnr. not unique id. Fix this in code
if __name__ == '__main__':
   print('bla')

