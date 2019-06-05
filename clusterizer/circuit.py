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

from . import globals

# The following line is required on Windows systems until this bug has been solved:
# https://github.com/pandas-dev/pandas/issues/15086
# (has been solved as of april 2019)

#sys._enablelegacywindowsfsencoding()


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
        circuit.cableconfig['Component type'].isin(['RMU', 'Termination (unknown)'])]
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

    def __init__(self, circuitnr, hide_warning=False):
        """
        Create a circuit from a directoy name coming from the DNV GL datadump
        Example: '1910 - 2018-08 - [201096] - MSR S V GENTSTEEG -- MSR ACHTEROMSTRAAT (OS Weesp veld 147)'
        Relevant information is extracted from this directory name on initialization using a regular expression.
        """
        if not hide_warning:
            print("Het aanmaken van een Circuit-object is waarschijnlijk niet wat je wilt: gebruik `MergedCircuit({0})` ipv `Circuit({0})`".format(circuitnr))
        self.circuitnr = circuitnr
        circuitdir = str(circuitnr)+" - 1970-01 - [-1] - REDACTED -- REDACTED"
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

    def build(self, force=False):
        """
        Read the relevant datafiles for the circuit. Either in .csv or in xlsx format
        """
        if self.built and not force:
            return

        try:
            self.build_pd()
            self.build_warning()
            #self.build_propagation()
            self.build_cableconfig()
            self.built = True
        except Exception as e:
            print('Circuit build failed: ', e, ' Circuit number:  ', self.circuitnr)

    def build_cableconfig(self):
        self.cableconfig = pd.read_csv(globals.datadir / (str(self.circuitnr) + '-cableconfig.csv'), sep=';')
        self.cableconfig['Cumulative length (m)'] = self.cableconfig['Cumulative length (m)'].fillna(method='ffill')
        self.circuitlength = self.cableconfig['Cumulative length (m)'].max()
        # If cableconfig start with NA, this is missed by previous line and means start of circuit, fill with 0.
        self.cableconfig['Cumulative length (m)'] = self.cableconfig['Cumulative length (m)'].fillna(0)

    def build_propagation(self):
        "NOT SUPPORTED"
        self.propagation = pd.read_csv(globals.datadumpdir / self.circuitdir / 'PropagationTime.csv', sep=';')
        if check_data(self.propagation):
            self.propagation['Date/time (UTC)'] = pd.to_datetime(self.propagation['Date/time (UTC)'],
                                                                 format='%Y-%m-%d %H:%M:%S')

    def build_pd(self):
        self.pd = pd.read_csv(globals.datadir / (str(self.circuitnr) + '-pd.csv'), parse_dates=[0], sep=';')
        last_dataset_date = self.pd['Date/time (UTC)'].max()
        self.date = date(last_dataset_date.year, last_dataset_date.month, 1)
        self.pd_occured = ~np.isnan(self.pd['Location in meters (m)'])

    def build_warning(self):

        self.warning = pd.read_csv(globals.datadir / (str(self.circuitnr) + '-warning.csv'), sep=';')
        # If enddate for warning level is missing, then the warning is active.
        # So we impute the max date in the dataset.
        if check_data(self.warning):
            if (self.pd is None) or not check_data(self.pd):
                last_day = calendar.monthrange(self.date.year, self.date.month)[1]
                self.warning = self.warning.fillna(date(self.date.year, self.date.month, last_day))
            else:
                self.warning = self.warning.fillna(self.pd['Date/time (UTC)'].max())
        self.warning["Start Date/time (UTC)"] = pd.to_datetime(self.warning["Start Date/time (UTC)"])
        self.warning["End Date/time (UTC)"] = pd.to_datetime(self.warning["End Date/time (UTC)"])

    def get_charge(self, location, bandwidth=0.01, relative=True, timeres='1min', startdate=None, enddate=None):
        """
        NOT SUPPORTED
        Returns charge timeseries with pd within distance +/- bandwidth
        If relative is true: bandwidth in percentage. If false: bandwidth in m.
        Warning, self.pd and self.circuitlength need to be filled.
        :param location: location in meters
        :param bandwidth: bandwidth in meters or in percentage (if relative is True).
        :param timeres: time resolution, 1 min is the minimum available in the data.
        :param relative: if relative is true, get charge around percentage bandwidth around distance.
        :param startdate: startdate
        :param enddate: enddate
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
        if timeres == '1min':
            return pd.Series(charge, index=pdframe['Date/time (UTC)'])
        return pd.Series(charge, index=pdframe['Date/time (UTC)']).resample(timeres).sum()

    def get_ewma_data(self, location, a=0.8, b=0.2, k=2, bandwidth=0.01, relative=True, timeres='1d'):
        "NOT SUPPORTED"
        raise NotImplementedError()

    def get_alarms(self):
        """NOT SUPPORTED - Get a pd series of number of alarms indexed by location"""
        return pd.Series()


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
            self.circuits.append(Circuit(self.circuitnr, hide_warning=True))
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
        else:
            self.warning = pd.DataFrame()

    def build_pd(self):
        pdlist = []
        for circ in self.circuits:
            circ.build_pd()
            if check_data(circ.pd):
                pdlist.append(circ.pd)
        if pdlist:
            self.pd = pd.concat(pdlist)
        self.pd_occured = ~np.isnan(self.pd['Location in meters (m)'])

    def build_propagation(self):
        """NOT SUPPORTED"""
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
            #self.build_propagation()
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
    circuit = MergedCircuit(1512)
    circuit.build_warning()
    print('Runtime: ', timeit.Timer(circuit.build).timeit())
