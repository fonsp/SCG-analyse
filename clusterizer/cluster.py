from functools import total_ordering
import functools
import numpy as np


@total_ordering
class Cluster:
    def __init__(self, location_range=None, time_range=None, found_by=None):
        """
        :param location_range: (Optional) 2-Tuple containing left and right cluster bounds
        :type location_range: tuple (float)

        :param time_range: (Optional) Tuple containing start and end dates of cluster
        :type time_range: tuple (numpy.datetime64)
        """
        self.location_range = location_range
        self.time_range = time_range
        if found_by is None:
            self.found_by = set()
        else:
            self.found_by = set(found_by)

    @staticmethod
    def from_circuit_warning(circuit, warning_index, cluster_width=None):
        """
        Create a new Cluster instance that corresponds to one of the (DNV-GL) warnings of the MergedCircuit. `warning_index` chooses which warning to convert from. (A MergedCircuit has multiple warnings, in general.)

        :param circuit: Circuit object containing warning series to convert from
        :type circuit: class:`clusterizer.circuit.MergedCircuit`

        :param warning_index: The index of the warning to convert from
        :type warning_index: int

        :param cluster_width: Width (m) of the Cluster to create. When set to `None`, 1% of the circuit length is used (0.5% at both sides).
        :type cluster_width: float, optional
        """
        w = circuit.warning.loc[warning_index]
        loc = w["Location in meters (m)"]
        dates = (w["Start Date/time (UTC)"], w["End Date/time (UTC)"])
        level = str(w["SCG warning level (1 to 3 or Noise)"])
        if cluster_width is None:
            cluster_width = circuit.circuitlength * 0.01

        loc_range = (loc - cluster_width * .5, loc + cluster_width * .5)

        return Cluster(location_range=loc_range, time_range=dates, found_by={"DNV GL warning {}".format(level)})

    def get_width(self):
        """The distance in m between the two cluster edges. `numpy.inf` if undefined.

        :rtype: float
        """
        if self.location_range is None:
            return np.inf
        return max(self.location_range) - min(self.location_range)

    def get_duration(self):
        """The time delta between the two cluster edges. `numpy.inf` if undefined.

        :rtype: numpy.timedelta64
        """
        if self.time_range is None:
            return float("inf")
        return max(self.time_range) - max(self.time_range)

    # Wordt opgeroepen als je `str(een_cluster)` of bv. `print(een_cluster)` schrijft.
    def __str__(self):
        sentences = []
        if self.location_range is not None:
            sentences.append("{0:.0f}m to {1:.0f}m".format(*self.location_range))
        if self.time_range is not None:
            sentences.append("{0} until {1}".format(*self.time_range))
        if self.found_by:
            sentences.append("Found by: " + ', '.join(['%s']*len(self.found_by)) % tuple(self.found_by))
        return "; ".join(sentences)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        if self.time_range is None and other.time_range is None:
            if other.location_range is None:
                return False
            if self.location_range is None:
                return True
            return self.location_range < other.location_range
        if other.time_range is None:
            return Cluster(self.location_range) < other
        if self.time_range is None:
            return self < Cluster(other.location_range)
        if self.time_range == other.time_range:
            return Cluster(self.location_range) < Cluster(other.location_range)
        return self.time_range < other.time_range

    def __eq__(self, other):
        return self.location_range == other.location_range and self.time_range == other.time_range

    def __hash__(self):
        return hash(self.location_range) + hash(self.time_range)

    def __and__(self, other):
        """
        Calculate the overlap between this cluster and another cluster
        """
        def overlap(first, second):
            if first is not None and second is not None:
                left_bound = max(first[0], second[0])
                right_bound = min(first[1], second[1])
                if left_bound > right_bound:
                    return None
                return (left_bound, right_bound)
            if first is not None:
                return first
            return second

        if other is None:
            return None
        overlap_cluster = Cluster(overlap(self.location_range, other.location_range), overlap(self.time_range, other.time_range), self.found_by | other.found_by)
        if overlap_cluster.location_range is None or overlap_cluster.time_range is None:
            return None
        return overlap_cluster

    def __rand__(self, other):
        """
        Right and. Needed to make things like 'None & Cluster' work
        """
        return self.__and__(other)

    def overlap(self, other):
        return self & other

    def __or__(self, other):
        """
        Calculate the smallest cluster which has both self and other as a subcluster
        """
        def least_common_superrange(first, second):
            if first is not None and second is not None:
                left_bound = min(first[0], second[0])
                right_bound = max(first[1], second[1])
                return (left_bound, right_bound)
            return None
        if other is None:
            return None
        return Cluster(least_common_superrange(self.location_range, other.location_range), least_common_superrange(self.time_range, other.time_range), self.found_by | other.found_by)

    def __ror__(self, other):
        return self.__or__(other)

    def disjunct(self, other):
        # return (self & other) is None
        def disjunct_range(first, second):
            if first is not None and second is not None:
                return first[1] <= second[0] or second[1] <= first[0]
            return False
        if other is None:
            return True
        return disjunct_range(self.location_range, other.location_range) or disjunct_range(self.time_range, other.time_range)

    def supercluster(self, other):
        return self | other

    def __sub__(self, other):
        """
        Calculate self without other (set theoretic: self \\ other)
        Returns a Set containing Clusters
        """
        def empty_range(r):
            if r is None:
                return False
            return r[0] == r[1]

        def empty_cluster(cluster):
            if cluster is None:
                return True
            if empty_range(cluster.location_range) or empty_range(cluster.time_range):
                return True
            return False

        if self.disjunct(other):
            return set([self])
        overlap = self & other
        left_locations = (self.location_range[0], overlap.location_range[0])
        middle_locations = overlap.location_range
        right_locations = (overlap.location_range[1], self.location_range[1])
        if self.time_range is not None and overlap.time_range is not None:
            lower_times = (self.time_range[0], overlap.time_range[0])
            middle_times = overlap.time_range
            upper_times = (overlap.time_range[1], self.time_range[1])
        else:
            lower_times, middle_times, upper_times = self.time_range, self.time_range, self.time_range
        lu = Cluster(left_locations, upper_times, found_by=self.found_by)
        mu = Cluster(middle_locations, upper_times, found_by=self.found_by)
        ru = Cluster(right_locations, upper_times, found_by=self.found_by)
        lm = Cluster(left_locations, middle_times, found_by=self.found_by)
        rm = Cluster(right_locations, middle_times, found_by=self.found_by)
        ll = Cluster(left_locations, lower_times, found_by=self.found_by)
        ml = Cluster(middle_locations, lower_times, found_by=self.found_by)
        rl = Cluster(right_locations, lower_times, found_by=self.found_by)
        mini_clusters = [lu, mu, ru, lm, rm, ll, ml, rl]
        result = set()
        for c in mini_clusters:
            if not empty_cluster(c):
                result.add(c)
        return result

    def __add__(self, other):
        if other is None:
            return set([self])
        if self.disjunct(other):
            return set([self, other])
        overlap = self & other
        smo = self - other
        oms = other - self
        return set([overlap]) | smo | oms

    def __mul__(self, other):
        return self & other

    def get_partial_discharges(self, circuit):
        """Returns all PDs that lie in the Cluster."""
        return circuit.pd[circuit.pd_occured].loc[self.get_partial_discharge_mask(circuit)]

    def get_partial_discharge_mask(self, circuit):
        """Returns a boolean array which indicates, for each PD, whether it lies in the Cluster."""
        locations = circuit.pd["Location in meters (m)"][circuit.pd_occured]
        times = circuit.pd["Date/time (UTC)"][circuit.pd_occured]

        return functools.reduce(np.logical_and, [
            self.location_range[0] < locations,
            self.location_range[1] >= locations,
            self.time_range[0] < times,
            self.time_range[1] >= times])
