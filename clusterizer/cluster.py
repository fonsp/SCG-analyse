import numpy as np


class Cluster:
    def __init__(self, location_range=None, time_range=None):
        """
        :param location_range: (Optional) 2-Tuple containing left and right cluster bounds
        :type location_range: tuple (float)

        :param time_range: (Optional) Tuple containing start and end dates of cluster
        :type time_range: tuple (numpy.datetime64)
        """
        self.location_range = location_range
        self.time_range = time_range

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
        if cluster_width is None:
            cluster_width = circuit.circuitlength * 0.01

        loc_range = (loc - cluster_width * .5, loc + cluster_width * .5)

        return Cluster(location_range=loc_range, time_range=dates)

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
        return "; ".join(sentences)

    def __repr__(self):
        return str(self)


def create_clusters_from_circuit_warnings(circuit, cluster_width=None):
    """Returns a set of Cluster instances corresponding to all warnings in the circuit. See `Cluster.from_circuit_warning` for details."""
    return set(Cluster.from_circuit_warning(circuit, i, cluster_width) for i in range(len(circuit.warning)))
