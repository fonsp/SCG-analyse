from functools import total_ordering
import numpy as np

@total_ordering
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

