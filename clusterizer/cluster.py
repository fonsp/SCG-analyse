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
        overlap_cluster = Cluster(overlap(self.location_range, other.location_range), overlap(self.time_range, other.time_range))
        if overlap_cluster.location_range is None and overlap_cluster.time_range is None:
            return None
        return overlap_cluster

    def __rand__(self, other):
        """
        Right and. Needed to make things like 'None & Cluster' work
        """
        return self.__and__(other)

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
        return Cluster(least_common_superrange(self.location_range, other.location_range), least_common_superrange(self.time_range, other.time_range))

    def __ror__(self, other):
        return self.__or__(other)


@total_ordering
class WeightedCluster:
    def __init__(self, cluster, weight=1):
        """
        :param cluster: Cluster object
        :type cluster: class: `clusterizer.cluster.Cluster`

        :param weight: The weight of the cluster, standard is 1
        :type weight: integer, optional
        """
        self.cluster = cluster
        self.weight = weight
        
    def __str__(self):
        return str(self.cluster) + ": Weight " + str(self.weight)
    
    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        if self.cluster < other.cluster:
            return True
        if self.cluster == other.cluster and self.weight < other.weight:
            return True
        return False

    def __eq__(self, other):
        return self.cluster == other.cluster and self.weight == other.weight

    def __hash__(self):
        return self.weight * hash(self.cluster)

    def __and__(self, other):
        return WeightedCluster(self.cluster & other.cluster, weight=self.weight + other.weight)

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        return WeightedCluster(self.cluster | other.cluster, weight=min(self.weight, other.weight))
    
    def __ror__(self, other):
        return self.__or__(other)


class WeightedClusterSet:
    def __init__(self, wcs):
        self.weighted_cluster_set = set(wcs)
        
    def __str__(self):
        result = ""
        for cluster in sorted(self.weighted_cluster_set):
            result += str(cluster) + "\n"
        return result
    
    def __repr__(self):
        return str(self)

    def __hash__(self):
        hashed = 0
        for cluster in self.weighted_cluster_set:
            hashed += hash(cluster)
        return hashed
    
    def as_set(self):
        return self.weighted_cluster_set
    
    def as_list(self):
        return list(self.weighted_cluster_set)
