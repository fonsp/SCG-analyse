from functools import total_ordering
import functools
import numpy as np
import operator


@total_ordering
class Cluster:
    """
    A set of Rectangle objects
    In a ClusterEnsemble, this represents one cluster
    """

    def __init__(self, rectangles):
        """
        :param rectangles: The clusters to be added to this Cluster
        :rtype: collection of class:`clusterizer.rectangle.Rectangle`
        """
        self.rectangles = set(rectangles)

    def __str__(self):
        return "[" + ("\n".join(str(r) for r in self.rectangles)) + "]"

    def __repr__(self):
        return str(self)

    def __bool__(self):
        """
        Boolean representation of a Cluster. Mirrors behaviour of sets.

        :return: False if self.rectangles is empty, True if non-empty
        :rtype: bool
        """
        return bool(self.rectangles)

    def __len__(self):
        """
        Number of Rectangles in the Cluster. Mirrors behaviour of sets.

        :return: The length of self.rectangles
        :rtype: int
        """
        return len(self.rectangles)

    def __iter__(self):
        """
        Provides an iterator over the Cluster. Mirrors behaviour of sets.
        This makes expressions like `for c in my_cluster` possible,
        so you don't have to type `for c in my_cluster.as_set()`

        :return: Iterator over the Clusters in this Cluster
        :rtype: iterator
        """
        return self.rectangles.__iter__()

    def __lt__(self, other):
        return self.get_bounding_rectangle() < other.get_bounding_rectangle()

    def __eq__(self, other):
        return self.rectangles == other.rectangles

    def __hash__(self):
        return sum(hash(r) for r in self.rectangles)

    def get_rectangles(self):
        """
        Returns the Rectangles in this Cluster, without casting to a specific collection type.

        :return: The Rectangles in this Cluster
        :rtype: Collection of class:`clusterizer.rectangle.Rectangle`
        """
        return self.rectangles

    def as_set(self):
        """
        Returns a set containing the Rectangles in this Cluster.

        :return: The Rectangles in this Cluster
        :rtype: set of class:`clusterizer.rectangle.Rectangle`
        """
        return set(self.rectangles)

    def as_list(self):
        """
        Returns a list containing the Rectangles in this Cluster.

        :return: The Rectangles in this Cluster
        :rtype: list of class:`clusterizer.rectangle.Rectangle`
        """
        return list(self.rectangles)

    def disjunct(self, other):
        """
        Determines if two Clusters are disjunct. That is, disjunct(self, other) returns:
        - True if self and other do not contain any overlap in any of their Clusters
        - False if self and other overlap in any of their Clusters
        So: self & other has to be empty for disjunct(self, other) to be True.

        :param other: The other Cluster to compare self to
        :type other: class:`clusterizer.ensemble.Cluster`

        :return: False if self and other have overlap, True otherwise.
        :rtype: bool
        """
        return not bool(self & other)

    def __and__(self, other):
        """
        Calculates the overlap between two Clusters, overloading the & operator.
        This is done by calculating the overlap between the underlying Rectangle objects.
        In doing so, it keeps track of which algorithms found the clusters.

        :param other: The other Cluster to calculate the overlap with
        :type other: class:`clusterizer.ensemble.Cluster`

        :return: The overlap between self and other
        :rtype: class:`clusterizer.ensemble.Cluster`
        """
        result = set()
        for c1 in self:
            for c2 in other:
                overlap = c1 & c2
                if overlap is not None:
                    result.add(overlap)
        return Cluster(result)

    def __mul__(self, other):
        """
        Calculates the overlap between two Clusters, overloading the * operator.
        Alternate alias for the & operator. Behaves exactly the same as &.

        :param other: The other Cluster to calculate the overlap with
        :type other: class:`clusterizer.ensemble.Cluster`

        :return: The overlap between self and other
        :rtype: class:`clusterizer.ensemble.Cluster`
        """
        return self & other

    def __or__(self, other):
        """
        Calculates the bounding box of two Clusters, overloading the | operator.h
        This is done by calculating the bounding box of the underlying Rectangle objects.
        For overlapping Rectangle objects, the bounding box is calculated.
        Separate Clusters are kept separate. If there are Rectangle objects that are disjunct from all others,
        the resulting Cluster will contain multiple Rectangle objects.

        :param other: The other Cluster to calculate the bounding box with
        :type other: class:`clusterizer.ensemble.Cluster`

        :return: The bounding box of the clusters in self and other
        :rtype: class:`clusterizer.ensemble.Cluster`
        """
        if self.disjunct(other):
            return Cluster(self.rectangles | other.rectangles)
        result = Cluster(self.rectangles)
        helper = Cluster(other.rectangles)
        while helper:
            helpercur = helper.rectangles.pop()
            for clust in result:
                if not helpercur.disjunct(clust):
                    helper.rectangles.add(helpercur | clust)
                    result.rectangles.remove(clust)
                    break
            else:
                result.rectangles.add(helpercur)
        return result

    def __add__(self, other):
        """
        Calculates the rich set-theoretic union of two Clusters, overloading the + operator.
        This is done by calling the + operator on the underlying Rectangle objects.
        In doing so, it keeps track of which algorithms found the clusters.
        The best analogy for the result is a Venn diagram. The overlapping parts are where the algorithms agree. Unlike &, + also remembers where the algorithms disagree.

        :param other: The other Cluster to calculate the union with
        :type other: class:`clusterizer.ensemble.Cluster`

        :return: The set-theoretic union of the clusters in self and other
        :rtype: class:`clusterizer.ensemble.Cluster`
        """
        if self.disjunct(other):
            return Cluster(self.rectangles | other.rectangles)
        result = Cluster(self.rectangles)
        helper = Cluster(other.rectangles)
        while helper:
            helpercur = helper.rectangles.pop()
            for clust in result:
                if not helpercur.disjunct(clust):
                    helper.rectangles |= helpercur + clust
                    result.rectangles.remove(clust)
                    break
            else:
                result.rectangles.add(helpercur)
        return result

    def get_partial_discharges(self, circuit):
        """Returns all PDs that lie in any of the Clusters in this Cluster."""
        return circuit.pd[circuit.pd_occured].loc[self.get_partial_discharge_mask(circuit)]

    def get_partial_discharge_mask(self, circuit):
        """Returns a boolean array which indicates, for each PD, whether it lies in any of the Clusters in this Cluster."""
        return functools.reduce(np.logical_or, [cs.get_partial_discharge_mask(circuit) for cs in self.rectangles])

    def most_confident(self):
        """Returns a new Cluster containing only those Clusters of `self` with the highest number of algorithms that found it.

        :return: The Rectangle object with the highest number of algorithms that found it
        :rtype: class:`clusterizer.ensemble.Cluster`
        """
        result = set()
        confidence = -1
        for c in self:
            if len(c.found_by) > confidence:
                result = {c}
                confidence = len(c.found_by)
            elif len(c.found_by) == confidence:
                result.add(c)
        return Cluster(result)

    def get_bounding_rectangle(self):
        """Returns a single rectangle that contains all the rectangles of this cluster.

        :rtype: class:`clusterizer.rectangle.Rectangle`
        """
        return functools.reduce(operator.__or__, self.rectangles)

    def get_width(self):
        """The distance in m between the two edges of the bounding rectangle. `numpy.inf` if undefined.

        :rtype: float
        """
        return self.get_bounding_rectangle().get_width()

    def get_duration(self):
        """The time delta between the two edges of the bounding rectangle. `numpy.inf` if undefined.

        :rtype: numpy.timedelta64
        """
        return self.get_bounding_rectangle().get_duration()

    @property
    def location_range(self):
        return self.get_bounding_rectangle().location_range

    @property
    def time_range(self):
        return self.get_bounding_rectangle().time_range

    @property
    def found_by(self):
        return self.get_bounding_rectangle().found_by
