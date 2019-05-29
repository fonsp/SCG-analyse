from functools import total_ordering
import functools
import numpy as np
import pandas as pd
from . import cluster


class ClusterSet:
    """
    A set of Cluster objects
    In a ClusterEnsemble, this represents one cluster
    """
    def __init__(self, clusters):
        """
        :param clusters: The clusters to be added to this ClusterSet
        :rtype: collection of class:`clusterizer.cluster.Cluster`
        """
        self.clusters = set(clusters)

    def __str__(self):
        result = "{"
        for c in self.clusters:
            result += str(c) + "\n"
        return result[:-1] + "}"

    def __repr__(self):
        return str(self)

    def __bool__(self):
        """
        Boolean representation of a ClusterSet. Mirrors behaviour of sets.
        
        :return: False if self.clusters is empty, True if non-empty
        :rtype: bool
        """
        return bool(self.clusters)

    def __len__(self):
        """
        Number of Clusters in the ClusterSet. Mirrors behaviour of sets.
        
        :return: The length of self.clusters
        :rtype: int
        """
        return len(self.clusters)

    def __iter__(self):
        """
        Provides an iterator over the ClusterSet. Mirrors behaviour of sets.
        This makes expressions like `for c in my_clusterset` possible,
        so you don't have to type `for c in my_clusterset.as_set()`

        :return: Iterator over the Clusters in this ClusterSet
        :rtype: iterator
        """
        return self.clusters.__iter__()

    def get_clusters(self):
        """
        Returns the Clusters in this ClusterSet, without casting to a specific collection type.

        :return: The Clusters in this ClusterSet
        :rtype: Collection of class:`clusterizer.cluster.Cluster`
        """
        return self.clusters

    def as_set(self):
        """
        Returns a set containing the Clusters in this ClusterSet.

        :return: The Clusters in this ClusterSet
        :rtype: set of class:`clusterizer.cluster.Cluster`
        """
        return set(self.clusters)

    def as_list(self):
        """
        Returns a list containing the Clusters in this ClusterSet.

        :return: The Clusters in this ClusterSet
        :rtype: list of class:`clusterizer.cluster.Cluster`
        """
        return list(self.clusters)

    def disjunct(self, other):
        """
        Determines if two ClusterSets are disjunct. That is, disjunct(self, other) returns:
        - True if self and other do not contain any overlap in any of their Clusters
        - False if self and other overlap in any of their Clusters
        So: self & other has to be empty for disjunct(self, other) to be True.

        :param other: The other ClusterSet to compare self to
        :type other: class:`clusterizer.ensemble.ClusterSet`

        :return: False if self and other have overlap, True otherwise.
        :rtype: bool
        """
        return not bool(self & other)

    def __and__(self, other):
        """
        Calculates the overlap between two ClusterSets, overloading the & operator.
        This is done by calculating the overlap between the underlying Cluster objects.
        In doing so, it keeps track of which algorithms found the clusters.

        :param other: The other ClusterSet to calculate the overlap with
        :type other: class:`clusterizer.ensemble.ClusterSet`

        :return: The overlap between self and other
        :rtype: class:`clusterizer.ensemble.ClusterSet`
        """
        result = set()
        for c1 in self:
            for c2 in other:
                overlap = c1 & c2
                if overlap is not None:
                    result.add(overlap)
        return ClusterSet(result)

    def __mul__(self, other):
        """
        Calculates the overlap between two ClusterSets, overloading the * operator.
        Alternate alias for the & operator. Behaves exactly the same as &.

        :param other: The other ClusterSet to calculate the overlap with
        :type other: class:`clusterizer.ensemble.ClusterSet`

        :return: The overlap between self and other
        :rtype: class:`clusterizer.ensemble.ClusterSet`
        """
        return self & other

    def __or__(self, other):
        """
        Calculates the bounding box of two ClusterSets, overloading the | operator.h
        This is done by calculating the bounding box of the underlying Cluster objects.
        For overlapping Cluster objects, the bounding box is calculated.
        Separate Clusters are kept separate. If there are Cluster objects that are disjunct from all others,
        the resulting ClusterSet will contain multiple Cluster objects.

        :param other: The other ClusterSet to calculate the bounding box with
        :type other: class:`clusterizer.ensemble.ClusterSet`

        :return: The bounding box of the clusters in self and other
        :rtype: class:`clusterizer.ensemble.ClusterSet`
        """
        if self.disjunct(other):
            return ClusterSet(self.clusters | other.clusters)
        result = ClusterSet(self.clusters)
        helper = ClusterSet(other.clusters)
        while helper:
            helpercur = helper.clusters.pop()
            for clust in result:
                if not helpercur.disjunct(clust):
                    helper.clusters.add(helpercur | clust)
                    result.clusters.remove(clust)
                    break
            else:
                result.clusters.add(helpercur)
        return result

    def __add__(self, other):
        """
        Calculates the rich set-theoretic union of two ClusterSets, overloading the + operator.
        This is done by calling the + operator on the underlying Cluster objects.
        In doing so, it keeps track of which algorithms found the clusters.
        The best analogy for the result is a Venn diagram. The overlapping parts are where the algorithms agree. Unlike &, + also remembers where the algorithms disagree.
        
        :param other: The other ClusterSet to calculate the union with
        :type other: class:`clusterizer.ensemble.ClusterSet`

        :return: The set-theoretic union of the clusters in self and other
        :rtype: class:`clusterizer.ensemble.ClusterSet`
        """
        if self.disjunct(other):
            return ClusterSet(self.clusters | other.clusters)
        result = ClusterSet(self.clusters)
        helper = ClusterSet(other.clusters)
        while helper:
            helpercur = helper.clusters.pop()
            for clust in result:
                if not helpercur.disjunct(clust):
                    helper.clusters |= helpercur + clust
                    result.clusters.remove(clust)
                    break
            else:
                result.clusters.add(helpercur)
        return result

    def get_partial_discharges(self, circuit):
        """Returns all PDs that lie in any of the Clusters in this ClusterSet."""
        return circuit.pd[circuit.pd_occured].loc[self.get_partial_discharge_mask(circuit)]

    def get_partial_discharge_mask(self, circuit):
        """Returns a boolean array which indicates, for each PD, whether it lies in any of the Clusters in this ClusterSet."""
        return functools.reduce(np.logical_or, [cs.get_partial_discharge_mask(circuit) for cs in self.clusters])

    def most_confident(self):
        """Returns a new ClusterSet containing only those Clusters of `self` with the highest number of algorithms that found it.

        :return: The Cluster object with the highest number of algorithms that found it
        :rtype: class:`clusterizer.ensemble.ClusterSet`
        """
        result = set()
        confidence = -1
        for c in self:
            if len(cluster.found_by) > confidence:
                result = {cluster}
                confidence = len(cluster.found_by)
            elif len(cluster.found_by) == confidence:
                result.add(cluster)
        return ClusterSet(result)


class ClusterEnsemble:
    """
    A Cluster Ensemble should be a set of ClusterSet objects
    The whole set makes the ensemble
    Each set in the ensemble represents a cluster of arbitrary shape
    The Cluster objects in each ClusterSet are 'rectangels' which combine to make a cluster
    """

    def __init__(self, sets):
        """
        :param sets: The ClusterSet objects to be added to this ClusterEnsemble
        :type sets: collection of class:`clusterizer.ensemble.ClusterSet`
        """
        self.sets = set(sets)

    @staticmethod
    def from_iterable(cluster_iterable):
        """
        Creates a ClusterEnsemble from an iterable containing Cluster objects.
        Each Cluster will be added to its own ClusterSet, creating a layered structure.
        This structure signifies that each Cluster in the iterable is disjunct from all others.

        :param cluster_iterable: The clusters to create a ClusterEnsemble with
        :type cluster_iterable: iterable of class:`clusterizer.cluster.Cluster`

        :return: A ClusterEnsemble containing ClusterSets with all Clusters from the iterable
        :rtype: class:`clusterizer.ensemble.ClusterEnsemble`
        """
        ensemble = set()
        for x in cluster_iterable:
            ensemble.add(ClusterSet([x]))
        return ClusterEnsemble(ensemble)

    def __str__(self):
        result = "{"
        for c in self:
            result += str(c) + "\n"
        return result[:-1] + "}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        """
        A hash function that computes a number based on the Clusters in the underlying ClusterSets
        Use case: Adding ClusterEnsembles to dictionaries, sets, etc.
        This function is cryptographically weak and should not be used for security purposes.

        :return: A hash of the ClusterEnsemble
        :rtype: int
        """
        hashed = 0
        for c in self.sets:
            hashed += hash(c)
        return hashed

    def __iter__(self):
        """
        Creates an iterator over the ClusterEnsemble. Mirrors behaviour of sets.
        This makes expressions like `for c in my_clusterensemble` possible,
        so you don't have to type `for c in my_clusterensemble.as_set()`

        :return: An iterator over the ClusterSets in this ClusterEnsemble
        :rtype: iterator
        """
        return self.sets.__iter__()

    def __bool__(self):
        """
        Boolean representation of a ClusterEnsemble. Mirrors behaviour of sets.
        
        :return: False if self.sets is empty, True if non-empty
        :rtype: bool
        """
        return bool(self.sets)

    def __len__(self):
        """
        Number of ClusterSets in the ClusterEnsemble. Mirrors behaviour of sets.
        
        :return: The length of self.sets
        :rtype: int
        """
        return len(self.sets)

    def get_clusters(self):
        """
        Creates a single set containing all the Clusters from each ClusterSet in this ClusterEnsemble.

        :return: All Clusters from all ClusterSets in this ClusterEnsemble
        :rtype: set of class:`clusterizer.cluster.Cluster`
        """
        result = set()
        for s in self.sets:
            for clust in s:
                result.add(clust)
        return result

    def flatten(self):
        """Returns a ClusterSet containing all Clusters contained in _any_ ClusterSet of this ClusterEnsemble.

        :return: A ClusterSet of all Clusters in all ClusterSets in this ClusterEnsemble
        :rtype: class:`clusterizer.ensemble.ClusterSet
        """
        result = set()
        for clusterset in self:
            result |= clusterset.as_set()
        return ClusterSet(result)

    def get_sets(self):
        """
        Returns the ClusterSets in this ClusterEnsemble, without casting to a specific collection type.

        :return: The ClusterSets in this ClusterEnsemble
        :rtype: Collection of class:`clusterizer.ensemble.ClusterSet`
        """
        return self.sets

    def as_set(self):
        """
        Returns a set containing the ClusterSets in this ClusterEnsemble.

        :return: The ClusterSets in this ClusterEnsemble
        :rtype: set of class:`clusterizer.ensemble.ClusterSet`
        """
        return set(self.sets)

    def as_list(self):
        """
        Returns a list containing the ClusterSets in this ClusterEnsemble.

        :return: The ClusterSets in this ClusterEnsemble
        :rtype: list of class:`clusterizer.ensemble.ClusterSet`
        """
        return list(self.sets)

    def disjunct(self, other):
        """
        Determines if two ClusterEnsembles are disjunct. That is, disjunct(self, other) returns:
        - True if self and other do not contain any overlap in any of their ClusterSets
        - False if self and other overlap in any of their ClusterSets
        So: self & other has to be empty for disjunct(self, other) to be True.

        :param other: The other ClusterEnsemble to compare self to
        :type other: class:`clusterizer.ensemble.ClusterEnsemble`

        :return: False if self and other have overlap, True otherwise.
        :rtype: bool
        """
        return not bool(self & other)

    def __and__(self, other):
        """
        Calculates the overlap between two ClusterEnsembles, overloading the & operator.
        This is done by calculating the overlap between the underlying ClusterSet objects.
        In doing so, it keeps track of which algorithms found the Clusters in the ClusterSets.

        :param other: The other ClusterEnsemble to calculate the overlap with
        :type other: class:`clusterizer.ensemble.ClusterEnsemble`

        :return: The overlap between self and other
        :rtype: class:`clusterizer.ensemble.ClusterEnsemble`
        """
        result = set()
        for cs1 in self:
            for cs2 in other:
                overlap = cs1 & cs2
                if overlap:
                    result.add(overlap)
        return result

    def __or__(self, other):
        """
        Calculates the bounding box of two ClusterEnsembles, overloading the | operator.h
        This is done by calculating the bounding box of the underlying ClusterSet objects.
        For overlapping Cluster objects in the ClusterSets, the bounding box is calculated.
        Separate ClusterSets are kept separate. If there are ClusterSet objects that are disjunct from all others,
        the resulting ClusterEnsemble will contain multiple ClusterSet objects.

        :param other: The other ClusterEnsemble to calculate the bounding box with
        :type other: class:`clusterizer.ensemble.ClusterEnsemble`

        :return: The bounding boxes of the clusters in the ClusterSets in self and other
        :rtype: class:`clusterizer.ensemble.ClusterEnsemble`
        """
        if self.disjunct(other):
            return ClusterEnsemble(self.sets | other.sets)
        result = ClusterEnsemble(self.sets)
        helper = ClusterEnsemble(other.sets)
        while helper:
            helpercur = helper.sets.pop()
            for rcur in result:
                if not helpercur.disjunct(rcur):
                    helper.sets.add(helpercur | rcur)
                    result.sets.remove(rcur)
                    break
            else:
                result.sets.add(helpercur)
        return result

    def __add__(self, other):
        """
        Calculates the rich set-theoretic union of two ClusterEnsembles, overloading the + operator.
        This is done by calling the + operator on the underlying ClusterSet objects.
        In doing so, it keeps track of which algorithms found the clusters.
        The best analogy for the result is a Venn diagram. The overlapping parts are where the algorithms agree. Unlike &, + also remembers where the algorithms disagree.
        
        :param other: The other ClusterEnsemble to calculate the union with
        :type other: class:`clusterizer.ensemble.ClusterEnsemble`

        :return: The set-theoretic union of the ClusterSets in self and other
        :rtype: class:`clusterizer.ensemble.ClusterEnsemble`
        """
        if self.disjunct(other):
            return ClusterEnsemble(self.sets | other.sets)
        result = ClusterEnsemble(self.sets)
        helper = ClusterEnsemble(other.sets)
        while helper:
            helpercur = helper.sets.pop()
            for rcur in result:
                if not helpercur.disjunct(rcur):
                    helper.sets.add(helpercur + rcur)
                    result.sets.remove(rcur)
                    break
            else:
                result.sets.add(helpercur)
        return result

    def most_confident(self):
    """
    Returns a ClusterEnsemble with for each ClusterSet in `self` the Clusters in that ClusterSet with the highest number of algorithms that found it.
    Confidence doesn't have to be the same for each ClusterSet. For example, one ClusterSet could contain overlap while another is only found by one algorithm. In this case, for the first ClusterSet, only the overlap is returned, while for the second ClusterSet, it is returned in its entirety.

    :return: The Clusters in each ClusterSet with the highest number of algorithms that found it
    :rtype: ClusterEnsemble
    """
        result = set()
        for clusterset in self:
            result.add(clusterset.most_confident())
        return ClusterEnsemble(result)
