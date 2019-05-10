import numpy as np
import scipy.stats
import functools
from clusterizer.cluster import Cluster
from sklearn.cluster import DBSCAN


def clusterize_poisson_1d(circuit, bin_size=4, weigh_charges=False, nominal_circuit_fraction=.80, certainty=.95, min_cluster_size=3, max_skipped_bins=2):
    """Identify clusters using the Poisson algorithm, as described in TODO

    :param circuit: The circuit the clusterize.
    :type circuit: class:`clusterizer.circuit.Circuit`

    :param bin_size: Location bin width (m)
    :type bin_size: float

    :param weigh_charges: When set to `True`, PD charges are accumulated, otherwise PD occurences are counted.
    :type weigh_charges: bool

    :param nominal_circuit_fraction: Lower bound for the fraction (of total circuit length) assumed to show nominal PD behaviour. Under this assumption, a statistic model is fitted to nominal PD behaviour.

    :param certainty: After a model is fitted to nominal behaviour, line sections with bin counts that are _abnormally high, with given certainty_ are identified as "highly suspicious".
    :type certainty: float

    :param min_cluster_size: Minimum number of highly suspicious line segments needed to form a cluster.
    :type min_cluster_size: int

    :param max_skipped_bins: Two sequences of highly suspicious line segments with a seperation less than this number are combined into a single cluster.
    :type max_skipped_bins: int

    :return: 4-element tuple containing
        (list of class:`clusterizer.cluster.Cluster`) found clusters;
        (np.ndarray) bin edges (including the right-most edge);
        (np.ndarray) bin counts;
        (float) the found 80% threshold of bin counts;
        (float) the rate parameter of the fitted Poisson model;
    :rtype: tuple
    """
    # TODO: the actual _certainty_ that a found cluster is abnormal is greater than 95%: it is the probability of finding _3 abnormal values, with at most 2 skipped values between them_. A lower bound would be
    # binomcdf(n=7, k=3, p=.05) = 0.999806421875
    # 7 is the length of X--X--X
    # right? ? ?

    locations = circuit.pd["Location in meters (m)"][circuit.pd_occured]
    charges = circuit.pd["Charge (picocoulomb)"][circuit.pd_occured]
    # %% Discretize PD locations
    # Could be sped up using more efficient methods, parallisation, and by taking advantage of the uniform bin size.
    # See: https://iscinumpy.gitlab.io/post/histogram-speeds-in-python/

    bins = np.arange(start=0., stop=circuit.circuitlength+bin_size, step=bin_size)
    # NP.HISTOGRAM bin_contents, _ = np.histogram(locations, bins=bins, weights=charges if weigh_charges else None)
    bin_contents = faster_histogram_1d(locations,
                                       bins_start=0.0,
                                       bin_width=bin_size,
                                       num_bins=int(circuit.circuitlength / bin_size)+1,
                                       weights=charges if weigh_charges else None,
                                       check_inside_bounds=False)

    # %% Find the 80% quantile: find the (lowest) value M such that at least 80% of bins have a content <= M.
    nominal_pd_quantile_level = np.sort(bin_contents)[int(nominal_circuit_fraction * len(bin_contents))] + 1

    # %% Fit a Poisson distribution on nominal data
    square = lambda x: x*x

    phieta = scipy.stats.norm.ppf(q=nominal_circuit_fraction)
    rate = .25*square(-phieta + np.sqrt(square(phieta) + 4*nominal_pd_quantile_level))

    # %% Find the threshold of bin contents that are abnormally high, with certainty `certainty`
    if rate == 0.0:
        # rate was zero: fault_pd_level should be zero, not nan
        fault_pd_level = 0.0
    else:
        # Als ik de ppf van de scipy.stats.poisson gebruik krijg ik alleen maar resultaten heel dichtbij M_{\eta}
        # dat hoort niet (toch?)
        # Normal approximation (by the Central Limit Theorem)
        fault_pd_level = scipy.stats.norm.ppf(q=certainty, loc=rate, scale=rate)

    # %% Identify clusters of bins with abnormally high bin counts
    cluster_edges = cluster_boolean_series(bin_contents > fault_pd_level, max_consecutive_false=max_skipped_bins, min_length=0, min_count=min_cluster_size)
    # It might be better to create a `clusterize_poisson_result` class containing all these intermediate values.
    # Similar to `OptimizeResult` in `scipy.optimize` (https://docs.scipy.org/doc/scipy/reference/optimize.html)

    clusters = set(Cluster(location_range=tuple(bin_size * np.array(c))) for c in cluster_edges)

    return clusters, bins, bin_contents, nominal_pd_quantile_level, rate


def clusterize_poisson(circuit, bin_size=4, weigh_charges=False, nominal_circuit_fraction=.80, certainty=.95, min_cluster_size=3, max_skipped_bins=2, time_bin_size=np.timedelta64(7, 'D'), magic_factor=10):
    # TODO: The magic factor should be the 95% quantile of X/Y, where X,Y are two iid Poisson variables.
    locations = circuit.pd["Location in meters (m)"][circuit.pd_occured]
    charges = circuit.pd["Charge (picocoulomb)"][circuit.pd_occured]
    times = circuit.pd["Date/time (UTC)"][circuit.pd_occured]
    times = np.float64(times)
    time_bin_size = np.float64(np.timedelta64(time_bin_size, 'ns'))
    # %% Apply the 1D algorithm
    loc_clusters, bins, bin_contents, nominal_pd_quantile_level, rate = clusterize_poisson_1d(
            circuit,
            bin_size=bin_size,
            weigh_charges=weigh_charges,
            nominal_circuit_fraction=nominal_circuit_fraction,
            certainty=certainty,
            min_cluster_size=min_cluster_size,
            max_skipped_bins=max_skipped_bins)

    # %% Find _nusters_
    is_below_quantile = bin_contents < nominal_pd_quantile_level

    # We group is boolean series to find the _nusters_: ranges of circuit that show nominal PD behaviour.
    below_quantile_groups = cluster_boolean_series(is_below_quantile)
    nuster_ranges = [np.array(g)*bin_size for g in below_quantile_groups]

    total_nusters_length = sum(b - a for a, b in nuster_ranges)

    # %% Did we find enough nusters to continue?
    # (Although we suspect that this situation is actually impossible)
    if total_nusters_length < circuit.circuitlength * 0.1:
        print("2D poisson model failed on Circuit {0}: there are not enough line segments with nominal PD behaviour. 1D clusters will be returned.".format(circuit.circuitnr))
        return loc_clusters

    # %% For each PD, determine whether it lies in one of the nusters

    def which_pds_inside_location_range(location_range):
        return np.logical_and(location_range[0] < locations.values, locations.values < location_range[1])

    in_a_nuster = functools.reduce(np.logical_or, map(which_pds_inside_location_range, nuster_ranges))
    # in_a_nuster is an np.array with boolean values, the length of which is the number of PDs.

    # in_a_nuster[i] == True
    #    if and only if
    # PD with index i is contained in one of the nusters

    times_in_nuster = times[in_a_nuster]
    if weigh_charges:
        charges_in_nuster = charges[in_a_nuster]

    # %% Discretize PD counts (only those PDs that lie inside a nuster) in second dimension
    # NP.HISTOGRAM: time_bins = np.arange(min(times), max(times) + time_bin_size, time_bin_size)
    # NP.HISTOGRAM: nuster_counts, _ = np.histogram(times_in_nuster, bins=time_bins)
    nuster_counts = faster_histogram_1d(times_in_nuster,
                                        bins_start=min(times),
                                        bin_width=time_bin_size,
                                        num_bins=int((max(times) - min(times))/time_bin_size)+1,
                                        weights=charges_in_nuster if weigh_charges else None,
                                        check_inside_bounds=False)

    # %% Discretize in second dimension
    found_2d_clusters = set()

    for loc_cluster in loc_clusters:
        in_current_loc_cluster = which_pds_inside_location_range(loc_cluster.location_range)
        times_in_loc_cluster = times[in_current_loc_cluster]
        if weigh_charges:
            charges_in_loc_cluster = charges[in_current_loc_cluster]
        # NP.HISTOGRAM: clust_counts, _ = np.histogram(times[which_pds_inside_location_range(loc_cluster.location_range)], bins=time_bins)
        clust_counts = faster_histogram_1d(times_in_loc_cluster,
                                           bins_start=min(times),
                                           bin_width=time_bin_size,
                                           num_bins=int((max(times) - min(times))/time_bin_size) + 1,
                                           weights=charges_in_loc_cluster if weigh_charges else None,
                                           check_inside_bounds=False)

        # We study the ratio of PDs
        cluster_length = loc_cluster.get_width()
        nominal_ratio = cluster_length / total_nusters_length

        with np.errstate(invalid="ignore"):
            found_ratio = clust_counts / nuster_counts
        found_ratio[np.isnan(found_ratio)] = 0.0

        is_suspiciously_high_ratio = found_ratio > magic_factor * nominal_ratio

        for start_index, end_index in cluster_boolean_series(is_suspiciously_high_ratio, max_consecutive_false=1, min_length=0, min_count=2):
            # NP.HISTOGRAM: time_range = (time_bins[start_index], time_bins[end_index])
            time_range = (np.array([start_index, end_index]) * time_bin_size + min(times)).astype("datetime64[ns]")
            cluster = Cluster(location_range=loc_cluster.location_range, time_range=tuple(time_range))
            found_2d_clusters.add(cluster)

    return found_2d_clusters


def faster_histogram_1d(a, bins_start, bin_width, num_bins, weights=None, check_inside_bounds=True):
    """When bin sizes are constant, this method is about 3x faster than `np.histogram`.
    It uses fast float -> integer casting to calculate the bin index of a value.
    Even greater speedups can be achieved. For example, using C bindings is claimed to be 8x faster than `np.histogram`.

    :param a: Input data
    :type a: array_like

    :param bins_start: The left edge of the first bin
    :type bins_start: float

    :param bin_width: Bin width. Constant for all bins (if not, use `np.histogram` instead)
    :type bin_width: float

    :param num_bins: Number of bins
    :type num_bins: int

    :param weights: Weights. Must have the same length as `a`
    :type weights: array_like

    :param check_inside_bounds: When True (default), the function first checks whether each value is contained in any of the bins. Values outside bounds `[bins_start ... bins_start + num_bins * bin_width]` are then ignored (not counted). Set to False when it is guaranteed that all values of `a` lie within the bounds, to skip the check for an additional speed-up.
    """
    if check_inside_bounds:
        inside = np.logical_and(bins_start < a, a < bins_start + (num_bins) * bin_width)
        a_inside = a[inside]
        weights_inside = None if weights is None else weights[inside]
    else:
        a_inside = a
        weights_inside = weights

    bin_indices = ((a_inside - bins_start) * (1.0 / bin_width)).astype(np.int64)
    return np.bincount(bin_indices, weights=weights_inside, minlength=num_bins)


def cluster_boolean_series(series, max_consecutive_false=5, min_length=5, min_count=0):
    """Imperative algorithm to identify sequences of mostly True values, under the conditions imposed by the parameters:

    :param series: Sequence of Booleans to cluster
    :type series: array_like

    :param max_consecutive_false: Maximum number of consecutive Falsey values to accept inside a cluster
    :type max_consecutive_false: int

    :param min_length: Minimum cluster length (right bound - left bound)
    :type min_length: int

    :param min_count: Minimum cluster size (number of Truthy values inside cluster bounds)
    :type min_count: int
    """

    clusters = set()

    cluster_start = 0   # Beginindex van het huidige cluster
    gap_size = 0        # Lengte van de rij nee'tjes die nu wordt belopen
    true_count = 0      # Aantal ja'tjes dat is gevonden in dit cluster

    for i, x in enumerate(series):
        if x:  # We doorlopen ja'tjes
            true_count += 1
            if gap_size > max_consecutive_false:   # Einde cluster
                cluster_end = i - gap_size
                if cluster_end - cluster_start >= min_length and true_count >= min_count:
                    # Cluster was lang genoeg en heeft genoeg ja'tjes:
                    clusters.add((cluster_start, cluster_end))

                # Begin een nieuw cluster
                cluster_start = i
                true_count = 0

            gap_size = 0  # We doorlopen geen nee'tjes (meer)

        if not x:  # We doorlopen nee'tjes
            gap_size += 1

    cluster_end = len(series) - gap_size
    if cluster_end - cluster_start >= min_length and true_count >= min_count:
        # Cluster was lang genoeg en heeft genoeg ja'tjes:
        clusters.add((cluster_start, cluster_end))
    return clusters


def clusterize_DBSCAN(circuit, binLengthX = 2, binLengthY = 1, epsilon = 3, minPts = 125, shave = 0.01):
    """Identify two-dimensional clusters based on DBSCAN, a density based clustering alogrithm from python library scikit-learn. It uses the following parameters:

    :param circuit: The circuit the clusterize.
    :type circuit: class:`clusterizer.circuit.Circuit`

    :param binLengthX: Location bin width (m)
    :type binLengthX: float

    :param binLengthY: time bin width (weeks)
    :type binLengthY: float

    :param epsilon: radius of the neighborhoods that DBSCAN uses
    :type epsilon: float

    :param minPts: minimum amount of points in an epsilon-neighborhood to be recognized as a core point by DBSCAN
    :type minPts: float

    :param shave: percentage of points that are removed from the edges of the clusters, to make them fit better
    :type shave: float

    :return: found clusters
    :rtype: list of class:`clusterizer.cluster.Cluster`
    """

    # loading data
    pds = circuit.pd[["Location in meters (m)", "Date/time (UTC)"]][circuit.pd_occured]
    times = pds["Date/time (UTC)"]
    times2 = circuit.pd["Date/time (UTC)"]
    locations = pds["Location in meters (m)"]

    # the following block of code is from https://iscinumpy.gitlab.io/post/histogram-speeds-in-python/
    # making a histogram of the data
    vals = np.array(pds)
    for val in vals:
        val[1] = val[1].value/1000000000/60/60/24/7/binLengthY
    vals = vals.T
    starttime = times2[0].value/1000000000/60/60/24/7/binLengthY
    endtime = times2[len(circuit.pd)-1].value/1000000000/60/60/24/7/binLengthY
    endlocation = circuit.circuitlength
    bins = (int(endlocation/binLengthX), int(endtime-starttime))
    ranges = ((0,endlocation),(starttime,endtime))
    bins = np.asarray(bins).astype(np.int64)
    ranges = np.asarray(ranges).astype(np.float64)
    edges = (np.linspace(*ranges[0,:], bins[0]+1), np.linspace(*ranges[1,:], bins[1]+1))
    cuts = (vals[0]>=ranges[0,0]) & (vals[0]<ranges[0,1]) & (vals[1]>=ranges[1,0]) & (vals[1]<ranges[1,1])
    c = ((vals[0,cuts] - ranges[0,0]) / (ranges[0,1] - ranges[0,0]) * bins[0]).astype(np.int_)
    c += bins[0]*((vals[1,cuts] - ranges[1,0]) / (ranges[1,1] - ranges[1,0]) * bins[1]).astype(np.int_)
    weights = np.bincount(c, minlength=bins[0]*bins[1]).reshape(*bins)

    # reshaping and scaling the data to fit DBSCAN
    weights = weights.reshape(bins[0]*bins[1],1)
    data = np.mgrid[0:bins[1], 0:bins[0]].reshape(2,-1).T.astype(np.float64)
    data[:,[0, 1]] = data[:,[1, 0]]
    weightedData = np.concatenate((data,weights), axis = 1)

    # removing empty bins
    weightedDataNoZero = np.array([row for row in weightedData if row[2] > 0])

    # DBSCAN
    labels = DBSCAN(eps=epsilon, min_samples=minPts).fit(weightedDataNoZero[:, [0,1]], sample_weight = weightedDataNoZero[:, 2] ).labels_

    # rescaling the data
    weightedDataNoZero[:,2] = labels
    weightedDataNoZero[:,0] *= endlocation/bins[0]
    weightedDataNoZero[:,0] += endlocation/bins[0]/2
    weightedDataNoZero[:,1] += (starttime + (endtime-starttime)/bins[1]/2)

    # make "rough" clusters
    clusterAmount = len(set(labels))-1
    locLower = [min([row[0] for row in weightedDataNoZero if row[2] == i]) - endlocation/bins[0]/2 for i in range(clusterAmount)]
    locUpper = [max([row[0] for row in weightedDataNoZero if row[2] == i]) + endlocation/bins[0]/2 for i in range(clusterAmount)]
    timeLower = [np.datetime64(int((min([row[1] for row in weightedDataNoZero if row[2] == i]) - ((endtime-starttime)/bins[1]/2))*60*60*24*7*binLengthY), 's') for i in range(clusterAmount)]
    timeUpper = [np.datetime64(int((max([row[1] for row in weightedDataNoZero if row[2] == i]) + ((endtime-starttime)/bins[1]/2))*60*60*24*7*binLengthY), 's') for i in range(clusterAmount)]
    clusters = set(Cluster(location_range=(locLower[i], locUpper[i]), time_range=(timeLower[i], timeUpper[i])) for i in range(clusterAmount))

    # fit the clusters by shaving a small amount of points from the edges
    clusters2 = set()
    for cluster in clusters:
        locationIndex = locations[locations>=cluster.location_range[0]][locations<=cluster.location_range[1]].index
        timeIndex = times[times>=cluster.time_range[0]][times<=cluster.time_range[1]].index
        index = [point for point in locationIndex if point in timeIndex]
        locations2 = locations.loc[index].sort_values()
        beginLoc = locations2.iloc[int(len(locations2)*shave)+1]
        endLoc = locations2.iloc[int(len(locations2)*(1-shave))-1]
        beginTime = np.datetime64(times.loc[index[int(len(index)*shave)+1]])
        endTime = np.datetime64(times.loc[index[int(len(index)*(1-shave))-1]])
        clusters2.add(Cluster(location_range=(beginLoc, endLoc), time_range=(beginTime, endTime)))
    return(clusters2)
