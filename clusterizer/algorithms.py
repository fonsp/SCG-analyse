import numpy as np
import scipy.stats
import functools
import operator
from clusterizer.rectangle import Rectangle
from clusterizer.ensemble import ClusterEnsemble
from clusterizer.cluster import Cluster
from sklearn.cluster import DBSCAN


def clusterize_poisson_1d(circuit, certainty=.95, loc_bin_size=4, nominal_circuit_fraction=.80, weigh_charges=False, min_bin_count=2, max_bins_skipped=2, return_intermediate_values=False, name="Poisson 1D"):
    """Identify location clusters using the Poisson algorithm, as described in TODO

    :param circuit: The circuit the clusterize.
    :type circuit: class:`clusterizer.circuit.MergedCircuit`

    :param certainty: After a model is fitted to nominal PD behaviour, line sections with bin counts that are _abnormally high, with given certainty_ are identified as "highly suspicious".
    :type certainty: float, optional

    :param loc_bin_size: Location bin width (m)
    :type loc_bin_size: float, optional

    :param nominal_circuit_fraction: Lower bound for the fraction (of total circuit length) assumed to show nominal PD behaviour. Under this assumption, a statistic model is fitted to nominal PD behaviour.
    :type nominal_circuit_fraction: float, optional

    :param weigh_charges: When set to `True`, PD charges are accumulated, otherwise PD occurences are counted.
    :type weigh_charges: bool, optional

    :param min_bin_count: Minimum number of highly suspicious line segments needed to form a cluster.
    :type min_bin_count: int, optional

    :param max_bins_skipped: Two sequences of highly suspicious line segments with a seperation less than this number are combined into a single cluster.
    :type max_bins_skipped: int, optional

    :param return_intermediate_values: Also return additional values used by the algorithm?
    :type return_intermediate_values: bool, optional

    :return: When return_intermediate_values is False, returns the found clusters.
    When return_intermediate_values is True, returns
    5-element tuple containing
        (object of class:`clusterizer.cluster.ClusterEnsemble`) found clusters;
        (np.ndarray) bin edges (including the right-most edge);
        (np.ndarray) bin counts;
        (float) the found 80% threshold of bin counts;
        (float) the rate parameter of the fitted Poisson model;
    :rtype: object of class:`clusterizer.cluster.ClusterEnsemble` or tuple
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

    bins = np.arange(start=0., stop=circuit.circuitlength+loc_bin_size, step=loc_bin_size)
    # NP.HISTOGRAM bin_contents, _ = np.histogram(locations, bins=bins, weights=charges if weigh_charges else None)
    bin_contents = faster_histogram_1d(locations,
                                       bins_start=0.0,
                                       bin_width=loc_bin_size,
                                       num_bins=int(circuit.circuitlength / loc_bin_size)+1,
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

    # %% Identify groups of bins with abnormally high bin counts
    group_edges = group_boolean_series(bin_contents > fault_pd_level, max_consecutive_false=max_bins_skipped, min_length=0, min_count=min_bin_count)
    # It might be better to create a `clusterize_poisson_result` class containing all these intermediate values.
    # Similar to `OptimizeResult` in `scipy.optimize` (https://docs.scipy.org/doc/scipy/reference/optimize.html)

    rectangles = set(Rectangle(location_range=tuple(loc_bin_size * np.array(c)), found_by=[name]) for c in group_edges)

    if return_intermediate_values:
        return rectangles, bins, bin_contents, nominal_pd_quantile_level, rate
    return rectangles


def clusterize_poisson(circuit, certainty=.95, loc_bin_size=4, time_bin_size=np.timedelta64(7, 'D'), nominal_circuit_fraction=.80, weigh_charges=False, min_loc_bin_count=2, max_loc_bins_skipped=2, magic_factor=4.0, min_time_bin_count=2, max_time_bins_skipped=1, return_intermediate_values=False, name="Poisson 2D"):
    """Identify clusters using the Poisson algorithm, as described in TODO

    :param circuit: The circuit the clusterize.
    :type circuit: class:`clusterizer.circuit.MergedCircuit`

    :param certainty: After a model is fitted to nominal PD behaviour, line sections with bin counts that are _abnormally high, with given certainty_ are identified as "highly suspicious". TODO
    :type certainty: float, optional

    :param loc_bin_size: Location bin width (m)
    :type loc_bin_size: float, optional

    :param time_bin_size: Time bin width (np.timedelta64). Defaults to one week.
    :type time_bin_size: np.timedelta64, optional

    :param nominal_circuit_fraction: Lower bound for the fraction (of total circuit length) assumed to show nominal PD behaviour. Under this assumption, a statistic model is fitted to nominal PD behaviour.
    :type nominal_circuit_fraction: float, optional

    :param weigh_charges: When set to `True`, PD charges are accumulated, otherwise PD occurences are counted.
    :type weigh_charges: bool, optional

    :param min_loc_bin_count: Minimum number of highly suspicious line segments needed to form a location cluster.
    :type min_loc_bin_count: int, optional

    :param max_loc_bins_skipped: Two sequences of highly suspicious line segments with a seperation less than this number are combined into a single location cluster.
    :type max_loc_bins_skipped: int, optional

    :param magic_factor: (NOTE: this argument might be removed in the future). Once a location cluster is found, PDs inside the location cluster are counted in periods set by `time_bin_size`, and compared to PD counts in nusters (line segments with nominal PD behaviour). If the ratio between these two counts exceeds this factor, it becomes _super suspicious_.
    :type magic_factor: float, optional

    :param min_time_bin_count: Minimum number of super suspicious 2D segments needed to form a 2D cluster.
    :type min_time_bin_count: int, optional

    :param max_time_bins_skipped: Two sequences of super suspicious line segments with a seperation less than this number are combined into a single cluster.
    :type max_time_bins_skipped: int, optional

    :param return_intermediate_values: Also return additional values used by the algorithm?
    :type return_intermediate_values: bool, optional

    :return: When return_intermediate_values is False, returns the found 2D clusters.
    When return_intermediate_values is True, returns
    7-element tuple containing
        (object of class:`clusterizer.cluster.ClusterEnsemble`) found 2D clusters;
        (object of class:`clusterizer.cluster.ClusterEnsemble`) found location clusters;
        (object of class:`clusterizer.cluster.ClusterEnsemble`) found nusters;
        (np.ndarray) bin edges (including the right-most edge);
        (np.ndarray) bin counts;
        (float) the found 80% threshold of bin counts;
        (float) the rate parameter of the fitted Poisson model;
    :rtype: object of class:`clusterizer.cluster.ClusterEnsemble` or tuple
    """
    # TODO: The magic factor should be the 95% quantile of X/Y, where X,Y are two iid Poisson variables.
    locations = circuit.pd["Location in meters (m)"][circuit.pd_occured]
    charges = circuit.pd["Charge (picocoulomb)"][circuit.pd_occured]
    times = circuit.pd["Date/time (UTC)"][circuit.pd_occured]
    times = np.float64(times)
    time_bin_size = np.float64(np.timedelta64(time_bin_size, 'ns'))
    # %% Apply the 1D algorithm
    loc_rectangles, loc_bins, loc_bin_contents, nominal_pd_quantile_level, rate = clusterize_poisson_1d(
            circuit,
            certainty=certainty,
            loc_bin_size=loc_bin_size,
            nominal_circuit_fraction=nominal_circuit_fraction,
            weigh_charges=weigh_charges,
            min_bin_count=min_loc_bin_count,
            max_bins_skipped=max_loc_bins_skipped,
            return_intermediate_values=True)

    # %% Find _nusters_
    is_below_quantile = loc_bin_contents < nominal_pd_quantile_level

    # We group is boolean series to find the _nusters_: ranges of circuit that show nominal PD behaviour.
    below_quantile_groups = group_boolean_series(is_below_quantile)
    nuster_ranges = [np.array(g)*loc_bin_size for g in below_quantile_groups]

    total_nusters_length = sum(b - a for a, b in nuster_ranges)

    # %% Did we find enough nusters to continue?
    # (Although we suspect that this situation is actually impossible)
    if total_nusters_length < circuit.circuitlength * 0.1:
        print("2D poisson model failed on Circuit {0}: there are not enough line segments with nominal PD behaviour. 1D clusters will be returned.".format(circuit.circuitnr))
        return loc_rectangles

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
    found_2d_rectangles = set()

    for loc_rectangle in loc_rectangles:
        # Potential speed-up: during the 1D algorithm, PDs were binned, so a list was created of _bin indices_. This list could be reused, to avoid the use of `which_pds_inside_location_range`.
        in_current_loc_rectangle = which_pds_inside_location_range(loc_rectangle.location_range)
        times_in_loc_rectangle = times[in_current_loc_rectangle]
        if weigh_charges:
            charges_in_loc_rectangle = charges[in_current_loc_rectangle]
        # NP.HISTOGRAM: clust_counts, _ = np.histogram(times[which_pds_inside_location_range(loc_rectangle.location_range)], bins=time_bins)

        clust_counts = faster_histogram_1d(times_in_loc_rectangle,
                                           bins_start=min(times),
                                           bin_width=time_bin_size,
                                           num_bins=int((max(times) - min(times))/time_bin_size) + 1,
                                           weights=charges_in_loc_rectangle if weigh_charges else None,
                                           check_inside_bounds=False)

        # We study the ratio of PDs
        rectangle_length = loc_rectangle.get_width()
        nominal_ratio = rectangle_length / total_nusters_length

        # Dividing non-zero by zero (which gives np.inf) is a desired result.
        # Dividing zero by zero (which gives np.nan) is accounted for.
        with np.errstate(invalid="ignore", divide="ignore"):
            found_ratio = clust_counts / nuster_counts
        found_ratio[np.isnan(found_ratio)] = 0.0

        is_suspiciously_high_ratio = found_ratio > magic_factor * nominal_ratio
        for start_index, end_index in group_boolean_series(is_suspiciously_high_ratio, max_consecutive_false=max_time_bins_skipped, min_length=0, min_count=min_time_bin_count):
            # NP.HISTOGRAM: time_range = (time_bins[start_index], time_bins[end_index])
            time_range = (np.array([start_index, end_index]) * time_bin_size + min(times)).astype("datetime64[ns]")
            rectangle = Rectangle(location_range=loc_rectangle.location_range, time_range=tuple(time_range), found_by=[name])
            found_2d_rectangles.add(rectangle)

        found_2d_clusters = ClusterEnsemble(Cluster({r}) for r in found_2d_rectangles)
    if return_intermediate_values:
        nusters = ClusterEnsemble(Cluster({Rectangle(location_range=tuple(r))}) for r in nuster_ranges)
        return found_2d_clusters, loc_rectangles, nusters, loc_bins, loc_bin_contents, nominal_pd_quantile_level, rate
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


def group_boolean_series(series, max_consecutive_false=5, min_length=5, min_count=0):
    """Imperative algorithm to identify sequences of mostly True values, under the conditions imposed by the parameters:

    :param series: Sequence of Booleans to group
    :type series: array_like

    :param max_consecutive_false: Maximum number of consecutive Falsey values to accept inside a group
    :type max_consecutive_false: int

    :param min_length: Minimum group length (right bound - left bound)
    :type min_length: int

    :param min_count: Minimum group size (number of Truthy values inside group bounds)
    :type min_count: int
    """

    groups = set()

    group_start = 0   # Beginindex van het huidige group
    gap_size = 0        # Lengte van de rij nee'tjes die nu wordt belopen
    true_count = 0      # Aantal ja'tjes dat is gevonden in dit group

    for i, x in enumerate(series):
        if x:  # We doorlopen ja'tjes
            true_count += 1
            if gap_size > max_consecutive_false:   # Einde group
                group_end = i - gap_size
                if group_end - group_start >= min_length and true_count >= min_count:
                    # group was lang genoeg en heeft genoeg ja'tjes:
                    groups.add((group_start, group_end))

                # Begin een nieuw group
                group_start = i
                true_count = 0

            gap_size = 0  # We doorlopen geen nee'tjes (meer)

        if not x:  # We doorlopen nee'tjes
            gap_size += 1

    group_end = len(series) - gap_size
    if group_end - group_start >= min_length and true_count >= min_count:
        # group was lang genoeg en heeft genoeg ja'tjes:
        groups.add((group_start, group_end))
    return groups


def clusterize_pinta(circuit, placeinterval=10, timeinterval=np.timedelta64(7, 'D'), sensitivity=1.0, name="Pinta"):
    """Algorithm that identifies clusters by using the fact that a lot of 2D-bins have the same amount of partial discharges. If the "gap" between two bins is too high, it means there is something going on with the bin. It uses the following parameters:

    :param circuit: The circuit the clusterize.
    :type circuit: class:`clusterizer.circuit.MergedCircuit`

    :param placeinterval: place bin width (m)
    :type binLengthY: float, optional

    :param timeinterval: time bin width (timedelta64)
    :type timeinterval: numpy.timedelta64, optional

    :param sensitivity: The higher this value, the more clusters the algorithm will find.
    :type minPts: float, optional

    :return: found clusters
    :rtype: object of class:`clusterizer.cluster.ClusterEnsemble`
    """
    def track_groups(elt, todo, ijlist):
        if elt in ijlist:
            todo += [elt]
            ijlist.remove(elt)

    def flood_fill(grid, condition):
        sizex, sizey = grid.shape
        ijlist = []
        for i in range(sizex):
            for j in range(sizey):
                if grid[i, j] >= condition:
                    ijlist += [[i, j]]
        groups = []
        groupcount = 0
        for ij in ijlist:
            groups += [[ij]]
            groupstart = True
            todo = [ij]
            ijlist.remove(ij)
            while len(todo) > 0:
                for p in todo:
                    track_groups([p[0]+1, p[1]], todo, ijlist)
                    track_groups([p[0]-1, p[1]], todo, ijlist)
                    track_groups([p[0], p[1]+1], todo, ijlist)
                    track_groups([p[0], p[1]-1], todo, ijlist)
                    if not groupstart:
                        groups[groupcount] += [p]
                    groupstart = False
                    todo.remove(p)
            groupcount += 1
        return np.array(groups)

    invsensitivity = 1.0 / sensitivity

    locations = circuit.pd["Location in meters (m)"][circuit.pd_occured]
    times = circuit.pd["Date/time (UTC)"][circuit.pd_occured]
    mintime = times.values[0]
    maxplace = circuit.circuitlength
    maxtime = times.values[-1]

    # create bins:

    times_float = times.values.astype(float)

    mintime_float = float(mintime)
    maxtime_float = float(maxtime)

    loc_bins = np.arange(0, maxplace, placeinterval)
    time_bins = np.arange(mintime_float, maxtime_float, float(np.timedelta64(timeinterval, "ns")))
    grid, _, _ = np.histogram2d(locations, times_float, bins=[loc_bins, time_bins])

    #determine the minimum amount of partial discharges needed for a bin to be part of a cluster

    grid_flattened = grid.flatten()
    grid_flattened = np.sort(grid_flattened)
    gridlength = len(grid_flattened)
    ratio = grid_flattened - invsensitivity * np.array(range(gridlength))

    minval = grid_flattened[np.argmin(ratio)]

    # group data

    groups = flood_fill(grid, minval)
    minplacezzzzz = np.zeros(len(groups))
    maxplacezzzzz = np.zeros(len(groups))
    mintimezzzzz = np.empty(len(groups), dtype='datetime64[s]')
    maxtimezzzzz = np.empty(len(groups), dtype='datetime64[s]')
    for i in range(len(groups)):
        # groups[0] = [[1,20],[3,3],[5,5]]
        # minc =
        minc = np.amin(groups[i], axis=0)
        maxc = np.amax(groups[i], axis=0)
        minplacezzzzz[i] = minc[0]*placeinterval
        maxplacezzzzz[i] = (maxc[0]+1)*placeinterval
        mintimezzzzz[i] = mintime + minc[1] * timeinterval
        maxtimezzzzz[i] = mintime + (maxc[1]+1)*timeinterval
    clusters = ClusterEnsemble(Cluster({Rectangle(location_range=(minplacezzzzz[i], maxplacezzzzz[i]), time_range=(mintimezzzzz[i], maxtimezzzzz[i]), found_by=[name])}) for i in range(len(groups)))
    return clusters


def clusterize_DBSCAN(circuit, binLengthX = 2, binLengthY = 1, epsilon = 3, minPts = 125, shave = 0.01, name="DBSCAN"):
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
    :rtype: object of class:`clusterizer.cluster.ClusterEnsemble`
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
    sample_weight = weightedDataNoZero[:, 2]

    # DBSCAN
    labels = DBSCAN(eps=epsilon, min_samples=minPts).fit(weightedDataNoZero[:, [0,1]], sample_weight = weightedDataNoZero[:, 2] ).labels_

    # rescaling the data
    clusterAmount = len(set(labels))-1
    weightedDataNoZero[:, 2] = labels
    weightedDataNoZero[:,0] *= endlocation/bins[0]
    weightedDataNoZero[:,0] += endlocation/bins[0]/2
    weightedDataNoZero[:,1] += (starttime + (endtime-starttime)/bins[1]/2)

     # make "rough" clusters

    locLower = [min([row[0] for row in weightedDataNoZero if row[2] == i]) - endlocation/bins[0]/2 for i in range(clusterAmount)]
    locUpper = [max([row[0] for row in weightedDataNoZero if row[2] == i]) + endlocation/bins[0]/2 for i in range(clusterAmount)]
    timeLower = [np.datetime64(int((min([row[1] for row in weightedDataNoZero if row[2] == i]) - ((endtime-starttime)/bins[1]/2))*60*60*24*7*binLengthY), 's') for i in range(clusterAmount)]
    timeUpper = [np.datetime64(int((max([row[1] for row in weightedDataNoZero if row[2] == i]) + ((endtime-starttime)/bins[1]/2))*60*60*24*7*binLengthY), 's') for i in range(clusterAmount)]
    rectangles = set(Rectangle(location_range=(locLower[i], locUpper[i]), time_range=(timeLower[i], timeUpper[i])) for i in range(clusterAmount))

    # fit the clusters by shaving a small amount of points from the edges
    rectangles2 = set()
    for rectangle in rectangles:
        locationIndex = locations[locations >= rectangle.location_range[0]][locations <= rectangle.location_range[1]].index
        timeIndex = times[times >= rectangle.time_range[0]][times <= rectangle.time_range[1]].index
        index = [point for point in locationIndex if point in timeIndex]
        locations2 = locations.loc[index].sort_values()
        beginLoc = locations2.iloc[int(len(locations2)*shave)+1]
        endLoc = locations2.iloc[int(len(locations2)*(1-shave))-1]
        beginTime = np.datetime64(times.loc[index[int(len(index)*shave)+1]])
        endTime = np.datetime64(times.loc[index[int(len(index)*(1-shave))-1]])
        rectangles2.add(Rectangle(location_range=(beginLoc, endLoc), time_range=(beginTime, endTime), found_by=[name]))
    return(ClusterEnsemble(Cluster({r}) for r in rectangles2))


def clusterize_ensemble_additive(circuit, algorithms=None, add=True):
    """
    Identify two dimensional clusters using multiple algorithms. The results are combined using the ClusterEnsemble class methods.
    algorithms should be an iterable containing algorithms. The algorithms should take as input a clusterizer.circuit.Circuit object and give as output a clusterizer.cluster.ClusterEnsemble object.
    If add is set to true, the clusters will be added together. This means that the overlap is found between the clusters and the clusters are combined in a venn diagram like way. If add is set to false, the clusters will be orred together. The result of an or is the bounding box of two clusters (if they have overlap, clusters without overlap will remain separate).

    :param circuit: The circuit the clusterize.
    :type circuit: class:`clusterizer.circuit.Circuit`


    :param algorithms: List of algorithms (methods from this submodule) to be used. Defaults to [clusterize_poisson, clusterize_DBSCAN, clusterize_pinta].
    :type algorithms: iterable, optional

    :param add: Set to true to __add__ (+) the clusters together, set to false to __or__ (|) them together
    :type add: bool, optional
    """
    result = ClusterEnsemble(set())
    if algorithms is None:
        algorithms = [clusterize_poisson, clusterize_DBSCAN, clusterize_pinta]
    for alg in algorithms:
        clusters = alg(circuit)
        if add:
            result += clusters
        else:
            result |= clusters
    return result


def warnings_to_clusters(circuit, include_noise_warnings=True, rectangle_width=None):
    """
    A clusterizer 'algorithm' that creates a Cluster for each warning given by DNV GL.

    :param circuit: The circuit to clusterize.
    :type circuit: class:`clusterizer.circuit.MergedCircuit`

    :param include_noise_warnings: When set to False, "Noise" warnings are skipped, and only level 1-3 warnings are converted.
    :type include_noise_warnings: bool, optional

    :param rectangle_width: Width (m) of the Rectangle to create. When set to `None`, 1% of the circuit length is used (0.5% at both sides).
    :type rectangle_width: float, optional
    """
    if circuit.warning is None or circuit.warning.empty or len(circuit.warning) == 0:
        return set()

    warning_rectangles = set()
    for i, w in circuit.warning.sort_values(by=['SCG warning level (1 to 3 or Noise)']).iterrows():
        # Using str key in dict instead of int to support 'Noise' warning
        level = str(w["SCG warning level (1 to 3 or Noise)"])

        if include_noise_warnings or not level == "N":
            warning_rectangles.add(Rectangle.from_circuit_warning(circuit, i, rectangle_width=rectangle_width))
    return ClusterEnsemble(Cluster({r}) for r in warning_rectangles)

def clusterize_Monte_Carlo(circuit, choices_div=100, found_div=50, loc_rect_size=32, time_rect_size=np.timedelta64(6, 'D'), name="Monte Carlo"):
    """
    Randomized Monte Carlo algorithm.
    Uses random PD choices to determine Clusters. Number of PDs is influenced by circuit length, total recorded circuit time and choices_div.
    Each random PD gets its own Rectangle of size loc_rect_size * time_rect_size.
    Takes the __or__ (|) over all Rectangles and looks at how many PDs are associated with the result.
    If there are enough (influenced by circuit length, total recorded circuit time and choices_div*found_div), the Cluster is accepted as valid

    Increasing choices_div makes the algorithm stricter
    Increasing found_div makes the algorithm more lenient

    :param circuit: The circuit to clusterize
    :type circuit: class:`clusterizer.circuit.MergedCircuit`

    :param choices_div: Magic factor that determines how many random choices are made
    :type choices_div: int, optional

    :param found_div: Magic factor that determines how many random PDs need to have found a cluster before it is seen as abnormal
    :type found_div: int, optional

    :param loc_rect_size: The size of rectangles in the location dimension
    :type loc_rect_size: int, optional

    :param time_rect_size: The size of rectangles in the time dimension
    :type time_rect_size: class:`numpy.timedelta64`, optional

    :param name: The name of the algorithm
    :type name: string, optional

    :return: found clusters
    :rtype: object of class:`clusterizer.cluster.ClusterEnsemble`
    """
    locations = circuit.pd["Location in meters (m)"][circuit.pd_occured]
    times = circuit.pd["Date/time (UTC)"][circuit.pd_occured]

    time_factor = (times[times.index[-1]] - times[times.index[0]]) / np.timedelta64(30, 'D')
    num = int(circuit.circuitlength * time_factor)
    chosen_pds = np.random.randint(len(locations), size=num // choices_div)
    chosen_locations = locations.iloc[chosen_pds].values
    chosen_times = times.iloc[chosen_pds].values

    xlength = loc_rect_size // 2
    ylength = time_rect_size // 2
    rectangles = set()
    for i, point in enumerate(zip(chosen_locations, chosen_times)):
        rectangles.add(Rectangle(
            location_range=(point[0]-xlength/2, point[0]+xlength/2),\
            time_range=(point[1]-ylength/2, point[1]+ylength/2),\
            found_by={str(i)}))

    clusters = {ClusterEnsemble({Cluster({r})}) for r in rectangles}
    reduced = functools.reduce(operator.__or__, clusters)
    highly_found = [x for x in reduced if len(x.found_by) > num // (choices_div*found_div)]
    result = set()
    for c in highly_found:
        for r in c:
            r.found_by = {name}
    return highly_found
