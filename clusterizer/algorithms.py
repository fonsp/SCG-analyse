import numpy as np
import scipy.stats

from clusterizer.cluster import Cluster


def clusterize_poisson(circuit, bin_size=4, weigh_charges=False, nominal_circuit_fraction=.80, certainty=.95, min_cluster_size=3, max_skipped_bins=2):
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
    bin_contents, _ = np.histogram(locations, bins=bins, weights=charges if weigh_charges else None)

    # %% Find the 80% quantile: find the (lowest) value M such that at least 80% of bins have a content <= M.
    nominal_pd_quantile_level = np.sort(bin_contents)[int(nominal_circuit_fraction * len(bin_contents))]

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

    return clusters, bins, bin_contents, rate


def cluster_boolean_series(series, max_consecutive_false=5, min_length=5, min_count=0):
    """
    Imperative algorithm to identify sequences of mostly True values, under the conditions imposed by the parameters:

    Parameters
    ----------

    series : array_like
        Sequence of Booleans to cluster

    max_consecutive_false : int
        Maximum number of consecutive Falsey values to accept inside a cluster

    min_length : int
        Minimum cluster length (right bound - left bound)

    min_count : int
        Minimum cluster size (number of Truthy values inside cluster bounds)
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

def clusterize_density(file,probability=0.01,placeint=10,timeint=7*24*60):
    """Algorithm finding two-dimensional clusters based on differences in density. It uses the following parameters:

    :param file: File containing all partial discharges of a certain cable (the data).
    :type file: file

    :param probability: All boxes 
    :type probability: float

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
        (float) the rate parameter of the fitted Poisson model;
    :rtype: tuple
    """
    #met data spelen

    data=open_file(file)
    maxplace=max(data[1])
    maxtime=max(data[0])
    maxcharge=max(data[2])
    boxesx=int(maxplace/placeint)
    boxesy=int(maxtime/timeint)
    grid=np.zeros((boxesx,boxesy))
    sstep=maxplace/boxesx
    tstep=maxtime/boxesy
    datalength=len(data[0])

    #gewoon scatterplot

    plt.figure("plaatjen")
    plt.axis([0,maxplace,0,maxtime])
    plt.xlabel("Plaats")
    plt.ylabel("Tijd")
    plt.scatter(data[1],data[0],marker="o",s=1,c="black")

    #data discretiseren

    for i in range(datalength):
        grid[get_box_x(i,boxesx,maxplace,data)][get_box_y(i,boxesy,maxtime,data)]+=1

    flatgrid=grid.flatten()
    maxdense=max(flatgrid)
    
    #kansverdelingen bepalen
    
    gridlength=len(flatgrid)
    graphrange=np.arange(0,maxdense)
    probabilities=np.zeros(maxdense)
    for i in graphrange:
        probabilities[i]=len(flatgrid[flatgrid>=i])/gridlength
    if maxdense<100:
        sensitivity=10000
    else:
        sensitivity=len(probabilities[probabilities>probability])

    #data groeperen

    groups=group_boxes(grid,sensitivity,boxesx,boxesy)

    colorcycle=["red","green","blue","yellow","orange","purple","pink","cyan","steelblue"]
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            plt.gca().add_patch(Rectangle((groups[i][j][0]*sstep,groups[i][j][1]*tstep),sstep,tstep,linewidth=1,edgecolor=colorcycle[i%len(colorcycle)],facecolor=colorcycle[i%len(colorcycle)],alpha=0.5))

    plt.show()
