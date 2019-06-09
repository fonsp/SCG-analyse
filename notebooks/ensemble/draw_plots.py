import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import sys
sys.path.append('..')
import clusterizer

plt.rcParams['figure.figsize'] = [8,5]

def split(ensemble):
    poisson = [c for c in ensemble.get_clusters() if c.found_by == {"Poisson 2D"}]
    DBSCAN = [c for c in ensemble.get_clusters() if c.found_by == {"DBSCAN"}]
    both = [c for c in ensemble.get_clusters() if len(c.found_by) > 1]
    return poisson, DBSCAN, both
    
for circuitnr in [2063]:
#for circuitnr in clusterizer.globals.available_circuits:
    circuit = clusterizer.circuit.MergedCircuit(circuitnr)
    circuit.build()
    
    clusters_poisson = clusterizer.algorithms.clusterize_poisson(circuit)
    clusters_DBSCAN = clusterizer.algorithms.clusterize_DBSCAN(circuit)
    
    ensemble_poisson = clusterizer.ensemble.ClusterEnsemble.from_iterable(clusters_poisson)
    ensemble_DBSCAN = clusterizer.ensemble.ClusterEnsemble.from_iterable(clusters_DBSCAN)
    
    ensemble_added = ensemble_poisson + ensemble_DBSCAN
    ensemble_orred = ensemble_poisson | ensemble_DBSCAN
    ensemble_conf = ensemble_added.most_confident()

    results = [ensemble_added, ensemble_orred, ensemble_conf]
    names = ["add", "or", "confident"]    

    for result, name in zip(results, names):
        print(name)
        only_poisson, only_DBSCAN, both = split(result)    
        fig, ax = plt.subplots()
        clusterizer.plot.draw_location_time_scatter(circuit, ax=ax)
        clusterizer.plot.overlay_cluster_collection(only_poisson, color="Red", ax=ax)
        clusterizer.plot.overlay_cluster_collection(only_DBSCAN, color="Blue", ax=ax)
        clusterizer.plot.overlay_cluster_collection(both, color="Green", ax=ax)
        fig.savefig("./plotjes/" + str(circuitnr) + "_" + name + ".pdf")
