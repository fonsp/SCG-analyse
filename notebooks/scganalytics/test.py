# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:17:57 2019

@author: matth
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

#import plotly.offline as py
#import plotly.graph_objs as go

from scipy import stats
from sklearn.cluster import DBSCAN 
from sklearn.neighbors import KernelDensity
import sys
sys.path.append('../..')
import clusterizer
import globals, circuit as cc
circuitje = load(4099)
clusters2 = clusterizer.algorithms.clusterize_pinta(circuitje,timeinterval = np.timedelta64(1,'D'))
clusterizer.plot.draw_location_time_scatter(circuitje)
clusterizer.plot.overlay_cluster_ensemble(clusters2)