#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Supporting the calculation of hopkin's statistic on runs"""

import numpy as np
import scipy
import sklearn.preprocessing

def hopkins(locs, subsample_n):
    """Create the hopkins statistic by sampling points out of locs,
    you'll want to bootstrap this"""
    n, dims = locs.shape
    assert subsample_n < n, "Sample fewer points than exist in locs"
    X = sklearn.preprocessing.MinMaxScaler().fit_transform(locs)
    Y = np.random.rand((subsample_n, dims))
    rand_dist_sum = scipy.spatial.distance.cdist(Y, X).min(1).sum()
    X_samp = X[np.random.choice(range(n), size=subsample_n, replace=False), :]
    X_dists = scipy.spatial.distance.cdist(X_samp, X)
    X_dists[X_dists == 0.0] = X_dists.max()  # eliminate self as nearest neighbor
    X_dist_sum = X_dists.min(1).sum()
    return rand_dist_sum / (rand_dist_sum + X_dist_sum)