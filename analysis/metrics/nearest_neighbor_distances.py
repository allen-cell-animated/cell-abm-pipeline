#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Supporting the calculation of nearest neighbor distances on runs"""

import numpy as np
import scipy
import sklearn.preprocessing


def nearest_neighbor_distances(locs):
    """Distance to nearest neighbor for each location"""
    dist_grid = scipy.spatial.distance.cdist(locs, locs)
    masked_grid = np.ma.array(dist_grid, mask=np.eye(len(locs)))
    return np.min(masked_grid, axis=1).data
