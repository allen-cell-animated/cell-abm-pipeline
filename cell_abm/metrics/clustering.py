#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Supporting the calculation of clustering metrics on runs"""

import numpy as np
import sklearn.cluster
import sklearn.preprocessing


def cluster_via_dbscan(locs, eps, min_samples=2):
    """Cluster using DBSCAN"""
    clustering = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(locs)
    return clustering


def cluster_via_single_link(locs, dist_thresh=90):
    """Cluster using single link agglomeration"""
    clusterer = sklearn.cluster.AgglomerativeClustering(
        n_clusters=None, linkage="single", distance_threshold=dist_thresh
    )
    clustering = clusterer.fit(locs)
    return clustering


def cluster_count(clustering):
    """The number of discrete clusters"""
    labels = clustering.labels_
    n_clusters = len(set(labels))
    if -1 in labels:
        n_noise = list(labels).count(-1)
    else:
        n_noise = sum([list(labels).count(l) == 1 for l in np.unique(labels)])
    n_clusters -= n_noise
    return n_clusters, n_noise


def cluster_sizes(clustering):
    """Sizes of clusters, individuals in each"""
    labels = clustering.labels_
    labels = labels[labels != -1]  # noise points aren't in cluster
    counts = np.unique(labels, return_counts=True)[1]
    return counts


def cluster_fractions(clustering):
    """Mean fraction of cells in any given cluster"""
    sizes = cluster_sizes(clustering)
    total = clustering.labels_.size
    return sizes / total