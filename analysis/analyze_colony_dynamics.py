import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.datasets
import sklearn.cluster
import pandas


def dbscan(locs, eps, min_samples=2):
    """Cluster using DBSCAN"""
    return skl.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(locs)

def cluster_count(clustering):
    """Try and determine the number of discrete clusters"""
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return n_clusters, n_noise

def plot_clusters(locs, cluster, ax=None):
    """Plot clusters with coloring, from https://tinyurl.com/s6zj6w9 """
    labels = cluster.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[cluster.core_sample_indices_] = True
    if ax is None:
        fig, ax = plt.subplots(1,1)
        ax.set(xlim=(0,2500),
               ylim=(0,2500),
               aspect=1,
               title="")
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in
               np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 0]
        class_member_mask = (labels == k)
        xy = locs[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o',
                markerfacecolor=tuple(col),
                markeredgecolor='k',
                markersize=5)
        xy = locs[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o',
                markerfacecolor=tuple(col),
                markeredgecolor='k',
                markersize=5)
    return ax

if not os.path.exists("output/"):
    os.makedirs("output/")

data = pandas.read_csv("../sample_data/cell_centroids.csv")

for t in range(data[['time']].max().values[0]):

    locs = data.loc[data['time'] == t][['x', 'y']].values

    clustering = dbscan(locs, 100)
    plot_clusters(locs, clustering)
    clusters = cluster_count(clustering)
    print("{} : {} clusters, {} singles".format(t, clusters[0], clusters[1]))

    t_string = str(t)
    while len(t_string) < 8:
        t_string = "0" + t_string

    plt.savefig("output/cluster{}.png".format(t_string), dpi=220)
    plt.close()
