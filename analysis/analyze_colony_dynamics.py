import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sklearn as skl
# import sklearn.datasets
import sklearn.cluster
import pandas
from sort import *


'''
Cluster using DBSCAN
'''
def dbscan(locs, eps, min_samples=2):

    return skl.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(locs)

'''
Get the number of discrete clusters and noise
'''
def count_clusters(clustering):

    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    return n_clusters, n_noise

'''
Get the center position of each cluster
'''
def get_cluster_centroids(locs, clustering):

    labels = clustering.labels_
    result = []
    for i in range(count_clusters(clustering)[0]):
        result.append([np.mean(locs[(labels == i)][:, 0]),
                       np.mean(locs[(labels == i)][:, 1])])

    return np.array(result)

'''
Get the bounding box for each cluster
'''
def get_cluster_bboxes(locs, clustering, size_multiplier):

    min_dimensions = 200.
    labels = clustering.labels_
    result = []
    for i in range(count_clusters(clustering)[0]):

        centroid = [np.mean(locs[(labels == i)][:, 0]),
                    np.mean(locs[(labels == i)][:, 1])]

        width = max(min_dimensions, size_multiplier * (locs[(labels == i)][:, 0].max()
                    - locs[(labels == i)][:, 0].min()))
        height = max(min_dimensions, size_multiplier * (locs[(labels == i)][:, 1].max()
                     - locs[(labels == i)][:, 1].min()))

        result.append([
            max(centroid[0] - width / 2., 0),
            max(centroid[1] - height / 2., 0),
            min(centroid[0] + width / 2., 2500),
            min(centroid[1] + height / 2., 2500)
        ])

    return np.array(result)

'''
Track clusters over time
'''
def track_clusters(cluster_bboxes):

    detection_score = 40.0
    bboxes = cluster_bboxes

    total_time = 0.0
    total_frames = 0
    result = {}

    cell_tracker = Sort() #create instance of the SORT tracker
    for frame in range(len(bboxes)):

        detections = []
        for i in range(len(bboxes[frame])):
            d = bboxes[frame][i].tolist()
            d.append(detection_score)
            detections.append(d)

        total_frames += 1

        start_time = time.time()
        trackers = cell_tracker.update(np.array(detections))
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:

            cluster_id = d[4]
            if cluster_id not in result:
                result[cluster_id] = {}

            result[cluster_id][frame] = [d[0]+(d[2]-d[0])/2., d[1]+(d[3]-d[1])/2.]

    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(
        total_time,total_frames,total_frames/total_time))

    return result

'''
format time for filenames
'''
def get_time_string(time):

    result = str(time)
    while len(result) < 8:
        result = "0" + result
    return result

'''
Plot clusters with coloring, from https://tinyurl.com/s6zj6w9
'''
def plot_clusters(locs, clustering, time):

    labels = clustering.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
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

    plt.savefig("output/cluster{}.png".format(get_time_string(time)), dpi=220)
    plt.close()

'''
Plot centroids for each cluster
'''
def plot_centroids(centroids, time):

    fig, ax = plt.subplots(1,1)
    ax.set(xlim=(0,2500),
           ylim=(0,2500),
           aspect=1,
           title="")
    ax.plot(centroids[:, 0], centroids[:, 1], '*',
        markerfacecolor=tuple([0, 0, 0, 1]),
        markeredgecolor='k',
        markersize=10)

    plt.savefig("output/centroid{}.png".format(get_time_string(time)), dpi=220)
    plt.close()

'''
Plot bounding boxes for each cluster
'''
def plot_bboxes(bboxes, time):

    fig, ax = plt.subplots(1,1)
    ax.set(xlim=(0,2500),
           ylim=(0,2500),
           aspect=1,
           title="")

    for bbox in bboxes:
        ax.add_patch(patches.Rectangle(
            (bbox[0],bbox[1]),
            bbox[2]-bbox[0],bbox[3]-bbox[1],
            linewidth=1,edgecolor='k',facecolor='none'))

    plt.savefig("output/bbox{}.png".format(get_time_string(time)), dpi=220)
    plt.close()

'''
Plot tracks for each cluster
'''
def plot_tracks(tracks, total_frames):

    track_length = 70

    for time in range(total_frames):

        positions = []
        for cluster in tracks:
            p = []
            if time in tracks[cluster]:
                for i in range(time, time - track_length, -1):
                    if i in tracks[cluster]:
                        p.append(tracks[cluster][i])
                    else:
                        break
            if len(p) > 1:
                positions.append(np.array(p))
        # positions = np.array(positions)

        fig, ax = plt.subplots(1,1)
        ax.set(xlim=(0,2500),
               ylim=(0,2500),
               aspect=1,
               title="")

        for cluster in range(len(positions)):
            plt.plot(positions[cluster][:,0], positions[cluster][:,1], color='k')

        plt.savefig("output/track{}.png".format(get_time_string(time)), dpi=220)
        plt.close()


# -----------------------------------------------------------------------------

if not os.path.exists("output/"):
    os.makedirs("output/")

data = pandas.read_csv("../sample_data/cell_centroids.csv")

frames = data[['time']].max().values[0]
cluster_bboxes = []
for t in range(frames):

    locs = data.loc[data['time'] == t][['x', 'y']].values

    clustering = dbscan(locs, 80)
    centroids = get_cluster_centroids(locs, clustering)
    cluster_bboxes.append(get_cluster_bboxes(locs, clustering, 2.))

    cluster_count = count_clusters(clustering)
    print("{} : {} clusters, {} singles".format(
        t, cluster_count[0], cluster_count[1]))

    plot_clusters(locs, clustering, t)
    plot_centroids(centroids, t)
    plot_bboxes(cluster_bboxes[t], t)

tracks = track_clusters(cluster_bboxes)
plot_tracks(tracks, frames)
