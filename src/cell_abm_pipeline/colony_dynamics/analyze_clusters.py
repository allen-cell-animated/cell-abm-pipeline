import numpy as np
import pandas as pd
from scipy.spatial import distance

from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_dataframe
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class AnalyzeClusters:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "NEIGHBORS", False),
            "output": make_folder_key(context.name, "analysis", "CLUSTERS", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["NEIGHBORS", "csv", "xz"], "", "%04d"),
            "output": make_file_key(context.name, ["CLUSTERS", "csv"], "%s", ""),
        }

    def run(self):
        self.analyze_clusters()

    def analyze_clusters(self):
        all_data = []

        for seed in self.context.seeds:
            file_key = make_full_key(self.folders, self.files, "input", seed)
            data = load_dataframe(self.context.working, file_key)

            if data.KEY.isnull().values.any():
                data.KEY = ""

            data = data[data.KEY.isin(self.context.keys)]
            all_data.append(data)

        for key, key_group in pd.concat(all_data).groupby("KEY"):
            output_key = make_full_key(self.folders, self.files, "output", key)
            output_df = self.calculate_cluster_metrics(key_group)
            save_dataframe(self.context.working, output_key, output_df, index=False)

    @staticmethod
    def calculate_cluster_metrics(clusters):
        all_metrics = []

        for (seed, tick), cluster in clusters.groupby(["SEED", "TICK"]):
            groups = cluster.groupby("GROUP")["ID"].unique()
            cluster_sizes, single_sizes = AnalyzeClusters.get_cluster_sizes(groups)
            centroids = AnalyzeClusters.make_centroid_dict(cluster)
            inter_mean, inter_std = AnalyzeClusters.get_inter_cluster_distances(groups, centroids)
            intra_mean, intra_std = AnalyzeClusters.get_intra_cluster_distances(groups, centroids)

            metrics = {
                "SEED": seed,
                "TICK": tick,
                "NUM_CLUSTERS": len(cluster_sizes),
                "NUM_SINGLES": len(single_sizes),
                "CLUSTER_SIZE_TOTAL": cluster_sizes.sum(),
                "CLUSTER_SIZE_MEAN": cluster_sizes.mean(),
                "CLUSTER_SIZE_STD": cluster_sizes.std(ddof=1),
                "INTER_DISTANCE_MEAN": inter_mean,
                "INTER_DISTANCE_STD": inter_std,
                "INTRA_DISTANCE_MEAN": intra_mean,
                "INTRA_DISTANCE_STD": intra_std,
            }

            all_metrics.append(metrics)

        return pd.DataFrame(all_metrics)

    @staticmethod
    def make_centroid_dict(df):
        """Create dictionary of id to (x, y, z) centroids."""
        entries = df[["ID", "CX", "CY", "CZ"]].to_records(index=False)
        return {i: [x, y, z] for i, x, y, z in entries}

    @staticmethod
    def get_cluster_sizes(groups):
        """Calculate sizes of clusters and singles for list of groups."""
        sizes = groups.map(len)
        cluster_sizes = sizes[sizes != 1]
        single_sizes = sizes[sizes == 1]
        return cluster_sizes, single_sizes

    @staticmethod
    def get_cluster_centroid(ids, centroid_dict):
        """Gets centroid of cluster by averaging across centroids."""
        centroids = [centroid_dict[i] for i in ids]
        x, y, z = np.mean(np.array(centroids), axis=0)
        return x, y, z

    @staticmethod
    def get_inter_cluster_distances(groups, centroid_dict):
        """Calculate distances between clusters."""
        cluster_centroids = [
            AnalyzeClusters.get_cluster_centroid(g, centroid_dict) for g in groups if len(g) != 1
        ]
        cluster_centroids = np.array(cluster_centroids)

        if cluster_centroids.shape[0] < 2:
            inter_distance_mean = np.nan
            inter_distance_std = np.nan
        else:
            inter_distances = distance.cdist(cluster_centroids, cluster_centroids, "euclidean")
            distances = np.ndarray.flatten(inter_distances)
            distances = np.delete(distances, range(0, len(distances), len(inter_distances) + 1), 0)

            inter_distance_mean = np.mean(distances)
            inter_distance_std = np.std(distances, ddof=1)

        return inter_distance_mean, inter_distance_std

    @staticmethod
    def get_intra_cluster_distances(groups, centroid_dict):
        """Calculate distances within clusters."""
        intra_distance_means = []
        intra_distance_stds = []

        for group in groups:
            if len(group) < 2:
                continue

            cluster_centroids = [centroid_dict[g] for g in group]
            cluster_centroids = np.array(cluster_centroids)

            intra_distances = distance.cdist(cluster_centroids, cluster_centroids, "euclidean")
            distances = np.ndarray.flatten(intra_distances)
            distances = np.delete(distances, range(0, len(distances), len(intra_distances) + 1), 0)

            intra_distance_means.append(np.mean(distances))
            intra_distance_stds.append(np.std(distances, ddof=1))

        return intra_distance_means, intra_distance_stds
