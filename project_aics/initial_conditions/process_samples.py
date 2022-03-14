import pandas as pd
import numpy as np
from skimage import measure
from scipy.spatial import distance
import matplotlib.pyplot as plt

from project_aics.initial_conditions.sample_images import SampleImages
from project_aics.initial_conditions.__config__ import (
    EDGE_THRESHOLD,
    CONNECTED_THRESHOLD,
    SCALE_MICRONS_XY,
    SCALE_MICRONS_Z,
)
from project_aics.utilities.load import load_dataframe
from project_aics.utilities.save import save_dataframe, save_image
from project_aics.utilities.keys import make_folder_key, make_file_key
from project_aics.utilities.plot import make_plot


class ProcessSamples:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "sample": make_folder_key(context.name, "samples", "RAW", False),
            "processed": make_folder_key(context.name, "samples", "PROCESSED", False),
            "contact": make_folder_key(context.name, "plots", "SAMPLE", False),
        }
        self.files = {
            "sample": make_file_key(context.name, ["RAW", "csv"], "%s", ""),
            "processed": make_file_key(context.name, ["PROCESSED", "csv"], "%s", ""),
            "contact": make_file_key(context.name, ["SAMPLE", "png"], "%s", ""),
        }
        # TODO: update contact sheet paths

    def run(self, grid, scale=None, edges=True, connected=True, contact=True):
        for key in self.context.keys:
            processed_df, samples_df = self.process_samples(key, grid, scale, edges, connected)

            if contact:
                self.plot_contact_sheet(key, {"processed": processed_df, "samples": samples_df})

    def process_samples(self, key, grid, scale, edges, connected):
        sample_key = self.folders["sample"] + self.files["sample"] % (key)
        samples_df = load_dataframe(self.context.working, sample_key)
        processed_df = samples_df.copy()

        if edges:
            print("Removing edge cells ...")
            processed_df = self.remove_edge_cells(processed_df, grid)

        if connected:
            print("Removing unconnected regions ...")
            processed_df = self.remove_unconnected_regions(processed_df, grid)

        if scale != None:
            print("Scaling coordinates ...")
            processed_df = self.scale_coordinates(processed_df, scale)

        processed_key = self.folders["processed"] + self.files["processed"] % (key)
        save_dataframe(self.context.working, processed_key, processed_df, index=False)

        return processed_df, samples_df

    def plot_contact_sheet(self, key, data):
        """Save contact sheet image for processed samples."""
        if len(data) == 0:
            return

        make_plot(sorted(data["samples"].z.unique()), data, self._plot_contact_sheet, size=3)

        plt.gca().invert_yaxis()
        plot_key = self.folders["contact"] + self.files["contact"] % key
        save_image(self.context.working, plot_key)
        plt.show()

    @staticmethod
    def _plot_contact_sheet(ax, data, key):
        samples = data["samples"]
        processed = data["processed"]
        filter = pd.merge(samples, processed, how="outer", indicator=True)
        removed = filter[filter._merge == "left_only"]

        z_slice = processed[processed.z == key]
        z_removed = removed[removed.z == key]

        max_id = int(samples.id.max())
        min_id = int(samples.id.min())

        ax.scatter(z_slice.x, z_slice.y, c=z_slice.id, vmin=min_id, vmax=max_id, s=1, cmap="jet")
        ax.scatter(z_removed.x, z_removed.y, s=0.5, c="#ccc")

    @staticmethod
    def scale_coordinates(df, scale_factor):
        df["x_scaled"] = df.x * SCALE_MICRONS_XY * scale_factor
        df["y_scaled"] = df.y * SCALE_MICRONS_XY * scale_factor
        df["z_scaled"] = df.z * SCALE_MICRONS_Z * scale_factor
        return df

    @staticmethod
    def remove_edge_cells(df, grid="rect", edge_threshold=EDGE_THRESHOLD):
        """Removes cells at edges of FOV."""

        # Get edge padding.
        x_padding = ProcessSamples.get_step_size(df.x) if grid == "hex" else 0
        y_padding = ProcessSamples.get_step_size(df.y) if grid == "hex" else 0

        # Get ids of cell at edge.
        x_edge_ids = ProcessSamples.find_edge_ids("x", df, x_padding, edge_threshold)
        y_edge_ids = ProcessSamples.find_edge_ids("y", df, y_padding, edge_threshold)

        # Filter df for cells not at edge.
        all_edge_ids = set(x_edge_ids + y_edge_ids)
        df_filtered = df[~df["id"].isin(all_edge_ids)]

        return df_filtered

    @staticmethod
    def remove_unconnected_regions(df, grid, connected_threshold=CONNECTED_THRESHOLD):
        """Removes unconnected regions of cells."""
        if grid == "rect":
            return ProcessSamples.remove_unconnected_by_connectivity(df)
        elif grid == "hex":
            return ProcessSamples.remove_unconnected_by_distance(df, connected_threshold)
        else:
            raise ValueError(f"invalid grid type {grid}")

    @staticmethod
    def remove_unconnected_by_connectivity(df):
        """Removes unconnected regions based on connectivity."""

        arr, steps, offsets = ProcessSamples.convert_to_integer_array(df)
        arr_conn = np.zeros(arr.shape, dtype="int")
        labels = measure.label(arr, connectivity=1)

        # Sort labeled regions by size.
        regions = np.bincount(labels.flatten())[1:]
        regions_sorted = sorted(
            [(i + 1, n) for i, n in enumerate(regions)],
            key=lambda tup: tup[1],
            reverse=True,
        )

        # Iterate through all regions and copy largest connected region to array.
        ids_added = set()
        for index, count in regions_sorted:
            cell_id = list(set(arr[labels == index]))[0]

            if cell_id not in ids_added:
                arr_conn[labels == index] = cell_id
                ids_added.add(cell_id)
            else:
                print(f"Skipping unconnected region for cell id {cell_id}")

        # Convert back to dataframe.
        df_connected = ProcessSamples.convert_to_dataframe(arr_conn, steps, offsets)

        return df_connected

    @staticmethod
    def remove_unconnected_by_distance(df, threshold):
        """Removes unconnected regions based on distance."""
        all_connected = []

        # Rescale coordinates to um.
        df["xs"] = df.x * SCALE_MICRONS_XY
        df["ys"] = df.y * SCALE_MICRONS_XY
        df["zs"] = df.z * SCALE_MICRONS_Z

        # Iterate through each id and filter out samples above the distance threshold.
        for name, group in df.groupby("id"):
            coords = group[["x", "y", "z", "xs", "ys", "zs"]].to_numpy()
            dists = [
                [ProcessSamples.get_minimum_distance(c[3:], coords[:, 3:]), *c[:3]] for c in coords
            ]
            connected = [[name, x, y, z] for d, x, y, z in dists if d < threshold]
            all_connected = all_connected + connected

        if len(all_connected) == 0:
            print("WARNING: no connected samples, try increasing connected threshold")
            return pd.DataFrame()

        # Convert back to dataframe.
        df_connected = pd.DataFrame(all_connected, columns=["id", "x", "y", "z"], dtype=int)

        return df_connected

    @staticmethod
    def get_step_size(arr):
        """Gets step size between array entries."""

        # Get steps between subsequent unique entries in array.
        unique = sorted(arr.unique())
        steps = [j - i for i, j in zip(unique[:-1], unique[1:])]
        step = set(steps)

        if len(step) > 1:
            print("WARNING: variable step size in array")

        return max(step)

    @staticmethod
    def find_edge_ids(axis, df, padding, threshold):
        """Finds ids of cells at edges of given axis."""

        # Get min and max coordinate for given axis.
        axis_min = df[axis].min() + padding
        axis_max = df[axis].max() - padding

        # Get min and max coordinate for each cell.
        df_mins = df.groupby("id")[axis].min()
        df_maxs = df.groupby("id")[axis].max()

        # Check for cell ids located at edges.
        edges = df.groupby("id").apply(
            lambda g: len(g[(g[axis] <= axis_min) | (g[axis] >= axis_max)])
        )
        edge_ids = edges[edges > threshold]

        return list(edge_ids.index)

    @staticmethod
    def convert_to_integer_array(df):
        """Converts dataframe of voxels into integer array."""

        # Get step size for voxels.
        step_x = ProcessSamples.get_step_size(df.x)
        step_y = ProcessSamples.get_step_size(df.y)
        step_z = ProcessSamples.get_step_size(df.z)

        # Rescale integers to step size 1.
        scaled_x = df.x.divide(step_x).astype("int32").values
        scaled_y = df.y.divide(step_y).astype("int32").values
        scaled_z = df.z.divide(step_z).astype("int32").values

        # Create integer array.
        offset_x = min(scaled_x)
        offset_y = min(scaled_y)
        offset_z = min(scaled_z)
        length = max(scaled_x) - offset_x + 1
        width = max(scaled_y) - offset_y + 1
        height = max(scaled_z) - offset_z + 1
        arr = np.zeros((height, width, length), dtype="int32")

        # Populate array.
        for x, y, z, i in zip(scaled_x, scaled_y, scaled_z, df.id):
            arr[z - offset_z, y - offset_y, x - offset_x] = i

        return arr, [step_x, step_y, step_z], [offset_x, offset_y, offset_z]

    @staticmethod
    def convert_to_dataframe(arr, steps, offsets):
        step_x, step_y, step_z = steps
        offset_x, offset_y, offset_z = offsets

        voxels = [
            (
                arr[z, y, x],
                step_x * (x + offset_x),
                step_y * (y + offset_y),
                step_z * (z + offset_z),
            )
            for z, y, x in zip(*np.where(arr != 0))
        ]

        return pd.DataFrame(voxels, columns=["id", "x", "y", "z"])

    @staticmethod
    def get_minimum_distance(point, points):
        """Get minimum distance of point to array of points."""
        dists = distance.cdist([point], points)
        return np.min(dists[dists != 0])
