import pandas as pd
import numpy as np
from skimage import measure
from sample_images import SampleImages
from constants import SCALE_MICRONS, SCALE_MICRONS_Z, OUTPUT_COLUMNS


class ProcessSamples:
    def __init__(self, file):
        self.file = file

    def process(
        self, edges, connected, scale, edge_threshold, scale_factor, grid_type, contact
    ):
        """Applies selected processing steps to samples."""

        # Load samples file.
        samples = pd.read_csv(self.file)

        if edges:
            print("Removing edge cells ...")
            samples = self.remove_edge_cells(samples, grid_type, edge_threshold)

        if connected:
            print("Removing unconnected regions ...")
            samples = self.remove_unconnected_regions(samples)

        if scale:
            print("Scaling coordinates ...")
            samples = self.scale_coordinates(samples, scale_factor)

        if contact:
            print("Saving contact sheet ...")
            contact_path = self.file.replace(
                f"{grid_type}_samples_", "contact_"
            ).replace(".csv", ".PROCESSED.png")
            SampleImages.save_contact_sheet(samples, contact_path)

        # Save processed samples.
        samples.to_csv(self.file.replace(".csv", ".PROCESSED.csv"), index=False)

    @staticmethod
    def remove_edge_cells(df, grid_type, edge_threshold):
        """Removes cells at edges of FOV."""

        # Get edge padding.
        x_padding = ProcessSamples._get_step_size(df.x) if grid_type == "hex" else 0
        y_padding = ProcessSamples._get_step_size(df.y) if grid_type == "hex" else 0

        # Get ids of cell at edge.
        x_edge_ids = ProcessSamples._find_edge_ids("x", df, x_padding, edge_threshold)
        y_edge_ids = ProcessSamples._find_edge_ids("y", df, y_padding, edge_threshold)

        # Filter df for cells not at edge.
        all_edge_ids = set(x_edge_ids + y_edge_ids)
        df_filtered = df[~df["id"].isin(all_edge_ids)]

        return df_filtered

    @staticmethod
    def remove_unconnected_regions(df):
        """Removes unconnected regions of cells."""

        arr, steps, offsets = ProcessSamples._convert_to_integer_array(df)
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
        df_conn = ProcessSamples._convert_to_dataframe(arr_conn, steps, offsets)

        # TODO: update method for hexagonal grid

        return df_conn

    @staticmethod
    def scale_coordinates(df, scale_factor):
        df.x = df.x * SCALE_MICRONS * scale_factor
        df.y = df.y * SCALE_MICRONS * scale_factor
        df.z = df.z * SCALE_MICRONS_Z * scale_factor
        return df

    @staticmethod
    def _find_edge_ids(axis, df, padding, threshold):
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
    def _convert_to_integer_array(df):
        """Converts dataframe of voxels into integer array."""

        # Get step size for voxels.
        step_x = ProcessSamples._get_step_size(df.x)
        step_y = ProcessSamples._get_step_size(df.y)
        step_z = ProcessSamples._get_step_size(df.z)

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
    def _convert_to_dataframe(arr, steps, offsets):
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

        return pd.DataFrame(voxels, columns=OUTPUT_COLUMNS)

    @staticmethod
    def _get_step_size(arr):
        """Gets step size between array entries."""

        # Get steps between subsequent unique entries in array.
        unique = sorted(arr.unique())
        steps = [j - i for i, j in zip(unique[:-1], unique[1:])]
        step = set(steps)

        if len(step) == 1:
            print("WARNING: variable step size in array")

        return max(step)
