import pandas as pd
import numpy as np
from skimage import measure
from constants import SCALE_MICRONS, SCALE_MICRONS_Z


class ProcessSamples:
    def __init__(self, file):
        self.file = file

    def process(self, edges, connected, scale, scale_factor):
        """Applies selected processing steps to samples."""

        # Load samples file.
        samples = pd.read_csv(self.file)

        if edges:
            print("Removing edge cells ...")
            samples = self.remove_edge_cells(samples)

        if connected:
            print("Removing unconnected regions ...")
            samples = self.remove_unconnected_regions(samples)

        if scale:
            print("Scaling coordinates ...")
            samples = self.scale_coordinates(samples, scale_factor)

        # Save processed samples.
        samples.to_csv(self.file.replace(".csv", ".PROCESSED.csv"), index=False)

    @staticmethod
    def remove_edge_cells(df):
        """Removes cells at edges of FOV."""

        # Get ids of cell at edge.
        x_edge_ids = ProcessSamples._find_edge_ids("x", df)
        y_edge_ids = ProcessSamples._find_edge_ids("y", df)
        all_edge_ids = set(x_edge_ids + y_edge_ids)

        # Filter df for cells not at edge.
        df_filtered = df[~df["id"].isin(all_edge_ids)]

        # TODO: update method for hexagonal grid

        return df_filtered

    @staticmethod
    def remove_unconnected_regions(df):
        """Removes unconnected regions of cells."""

        arr, steps, offsets = ProcessSamples._convert_to_integer_array(df)
        arr_connected = np.zeros(arr.shape, dtype="int")
        labels = measure.label(arr)

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
                arr_connected[labels == index] = cell_id
                ids_added.add(cell_id)
            else:
                print(f"Skipping unconnected region for cell id {cell_id}")

        # Convert back to dataframe.
        df_connected = ProcessSamples._convert_to_dataframe(arr_connected, steps, offsets)

        # TODO: update method for hexagonal grid

        return df_connected

    @staticmethod
    def scale_coordinates(df, scale_factor):
        df.x = df.x * SCALE_MICRONS * scale_factor
        df.y = df.y * SCALE_MICRONS * scale_factor
        df.z = df.z * SCALE_MICRONS_Z * scale_factor
        return df

    @staticmethod
    def _find_edge_ids(axis, df):
        """Finds ids of cells at edges of given axis."""

        # Get min and max coordinate for given axis.
        axis_min = df[axis].min()
        axis_max = df[axis].max()

        # Get min and max coordinate for each cell.
        df_mins = df.groupby("id")[axis].min()
        df_maxs = df.groupby("id")[axis].max()

        # Check for cell ids located at edges.
        edge_ids = []
        edge_ids = edge_ids + df_mins[df_mins <= axis_min].index.tolist()
        edge_ids = edge_ids + df_maxs[df_maxs >= axis_max].index.tolist()

        return list(set(edge_ids))

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
                step_x * (x + offset_x),
                step_y * (y + offset_y),
                step_z * (z + offset_z),
                arr[z, y, x],
            )
            for z, y, x in zip(*np.where(arr != 0))
        ]

        return pd.DataFrame(voxels, columns=["x", "y", "z", "id"])

    @staticmethod
    def _get_step_size(arr):
        """Gets step size between array entries."""

        # Get steps between subsequent unique entries in array.
        unique = sorted(arr.unique())
        steps = [j - i for i, j in zip(unique[:-1], unique[1:])]
        step = set(steps)

        # Check that array only has one step size.
        assert len(step) == 1, "variable step size in array -- check scaling"

        return step.pop()
