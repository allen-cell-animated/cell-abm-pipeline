from typing import List

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_fill_holes

from cell_abm_pipeline.initial_conditions.__main__ import Context
from cell_abm_pipeline.utilities.load import load_image
from cell_abm_pipeline.utilities.save import save_image
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class CreateVoronoi:
    """
    Task to create Voronoi tessellation from given starting image.

    Working location structure for a given context:

    .. code-block:: bash

        (name)
        └── images
            ├── (name)_(image key 1).ome.tiff
            ├── (name)_(image key 2).ome.tiff
            ├── ...
            ├── (name)_(image key n).ome.tiff
            ├── ...
            ├── (name)_(image key 1)_(channel)_voronoi.ome.tiff
            ├── (name)_(image key 2)_(channel)_voronoi.ome.tiff
            ├── ...
            └── (name)_(image key n)_(channel)_voronoi.ome.tiff

    Voronoi images are saved to the same directory as the input images with
    appended channel value and a "voronoi" label.

    Attributes
    ----------
    context
        **Context** object defining working location and name.
    folders
        Dictionary of input and output folder keys.
    files
        Dictionary of input and output file keys.
    """

    def __init__(self, context: Context):
        self.context = context
        self.folders = {
            "image": make_folder_key(context.name, "images", "", False),
            "output": make_folder_key(context.name, "images", "", False),
        }
        self.files = {
            "image": make_file_key(context.name, ["ome", "tiff"], "%s", ""),
            "output": make_file_key(context.name, ["ome", "tiff"], "%s", "%02d_voronoi"),
        }

    def run(self, iterations: int = 10, channels: List[int] = [0]) -> None:
        """
        Runs create voronoi task for given context.

        Parameters
        ----------
        iterations
            Number of boundary estimation steps.
        channels
            Image channel indices.
        """
        for key in self.context.keys:
            for channel in channels:
                self.create_voronoi(key, iterations, channel)

    def create_voronoi(self, key: str, iterations: int, channel: int) -> None:
        """
        Create Voronoi task.

        Loads image from working location.
        Creates boundary for Voronoi using binary dilation, then performs
        the Voronoi tessellation using the selected channel of the image.

        Parameters
        ----------
        key
            Key for image.
        iterations
            Number of boundary estimation steps.
        channel
            Image channel index.
        """
        image_key = make_full_key(self.folders, self.files, "image", key)
        image = load_image(self.context.working, image_key)

        array = image.get_image_data("ZYX", T=0, C=channel)

        # Create artificial boundary for voronoi.
        mask = self.create_boundary_mask(array, iterations)
        mask_id = np.iinfo(array.dtype).max
        array[mask == 0] = mask_id

        xmin, xmax, ymin, ymax, zmin, zmax = self.get_bounded_array(mask)
        subarray = array[zmin:zmax, ymin:ymax, xmin:xmax]

        distances = distance_transform_edt(
            subarray == 0, return_distances=False, return_indices=True
        )
        distances = distances.astype("uint16", copy=False)

        coordz = distances[0].flatten()
        coordy = distances[1].flatten()
        coordx = distances[2].flatten()
        voronoi = subarray[coordz, coordy, coordx].reshape(subarray.shape)

        # Remove masking ids.
        array[zmin:zmax, ymin:ymax, xmin:xmax] = voronoi
        array[mask == 0] = 0
        array[array == mask_id] = 0

        output_key = make_full_key(self.folders, self.files, "output", (key, channel))
        save_image(self.context.working, output_key, array)

    @staticmethod
    def create_boundary_mask(array: np.ndarray, iterations: int = 10) -> np.ndarray:
        """
        Creates filled boundary mask around regions in array.

        Parameters
        ----------
        array
            Image array.
        iterations
            Number of boundary estimation steps.

        Returns
        -------
        :
            Boundary mask array.
        """
        mask = np.zeros(array.shape, dtype="uint8")
        mask[array != 0] = 1

        # Expand using binary dilation to create a border.
        binary_dilation(mask, output=mask, iterations=iterations)

        # Fill holes in the mask in each z slice.
        for z in range(array.shape[0]):
            binary_fill_holes(mask[z, :, :], output=mask[z, :, :])

        return mask

    @staticmethod
    def get_bounded_array(array):
        """Finds bounding box around binary array."""
        z, y, x = array.shape

        zbounds = np.any(array, axis=(1, 2))
        ybounds = np.any(array, axis=(0, 2))
        xbounds = np.any(array, axis=(0, 1))

        zmin, zmax = np.where(zbounds)[0][[0, -1]]
        ymin, ymax = np.where(ybounds)[0][[0, -1]]
        xmin, xmax = np.where(xbounds)[0][[0, -1]]

        xmin = max(xmin - 1, 0)
        xmax = min(xmax + 2, x)

        ymin = max(ymin - 1, 0)
        ymax = min(ymax + 2, y)

        zmin = max(zmin - 1, 0)
        zmax = min(zmax + 2, z)

        return xmin, xmax, ymin, ymax, zmin, zmax
