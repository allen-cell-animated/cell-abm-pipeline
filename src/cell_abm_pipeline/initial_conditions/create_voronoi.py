from typing import List, Tuple, Optional
from math import floor

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_fill_holes

from cell_abm_pipeline.initial_conditions.__main__ import Context
from cell_abm_pipeline.utilities.load import load_image
from cell_abm_pipeline.utilities.save import save_image
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class CreateVoronoi:
    """
    Task to create Voronoi tessellation from starting image.

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

    def run(
        self, iterations: int = 2, channels: Optional[List[int]] = None, height: int = 10
    ) -> None:
        """
        Runs create voronoi task for given context.

        Parameters
        ----------
        iterations
            Number of boundary estimation steps.
        channels
            Image channel indices.
        height
            Target height for tesselation.
        """
        if channels is None:
            channels = [0]

        for key in self.context.keys:
            for channel in channels:
                self.create_voronoi(key, iterations, channel, height)

    def create_voronoi(self, key: str, iterations: int, channel: int, height: int) -> None:
        """
        Create Voronoi task.

        Loads image from working location.
        Creates boundary for Voronoi using binary dilation, clamped to the
        target height, then performs the Voronoi tessellation using the selected
        channel of the image.

        Parameters
        ----------
        key
            Key for image.
        iterations
            Number of boundary estimation steps.
        channel
            Image channel index.
        height
            Target height for tesselation.
        """
        image_key = make_full_key(self.folders, self.files, "image", key)
        image = load_image(self.context.working, image_key)

        array = image.get_image_data("ZYX", T=0, C=channel)

        # Create artificial boundary for voronoi.
        mask = self.create_boundary_mask(array, iterations)
        lower_bound, upper_bound = self.adjust_mask_bounds(array, height)
        mask_id = np.iinfo(array.dtype).max
        array[mask == 0] = mask_id
        mask[:lower_bound, :, :] = 0
        mask[upper_bound:, :, :] = 0

        # Calculate voronoi on bounded array.
        zslice, yslice, xslice = self.get_array_slices(mask)
        voronoi = self.calculate_voronoi_array(array[zslice, yslice, xslice])

        # Remove masking ids.
        array[zslice, yslice, xslice] = voronoi
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
    def adjust_mask_bounds(array: np.ndarray, target_height: int) -> Tuple[int, int]:
        lower_bound, upper_bound = np.where(np.any(array, axis=(1, 2)))[0][[0, -1]]
        current_height = upper_bound - lower_bound + 1

        if current_height < target_height:
            height_delta = target_height - current_height
            lower_offset = floor(height_delta / 2)
            upper_offset = height_delta - lower_offset
            lower_bound = lower_bound - lower_offset
            upper_bound = upper_bound + upper_offset + 1
        else:
            upper_bound = upper_bound + 1

        return (lower_bound, upper_bound)

    @staticmethod
    def get_array_slices(array: np.ndarray) -> Tuple[slice, slice, slice]:
        """
        Calculate bounding box slices around binary array.

        Parameters
        ----------
        array
            Binary array.

        Returns
        -------
        :
            Slices in the z, y, and x directions.
        """
        zsize, ysize, xsize = array.shape

        zmin, zmax = np.where(np.any(array, axis=(1, 2)))[0][[0, -1]]
        ymin, ymax = np.where(np.any(array, axis=(0, 2)))[0][[0, -1]]
        xmin, xmax = np.where(np.any(array, axis=(0, 1)))[0][[0, -1]]

        zslice = slice(max(zmin - 1, 0), min(zmax + 2, zsize))
        yslice = slice(max(ymin - 1, 0), min(ymax + 2, ysize))
        xslice = slice(max(xmin - 1, 0), min(xmax + 2, xsize))

        slices = (zslice, yslice, xslice)
        return slices

    @staticmethod
    def calculate_voronoi_array(array: np.ndarray) -> np.ndarray:
        """
        Calculates voronoi on image array using distance transform.

        Parameters
        ----------
        array
            Image array.

        Returns
        -------
        :
            Voronoi array.
        """
        distances = distance_transform_edt(array == 0, return_distances=False, return_indices=True)
        distances = distances.astype("uint16", copy=False)

        coordinates_z = distances[0].flatten()
        coordinates_y = distances[1].flatten()
        coordinates_x = distances[2].flatten()
        voronoi = array[coordinates_z, coordinates_y, coordinates_x].reshape(array.shape)

        return voronoi
