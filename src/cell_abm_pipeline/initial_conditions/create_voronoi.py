import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_fill_holes

from cell_abm_pipeline.utilities.load import load_image
from cell_abm_pipeline.utilities.save import save_image
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class CreateVoronoi:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "image": make_folder_key(context.name, "images", "", False),
            "output": make_folder_key(context.name, "images", "", False),
        }
        self.files = {
            "image": make_file_key(context.name, ["ome", "tiff"], "%s", ""),
            "output": make_file_key(context.name, ["ome", "tiff"], "%s", "%02d_voronoi"),
        }

    def run(self, iterations=10, channels=[0]):
        for key in self.context.keys:
            for channel in channels:
                self.create_voronoi(key, iterations, channel)

    def create_voronoi(self, key, iterations, channel):
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
    def create_boundary_mask(array, iterations=10):
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
