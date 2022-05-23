import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

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

    def run(self, channels=[0]):
        for key in self.context.keys:
            for channel in channels:
                self.create_voronoi(key, channel)

    def create_voronoi(self, key, channel):
        image_key = make_full_key(self.folders, self.files, "image", key)
        image = load_image(self.context.working, image_key)

        array = image.get_image_data("ZYX", T=0, C=channel)

        # Create mask and expand using binary dilation to create an artifical
        # border for the voronoi.
        mask = np.zeros(array.shape, dtype="int")
        mask[array != 0] = 1
        mask = binary_dilation(mask, iterations=50)
        mask_id = np.iinfo(array.dtype).max
        array[mask == 0] = mask_id

        distances = distance_transform_edt(array == 0, return_distances=False, return_indices=True)
        distances = distances.astype("uint16", copy=False)

        coordz = distances[0].flatten()
        coordy = distances[1].flatten()
        coordx = distances[2].flatten()
        voronoi = array[coordz, coordy, coordx].reshape(array.shape)

        # Remove masking ids.
        voronoi[mask == 0] = 0
        voronoi[voronoi == mask_id] = 0

        output_key = make_full_key(self.folders, self.files, "output", (key, channel))
        save_image(self.context.working, output_key, voronoi)
