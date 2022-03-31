from scipy.ndimage import distance_transform_edt

from cell_abm_pipeline.utilities.load import load_image
from cell_abm_pipeline.utilities.save import save_image
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key


class CreateVoronoi:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "image": make_folder_key(context.name, "images", "", False),
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
        image_key = self.folders["image"] + self.files["image"] % (key)
        image = load_image(self.context.working, image_key)

        array = image.get_image_data("ZYX", T=0, C=channel)
        distances = distance_transform_edt(array == 0, return_distances=False, return_indices=True)

        coordz = distances[0].flatten()
        coordy = distances[1].flatten()
        coordx = distances[2].flatten()
        voronoi = array[coordz, coordy, coordx].reshape(array.shape)

        output_key = self.folders["image"] + self.files["output"] % (key, channel)
        save_image(self.context.working, output_key, voronoi)
