import numpy as np

from cell_abm_pipeline.utilities.load import load_tar, load_tar_member
from cell_abm_pipeline.utilities.save import save_image
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class ArcadeToImage:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "data", "LOCATIONS", False),
            "output": make_folder_key(context.name, "converted", "IMAGE", False),
        }
        self.files = {
            "input": make_file_key(context.name, ["LOCATIONS", "tar", "xz"], "%s", ""),
            "output": make_file_key(context.name, ["tiff"], "%s_%02d_%02d", ""),
        }

    def run(self, box=(100, 100, 10)):
        for key in self.context.keys:
            self.arcade_to_image(key, box)

    def arcade_to_image(self, key, box):
        data_key = make_full_key(self.folders, self.files, "input", (key))
        data_tar = load_tar(self.context.working, data_key)

        frames = len(data_tar.getmembers())
        length, width, height = box
        array = np.zeros((frames, 1, height, width, length), "uint16")

        # Iterate through each frame.
        for i, member in enumerate(data_tar.getmembers()):
            tar_member = load_tar_member(data_tar, member.name)
            self.convert_image_frame(array, tar_member, i)

        # Split image into chunks and save.
        chunks = self.split_array_chunks(array)
        for i, j, chunk in chunks:
            output_key = make_full_key(self.folders, self.files, "output", (key, i, j))
            save_image(self.context.working, output_key, chunk)

    @staticmethod
    def convert_image_frame(array, locations, index):
        for location in locations:
            location_id = location["id"]
            voxels = [(z, y, x) for region in location["location"] for x, y, z in region["voxels"]]
            voxels = tuple(np.transpose(voxels))
            array[index, 0][voxels] = location_id

    @staticmethod
    def split_array_chunks(array):
        chunks = []
        length = array.shape[4]
        width = array.shape[3]

        # Calculate chunk splits.
        length_section = [0, 101] + (int(length / 100) - 2) * [100] + [101]
        length_splits = np.array(length_section, dtype=np.int32).cumsum()
        width_section = [0, 101] + (int(width / 100) - 2) * [100] + [101]
        width_splits = np.array(width_section, dtype=np.int32).cumsum()

        # Iterate through each chunk split.
        for i in range(len(length_splits) - 1):
            length_start = length_splits[i]
            length_end = length_splits[i + 1]

            for j in range(len(width_splits) - 1):
                width_start = width_splits[j]
                width_end = width_splits[j + 1]

                # Extract chunk from full contents.
                chunk = array[:, :, :, length_start:length_end, width_start:width_end]

                if np.sum(chunk) != 0:
                    chunks.append((i, j, chunk))

        return chunks
