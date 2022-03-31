import os
import io

import quilt3
import pandas as pd

from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_dataframe, save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key

QUILT_PACKAGE = "aics/hipsc_single_cell_image_dataset"
QUILT_REGISTRY = "s3://allencell"


class DownloadImages:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "manifest": make_folder_key(context.name, "", "", False),
            "image": make_folder_key(context.name, "images", "", False),
        }
        self.files = {
            "manifest": make_file_key(context.name, ["csv"], "", ""),
            "image": make_file_key(context.name, [], "%s", ""),
        }

    def run(self, num_images=0):
        self.download_images(num_images)

    def download_images(self, num_images):
        # Skip if requested number images is invalid.
        if num_images <= 0:
            print("No images downloaded.")
            return

        # Connect to Quilt package.
        pkg = quilt3.Package.browse(QUILT_PACKAGE, QUILT_REGISTRY)

        # Get list of FOV images.
        manifest_key = self.folders["manifest"] + self.files["manifest"]
        fov_files = self.get_fov_files(self.context.working, manifest_key, pkg)

        image_path = self.context.working + self.folders["image"]
        downloaded_fov_files = []

        # Iterate through downloadable images until no more image are available
        # or requested number of images have been downloaded.
        for fov_file, status in fov_files.to_records(index=False):
            if len(downloaded_fov_files) >= num_images:
                break

            if status == "downloaded":
                continue

            # Download image from Quilt as bytes, then save buffer.
            print(f"Downloading {fov_file} ...")
            contents = io.BytesIO(pkg["fov_seg_path"][fov_file].get_bytes())
            image_key = self.folders["image"] + self.files["image"] % fov_file
            save_buffer(self.context.working, image_key, contents)
            downloaded_fov_files.append(fov_file)

        # Update status for downloaded FOV images.
        fov_files.loc[fov_files["fov_seg_path"].isin(downloaded_fov_files), "status"] = "downloaded"
        save_dataframe(self.context.working, manifest_key, fov_files, index=False)

    @staticmethod
    def get_fov_files(path, key, pkg):
        """Gets list of all available FOV files."""
        full_path = f"{path}{key}"

        if not os.path.isfile(full_path):
            fov_file_list = list(pkg["fov_seg_path"].map(lambda lk, entry: lk))
            fov_files = pd.DataFrame(fov_file_list, columns=["fov_seg_path"])
            fov_files["status"] = "available"
            save_dataframe(path, key, fov_files, index=False)
        else:
            fov_files = load_dataframe(path, key)

        return fov_files
