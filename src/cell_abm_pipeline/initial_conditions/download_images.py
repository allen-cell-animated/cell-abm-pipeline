import os
import io
from typing import List

import quilt3
import pandas as pd

from cell_abm_pipeline.initial_conditions.__config__ import QUILT_REGISTRY, QUILT_PACKAGE
from cell_abm_pipeline.initial_conditions.__main__ import Context
from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_dataframe, save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class DownloadImages:
    """
    Task to download images from Quilt package.

    Working location structure for a given context:

    .. code-block:: bash

        (name)/
        ├── (name).csv
        └── images
            ├── (name)_(image_key_1).ome.tiff
            ├── (name)_(image_key_2).ome.tiff
            ├── ...
            └── (name)_(image_key_n).ome.tiff

    The manifest ``(name).csv`` lists FOV segmentations paths (extracted from
    the Quilt package manifest) and file status ("downloaded" or "available").
    Files that are marked as "downloaded" will not be re-downloaded.
    This file can be edited to filter for specific files to download.
    Delete this file to regenerate it from the Quilt package manifest.

    Attributes
    ----------
    context
        ``Context`` object defining working location and name.
    folders
        Dictionary of input and output folder keys.
    files
        Dictionary of input and output file keys.
    """

    def __init__(self, context: Context):
        self.context = context
        self.folders = {
            "manifest": make_folder_key(context.name, "", "", False),
            "image": make_folder_key(context.name, "images", "", False),
        }
        self.files = {
            "manifest": make_file_key(context.name, ["csv"], "", ""),
            "image": make_file_key(context.name, [], "%s", ""),
        }

    def run(self, num_images: int = 0) -> None:
        """
        Runs download image task for given context.

        Parameters
        ----------
        num_images
            Number of images to download
        """
        self.download_images(num_images)

    def download_images(self, num_images: int) -> None:
        """
        Download image task.

        Connects to Quilt package specified by ``QUILT_PACKAGE`` and ``QUILT_REGISTRY``.
        Downloads the requested number of images (up to available images).
        Downloaded images are marked as "downloaded" in the manifest.

        Parameters
        ----------
        num_images
            Number of images to download
        """

        # Skip if requested number images is invalid.
        if num_images <= 0:
            print("No images downloaded.")
            return

        # Connect to Quilt package.
        pkg = quilt3.Package.browse(QUILT_PACKAGE, QUILT_REGISTRY)

        # Get list of FOV images.
        manifest_key = make_full_key(self.folders, self.files, "manifest")
        fov_files = self.get_fov_files(self.context.working, manifest_key, pkg)

        # Iterate through downloadable images until no more images are available
        # or requested number of images have been downloaded.
        downloaded_fov_files: List[str] = []
        for fov_seg_path, _, status in fov_files.to_records(index=False):
            if len(downloaded_fov_files) >= num_images:
                break

            if status == "downloaded":
                continue

            # Download image from Quilt as bytes, then save buffer.
            print(f"Downloading {fov_seg_path} ...")
            contents = io.BytesIO(pkg[fov_seg_path].get_bytes())
            fov_seg_path_key = fov_seg_path.split("/")[-1]
            image_key = make_full_key(self.folders, self.files, "image", fov_seg_path_key)
            save_buffer(self.context.working, image_key, contents)
            downloaded_fov_files.append(fov_seg_path)

        # Update status for downloaded FOV images.
        fov_files.loc[fov_files["fov_seg_path"].isin(downloaded_fov_files), "status"] = "downloaded"
        save_dataframe(self.context.working, manifest_key, fov_files, index=False)

    @staticmethod
    def get_fov_files(working: str, key: str, pkg: quilt3.Package) -> pd.DataFrame:
        """
        Gets dataframe of all FOV file paths and statuses.

        ============  =========================================
        Column        Description
        ============  =========================================
        fov_seg_path  path to FOV segmentation file
        fov_path      path to FOV raw file
        status        file status ("available" or "downloaded")
        ============  =========================================

        If saved manifest exists, it is loaded directly.
        If saved manifest does not exist, the manifest is extracted from the
        Quilt manifest column ``fov_seg_path`` and ``fov_path``.

        Parameters
        ----------
        working
            Working location (local path or S3 bucket)
        key
            Key for download manifest
        pkg
            Quilt package containing FOV image segmentation paths
        """
        full_path = f"{working}{key}"

        if not os.path.isfile(full_path):
            all_fov_entries = set()

            for pkg_entry in pkg.manifest:
                try:
                    associates = pkg_entry["meta"]["user_meta"]["associates"]
                    fov_entry = (associates["fov_seg_path"], associates["fov_path"], "available")
                    all_fov_entries.add(fov_entry)
                except KeyError:
                    pass

            sorted_fov_entries = sorted(list(all_fov_entries))
            fov_columns = ["fov_seg_path", "fov_path", "status"]
            fov_files = pd.DataFrame(sorted_fov_entries, columns=fov_columns)
            save_dataframe(working, key, fov_files, index=False)
        else:
            fov_files = load_dataframe(working, key)

        return fov_files
