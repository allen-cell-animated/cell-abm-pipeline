import numpy as np
import quilt3
import os
import imageio
from glob import glob
from aicsimageio import AICSImage
from constants import QUILT_PACKAGE, QUILT_REGISTRY, IMAGE_EXTENSION


class DownloadImages:
    def __init__(self):
        pass

    def download(self, num_images, output_path):
        """Download segmentation data for FOVs and save projections."""

        # Skip if requested number images is invalid.
        if num_images <= 0:
            print("No images downloaded.")
            return

        # Connect to Quilt package.
        pkg = quilt3.Package.browse(QUILT_PACKAGE, QUILT_REGISTRY)

        # Create directory to save downloaded images.
        root_path = os.path.join(output_path, "downloaded_images")
        if not os.path.isdir(root_path):
            os.mkdir(root_path)

        # Get list of downloadable images (exluding those already downloaded).
        fov_files = self.get_fov_files(output_path, pkg)
        downloadable_images = self.get_downloadable_images(fov_files, root_path)

        # Iterate through downloadable images until no more image are available
        # or requested number of images have been downloaded.
        for i, image in enumerate(downloadable_images):
            # Check if request number of image have been downloaded.
            if i == num_images:
                return

            # Create directory for image files.
            image_path = os.path.join(root_path, image)
            os.mkdir(image_path)

            # Download image from Quilt.
            print(f"Downloading {image}")
            pkg["fov_seg_path"][image + IMAGE_EXTENSION].fetch(f"{image_path}/")

            # Save XY projection of segmented FOV as png
            self.save_png_projection(image_path, image)

    @staticmethod
    def get_fov_files(output_path, pkg):
        """Gets list of all available FOV files."""
        fov_path = os.path.join(output_path, "fovs.txt")

        if not os.path.isfile(fov_path):
            # Extract all paths in fov_seg_path folder from Quilt package.
            fov_files = list(pkg["fov_seg_path"].map(lambda lk, entry: lk))

            # Save a copy of fov files list.
            with open(fov_path, "w") as file:
                file.write("\n".join(fov_files))
        else:
            # Load FOV files list if it already exists.
            with open(fov_path, "r") as file:
                fov_files = [row.strip() for row in file.readlines()]

        return fov_files

    @staticmethod
    def get_downloadable_images(fov_files, path):
        """Get list downloadable images (that are not already downloaded)."""

        # Get available and downloaded images without extensions.
        available_images = [file.replace(IMAGE_EXTENSION, "") for file in fov_files]
        downloaded_images = [file.replace(f"{path}/", "") for file in glob(f"{path}/*")]

        # Filter list of available image for those that have not been downloaded.
        downloadable_images = list(set(available_images) - set(downloaded_images))

        # Shuffle list of images.
        np.random.shuffle(downloadable_images)

        return downloadable_images

    @staticmethod
    def save_png_projection(image_path, image):
        """Saves XY projection of image for all Z slices."""
        img = AICSImage(os.path.join(image_path, image + IMAGE_EXTENSION))
        for z in range(img.shape[3]):
            imageio.imwrite(
                os.path.join(image_path, f"{image}_z{z}.png"),
                img.get_image_data("XYZ", S=0, T=0, C=1)[:, :, z],
            )
