QUILT_PACKAGE = "aics/hipsc_single_cell_image_dataset"
QUILT_REGISTRY = "s3://allencell"
IMAGE_EXTENSION = ".ome.tiff"
SCALE_MICRONS = 0.108333  # um/pixel, from metadata.csv
SCALE_MICRONS_Z = 0.29  # um/pixel,from preprint
OUTPUT_COLUMNS = ["id", "x", "y", "z"]
