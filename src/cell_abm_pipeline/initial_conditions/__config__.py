from typing import List

# Pixel resolution for images (um/pixel) in x/y
SCALE_MICRONS_XY: float = 0.108333

# Pixel resolution for images (um/pixel) in z
SCALE_MICRONS_Z: float = 0.108333

# Threshold for number of samples touching edge of FOV to be considered edge
EDGE_THRESHOLD: int = 1

# Maximum distance (in um) to nearest neighbor to be considered connected
CONNECTED_THRESHOLD: float = 2.0

# List of Potts Hamiltonian terms for setup file
POTTS_TERMS: List[str] = ["volume", "surface", "adhesion", "height", "substrate", "persistence"]

# Name of Quilt package for downloading images
QUILT_PACKAGE: str = "aics/hipsc_single_cell_image_dataset"

# Name of Quilt registry
QUILT_REGISTRY: str = "s3://allencell"
