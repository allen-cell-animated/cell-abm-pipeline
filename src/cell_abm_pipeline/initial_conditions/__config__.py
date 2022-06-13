from typing import List, Dict

# Pixel resolution for images (um/pixel) in x/y
SCALE_MICRONS_XY: float = 0.108333

# Pixel resolution for images (um/pixel) in z
SCALE_MICRONS_Z: float = 0.29

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

# Volume distribution averages.
VOLUME_MU: Dict[str, float] = {
    "DEFAULT": 1865.0,
    "NUCLEUS": 542.0,
}

# Volume distribution standard deviations.
VOLUME_SIGMA: Dict[str, float] = {
    "DEFAULT": 517.0,
    "NUCLEUS": 157.0,
}

# Critical volume distribution averages.
CRITICAL_VOLUME_MU: Dict[str, float] = {
    "DEFAULT": 1300.0,
    "NUCLEUS": 400.0,
}

# Critical volume distribution deviations.
CRITICAL_VOLUME_SIGMA: Dict[str, float] = {
    "DEFAULT": 200.0,
    "NUCLEUS": 50.0,
}

# Height distribution averages.
HEIGHT_MU: Dict[str, float] = {
    "DEFAULT": 9.65,
    "NUCLEUS": 6.75,
}

# Height distribution standard deviations.
HEIGHT_SIGMA: Dict[str, float] = {
    "DEFAULT": 2.4,
    "NUCLEUS": 1.7,
}

# Critical height distribution averages.
CRITICAL_HEIGHT_MU: Dict[str, float] = {
    "DEFAULT": 4.5,
    "NUCLEUS": 3.25,
}

# Critical height distribution standard deviations.
CRITICAL_HEIGHT_SIGMA: Dict[str, float] = {
    "DEFAULT": 1,
    "NUCLEUS": 0.75,
}
