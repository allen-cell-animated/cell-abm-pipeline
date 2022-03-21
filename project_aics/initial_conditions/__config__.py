# Pixel resolution for images (um/pixel) in x/y
SCALE_MICRONS_XY = 0.108333

# Pixel resolution for images (um/pixel) in z
SCALE_MICRONS_Z = 0.108333

# Threshold for number of samples touching edge of FOV to be considered edge
EDGE_THRESHOLD = 1

# Maximum distance (in um) to nearest neighbor to be considered connected
CONNECTED_THRESHOLD = 2.0

# List of Potts Hamiltonian terms for setup file
POTTS_TERMS = ["volume", "surface", "adhesion", "height", "substrate", "persistence"]
