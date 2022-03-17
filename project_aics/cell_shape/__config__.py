# Order of the spherical harmonics coefficient parametrization
COEFF_ORDER = 16

# Number of principal components (i.e. shape modes)
PCA_COMPONENTS = 8

# Valid cell features and corresponding ranges.
CELL_FEATURES = {
    "NUM_VOXELS": [500, 2000],
    "TICK": [0, 24],
}

# Valid cell phases for calculating shapes.
VALID_PHASES = ["PROLIFERATIVE_G1", "PROLIFERATIVE_S", "PROLIFERATIVE_G2", "M0"]
