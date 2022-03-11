import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from project_aics.cell_shape.__config__ import PCA_COMPONENTS
from project_aics.cell_shape.calculate_coefficients import CalculateCoefficients
from project_aics.utilities.load import load_dataframe
from project_aics.utilities.save import save_pickle_to_fs
from project_aics.utilities.keys import make_folder_key, make_file_key


class PerformPCA:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "SH", False),
            "output": make_folder_key(context.name, "analysis", "SHPCA", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["SH", "csv", "xz"], "", "%04d"),
            "output": lambda r: make_file_key(context.name, ["SHPCA", r, "pkl"], "%s", ""),
        }

    def run(self, region=None):
        self.perform_pca(region)

    def perform_pca(self, region):
        all_data = []

        for seed in self.context.seeds:
            file_key = self.folders["input"] + self.files["input"] % (seed)
            data = load_dataframe(self.context.working, file_key)
            data = data[data.KEY.isin(self.context.keys)]
            all_data.append(data)

        # TODO: add filter by cell phase

        data_df = pd.concat(all_data).set_index("KEY")
        coeff_names = CalculateCoefficients.get_coeff_names()

        for key, key_group in data_df.groupby("KEY"):
            output_key = self.folders["output"] + self.files["output"](region) % (key)
            pca = self.fit_feature_pca(key_group[coeff_names], key_group["NUM_VOXELS"])
            output = {"data": key_group, "pca": pca}
            save_pickle_to_fs(self.context.working, output_key, output)

    @staticmethod
    def fit_feature_pca(features, ordering, components=PCA_COMPONENTS):
        """Perform PCA on given features."""
        if isinstance(features, pd.DataFrame):
            features = features.values.copy()

        # Fit data.
        pca = PCA(n_components=components)
        pca = pca.fit(features)

        # Reorient features by ordering data.
        transformed = pca.transform(features)
        for i in range(components):
            pearson = np.corrcoef(ordering, transformed[:, i])
            if pearson[0, 1] < 0:
                pca.components_[i] = pca.components_[i] * -1

        return pca
