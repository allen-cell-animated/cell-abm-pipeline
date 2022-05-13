import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from cell_abm_pipeline.cell_shape.__config__ import PCA_COMPONENTS, VALID_PHASES
from cell_abm_pipeline.cell_shape.calculate_coefficients import CalculateCoefficients
from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_pickle
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class PerformPCA:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "SH", False),
            "region": make_folder_key(context.name, "analysis", "SH", False),
            "output": make_folder_key(context.name, "analysis", "SHPCA", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["SH", "csv", "xz"], "", "%04d"),
            "region": lambda r: make_file_key(context.name, ["SH", r, "csv", "xz"], "", "%04d"),
            "output": lambda r: make_file_key(context.name, ["SHPCA", r, "pkl"], "%s", ""),
        }

    def run(self, region=None):
        self.perform_pca(region)

    def perform_pca(self, region):
        all_data = []

        for seed in self.context.seeds:
            file_key = make_full_key(self.folders, self.files, "input", seed)
            data = load_dataframe(self.context.working, file_key)

            if data.KEY.isnull().values.any():
                data.KEY = ""

            if region:
                region_file_key = make_full_key(self.folders, self.files, "region", seed, region)
                region_data = load_dataframe(self.context.working, region_file_key)

                if region_data.KEY.isnull().values.any():
                    region_data.KEY = ""

                join_columns = ["KEY", "ID", "SEED", "TICK"]
                region_data = region_data.set_index(join_columns)
                data = data.join(region_data, on=join_columns, rsuffix=f".{region}")

            data = data[data.KEY.isin(self.context.keys)]
            all_data.append(data)

        data_df = pd.concat(all_data).set_index("KEY")
        data_df = data_df[data_df.PHASE.isin(VALID_PHASES)]

        coeff_names = CalculateCoefficients.get_coeff_names()

        if region:
            coeff_names = coeff_names + CalculateCoefficients.get_coeff_names(suffix=f".{region}")

        for key, key_group in data_df.groupby("KEY"):
            output_key = make_full_key(self.folders, self.files, "output", key, region)
            pca = self.fit_feature_pca(key_group[coeff_names], key_group["NUM_VOXELS"])
            output = {"data": key_group, "pca": pca}
            save_pickle(self.context.working, output_key, output)

    @staticmethod
    def fit_feature_pca(features, ordering, components=PCA_COMPONENTS):
        """Perform PCA on given features."""
        if isinstance(features, pd.DataFrame):
            features = features.values.copy()

        # Drop nans.
        nan_rows = np.isnan(features).any(axis=1)
        features = features[~nan_rows]
        ordering = ordering[~nan_rows]

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

    @staticmethod
    def apply_data_transform(features, pca):
        if isinstance(features, pd.DataFrame):
            features = features.values.copy()

        nan_indices = np.isnan(features).any(axis=1)
        features = features[~nan_indices]

        return pca.transform(features), nan_indices
