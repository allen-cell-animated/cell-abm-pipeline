import re
import os

import boto3
import pandas as pd
from tqdm import tqdm

from cell_abm_pipeline.resource_usage.__config__ import STORAGE_CONVERSION
from cell_abm_pipeline.utilities.load import load_dataframe, load_keys
from cell_abm_pipeline.utilities.save import save_dataframe
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class CalculateStorage:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": lambda group: make_folder_key(context.name, "data", group, False),
            "output": make_folder_key(context.name, "analysis", "RESOURCES", False),
        }
        self.files = {
            "input": lambda group: f"*.{group}.tar.xz",
            "output": make_file_key(context.name, ["RESOURCES", "csv"], "%s", ""),
        }

    def run(self):
        file_key = make_full_key(self.folders, self.files, "output", "storage")

        try:
            load_dataframe(self.context.working, file_key)
        except:
            self.calculate_storage()

    def calculate_storage(self):
        groups = ["CELLS", "LOCATIONS"]
        storage = []

        for group in groups:
            key_pattern = make_full_key(self.folders, self.files, "input", arguments=group)
            file_keys = load_keys(self.context.working, key_pattern)

            # Iterate through file keys to extract storage size.
            for file_key in tqdm(file_keys):
                summary = CalculateStorage.get_file_summary(
                    self.context.working, self.context.name, file_key, group
                )
                storage.append(summary)

        # Store results in a data frame.
        storage_df = pd.DataFrame(storage)
        storage_df.sort_values(by=["KEY", "GROUP", "SEED"], ignore_index=True, inplace=True)
        storage_df = storage_df.astype({"STORAGE": "float64"})
        storage_df = storage_df.set_index("KEY")

        file_key = make_full_key(self.folders, self.files, "output", "storage")
        save_dataframe(self.context.working, file_key, storage_df)

    @staticmethod
    def get_file_summary(working, name, file_key, group):
        keys = ["KEY", "SEED"]
        pattern = "[_]*([A-z0-9\s\_]*)_([0-9]{4})\."
        key, seed = re.findall(pattern, file_key.split(name)[-1])[0]

        if working[:5] == "s3://":
            s3_resource = boto3.resource("s3")
            summary = s3_resource.ObjectSummary(working[5:], file_key)
            storage = summary.size
        else:
            storage = os.path.getsize(f"{working}{file_key}")

        return {"KEY": key, "SEED": seed, "GROUP": group, "STORAGE": storage * STORAGE_CONVERSION}
