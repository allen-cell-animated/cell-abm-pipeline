import re

import pandas as pd
from tqdm import tqdm

from cell_abm_pipeline.utilities.load import load_dataframe, load_gzip
from cell_abm_pipeline.utilities.save import save_dataframe
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, get_keys


class ExtractClock:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "logs", "", False),
            "output": make_folder_key(context.name, "analysis", "RESOURCES", False),
        }
        self.files = {
            "input": "*.gz",
            "output": make_file_key(context.name, ["RESOURCES", "csv"], "%s", ""),
        }

    def run(self):
        file_key = self.folders["output"] + self.files["output"] % "clock"

        try:
            load_dataframe(self.context.working, file_key)
        except:
            self.extract_clock()

    def extract_clock(self):
        key_pattern = self.folders["input"] + self.files["input"]
        log_file_keys = get_keys(self.context.working, key_pattern)

        # Iterate through log file keys to extract wall clock time.
        clock = []
        for log_file_key in tqdm(log_file_keys):
            contents = load_gzip(self.context.working, log_file_key)
            clock = clock + ExtractClock.parse_log_file(contents)

        # Store results in a data frame.
        clock_df = pd.DataFrame(clock)
        clock_df.sort_values(by=["KEY", "SEED"], ignore_index=True, inplace=True)
        clock_df = clock_df.astype({"CLOCK": "float64"})
        clock_df = clock_df.set_index("KEY")

        file_key = self.folders["output"] + self.files["output"] % "clock"
        save_dataframe(self.context.working, file_key, clock_df)

    @staticmethod
    def parse_log_file(contents):
        keys = ["KEY", "SEED", "CLOCK"]

        pattern = (
            "simulation \[ ([A-z0-9\s\_]+) \\| ([0-9]{4}) \] finished in ([0-9\.]+) minutes \n"
        )
        matches = re.findall(pattern, contents)
        matches_dicts = [{key: entry for key, entry in zip(keys, match)} for match in matches]

        return matches_dicts
