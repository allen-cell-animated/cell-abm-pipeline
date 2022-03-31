import io
import lzma
from tqdm import tqdm

from cell_abm_pipeline.utilities.load import load_buffer
from cell_abm_pipeline.utilities.save import save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key


class MergeCoefficients:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "SH", False),
            "output": make_folder_key(context.name, "analysis", "SH", True),
        }
        self.files = {
            "input": lambda r: make_file_key(context.name, ["SH", r, "csv"], "%s", "%04d"),
            "output": lambda r: make_file_key(context.name, ["SH", r, "csv", "xz"], "", "%04d"),
        }

    def run(self, region=None):
        for seed in self.context.seeds:
            self.merge_coefficients(seed, region)

    def merge_coefficients(self, seed, region):
        """
        Merge individual coefficients files into single file.
        """
        file_keys = [
            self.folders["input"] + self.files["input"](region) % (key, seed)
            for key in self.context.keys
        ]

        with io.BytesIO() as buffer:
            with lzma.open(buffer, "wb") as lzf:
                for file_key in tqdm(file_keys):
                    contents = load_buffer(self.context.working, file_key)
                    file_contents = contents.read().decode("utf-8").split("\n")

                    if file_key == file_keys[0]:
                        header = file_contents[0] + "\n"
                        lzf.write(header.encode("utf-8"))

                    rows = [entry.replace("0.0,", "0,") for entry in file_contents[1:]]
                    lzf.write("\n".join(rows).encode("utf-8"))

            analysis_key = self.folders["output"] + self.files["output"](region) % (seed)
            save_buffer(self.context.working, analysis_key, buffer)
