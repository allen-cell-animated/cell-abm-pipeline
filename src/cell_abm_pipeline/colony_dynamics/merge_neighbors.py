import io
import lzma
from tqdm import tqdm

from cell_abm_pipeline.utilities.load import load_buffer
from cell_abm_pipeline.utilities.save import save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key


class MergeNeighbors:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "NEIGHBORS", False),
            "output": make_folder_key(context.name, "analysis", "NEIGHBORS", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["NEIGHBORS", "csv"], "%s", "%04d"),
            "output": make_file_key(context.name, ["NEIGHBORS", "csv", "xz"], "", "%04d"),
        }

    def run(self):
        for seed in self.context.seeds:
            self.merge_neighbors(seed)

    def merge_neighbors(self, seed):
        """
        Merge individual neighbors files into single file.
        """
        file_keys = [
            self.folders["input"] + self.files["input"] % (key, seed) for key in self.context.keys
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

            analysis_key = self.folders["output"] + self.files["output"] % (seed)
            save_buffer(self.context.working, analysis_key, buffer)
