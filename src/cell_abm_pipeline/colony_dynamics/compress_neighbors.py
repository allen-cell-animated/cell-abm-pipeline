import io
import tarfile
from tqdm import tqdm

from cell_abm_pipeline.utilities.load import load_buffer
from cell_abm_pipeline.utilities.save import save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key


class CompressNeighbors:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "NEIGHBORS", False),
            "output": make_folder_key(context.name, "analysis", "NEIGHBORS", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["NEIGHBORS", "csv"], "%s", "%04d"),
            "output": make_file_key(context.name, ["NEIGHBORS", "tar", "xz"], "", "%04d"),
        }

    def run(self):
        for seed in self.context.seeds:
            self.compress_neighbors(seed)

    def compress_neighbors(self, seed):
        """
        Compress individual neighbors files into single archive.
        """
        file_keys = [
            self.folders["input"] + self.files["input"] % (key, seed) for key in self.context.keys
        ]

        with io.BytesIO() as buffer:
            with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
                for file_key in tqdm(file_keys):
                    contents = load_buffer(self.context.working, file_key)

                    info = tarfile.TarInfo(file_key.split("/")[-1])
                    info.size = contents.getbuffer().nbytes

                    tar.addfile(info, fileobj=contents)

            analysis_key = self.folders["output"] + self.files["output"] % (seed)
            save_buffer(self.context.working, analysis_key, buffer)
