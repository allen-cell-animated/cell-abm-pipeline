import io
import tarfile
from tqdm import tqdm

from cell_abm_pipeline.utilities.load import load_buffer
from cell_abm_pipeline.utilities.save import save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class CompressCoefficients:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "SH", False),
            "output": make_folder_key(context.name, "analysis", "SH", True),
        }
        self.files = {
            "input": lambda r: make_file_key(context.name, ["SH", r, "csv"], "%s", "%04d"),
            "output": lambda r: make_file_key(context.name, ["SH", r, "tar", "xz"], "", "%04d"),
        }

    def run(self, region=None):
        for seed in self.context.seeds:
            self.compress_coefficients(seed, region)

    def compress_coefficients(self, seed, region):
        """
        Compress individual coefficients files into single archive.
        """
        file_keys = [
            make_full_key(self.folders, self.files, "input", (key, seed), region)
            for key in self.context.keys
        ]

        with io.BytesIO() as buffer:
            with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
                for file_key in tqdm(file_keys):
                    contents = load_buffer(self.context.working, file_key)

                    info = tarfile.TarInfo(file_key.split("/")[-1])
                    info.size = contents.getbuffer().nbytes

                    tar.addfile(info, fileobj=contents)

            analysis_key = make_full_key(self.folders, self.files, "output", seed, region)
            save_buffer(self.context.working, analysis_key, buffer)
