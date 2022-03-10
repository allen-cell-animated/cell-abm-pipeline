import io
import tarfile

from tqdm import tqdm

from project_aics.utilities import (
    load_buffer,
    save_buffer,
    make_folder_key,
    make_file_key,
)


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
            self.folders["input"] + self.files["input"](region) % (key, seed)
            for key in self.context.keys
        ]

        with io.BytesIO() as buffer:
            with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
                for file_key in tqdm(file_keys):
                    contents = load_buffer(self.context.working, file_key)

                    info = tarfile.TarInfo(file_key.split("/")[-1])
                    info.size = contents.getbuffer().nbytes

                    tar.addfile(info, fileobj=contents)

            analysis_key = self.folders["output"] + self.files["output"](region) % (seed)
            save_buffer(self.context.working, analysis_key, buffer)
