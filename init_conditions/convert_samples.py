import json
import re
from random import random


class ConvertSamples:
    def __init__(self, file):
        self.file = file

    @staticmethod
    def save_json(filename, contents, extension):
        """Save contents as JSON."""
        with open(filename + extension + ".json", "w") as f:
            jn = json.dumps(contents, indent=2, separators=(",", ":"))
            f.write(ConvertSamples.format_json(jn).replace("NaN", '"nan"'))

    @staticmethod
    def format_json(jn):
        """Format JSON contents."""
        jn = jn.replace(":", ": ")
        for arr in re.findall('\[\n\s+[A-z0-9$",\-\.\n\s]*\]', jn):
            jn = jn.replace(arr, re.sub(r",\n\s+", r",", arr))
        jn = re.sub(r'\[\n\s+([A-Za-z0-9,"$\.\-]+)\n\s+\]', r"[\1]", jn)
        jn = jn.replace("],[", "],\n          [")
        jn = re.sub("([0-9]{1}),", r"\1, ", jn)
        return jn
