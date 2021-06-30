import pandas as pd
from math import ceil
from process_samples import ProcessSamples
from convert_samples import ConvertSamples


class ConvertSamplesToARCADE(ConvertSamples):
    def __init__(self, file):
        super().__init__(file)

    def convert(self, box):
        """Converts samples to ARCADE format."""
        print("Converting to ARCADE format ...")

        # Load samples file.
        samples = self.load_samples(*box)

        # Initialize outputs.
        locations, cells = [], []
        samples_by_id = samples.groupby("id")

        # Iterate through each cell in samples and convert.
        for i, (cell_id, group) in enumerate(samples_by_id):
            cells.append(self.convert_to_cell(i + 1, group))
            locations.append(self.convert_to_location(i + 1, group))

        # Save converted ARCADE files.
        file_path = self.file.replace(".csv", "").replace(".PROCESSED", "")
        self.save_json(file_path, locations, ".LOCATIONS")
        self.save_json(file_path, cells, ".CELLS")

        # Save setup file.
        with open(file_path + ".xml", "w") as out:
            setup_file = self.make_setup_file(len(cells), *box)
            out.write(setup_file)

    def load_samples(self, length, width, height):
        """Load samples and reposition to center of bounding box."""
        df = pd.read_csv(self.file)

        # Get step size for voxels.
        step_x = ProcessSamples._get_step_size(df.x)
        step_y = ProcessSamples._get_step_size(df.y)
        step_z = ProcessSamples._get_step_size(df.z)

        # Rescale integers to step size 1.
        df["x"] = df["x"].divide(step_x).astype("int32")
        df["y"] = df["y"].divide(step_y).astype("int32")
        df["z"] = df["z"].divide(step_z).astype("int32")

        # Check bound sizing.
        delta_x = df["x"].max() - df["x"].min()
        delta_y = df["y"].max() - df["y"].min()
        delta_z = df["z"].max() - df["z"].min()
        assert (
            delta_x + 2 < length
        ), f"x coordinate(s) out of bounds -- increase bounding box size to > {delta_x + 2}"
        assert (
            delta_y + 2 < width
        ), f"y coordinate(s) out of bounds -- increase bounding box size to > {delta_y + 2}"
        assert (
            delta_z + 2 < height
        ), f"z coordinate(s) out of bounds -- increase bounding box size to > {delta_z + 2}"

        # Adjust bounds.
        df["x"] = df["x"] - df["x"].unique().mean() + length / 2
        df["y"] = df["y"] - df["y"].unique().mean() + width / 2
        df["z"] = df["z"] - df["z"].unique().mean() + height / 2

        return df

    def make_setup_file(self, init, length, width, height):
        """Create empty setup file for converted samples."""
        return "\n".join(
            [
                "<set>",
                '    <series name="ARCADE" interval="1" start="0" end="0"',
                f'            dt="0.5" ticks="48"',
                f'            ds="1" height="{height}" length="{length}" width="{width}">',
                "        <agents>",
                "            <populations>",
                f'                <population id="X" init="{init}" />',
                "            </populations>",
                "        </agents>",
                "    </series>",
                "</set>",
            ]
        )

    def convert_to_cell(self, cell_id, samples):
        """Convert samples to .CELL json format."""
        volume, surface = self._get_cell_targets(samples)
        state, phase = self._get_cell_phase(volume)

        return {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "state": state,
            "phase": phase,
            "voxels": volume,
            "targets": [volume, surface],
        }

    def convert_to_location(self, cell_id, samples):
        """Convert samples to .LOCATIONS json format."""
        center = self._get_location_center(samples)
        voxels = self._get_location_voxels(samples)

        return {
            "id": cell_id,
            "center": center,
            "location": [{"region": "UNDEFINED", "voxels": voxels}],
        }

    def _get_cell_targets(self, samples):
        """Convert samples to volume and discrete surface area."""
        critical_height = 10  # um, base height for cells
        volume = len(samples)
        surface = self._volume_to_surface(volume, critical_height)
        return volume, surface

    def _get_cell_phase(self, volume):
        """Converts volume to cell state and phase."""
        critical_volume = 2000  # um^3, base size for cells

        if volume < 0.5 * critical_volume:
            state = "APOPTOTIC"
            phase = "APOPTOTIC_EARLY"
        else:
            state = "PROLIFERATIVE"
            if volume < 2 * 0.95 * critical_volume:
                phase = "PROLIFERATIVE_G1"
            else:
                phase = "PROLIFERATIVE_S" if random() < 0.5 else "PROLFERATIVE_G2"

        return state, phase

    def _volume_to_surface(self, volume, height):
        """Calculate discrete surface area from volume."""
        EQUATION_PARAMETER_N = 0.66880553
        EQUATION_PARAMETER_A = 3.70912871
        EQUATION_PARAMETER_B = -6.82145792

        surface = (3 * volume) / (2 * height)
        correction = (
            EQUATION_PARAMETER_A * (volume ** EQUATION_PARAMETER_N)
            + EQUATION_PARAMETER_B
        )

        return ceil(surface + correction)

    def _get_location_center(self, samples):
        """Gets coordinates of center of samples."""
        return [int(samples[v].mean()) for v in ["x", "y", "z"]]

    def _get_location_voxels(self, samples):
        """Get list of voxel coordinates from samples dataframe."""
        return [
            [int(v) for v in voxel]
            for voxel in samples[["x", "y", "z"]].to_records(index=False)
        ]
