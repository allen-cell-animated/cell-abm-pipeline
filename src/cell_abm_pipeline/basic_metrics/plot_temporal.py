from itertools import groupby

import numpy as np
import pandas as pd
from scipy.stats import gamma

from cell_abm_pipeline.basic_metrics.__config__ import PHASE_COLORS
from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_plot
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key
from cell_abm_pipeline.utilities.plot import make_plot


class PlotTemporal:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "results", "", False),
            "output": make_folder_key(context.name, "plots", "BASIC", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["csv"], "%s", "%04d"),
            "output": lambda r: make_file_key(context.name, ["BASIC", r, "png"], "%s", ""),
        }

    def run(self, ds=1, dt=1, region=None, reference=None):
        data = {}

        for key in self.context.keys:
            key_data = []

            for seed in self.context.seeds:
                key_file = make_full_key(self.folders, self.files, "input", (key, seed))
                seed_data = load_dataframe(self.context.working, key_file)
                seed_data["SEED"] = seed
                self.convert_data_units(seed_data, ds, dt, region)
                key_data.append(seed_data)

            data[key] = pd.concat(key_data)

        if reference:
            data["_reference"] = load_dataframe(self.context.working, reference)

        self.plot_individual_volume(data, region)
        self.plot_average_volume(data, region)
        self.plot_volume_distribution(data, region)
        self.plot_height_distribution(data, region)

        if not region:
            self.plot_total_counts(data)
            self.plot_cell_phases(data)
            self.plot_phase_durations(data)

    @staticmethod
    def convert_data_units(data, ds, dt, region=None):
        data["TIME"] = dt * data["TICK"]
        data["VOLUME"] = ds * ds * ds * data["NUM_VOXELS"]
        data["HEIGHT"] = ds * (data["MAX_Z"] - data["MIN_Z"] + 1)

        if region:
            data[f"VOLUME.{region}"] = ds * ds * ds * data[f"NUM_VOXELS.{region}"]
            data[f"HEIGHT.{region}"] = ds * (data[f"MAX_Z.{region}"] - data[f"MIN_Z.{region}"])

    def plot_total_counts(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_total_counts,
            xlabel="Time (hrs)",
            ylabel="Number of Cells",
        )

        plot_key = make_full_key(self.folders, self.files, "output", "total_counts")
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_total_counts(ax, data, key):
        total_count = data[key].groupby(["SEED", "TIME"]).size()
        mean = total_count.groupby(["TIME"]).mean()
        std = total_count.groupby(["TIME"]).std()
        ticks = mean.index
        ax.plot(ticks, mean, c="#000")
        ax.fill_between(ticks, mean - std, mean + std, facecolor="#bbb")

    def plot_cell_phases(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_cell_phases,
            xlabel="Time (hrs)",
            ylabel="Fraction of Cells",
        )

        plot_key = make_full_key(self.folders, self.files, "output", "cell_phases")
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_cell_phases(ax, data, key):
        data = data[key]
        total_proliferative_count = (
            data[data["STATE"] == "PROLIFERATIVE"].groupby(["SEED", "TIME"]).size()
        )

        for phase, color in PHASE_COLORS.items():
            phase_count = data[data["PHASE"] == phase].groupby(["SEED", "TIME"]).size()
            phase_fraction = phase_count / total_proliferative_count

            mean = phase_fraction.groupby("TIME").mean()
            std = phase_fraction.groupby(["TIME"]).std()
            ticks = mean.index
            ax.plot(ticks, mean, label=phase, color=color)
            ax.fill_between(ticks, mean - std, mean + std, alpha=0.5, color=color)

    def plot_phase_durations(self, data):
        phase_settings = {
            "PROLIFERATIVE_G1": {
                "bins": np.arange(0, 7, 1),
                "lambda": 8.33,
                "k": 17,
            },
            "PROLIFERATIVE_S": {
                "bins": np.arange(0, 20, 1),
                "lambda": 4.35,
                "k": 43,
            },
            "PROLIFERATIVE_G2": {
                "bins": np.arange(0, 20, 1),
                "lambda": 0.752,
                "k": 3,
            },
            "PROLIFERATIVE_M": {
                "bins": np.arange(0, 20, 1),
                "lambda": 28,
                "k": 14,
            },
        }

        for phase, settings in phase_settings.items():
            make_plot(
                self.context.keys,
                {"data": data, "settings": settings},
                lambda ax, data, key: self._plot_phase_durations(ax, data, key, phase),
                xlabel="Duration (hours)",
                ylabel="Frequency",
            )

            plot_key = make_full_key(self.folders, self.files, "output", f"cell_phases_{phase}")
            save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_phase_durations(ax, data, key, phase):
        color = PHASE_COLORS[phase]
        phase_durations = PlotTemporal.get_phase_durations(data["data"][key])

        if phase not in phase_durations.keys():
            return

        settings = data["settings"]
        durations = np.array(phase_durations[phase])

        m = durations.mean()

        counts, labels = np.histogram(durations, bins=settings["bins"])
        sums = np.sum(counts)
        counts = counts / sums
        ax.bar(labels[:-1], counts, align="center", color=color, alpha=0.7)

        scale = 1.0 / settings["lambda"]
        k = settings["k"]
        x = np.linspace(gamma.ppf(0.001, k, scale=scale), gamma.ppf(0.999, k, scale=scale), 100)
        ax.plot(x, gamma.pdf(x, k, scale=scale), color=color, lw=2)

    @staticmethod
    def get_phase_durations(data):
        """Calculates phase durations for given dataframe."""
        phase_durations = {}

        for name, group in data.groupby(["SEED", "ID"]):
            group.sort_values("TICK", inplace=True)
            phase_list = group[["PHASE", "TIME"]].to_records(index=False)
            phase_groups = [list(g) for k, g in groupby(phase_list, lambda g: g[0])]

            for group, next_group in zip(phase_groups[:-1], phase_groups[1:]):
                key, start_time = group[0]
                _, stop_time = next_group[0]

                if key not in phase_durations.keys():
                    phase_durations[key] = []

                duration = stop_time - start_time
                phase_durations[key].append(duration)

        return phase_durations

    def plot_individual_volume(self, data, region=None):
        make_plot(
            self.context.keys,
            data,
            lambda a, d, k: self._plot_individual_volume(a, d, k, region),
            xlabel="Time (hrs)",
            ylabel="Volume ($\mu m^3$)",
        )

        plot_key = make_full_key(self.folders, self.files, "output", "individual_volume", region)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_individual_volume(ax, data, key, region=None, num=5):
        value = f"VOLUME.{region}" if region else "VOLUME"
        counter = 0

        for _, group in data[key].groupby(["SEED", "ID"]):
            group.sort_values("TIME", inplace=True)
            counter = counter + 1

            volume = group[value].values
            ticks = group["TIME"]

            ax.plot(ticks, volume)

            if counter >= num:
                break

    def plot_average_volume(self, data, region=None):
        make_plot(
            self.context.keys,
            data,
            lambda a, d, k: self._plot_average_volume(a, d, k, region),
            xlabel="Time (hrs)",
            ylabel="Average Volume ($\mu m^3$)",
        )

        plot_key = make_full_key(self.folders, self.files, "output", "average_volume", region)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_average_volume(ax, data, key, region=None):
        value = f"VOLUME.{region}" if region else "VOLUME"

        volume = data[key].groupby(["SEED", "TIME"]).mean()
        mean = volume.groupby(["TIME"])[value].mean()
        std = volume.groupby(["TIME"])[value].std()
        ticks = mean.index

        if "_reference" in data:
            ax.plot(ticks, [data["_reference"][value].mean()] * len(ticks), c="#555", lw=0.5)

        ax.plot(ticks, mean, c="#000")
        ax.fill_between(ticks, mean - std, mean + std, facecolor="#bbb")

    def plot_volume_distribution(self, data, region=None):
        make_plot(
            self.context.keys,
            data,
            lambda a, d, k: self._plot_volume_distribution(a, d, k, region),
            xlabel="Volume ($\mu m^3$)",
            ylabel="Frequency",
            legend=True,
        )

        plot_key = make_full_key(self.folders, self.files, "output", "volume_distribution", region)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_volume_distribution(ax, data, key, region=None):
        value = f"VOLUME.{region}" if region else "VOLUME"
        bins = np.arange(0, 5000, 100)

        if "_reference" in data:
            reference = data["_reference"][value]
            ax.hist(reference, bins=bins, density=True, color="#999999", alpha=0.7, label="ref")

        volumes = data[key][value]
        print(f"volume mean    = {volumes.mean()}")
        print(f"volume std dev = {volumes.std()}")
        ax.hist(volumes, bins=bins, density=True, histtype="step", color="k", label="sim")

    def plot_height_distribution(self, data, region=None):
        make_plot(
            self.context.keys,
            data,
            lambda a, d, k: self._plot_height_distribution(a, d, k, region),
            xlabel="Height ($\mu m$)",
            ylabel="Frequency",
            legend=True,
        )

        plot_key = make_full_key(self.folders, self.files, "output", "height_distribution", region)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_height_distribution(ax, data, key, region=None):
        value = f"HEIGHT.{region}" if region else "HEIGHT"
        bins = np.arange(0, 30, 1)

        if "_reference" in data:
            reference = data["_reference"][value]
            ax.hist(reference, bins=bins, density=True, color="#999999", alpha=0.7, label="ref")

        heights = data[key][value]
        print(f"height mean    = {heights.mean()}")
        print(f"height std dev = {heights.std()}")
        ax.hist(heights, bins=bins, density=True, histtype="step", color="k", label="sim")
