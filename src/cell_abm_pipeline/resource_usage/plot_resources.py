import matplotlib.pyplot as plt

from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_plot
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key


class PlotResources:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "RESOURCES", False),
            "output": make_folder_key(context.name, "plots", "RESOURCES", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["RESOURCES", "csv"], "%s", ""),
            "output": make_file_key(context.name, ["RESOURCES", "png"], "%s", ""),
        }

    def run(self):
        self.plot_wall_clock()
        self.plot_object_storage()

    def plot_wall_clock(self):
        key_file = self.folders["input"] + self.files["input"] % "clock"
        data = load_dataframe(self.context.working, key_file)
        keys = data.KEY.unique()

        fig, axs = plt.subplots(1, 1, figsize=(len(keys), 3))
        values = [data[data["KEY"] == key]["CLOCK"] for key in keys]

        axs.boxplot(values, labels=keys, positions=range(0, len(keys)))
        axs.scatter(data["KEY"], data["CLOCK"], s=10, alpha=0.3, c="k", edgecolors="none")

        axs.set_ylabel("Wall Clock Time (minutes)")

        plot_key = self.folders["output"] + self.files["output"] % "clock"
        save_plot(self.context.working, plot_key)

    def plot_object_storage(self):
        key_file = self.folders["input"] + self.files["input"] % "storage"
        data = load_dataframe(self.context.working, key_file)
        keys = data.KEY.unique()

        fig, axs = plt.subplots(1, 2, figsize=(len(keys) * 2, 3))

        for i, (name, group) in enumerate(data.groupby("GROUP")):
            values = [group[group["KEY"] == key]["STORAGE"] for key in keys]

            axs[i].boxplot(values, labels=keys, positions=range(0, len(keys)))
            axs[i].scatter(
                group["KEY"], group["STORAGE"], s=10, alpha=0.3, c="k", edgecolors="none"
            )

            axs[i].set_ylabel("Size (KiB)")
            axs[i].set_title(f"*.{name}.tar.xz", fontsize=16)

        plot_key = self.folders["output"] + self.files["output"] % "storage"
        save_plot(self.context.working, plot_key)
