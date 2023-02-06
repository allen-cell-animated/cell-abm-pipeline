from itertools import groupby
from math import sqrt

import numpy as np
from prefect import task
from scipy.stats import gamma

from cell_abm_pipeline.tasks.plot_phase_fractions import PHASE_COLORS
from cell_abm_pipeline.utilities.plot import make_grid_figure

PHASE_SETTINGS = {
    "PROLIFERATIVE_G1": {
        "bins": np.arange(0, 7, 1),
        "lambda": 8.33,
        "k": 17,
    },
    "PROLIFERATIVE_S": {
        "bins": np.arange(0, 21, 1),
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


@task
def plot_phase_durations(keys, data, phase):
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Duration (hrs)")
        ax.set_ylabel("Frequency")

        phase_durations = get_phase_durations(data[key])

        if phase not in PHASE_SETTINGS or phase not in phase_durations:
            return None

        durations = np.array(phase_durations[phase])
        settings = PHASE_SETTINGS[phase]
        color = PHASE_COLORS[phase]

        counts, labels = np.histogram(durations, bins=settings["bins"])
        counts = counts / np.sum(counts)
        label = f"simulated ({durations.mean():.2f} $\pm$ {durations.std():.2f} hr)"
        ax.bar(labels[:-1], counts, align="center", color=color, alpha=0.7, label=label)

        scale = 1.0 / settings["lambda"]
        k = settings["k"]
        x = np.linspace(gamma.ppf(0.001, k, scale=scale), gamma.ppf(0.999, k, scale=scale), 100)

        ref_mean = k / settings["lambda"]
        ref_std = sqrt(k / settings["lambda"] ** 2)
        ref_label = (
            f"reference ({ref_mean:.2f} $\pm$ {ref_std:.2f} hr)"
            + f"\nk = {k}, $\lambda$ = {settings['lambda']}"
        )
        ax.plot(x, gamma.pdf(x, k, scale=scale), color=color, lw=2, label=ref_label)

        ax.legend(loc="upper right")

    return fig


def get_phase_durations(data):
    """Calculates phase durations for given dataframe."""
    phase_durations = {}

    for _, group in data.groupby(["SEED", "ID"]):
        group.sort_values("TICK", inplace=True)
        phase_list = group[["PHASE", "time"]].to_records(index=False)
        phase_groups = [list(g) for k, g in groupby(phase_list, lambda g: g[0])]

        for group, next_group in zip(phase_groups[:-1], phase_groups[1:]):
            key, start_time = group[0]
            _, stop_time = next_group[0]

            if key not in phase_durations:
                phase_durations[key] = []

            duration = stop_time - start_time
            phase_durations[key].append(duration)

    return phase_durations
