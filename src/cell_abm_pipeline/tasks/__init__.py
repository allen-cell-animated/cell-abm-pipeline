"""Tasks for working with cell-scale agent-based models."""

import matplotlib

from .bin_to_hex import bin_to_hex
from .build_svg_image import build_svg_image
from .calculate_category_durations import calculate_category_durations
from .calculate_data_bins import calculate_data_bins
from .check_data_bounds import check_data_bounds
from .make_bar_figure import make_bar_figure
from .make_box_figure import make_box_figure
from .make_centroids_figure import make_centroids_figure
from .make_contour_figure import make_contour_figure
from .make_density_figure import make_density_figure
from .make_graph_figure import make_graph_figure
from .make_heatmap_figure import make_heatmap_figure
from .make_histogram_figure import make_histogram_figure
from .make_line_figure import make_line_figure
from .make_range_figure import make_range_figure
from .make_scatter_figure import make_scatter_figure

matplotlib.use("agg")
