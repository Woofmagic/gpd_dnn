from pathlib import Path
import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def tag_figure_with_date(matplotlib_axis: plt.axes):
    """
    Add a date "stamp" on a Matplotlib figure.
    """

    # position for the date label:
    _DEFAULT_X_POSITION = 0.0
    _DEFAULT_Y_POSITION = -0.05

    # default string:
    _DEFAULT_DATESTRING = f"Figure rendered {datetime.datetime.now():%Y%m%d-%H%M%S}"

    # main function flow:
    matplotlib_axis.text(
            x = _DEFAULT_X_POSITION, y = _DEFAULT_Y_POSITION,
            s = _DEFAULT_DATESTRING,
            transform = matplotlib_axis.transAxes,
        )

    return matplotlib_axis

def save_matplotlib_figure(
        figure: plt.figure,
        output_directory: Path,
        figure_name: str = "sample"):

    # file extensions:
    _FIGURE_FILE_EXTENSIONS = ("png", "eps", "svg")

    # figure facecolor:
    _FIGURE_FACECOLOR = "white"
    _FIGURE_TRANSPENCY = False # i.e. is the thing transparent or not

    # compute the figurename
    computed_figure_pathname = output_directory / figure_name

    for extension in _FIGURE_FILE_EXTENSIONS:
        figure.savefig(
            computed_figure_pathname.with_suffix(f".{extension}"),
            facecolor = _FIGURE_FACECOLOR, transparent = _FIGURE_TRANSPENCY,
        )

    # your responsibility to close the figure
    return figure

def construct_cff_histogram_plot(
    cff_values: np.ndarray,
    km15_value,
    cff_label,
    good_kinematic_set,
    kinematic_title,
    km15_cff_string,
    ):
    """
    Plot the distribution of a CFF over all replicas.
    """
    # figure size:
    _FIGURE_WIDTH = 10
    _FIGURE_HEIGHT = 8

    # histogram bin settings:
    _HISTOGRAM_BIN_NUMBER = 30
    _HISTOGRAM_ALPHA_VALUE = 0.6
    _HISTOGRAM_SOLID_COLOR = "skyblue"
    _HISTOGRAM_EDGE_COLOR = "black"

    # gaussian fit settings:
    _GAUSSIAN_LINE_COLOR = "red"
    _GAUSSIAN_LINE_STYLE = "--"

    # vertical KM15 line:
    _VERTICAL_LINE_COLOR = "green"
    _VERTICAL_LINE_STYLE = "-"
    _VERTICAL_LINE_THICKNESS = 2.0

    fig, ax = plt.subplots(figsize = (_FIGURE_WIDTH, _FIGURE_HEIGHT))

    ax.hist(
        cff_values,
        bins = _HISTOGRAM_BIN_NUMBER,
        alpha = _HISTOGRAM_ALPHA_VALUE, color = _HISTOGRAM_SOLID_COLOR, edgecolor = _HISTOGRAM_EDGE_COLOR,
    )

    # [TODO]: decide if we want to compute this here or not:#

    # ax.plot(
    #     gaussian_x,
    #     norm.pdf(gaussian_x, gaussian_mean, gaussian_std),
    #     color = _GAUSSIAN_LINE_COLOR, linestyle = _GAUSSIAN_LINE_STYLE,
    #     label = (
    #         rf"Gaussian Fit: $\mu = {gaussian_mean:.3f}$, "
    #         rf"$\sigma = {gaussian_std:.3f}$"
    #     ),
    # )

    ax.axvline(
        km15_value,
        color = _VERTICAL_LINE_COLOR, linestyle = _VERTICAL_LINE_STYLE, linewidth = _VERTICAL_LINE_THICKNESS,
        label = f"KM15: {km15_value:.3f}",
    )

    ax.set_xlabel(cff_label, fontsize = 15)
    ax.set_ylabel("Frequency", fontsize = 15)

    ax.set_title(
        rf"(Set {good_kinematic_set}) {cff_label} Distribution, "
        f"{kinematic_title}\n"
        f"(KM15): {km15_cff_string}",
        fontsize = 16,
    )

    ax.legend(fontsize = 16)

    tag_figure_with_date(ax)
    save_matplotlib_figure(fig, "./", "blah")


# names of the various things you can put into a log message:
# https://docs.python.org/3/library/logging.html#logrecord-attributes
