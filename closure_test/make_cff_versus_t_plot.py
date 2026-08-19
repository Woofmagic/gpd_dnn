"""
Makes a plot of a given CFF versus |t|.
Created: 20260818
Last changed: 20260818
Notes:
    1. 2026/08/18: Using np.isclose() because of floating point
    issues, as in t = -0.5 and t = -0.50000000001.
"""

import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_cff_plot_label(
        cff_label: str
    ):
    """
    I am going to customize LaTeX representation of a given CFF
    based on the string version that I received from a datafile.
    """

    # why does this work? because the variable is a STRING
    real_or_imag = cff_label[:2]
    cff_name = cff_label[2:]

    component = { "Re": "Re", "Im": "Im" }[real_or_imag]

    cff_symbol = {
        "H": r"\mathcal{H}",
        "Ht": r"\widetilde{\mathcal{H}}",
        "E": r"\mathcal{E}",
        "Et": r"\widetilde{\mathcal{E}}",
    }[cff_name]

    return rf"{component}$[{cff_symbol}]$"

def configure_matplotlib(
        verbose: bool = False
    ):
    """
    Let me choose the Matplotlib aesthetic parameters that I like
    the most to construct all my plots.
    """

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "savefig.dpi": 300,
        "axes.labelsize": 16,
        "xtick.direction": "in",
        "xtick.major.size": 8.5,
        "xtick.major.width": 1.0,
        "xtick.minor.size": 4.5,
        "xtick.minor.width": 1.0,
        "xtick.minor.visible": True,
        "xtick.top": True,
        "ytick.direction": "in",
        "ytick.major.size": 8.5,
        "ytick.major.width": 1.0,
        "ytick.minor.size": 4.5,
        "ytick.minor.width": 1.0,
        "ytick.minor.visible": True,
        "ytick.right": True,
        "xtick.labelsize": 15.0,
        "ytick.labelsize": 15.0,
    })

    if verbose:
        print("[VERBOSE]: I globally updated Matplotlib's rcParams.")

def load_cff_total_fitting_data(
        version_number: str,
        verbose: bool = False,
    ):
    """
    I need to be able to read the .csv files in order to get the relevant data
    to make the historgrams.
    """

    if verbose:
        print(f"[VERBOSE]: I'm now reading the relevant datafiles for experiment v{version_number}")

    data_directory = f"./hpc/version_{version_number}"
    cff_fitting_data = pd.read_csv(f"{data_directory}/cff_summary_statistics_v{version_number}.csv")

    return cff_fitting_data

def inspect_t_lines(
        dataframe: pd.DataFrame,
        cff_label: str | None = None,
        verbose: bool = False
    ):

    if verbose:
        print(
            "[VERBOSE]: I'm taking a look at what t-trajectories we can make based on the "
            "data you sent me"
            )

    data = dataframe.copy()

    if cff_label is not None:
        data = data[data["cff"] == cff_label]

    t_line_candidates = (
        data
        .groupby(["k", "xb", "q_squared"])
        .agg(
            number_of_t_points = ("t", "nunique"),
            t_min = ("t", "min"),
            t_max = ("t", "max"),
        )
        .reset_index()
    )

    t_line_candidates = t_line_candidates[t_line_candidates["number_of_t_points"] > 1]

    return t_line_candidates.sort_values(
        ["k", "xb", "q_squared"]
    )

def plot_cff_vs_t(
        dataframe: pd.DataFrame,
        cff_label: str,
        fixed_k: float,
        fixed_xb: float,
        fixed_q_squared: float,
        verbose: bool = False
    ):
    """
    I make the actual CFF vs. -t plot here. It requires that I know which CFF
    I'm plotting, as well as the fixed values of xB and Q^2.
    """

    this_kinematic_set_title_string = (
        rf"$k = {fixed_k:.3f}$ GeV, "
        rf"$x_B = {fixed_xb:.3f}$, "
        rf"$Q^2 = {fixed_q_squared:.3f}$ GeV$^2$"
    )

    data = dataframe[
        (dataframe["cff"] == cff_label)
        & np.isclose(dataframe["k"], fixed_k)
        & np.isclose(dataframe["xb"], fixed_xb)
        & np.isclose(dataframe["q_squared"], fixed_q_squared)
    ].copy()

    data = data.sort_values("t")

    if data.empty:
        raise ValueError(
            f"No data found for {cff_label} with "
            f"k = {fixed_k}, xb = {fixed_xb}, Q^2 = {fixed_q_squared}."
        )

    if data["t"].nunique() < 2:
        raise ValueError(
            f"Only {data['t'].nunique()} unique t value(s) found. "
            f"At least two are required for a line plot."
        )

    figure, axis = plt.subplots(1, 1, figsize = (10, 8))

    # [NOTE]: multiplied t by -1:
    axis.errorbar(
        -1.0 * data["t"],
        data["mean"],
        yerr = data["stddev"],
        fmt = "o-",
        capsize = 4.,
        label = "DNN prediction")

    # [NOTE]: multiplied t by -1:
    axis.plot(
        -1.0 * data["t"],
        data["km15"],
        linestyle = "--",
        marker = "s",
        label = "KM15 value")

    axis.set_xlabel(
        r"$-t\;[\mathrm{GeV}^2]$",
        fontsize = 16.)

    axis.set_ylabel(
        make_cff_plot_label(cff_label),
        rotation = 90.,
        fontsize = 16.)

    axis.set_title(
        rf"{make_cff_plot_label(cff_label)} vs. $t$"
        "\n"
        rf"{this_kinematic_set_title_string}",
        fontsize = 16.)

    axis.legend(fontsize = 16.0)

    axis.grid(True, alpha = 0.50)

    figure.tight_layout()

    if verbose:
        print("[VERBOSE]: I prepared the figure. Returning data now...")

    return figure, axis

def plot_all_t_trends(
        version_number: int,
        dataframe: pd.DataFrame,
        cff_label: str,
        verbose: bool = False
    ):

    t_line_candidates = inspect_t_lines(
        dataframe = dataframe,
        cff_label = "ImHt",
        verbose = verbose
    )

    if verbose:
        print(
            f"[VERBOSE]: I found {len(t_line_candidates)} possible t-trends for "
            f"{cff_label} that we can plot"
        )

    for _, candidate in t_line_candidates.iterrows():

        fixed_k = candidate["k"]
        fixed_xb = candidate["xb"]
        fixed_q_squared = candidate["q_squared"]

        if verbose:
            print(
                f"[VERBOSE]: I'm ready to plot {cff_label} vs. t with k = {fixed_k:.6f}, xb = {fixed_xb:.6f}, "
                f"Q^2 = {fixed_q_squared:.6f}, and using {candidate['number_of_t_points']} points."
            )

        cff_figure, cff_axis = plot_cff_vs_t(
            dataframe = dataframe,
            cff_label = cff_label,
            fixed_k = fixed_k,
            fixed_xb = fixed_xb,
            fixed_q_squared = fixed_q_squared,
            verbose = verbose
        )

        # [NOTE]: You must ensure this :4f business does not lead to
        # multiple filesnames that are the SAME. 4 figures should be 
        # good enough to avoid this degeneracy, but just keep this 
        # design decision in mind!
        output_filename = (
            f"cff_{cff_label}_vs_t_k_{fixed_k:.4f}"
            f"_xb_{fixed_xb:.4f}_q2_{fixed_q_squared:.4f}"
            )

        for extension in ["png", "eps"]:
            cff_figure.savefig(
                f"./hpc/version_{version_number}/{output_filename}.{extension}",
                facecolor = "white",
                transparent = False
            )

        plt.close(cff_figure)

def main(
    version_number: int,
    verbose: bool = False
    ):

    if verbose:
        print("[VERBOSE]: First, I'm going to set up my Matplotlib aethestics...")

    configure_matplotlib(verbose)

    cff_fitting_dataframe = load_cff_total_fitting_data(
        version_number = version_number,
        verbose = verbose
        )

    plot_all_t_trends(
        version_number = version_number,
        dataframe = cff_fitting_dataframe,
        cff_label = "ImHt",
        verbose = verbose
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = (
            "We're going to make a plot of the CFF that we just fit across the "
            "-t variable while *fixing* both xB and Q^2. This requires that we know "
            "how the CFF fit across many different values of t."
        )
    )

    parser.add_argument(
        "-ver",
        "--version-number",
        type = str,
        required = True,
        help = "What is the version number of this plot?"
    )

    parser.add_argument(
        '-v',
        '--verbose',
        action = "store_true",
        required = False,
        default = False,
        help = (
            "Do you want me to print out a bunch of debugging "
            "statements? Careful what you wish for!"
        )
    )

    arguments = parser.parse_args()

    print("[INFO]: Script began running!")

    main(
        version_number = arguments.version_number,
        verbose = arguments.verbose
        )

    print("[INFO]: End of script reached!")
