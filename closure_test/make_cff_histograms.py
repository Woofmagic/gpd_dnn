"""
Makes a frequency distribution (histogram) for the number of 
freely-fit CFFs for a given range of kinematic set numbers.
Created: 20260817
Last changed: 20260818
Notes:
"""

import argparse
import datetime
import os

from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def find_kinematic_sets(
        version_number: str,
        verbose: bool = False,
    ):
    """
    I need to look through the local directory structure to 
    determine which kinematic sets await the generation of
    CFF histograms!
    """

    if verbose:
        print(
            "[VERBOSE]: I'm now looking through the various kinematic settings "
            f"for experiment v{version_number}"
        )

    version_directory = f"./hpc/version_{version_number}"

    kinematic_sets = []

    for subdirectory in os.listdir(version_directory):

        if not subdirectory.startswith("kinematic_set_"):
            continue

        # extracting the integer of the kinematic set:
        kinematic_set_number = int(subdirectory.replace("kinematic_set_", ""))

        data_directory = f"{version_directory}/kinematic_set_{kinematic_set_number}/data"
        observable_file = f"{data_directory}/observable_preds_v{version_number}.csv"
        cff_file = f"{data_directory}/cff_replica_average_preds_v{version_number}.csv"

        if os.path.exists(observable_file) and os.path.exists(cff_file):
            kinematic_sets.append(kinematic_set_number)

    return sorted(kinematic_sets)

def load_kinematic_set_data(
        version_number: str,
        kinematic_set_number: int,
        verbose: bool = False,
    ):
    """
    I need to be able to read the .csv files in order to get the relevant data
    to make the historgrams.
    """

    if verbose:
        print(f"[VERBOSE]: I'm now reading the relevant datafiles for experiment v{version_number}")

    data_directory = f"./hpc/version_{version_number}/kinematic_set_{kinematic_set_number}/data"
    observable_statistics = pd.read_csv(f"{data_directory}/observable_preds_v{version_number}.csv")
    cff_statistics = pd.read_csv(f"{data_directory}/cff_replica_average_preds_v{version_number}.csv")

    return observable_statistics, cff_statistics

def extract_kinematic_information(
        observable_statistics
    ):
    """
    All I need to do here is to extract the *kinematic setting*, which is
    just the combination of (k, x_{B}, t, Q^{2}). So, I need to just read
    the datafile and then, *because I presume the datafile has fixed values*
    of the kinematics in its columns, I just take the first entry.
    """

    beam_energy = observable_statistics["k"].iloc[0]
    x_bjorken = observable_statistics["xb"].iloc[0]
    t_value = observable_statistics["t"].iloc[0]
    q_squared = observable_statistics["q_squared"].iloc[0]

    return {
        "k": beam_energy,
        "xb": x_bjorken,
        "t": t_value,
        "q_squared": q_squared,
    }

def extract_km15_values(
        observable_statistics
    ):
    """
    In order for me to display the KM15 prediction on the histogram, I need
    to extract its value from the datafile that you are supplying me.
    """

    return {
        "ReH": observable_statistics["Re[H]"].iloc[0],
        "ImH": observable_statistics["Im[H]"].iloc[0],
        "ReE": observable_statistics["Re[E]"].iloc[0],
        "ImE": observable_statistics["Im[E]"].iloc[0],
        "ReHt": observable_statistics["Re[Ht]"].iloc[0],
        "ImHt": observable_statistics["Im[Ht]"].iloc[0],
        "ReEt": observable_statistics["Re[Et]"].iloc[0],
        "ImEt": observable_statistics["Im[Et]"].iloc[0],
    }

def make_km15_cff_string(
        km15_values
    ):
    """
    If we want to show the KM15-predicted values at the top of the histogram, 
    then I need to compute a string that does this.
    """

    cff_h_km15 = complex(km15_values["ReH"],km15_values["ImH"])
    cff_e_km15 = complex(km15_values["ReE"],km15_values["ImE"])
    cff_ht_km15 = complex(km15_values["ReHt"],km15_values["ImHt"])
    cff_et_km15 = complex(km15_values["ReEt"],km15_values["ImEt"])

    return (
        rf"$\mathcal{{H}} = {cff_h_km15:.3f}$, "
        rf"$\mathcal{{E}} = {cff_e_km15:.3f}$, "
        rf"$\widetilde{{\mathcal{{H}}}} = {cff_ht_km15:.3f}$, "
        rf"$\widetilde{{\mathcal{{E}}}} = {cff_et_km15:.3f}$"
    )

def make_kinematic_title_string(
        kinematics
    ):
    """
    If I want to show the fixed kinematic bin at the top of the histogram,
    then I need to use the values from the datafile you provided me and
    compute a string representation of them.
    """

    return (
        rf"$k = {kinematics['k']:.3f}$ GeV, "
        rf"$x_B = {kinematics['xb']:.3f}$, "
        rf"$t = {kinematics['t']:.3f}$ GeV$^2$, "
        rf"$Q^2 = {kinematics['q_squared']:.3f}$ GeV$^2$"
    )

def make_cff_histogram(
    version_number: int,
    cff_label: str,
    kinematic_set: int,
    cff_statistics: pd.DataFrame,
    kinematics: dict,
    km15_values: dict,
    km15_cff_string: str,
    verbose: bool = False):
    """
    I make the CFF histogram here! Presuming all of the subroutines finished successfully,
    then I have all the data needed to spit out a histogram.
    """
    
    _NUMBER_OF_STDDEVS = 4.
    _NUMBER_OF_HISTOGRAM_BINS = 30
    _NUMBER_OF_GAUSSIAN_POINTS = 200

    corresponding_key = f"{cff_label}_pred"

    cff_prediction_per_replica = cff_statistics[corresponding_key]

    cff_mean, cff_stddev = norm.fit(cff_prediction_per_replica)

    cff_km15_value = km15_values[cff_label]

    gaussian_x_values = np.linspace(
        cff_mean - _NUMBER_OF_STDDEVS * cff_stddev,
        cff_mean + _NUMBER_OF_STDDEVS * cff_stddev,
        _NUMBER_OF_GAUSSIAN_POINTS
    )

    kinematic_title = make_kinematic_title_string(kinematics)

    if verbose:
        print(f"[VERBOSE]: I'm now starting to make a histogram for {corresponding_key}")

    cff_figure, cff_axis = plt.subplots(1, 1, figsize = (10, 8))

    cff_axis.hist(
        cff_prediction_per_replica,
        bins = _NUMBER_OF_HISTOGRAM_BINS,
        alpha = 0.6,
        color = "skyblue",
        edgecolor = "black")

    cff_axis.plot(
        gaussian_x_values,
        norm.pdf(
            gaussian_x_values,
            cff_mean,
            cff_stddev
        ),
        color = "red",
        linestyle = "--",
        label = (
            fr"Gaussian Fit: $\mu = {cff_mean:.3f}$, $\sigma = {cff_stddev:.3f}$"
        )
    )

    cff_axis.axvline(
        cff_km15_value,
        color = "green",
        linestyle = "-",
        linewidth = 2.0,
        label = f"KM15: {cff_km15_value:.3f}")

    cff_axis.set_ylabel(
        "Frequency",
        rotation = 90.,
        fontsize = 16.0)

    cff_axis.set_xlabel(
        make_cff_plot_label(cff_label),
        fontsize = 16.0)

    cff_axis.set_title(
        rf"(Set {kinematic_set}) {make_cff_plot_label(cff_label)} Distribution, "
        rf"{kinematic_title}"
        "\n"
        rf"(KM15): {km15_cff_string}",
        fontsize = 16.0)

    cff_axis.legend(fontsize = 16.0)

    cff_axis.text(
        0.00, -0.05,
        f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        transform = cff_axis.transAxes)

    plt.tight_layout()

    os.makedirs(
        f"./hpc/version_{version_number}/kinematic_set_{kinematic_set}/plots", 
        exist_ok = True)

    for extension in ["png", "eps"]:
        cff_figure.savefig(
            f"./hpc/version_{version_number}/kinematic_set_{kinematic_set}/plots/"
            f"cff_{cff_label}_fits_v{version_number}.{extension}",
            facecolor = "white",
            transparent = False
        )

    plt.close(cff_figure)

    if verbose:
        print("[VERBOSE]: All done saving the histogram!")

    return {
        "kinematic_set": kinematic_set,
        "k": kinematics["k"],
        "xb": kinematics["xb"],
        "t": kinematics["t"],
        "q_squared": kinematics["q_squared"],
        "cff": cff_label,
        "mean": cff_mean,
        "stddev": cff_stddev,
        "km15": cff_km15_value,
    }

def main(
    version_number: int = 1,
    kinematic_sets: list | None = None,
    verbose: bool = False):

    if verbose:
        print("[VERBOSE]: First, I'm going to set up my Matplotlib aethestics...")

    configure_matplotlib()

    if kinematic_sets is None:
        kinematic_sets = find_kinematic_sets(
            version_number,
            verbose)

    if verbose:
        print(f"[VERBOSE]: I found {len(kinematic_sets)} kinematic sets: {kinematic_sets}")

    cff_fitting_results = []

    for kinematic_set in kinematic_sets:

        if verbose:
            print(f"[VERBOSE]: I'm now analyzing kinematic set #{kinematic_set}")

        observable_statistics, cff_statistics = (
            load_kinematic_set_data(
                version_number,
                kinematic_set,
                verbose
            )
        )

        if verbose:
            print(f"[VERBOSE]: I'm now going to extract the kinematics of kinematic set #{kinematic_set}...")

        kinematics = extract_kinematic_information(observable_statistics)

        if verbose:
            print("[VERBOSE]: I'm now looking at the KM15 values at this kinematic setting...")

        km15_values = extract_km15_values(observable_statistics)

        if verbose:
            print("[VERBOSE]: I'm now computing a string with the KM15 CFF values for use in plotting...")

        km15_cff_string = make_km15_cff_string(km15_values)

        for cff_label in km15_values:
            if verbose:
                print(f"[VERBOSE]: I'm starting to make a histogram for CFF {cff_label}...")
                
            corresponding_key = f"{cff_label}_pred"

            if corresponding_key not in cff_statistics.columns:
                if verbose:
                    print(f"[VERBOSE]: No {cff_label} prediction for kinematic set #{kinematic_set}. I'm skipping this one...")
                continue

            record = make_cff_histogram(
                version_number = version_number,
                cff_label = cff_label,
                kinematic_set = kinematic_set,
                cff_statistics = cff_statistics,
                kinematics = kinematics,
                km15_values = km15_values,
                km15_cff_string = km15_cff_string,
                verbose = verbose)

            cff_fitting_results.append(record)

            if verbose:
                print("[VERBOSE]: Appended the relevant CFF data to the array!")

    cff_summary_statistics = pd.DataFrame(cff_fitting_results)

    summary_filename = f"./hpc/version_{version_number}/cff_summary_statistics_v{version_number}.csv"

    cff_summary_statistics.to_csv(
        summary_filename,
        index = False
    )

    if verbose:
        print(f"[VERBOSE]: I wrote a summary of the CFF data to: {summary_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = (
            "We'll produce a frequentist statistical distribution of a given "
            "CFF and fit it with a Gaussian to extract its mean and variance."
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
        "-kin-set",
        "--kinematic_sets",
        nargs = "+",
        type = int,
        required = False,
        default = None,
        help = (
            "I need to know the number of the kinematic setting we are "
            "extracting CFF data from."
            )
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
        kinematic_sets = arguments.kinematic_sets,
        verbose = arguments.verbose
        )

    print("[INFO]: End of script reached!")
