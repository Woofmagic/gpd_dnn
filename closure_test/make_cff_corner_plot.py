"""
Makes a 2n-by-2n CFF corner plot (correlations).
Created: 20260817
Last changed: 20260819
"""

import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner

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

def make_cff_corner_plot(
    version_number,
    kinematic_set_number,
    cff_statistics_datafile,
    verbose: bool = False):
    """
    I'm now prepared to iterate through each of the predicted CFFs
    and figure out their correlations through the construction of a corner plot.
    """

    _CFF_LABELS = (
        "ReH", "ImH",
        "ReHt", "ImHt",
        "ReE", "ImE",
        "ReEt", "ImEt",
    )
    _HISTOGRAM_TITLE_FONTSIZE = 16.
    _HISTOGRAM_LABEL_FONTSIZE = 16.
    _HISTOGRAM_FACECOLOR = "skyblue"
    _HISTOGRAM_EDGECOLOR = "black"

    cff_samples = []
    cff_labels = []

    for cff_label in _CFF_LABELS:

        statistics_key = f"{cff_label}_pred"

        if statistics_key not in cff_statistics_datafile.columns:
            continue

        if verbose:
            print(f"[VERBOSE]: I'm now analyzing CFF {cff_label} using column '{statistics_key}'")

        cff_samples.append(cff_statistics_datafile[statistics_key].to_numpy())
        cff_labels.append(make_cff_plot_label(cff_label))

    if not cff_samples:

        print(
            f"[WARN]: No CFF predictions for kinematic set #{kinematic_set_number}. "
            "Skipping corner plot..."
        )

        return

    cff_samples_combined = np.column_stack(cff_samples)

    if verbose:
        print(
            f"[VERBOSE]: This plot will contain {cff_samples_combined.shape[1]} CFF dimensions "
            f"and {cff_samples_combined.shape[0]} replicas."
        )

    corner_fig = corner.corner(
        cff_samples_combined,
        labels = cff_labels,
        show_titles = True,
        label_kwargs = {
            "fontsize": _HISTOGRAM_LABEL_FONTSIZE
            },
        title_fmt = ".3f",
        title_kwargs = {
            "fontsize": _HISTOGRAM_TITLE_FONTSIZE
            },
        hist_kwargs = {
            "fc": _HISTOGRAM_FACECOLOR, # facecolor
            "ec": _HISTOGRAM_EDGECOLOR # edgecolor
        })

    corner_fig.text(
        0.750, 0.900,
        f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        transform = corner_fig.transFigure)

    base_path = Path(
        f"./hpc/version_{version_number}/kinematic_set_{kinematic_set_number}"
    )

    plot_directory = base_path / "plots"
    plot_directory.mkdir(parents = True, exist_ok = True)

    for extension in ["png", "eps"]:
        corner_fig.savefig(
            plot_directory / f"cff_corner_v{version_number}.{extension}",
            facecolor = "white", 
            transparent = False
        )

    plt.close(corner_fig)

def main(
    version_number: int,
    kinematic_sets: list | None = None,
    verbose: bool = False):

    if kinematic_sets is None:
        kinematic_sets = find_kinematic_sets(
            version_number,
            verbose)

    if verbose:
        print(f"[VERBOSE]: I found {len(kinematic_sets)} kinematic sets: {kinematic_sets}")

    for kinematic_set in kinematic_sets:
    
        if verbose:
            print(f"[VERBOSE]: I'm now analyzing kinematic set #{kinematic_set}")

        _, cff_statistics = (
            load_kinematic_set_data(
                version_number,
                kinematic_set,
                verbose
            )
        )
        
        make_cff_corner_plot(
            version_number = version_number,
            kinematic_set_number = kinematic_set,
            cff_statistics_datafile = cff_statistics,
            verbose = verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = (
            "We'll produce a corner plot that shows how all of the CFFs "
            "that were free during the simultaneous fit correlate with each other."
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
