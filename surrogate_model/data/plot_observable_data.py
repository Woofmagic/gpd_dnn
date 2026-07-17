#################################################################################
# FILE INFORMATION:
# Purpose: Makes a million plots of the surrogate-derived data for simult. fits.
# Created: 20260708
# Last changed: 20260708
# Notes:
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#################################################################################
# Version numbers!
#################################################################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

#################################################################################
# obtain the path:
#################################################################################

output_directory = Path.cwd()

#################################################################################
# load the datafile:
#################################################################################

surrogate_dataframe = pd.read_csv(
    output_directory
    / f"version_{MAJOR_MINOR_NUMBER}"
    / "data"
    / f"surrogate_observable_data_v{MAJOR_MINOR_NUMBER}.csv"
)

#################################################################################
# Making plots of the surrogate data...
#################################################################################

grouped = surrogate_dataframe.groupby(['k', 't', 'x_b', 'q_squared'])

for (k_value, t_value, xb_value, qsquared_value), group in grouped:
    print(f"[INFO]: Processing k = {k_value}, t = {t_value}, xb = {xb_value}, Q2 = {qsquared_value}")

    group = group.sort_values("phi")
    phi = group["phi"].to_numpy()

    xsec = group["unp_beam_unp_target_xsec"].to_numpy()
    xsec_std = group["unp_beam_unp_target_xsec_std"].to_numpy()
    bsa = group["unp_target_bsa"].to_numpy()
    bsa_std = group["unp_target_bsa_std"].to_numpy()

    surrogate_data_figure, surrogate_data_axes = plt.subplots(2, 1, figsize = (10, 8), sharex = 'col', layout = "tight")

    surrogate_data_axes[1].text(
        -0.1, -0.1,
        fr"Figure rendered {datetime.datetime.now().strftime('%y%m%d-%H%M%S')}", 
        transform = surrogate_data_axes[1].transAxes)

    surrogate_data_axes[0].scatter(phi, bsa, s = 5.0, color = 'red', label = 'Surrogate Model BSA Data')
    surrogate_data_axes[1].scatter(phi, xsec, s = 5.0, color = 'red', label = 'Surrogate Model Cross-Section Data')
    surrogate_data_axes[1].set_xlabel(r"$\phi$ (radians)", fontsize = 16.)
    surrogate_data_axes[0].set_ylabel(r"BSA", fontsize = 16.)
    surrogate_data_axes[1].set_ylabel(r"$d^{4}\sigma^{UU}$ $(nb/GeV^{-4})$", fontsize = 16.)
    surrogate_data_axes[0].set_title(
        "Surrogate-Generated Data at Kinematic Setting:\n"
        rf"$k = {k_value}$, $t = {t_value}$, $x_\mathrm{{B}} = {xb_value}$, $Q^2 = {qsquared_value}$", 
        fontsize = 18.)
    surrogate_data_axes[0].legend(fontsize = 14.)
    surrogate_data_axes[1].legend(fontsize = 14.)
    surrogate_data_axes[0].grid(True, linestyle = ':', alpha = 0.6)
    surrogate_data_axes[1].grid(True, linestyle = ':', alpha = 0.6)

    filename = (
        output_directory /
        f"version_{MAJOR_MINOR_NUMBER}" / 
        "plots" / 
        f"surrogate_bsa_k{k_value:.3f}_t{t_value:.3f}_xb{xb_value:.3f}_q2{qsquared_value:.3f}"
    )

    for extension in ["png", "eps"]:
        surrogate_data_figure.savefig(
            f"{filename}.{extension}",
            facecolor = "white", transparent = False)

    plt.close(surrogate_data_figure)
