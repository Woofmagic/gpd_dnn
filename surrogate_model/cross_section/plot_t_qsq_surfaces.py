#################################################################################
# FILE INFORMATION:
# Purpose: makes surface plots of DNN surrogate fits across xb and Q^{2}
# Created: 20260603
# Last changed: 20260603
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#################################################################################
# Matplotlib Plotting Customizability
#################################################################################

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 3.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 8.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 3.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['savefig.dpi'] = 300

#################################################################################
# Version numbers!
#################################################################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

#################################################################################
# Retreive the proper datafiles:
#################################################################################

xb_q2_groups = test_dataframe.groupby(['x_b', 'q_squared'])
number_of_xb_qsquared_settings = xb_q2_groups.ngroups

print(f"[INFO]: preparing to make {number_of_xb_qsquared_settings} plot")

phi_grid = np.linspace(-np.pi, np.pi, 361)

for (xb_value, qsquared_value), group in xb_q2_groups:

    group = group.sort_values(['t', 'phi'])

    t_values = np.sort(group['t'].unique())

    phi_meshgrid, t_meshgrid = np.meshgrid(phi_grid, t_values)

    phi_data = group['phi'].values
    t_data = group['t'].values

    indices = group.index.values

    xsec_pred = average_prediction[indices, 0]

    xsec_actual = group['unp_beam_unp_target_xsec'].values
    bsa_actual = group['unp_target_bsa'].values

    xsec_res = xsec_actual - xsec_pred

    colors_xsec = np.where(xsec_res >= 0, 'red', 'blue')

    model_surface_input = np.column_stack([
        t_meshgrid.ravel(),
        np.full(t_meshgrid.size, xb_value),
        np.full(t_meshgrid.size, qsquared_value),
        phi_meshgrid.ravel()
    ])

    surface_preds_all = np.array([
        model.predict(model_surface_input) for model in models
    ])

    surface_mean = np.mean(surface_preds_all, axis = 0)
    surface_std_dev = np.std(surface_preds_all, axis = 0)

    xsec_surface = surface_mean[:, 0].reshape(t_meshgrid.shape)

    xsec_stddev_surface = surface_std_dev[:, 0].reshape(t_meshgrid.shape)

    zero_plane_xsec = np.zeros_like(xsec_surface)

    fig = plt.figure(figsize = (14, 7), layout = "tight")

    ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
    ax2 = fig.add_subplot(1, 2, 2, projection = '3d')

    # [NOTE]: this order actually determines some z-ordering stuff...
    ax1.plot_surface(
        phi_meshgrid, t_meshgrid, xsec_surface + xsec_stddev_surface,
        color = "gray", alpha = 0.30)
    ax1.plot_surface(
        phi_meshgrid, t_meshgrid, xsec_surface - xsec_stddev_surface,
        color = "gray", alpha = 0.30)
    ax1.plot_surface(
        phi_meshgrid, t_meshgrid, xsec_surface,
        cmap = 'viridis', alpha = 0.30)
    ax1.scatter(
        phi_data, t_data, xsec_actual, 
        facecolors = 'white', edgecolors = 'black', s = 20, linewidths = 0.5, alpha = 1.0)

    ax1.set_xlabel(r'$\phi$ [Radians]',
                   labelpad = 16, fontsize = 16.)
    ax1.set_ylabel(r'$t$ [GeV$^{2}$]',
                   labelpad = 16, fontsize = 16.)
    ax1.set_zlabel(r'$d^{4}\sigma^{UU}$ [nb GeV$^{-4}$]',
                   labelpad = 7, fontsize = 16.)
    ax1.set_title('Cross Section', fontsize = 18.)

    ax2.plot_surface(phi_meshgrid, t_meshgrid, zero_plane_xsec, color = 'gray', alpha = 0.15)
    ax2.scatter(phi_data, t_data, xsec_res, color = colors_xsec, s = 20)
 
    ax2.set_xlabel(r'$\phi$ [Radians]',
                   labelpad = 16, fontsize = 16.)
    ax2.set_ylabel(r'$t$ [GeV$^{2}$] ',
                   labelpad = 16, fontsize = 16.)
    ax2.set_zlabel('Residuals',
                   labelpad = 7, fontsize = 16.)
    ax2.set_title('Cross Section Residuals', fontsize = 18)

    fig.suptitle(
        r"DNN Interpolations Across $t$ and $\phi$"
        "\n"
        rf"Kinematic Setting: $x_\textrm{{B}} = {xb_value:.4g}$, $Q^2 = {qsquared_value:.4g}$ GeV$^{{2}}$",
        fontsize = 16)

    plot_filename = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/surface_xb{xb_value:.4g}_q2{qsquared_value:.4g}_v{MAJOR_MINOR_NUMBER}"
    
    for extension in ['png', 'eps']:
        fig.savefig(f"{plot_filename}.{extension}", facecolor = 'white')

    plt.close(fig)

    # cleanup:
    del fig
    del ax1
    del ax2

#################################################################################
# Script finishes:
#################################################################################

print("[INFO]: Processing complete!")
