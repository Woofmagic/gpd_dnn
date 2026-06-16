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

import glob
import datetime

import tensorflow as tf
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
# Scratch path
#################################################################################

SCRATCH_PATH = 'placeholder!'

#################################################################################
# Retreive the proper datafiles:
#################################################################################

# find replicas:
replica_paths = sorted(
    glob.glob(
        f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/replicas/replica_*_v{MAJOR_MINOR_NUMBER}.keras"))

replicas = [tf.keras.models.load_model(
    path,
    compile = False,
    safe_mode = False) for path in replica_paths]

print(f"[INFO]: Loaded {len(replicas)} replica models.")

test_dataframe = pd.read_csv(
    f'{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/surrogate_cross_section_replica_average_v{MAJOR_MINOR_NUMBER}')

grouped = test_dataframe.groupby(['t', 'x_b', 'q_squared'])

average_prediction = test_dataframe['mean_predicted_cross_section']
standard_dev_prediction = test_dataframe['std_predicted_cross_section']

# this is trento convention: -pi to pi:
phi_smooth = np.linspace(-np.pi, np.pi, 361)
special_phis = [0, np.pi/2., -np.pi/2., np.pi]

for (t_value, xb_value, qsquared_value), group in grouped:
    print(f"[INFO]: Processing t = {t_value}, xb = {xb_value}, Q2 = {qsquared_value}")

    group = group.sort_values('phi')

    xsec_err = group['unp_beam_unp_target_xsec_err'].values

    indices = group.index.values

    xsec_pred = average_prediction[indices, 0]
    xsec_std = standard_dev_prediction[indices, 0]

    x_smooth = np.column_stack([
        np.full_like(phi_smooth, t_value),
        np.full_like(phi_smooth, xb_value),
        np.full_like(phi_smooth, qsquared_value),
        phi_smooth
    ])

    smooth_preds_all = np.array([ model.predict(x_smooth, verbose = 0) for model in replicas ])

    smooth_mean = np.mean(smooth_preds_all, axis = 0)
    smooth_std = np.std(smooth_preds_all, axis = 0)

    xsec_smooth_mean = smooth_mean[:, 0]
    xsec_smooth_std = smooth_std[:, 0]

    for phi_target in special_phis:
        phi_index = np.argmin(np.abs(phi_smooth - phi_target))
        phi_actual = phi_smooth[phi_index]
        sigma_value = xsec_smooth_std[phi_index]
        print(
            f"[INFO]: "
            f"(xb = {xb_value:.3f}, t = {t_value:.3f}, Q2 = {qsquared_value:.3f}) "
            f"cross-section 1σ uncertainty at phi = {phi_actual:.3f} rad is ±{sigma_value:.6f}"
        )

    phi = group['phi'].values
    xsec_actual = group['unp_beam_unp_target_xsec'].values

    xsec_res = xsec_actual - xsec_pred
    chi2_xsec = np.sum(xsec_res**2) / len(phi)

    residuals_figure, axes = plt.subplots(2, 1, figsize = (10, 8), sharex = 'col', layout = "tight")

    axes[1].text(
        -0.1, -0.1,
        fr"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", 
        transform = axes[1].transAxes)

    axes[0].plot(phi_smooth, xsec_smooth_mean, color = 'red', lw = 2, label = rf'Replica Average ($N = {NUMBER_OF_REPLICAS}$)')
    axes[0].fill_between(
        phi_smooth, xsec_smooth_mean - xsec_smooth_std, xsec_smooth_mean + xsec_smooth_std,
        color = 'red', alpha = 0.3,
        label = r'$\sigma$ band')
    
    axes[0].errorbar(
        phi, xsec_actual, yerr = xsec_err,
        fmt = 'o', mfc = 'white', mec = 'black', ms = 5, ecolor = 'black', elinewidth = 1, capsize = 2, alpha = 0.8,
        label = 'Experimental Data')
    axes[0].set_ylabel(r"$d^{4}\sigma$ [nb / GeV$^{4}$]", fontsize = 16.)
    axes[0].set_title(rf"Cross Section ($\chi^2_\nu = {chi2_xsec:.4f}$)", fontsize = 18.)
    axes[0].legend(fontsize = 14.)
    axes[0].grid(True, linestyle = ':', alpha = 0.6)

    axes[1].scatter(phi, xsec_res, color = 'blue', alpha = 0.6)
    axes[1].axhline(0, color = 'black', linestyle = '--')
    axes[1].set_title("Residuals", fontsize = 18.)
    axes[1].grid(True, linestyle = ':', alpha = 0.6)
    
    residuals_figure.suptitle(
        "Kinematic Setting:\n"
        rf"$t = {t_value:.3f}$, $x_\textrm{{B}} = {xb_value:.3f}$, $Q^2 = {qsquared_value:.3f}$",
        fontsize = 16
    )

    filename = f"{SCRATCH_PATH}/hpc/version_{MAJOR_MINOR_NUMBER}/plots/t{t_value:.3f}_xb{xb_value:.3f}_q2{qsquared_value:.3f}_residuals_v{MAJOR_MINOR_NUMBER}"
    
    for extension in ['png', 'eps']:
        residuals_figure.savefig(
            fname = f"{filename}.{extension}",
            facecolor = 'white', transparent = False)

    plt.close(residuals_figure)