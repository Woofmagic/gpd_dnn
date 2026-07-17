#################################################################################
# FILE INFORMATION:
# Purpose: makes surface plots of DNN surrogate fits across xb and Q^{2}
# Created: 20260702
# Last changed: 20260702
# Notes:
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

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

SCRATCH_PATH = Path("placeholder!")

#################################################################################
# TensorFlow model!
#################################################################################

class BSASurrogateModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.hidden_layers = [
            tf.keras.layers.Dense(
                128, activation = "silu", kernel_initializer = "glorot_normal"
            )
            for _ in range(4)
        ]

        # linear activation is default activation if `activation` key is not specified: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        self.bsa_output = tf.keras.layers.Dense(1)

    def call(self, x):

        # nothing fancy here!
        for layer in self.hidden_layers:
            x = layer(x)
        
        return self.bsa_output(x)
    
#################################################################################
# Useful functions for plotting
#################################################################################

# have to go through the intermediate transformations:
def predict_bsa(model, x_dataframe, x_scaler, y_scaler):
    
    x_scaled = x_scaler.transform(x_dataframe)
    model_prediction_in_z = model.predict(x_scaled, verbose = 0)
    bsa = y_scaler.inverse_transform(model_prediction_in_z)
    return bsa

#################################################################################
# This provides a *reference dataframe*
#################################################################################

reference_dataframe = pd.read_csv(
    SCRATCH_PATH /
    f"version_{MAJOR_MINOR_NUMBER}" /
    "data" /
    f"refined_bsa_data_v{MAJOR_MINOR_NUMBER}.csv")

# phi -> v(phi)
reference_dataframe["v"] = np.sin(reference_dataframe["phi"])

#################################################################################
# We need this for the transform/inverse-transform of the data:
#################################################################################

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_data = reference_dataframe[["k", "q_squared", "x_b", "t", "v"]]
# [NOTE]: we do NOT LOG THE BSA DATA!
y_data = reference_dataframe[["unp_target_bsa"]]

x_scaler.fit(x_data)
y_scaler.fit(y_data)

#################################################################################
# Retreive the proper datafiles:
#################################################################################

# find replicas:
replica_paths = sorted(
    (
        SCRATCH_PATH /
        f"version_{MAJOR_MINOR_NUMBER}" /
        "replicas"
    ).glob(f"replica_*_v{MAJOR_MINOR_NUMBER}.keras")
)

replicas = [tf.keras.models.load_model(
    path,
    custom_objects = {
            "BSASurrogateModel": BSASurrogateModel
        },
    compile = False,
    safe_mode = False) for path in replica_paths]

print(f"[INFO]: Loaded {len(replicas)} replica models.")

#################################################################################
# Grouping by setting *and* replica averaging!
#################################################################################

grouped = reference_dataframe.groupby(['k', 't', 'x_b', 'q_squared'])

# this is trento convention: -pi to pi:
phi_smooth = np.linspace(-np.pi, np.pi, 361)

# for printing stuff later:
special_phis = [0, np.pi/2., -np.pi/2., np.pi]

#################################################################################
# Making plots of the local fits
#################################################################################

for (k_value, t_value, xb_value, qsquared_value), group in grouped:
    print(f"[INFO]: Processing k = {k_value}, t = {t_value}, xb = {xb_value}, Q2 = {qsquared_value}")

    group = group.sort_values("phi")

    smooth_dataframe = pd.DataFrame({
        "k": np.full_like(phi_smooth, k_value),
        "q_squared": np.full_like(phi_smooth, qsquared_value),
        "x_b": np.full_like(phi_smooth,xb_value),
        "t": np.full_like(phi_smooth, t_value),
        "v": np.sin(phi_smooth)
    })

    smooth_predictions_all = np.array([
        predict_bsa(model, smooth_dataframe, x_scaler, y_scaler)
        for model in replicas
    ])

    smooth_mean = np.mean(smooth_predictions_all, axis = 0)
    smooth_std = np.std(smooth_predictions_all, axis = 0)

    bsa_smooth_mean = smooth_mean[:, 0]
    bsa_smooth_std = smooth_std[:, 0]

    point_dataframe = pd.DataFrame({
        "k": group["k"],
        "q_squared": group["q_squared"],
        "x_b": group["x_b"],
        "t": group["t"],
        "v": np.sin(group["phi"])
    })

    point_predictions_all = np.array([
        predict_bsa(model, point_dataframe, x_scaler, y_scaler)
        for model in replicas
    ])

    point_mean = np.mean(point_predictions_all, axis = 0)
    point_std = np.std(point_predictions_all, axis = 0)

    bsa_predicted = point_mean[:, 0]
    # not actually used:
    bsa_predicted_stddev = point_std[:, 0]

    # these are experimental values:
    phi = group["phi"].to_numpy()
    bsa_error = group["unp_target_bsa_err"].to_numpy()
    bsa_actual = group["unp_target_bsa"].to_numpy()

    pulls = (bsa_actual - bsa_predicted) / bsa_error

    chi_squared = np.sum(pulls**2)

    chi2_per_point = chi_squared / len(phi)

    bsa_residuals = bsa_actual - bsa_predicted

    residuals_figure, residuals_axes = plt.subplots(2, 1, figsize = (10, 8), sharex = 'col', layout = "tight")

    residuals_axes[1].text(
        -0.1, -0.1,
        fr"Figure rendered {datetime.datetime.now().strftime('%y%m%d-%H%M%S')}", 
        transform = residuals_axes[1].transAxes)

    residuals_axes[0].plot(phi_smooth, bsa_smooth_mean, color = 'red', lw = 2, label = rf'Replica Average ($N = {len(replicas)}$)')
    residuals_axes[0].fill_between(
        phi_smooth, bsa_smooth_mean - bsa_smooth_std, bsa_smooth_mean + bsa_smooth_std,
        color = 'red', alpha = 0.3,
        label = r'$\sigma$ band')

    residuals_axes[0].errorbar(
        phi, bsa_actual, yerr = bsa_error,
        fmt = 'o', mfc = 'white', mec = 'black', ms = 5, ecolor = 'black', elinewidth = 1, capsize = 2, alpha = 0.8,
        label = 'Experimental Data')
    residuals_axes[0].set_ylabel(r"BSA", fontsize = 16.)
    residuals_axes[0].set_title(rf"BSA ($\chi^2/N = {chi2_per_point:.7f}$)", fontsize = 18.)
    residuals_axes[0].legend(fontsize = 14.)
    residuals_axes[0].grid(True, linestyle = ':', alpha = 0.6)

    residuals_axes[1].scatter(phi, bsa_residuals, color = 'blue', alpha = 0.6)
    residuals_axes[1].axhline(0, color = 'black', linestyle = '--')
    residuals_axes[1].set_xlabel(r"$\phi$ (radians)", fontsize = 16.)
    residuals_axes[1].set_title("Residuals", fontsize = 18.)
    residuals_axes[1].grid(True, linestyle = ':', alpha = 0.6)

    residuals_figure.suptitle(
        "Kinematic Setting:\n"
        rf"$k = {k_value}$, $t = {t_value}$, $x_\mathrm{{B}} = {xb_value}$, $Q^2 = {qsquared_value}$",
        fontsize = 16.
    )

    filename = (
        SCRATCH_PATH /
        f"version_{MAJOR_MINOR_NUMBER}" / 
        "plots" / 
        f"k{k_value:.3f}_t{t_value:.3f}_xb{xb_value:.3f}_q2{qsquared_value:.3f}_residuals"
    )

    for extension in ["png", "eps"]:
        residuals_figure.savefig(
            f"{filename}.{extension}",
            facecolor = "white", transparent = False)

    plt.close(residuals_figure)

#################################################################################
# Making plots of the interpolate surfaces:
#################################################################################

unique_k_values = reference_dataframe.groupby(['k'])
print(f"[INFO]: there are {unique_k_values.ngroups} unique values for k.")

for k_value, k_group in unique_k_values:
    xb_q2_groups = k_group.groupby(['x_b', 'q_squared'])
    xb_q2_combinations = xb_q2_groups.ngroups

    for (xb_value, qsquared_value), group in xb_q2_groups:

        group = group.sort_values(['t', 'phi'])

        t_values = np.sort(group['t'].unique())
    
        phi_meshgrid, t_meshgrid = np.meshgrid(
            # this is a dense grid of phi values:
            np.linspace(-np.pi, np.pi, 361),
            t_values)

        phi_data = group['phi'].values
        t_data = group['t'].values

        point_dataframe = pd.DataFrame({
            "k": group["k"],
            "q_squared": group["q_squared"],
            "x_b": group["x_b"],
            "t": group["t"],
            "v": np.sin(group["phi"])
        })

        point_predictions_all = np.array([
            predict_bsa(model, point_dataframe, x_scaler, y_scaler)
            for model in replicas
        ])

        point_mean = np.mean(point_predictions_all, axis = 0)
        point_std = np.std(point_predictions_all, axis = 0)

        bsa_predictions = point_mean.ravel()
        # we don't actually use this:
        # bsa_stddev = point_std.ravel()

        bsa_actual = group["unp_target_bsa"].to_numpy()

        bsa_residuals = bsa_actual - bsa_predictions

        colors_bsa = np.where(bsa_residuals >= 0, 'red', 'blue')

        surface_dataframe = pd.DataFrame({
            "k": np.full(phi_meshgrid.size, k_value),
            "q_squared": np.full(phi_meshgrid.size, qsquared_value),
            "x_b": np.full(phi_meshgrid.size, xb_value),
            "t": t_meshgrid.ravel(),
            "v": np.sin(phi_meshgrid.ravel())
        })

        surface_predictions_all = np.array([
            predict_bsa(model, surface_dataframe, x_scaler, y_scaler)
            for model in replicas
        ])

        surface_mean = np.mean(surface_predictions_all, axis = 0)
        surface_std_dev = np.std(surface_predictions_all, axis = 0)
        bsa_surface = surface_mean[:, 0].reshape(phi_meshgrid.shape)
        bsa_stddev_surface = surface_std_dev[:, 0].reshape(phi_meshgrid.shape)

        zero_plane_bsa = np.zeros_like(bsa_surface)

        fig = plt.figure(figsize = (14, 7), layout = "tight")

        ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
        ax2 = fig.add_subplot(1, 2, 2, projection = '3d')

        # [NOTE]: this order actually determines some z-ordering stuff...
        ax1.plot_surface(
            phi_meshgrid, t_meshgrid, bsa_surface + bsa_stddev_surface,
            color = "gray", alpha = 0.30)
        ax1.plot_surface(
            phi_meshgrid, t_meshgrid, bsa_surface - bsa_stddev_surface,
            color = "gray", alpha = 0.30)
        ax1.plot_surface(
            phi_meshgrid, t_meshgrid, bsa_surface,
            cmap = 'viridis', alpha = 0.30)
        ax1.scatter(
            phi_data, t_data, bsa_actual,
            facecolors = 'white', edgecolors = 'black', s = 20, linewidths = 0.5, alpha = 1.0)

        ax1.set_xlabel(r'$\phi$ [Radians]',
                    labelpad = 16, fontsize = 16.)
        ax1.set_ylabel(r'$t$ [GeV$^{2}$]',
                    labelpad = 16, fontsize = 16.)
        ax1.set_zlabel(r'BSA',
                    labelpad = 7, fontsize = 16.)
        ax1.set_title('BSA', fontsize = 18.)

        ax2.plot_surface(
            phi_meshgrid, t_meshgrid, zero_plane_bsa,
            color = 'gray', alpha = 0.15)
        ax2.scatter(
            phi_data, t_data, bsa_residuals,
            color = colors_bsa, s = 20)

        ax2.set_xlabel(r'$\phi$ [Radians]',
                    labelpad = 16, fontsize = 16.)
        ax2.set_ylabel(r'$t$ [GeV$^{2}$] ',
                    labelpad = 16, fontsize = 16.)
        ax2.set_zlabel('Residuals',
                    labelpad = 7, fontsize = 16.)
        ax2.set_title('BSA Residuals', fontsize = 18)

        fig.suptitle(
            r"DNN Interpolations Across $t$ and $\phi$"
            "\n"
            rf"Kinematic Setting: $k = {k_value}$ GeV, $x_\textrm{{B}} = {xb_value}$, $Q^2 = {qsquared_value}$ GeV$^{{2}}$",
            fontsize = 16.0
        )
        
        for extension in ['png', 'eps']:
            fig.savefig(
                fname =
                    SCRATCH_PATH /
                    f"version_{MAJOR_MINOR_NUMBER}" /
                    "plots" /
                    f"surface_k{k_value}_xb{xb_value}_q2{qsquared_value}_v{MAJOR_MINOR_NUMBER}.{extension}", 
                facecolor = 'white')

        plt.close(fig)

print("[INFO]: End of script reached!")
