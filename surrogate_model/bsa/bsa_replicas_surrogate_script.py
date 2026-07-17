#################################################################################
# FILE INFORMATION:
# Purpose: script version of the surrogate model
# Created: 20260701
# Last changed: 20260701
# Notes:
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#################################################################################
# Version numbers!
#################################################################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

#################################################################################
# model hyperparameters:
#################################################################################

BASE_LEARNING_RATE = 5e-4
BASE_WEIGHT_DECAY_RATE = 1e-7
TOTAL_EPOCHS = 3000
NUMBER_OF_REPLICAS = 15

output_directory = Path("./local")

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

#################################################################################
# Reading the pseudodata file:
#################################################################################

test_dataframe = pd.read_csv(
    output_directory /
    f"version_{MAJOR_MINOR_NUMBER}" / 
    "data" / 
    f"refined_bsa_data_v{MAJOR_MINOR_NUMBER}.csv"
)

# phi -> v(phi)
test_dataframe["v"] = np.sin(test_dataframe["phi"])

#################################################################################
# Data loading *and* preprocessing:
#################################################################################

# we will use this to make predictions across the *entire* dataset!
x_data = test_dataframe[["k", "q_squared", "x_b", "t", "v"]]
# [NOTE]: we do NOT LOG THE BSA DATA!
y_data = test_dataframe[["unp_target_bsa"]]

x_scaler = StandardScaler()
y_scaler = StandardScaler()

preprocessed_x_data = x_scaler.fit_transform(x_data)
preprocessed_y_data = y_scaler.fit_transform(y_data)

x_training, x_validation, y_training, y_validation = train_test_split(
    preprocessed_x_data, preprocessed_y_data, test_size = 0.20, random_state = 42
)

print(f"[INFO]: length of training: {len(x_training)}")
print(f"[INFO]: length of validation: {len(x_validation)}")

#################################################################################
# Useful functions for plotting
#################################################################################

def plot_learning_curve(
    replica_number, history, validation_loss, output_directory):

    figure, axis = plt.subplots(figsize = (8, 8))

    axis.plot(history.history["loss"], label = "Training Loss")
    axis.plot(history.history["val_loss"], label = "Validation Loss")

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")

    axis.set_title(f"Cross-Section Surrogate\nValidation Loss = {validation_loss:.6e}")

    axis.legend()

    figure.tight_layout()

    for extension in ['png', 'eps']:
        figure.savefig(
            output_directory /
            f"version_{MAJOR_MINOR_NUMBER}" /
            "learning_curves" /
            f"bsa_surrogate_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.{extension}"
        )

    plt.close(figure)

def plot_log_learning_curve(
    replica_number, history, validation_loss, output_directory):

    figure, axis = plt.subplots(figsize = (8, 8))

    axis.plot(history.history["loss"], label = "Training Loss")
    axis.plot(history.history["val_loss"], label = "Validation Loss")   

    axis.set_yscale("log")

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")

    axis.set_title(f"Cross-Section Surrogate\nValidation Loss = {validation_loss:.6e}")

    axis.legend()

    figure.tight_layout()

    for extension in ['png', 'eps']:
        figure.savefig(
            output_directory / 
            f"version_{MAJOR_MINOR_NUMBER}" /
            "learning_curves" /
            f"bsa_surrogate_log_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.{extension}"
        )

    plt.close(figure)

def plot_prediction_vs_truth(truth, prediction, output_path):

    r_squared = r2_score(truth, prediction)

    fig, ax = plt.subplots(figsize = (9, 9))

    ax.scatter(truth, prediction, s = 4.0, alpha = 0.6, color = "blue")

    minimum = min(np.min(truth), np.min(prediction))
    maximum = max(np.max(truth), np.max(prediction))

    ax.plot(
        [minimum, maximum], [minimum, maximum],
        color = "red", linestyle = "-", label = "Perfect Fit"
    )

    ax.set_xlabel("BSA Data", fontsize = 14.0)
    ax.set_ylabel("DNN Prediction", fontsize = 14.0)

    ax.set_title(f"Replica {replica_number} Performance\nR^2 = {r_squared:.5f}")

    ax.legend()

    fig.tight_layout()

    for extension in ("png", "eps"):
        fig.savefig(output_path.with_suffix(f".{extension}"))

    plt.close(fig)

    return r_squared

# have to go through the intermediate transformations:
def predict_bsa(model, x_dataframe, x_scaler, y_scaler,):
    
    x_scaled = x_scaler.transform(x_dataframe)
    model_prediction_in_z = model.predict(x_scaled, verbose = 0)
    bsa = y_scaler.inverse_transform( model_prediction_in_z)
    return bsa

#################################################################################
# DNN model
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
# Begin main program flow!
#################################################################################

all_histories = []
all_point_predictions = []
models = []

for replica_index in range(NUMBER_OF_REPLICAS):
    replica_number = replica_index + 1
    print(f'[INFO]: Training replica {replica_number}')

    dnn_model = BSASurrogateModel()
    dnn_model.compile(
        # LR is alpha in ADAM, which is stepsize:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate = BASE_LEARNING_RATE,
            weight_decay = BASE_WEIGHT_DECAY_RATE),
        loss = "mse",
        metrics = ["mae"])

    history = dnn_model.fit(
        x_training,
        y_training,
        validation_data = (x_validation, y_validation),
        epochs = TOTAL_EPOCHS,
        batch_size = len(x_training),
        verbose = 0
    )

    evaluation_metrics = dnn_model.evaluate(
        x_validation,
        y_validation,
        verbose = 0
    )

    print(f"[INFO] Evaluation metrics: {evaluation_metrics}")

    metrics_dictionary = dict(zip(dnn_model.metrics_names, evaluation_metrics))

    validation_loss = metrics_dictionary["loss"]

    print(f"[INFO] Validation loss = {validation_loss}")

    plot_learning_curve(
        replica_number = replica_number,
        history = history,
        validation_loss = validation_loss,
        output_directory = output_directory
    )
    plot_log_learning_curve(
        replica_number = replica_number,
        history = history,
        validation_loss = validation_loss,
        output_directory = output_directory
    )

    # make the predictions:
    predictions_z = dnn_model.predict(preprocessed_x_data, verbose = 0)
    predicted_bsa = y_scaler.inverse_transform(predictions_z)

    prediction_dataframe = test_dataframe.copy()

    prediction_dataframe["model_bsa"] = predicted_bsa

    prediction_dataframe.to_csv(
        output_directory /
        f"version_{MAJOR_MINOR_NUMBER}" /
        "data" / 
        f"replica_{replica_number}_predictions.csv",
        index = False
    )

    r2 = plot_prediction_vs_truth(
        truth = test_dataframe["unp_target_bsa"],
        prediction = predicted_bsa,
        output_path = (
            output_directory
            / f"version_{MAJOR_MINOR_NUMBER}"
            / "plots"
            / f"data_vs_prediction_replica_{replica_number}"
        ),
    )

    print(f"[INFO]: (replica {replica_number}) R^2 = {r2}")

    dnn_model.save(
        output_directory / 
        f"version_{MAJOR_MINOR_NUMBER}" / 
        "replicas" / 
        f"replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras"
    )

    all_histories.append(history.history)
    models.append(dnn_model)
    all_point_predictions.append(predicted_bsa)

all_point_predictions = np.array(all_point_predictions)
average_prediction = np.mean(all_point_predictions, axis = 0)
standard_dev_prediction = np.std(all_point_predictions, axis = 0)

grouped = test_dataframe.groupby(['k', 't', 'x_b', 'q_squared'])

plot_directory = (
    output_directory /
    f"version_{MAJOR_MINOR_NUMBER}" /
    "plots" /
    "local_fits"
)

plot_directory.mkdir(parents = True, exist_ok = True)

# this is trento convention: -pi to pi:
phi_smooth = np.linspace(-np.pi, np.pi, 361)

# for printing stuff later:
special_phis = [0, np.pi/2., -np.pi/2., np.pi]

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
        for model in models
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
        for model in models
    ])

    point_mean = np.mean(point_predictions_all, axis = 0)
    point_std = np.std(point_predictions_all, axis = 0)

    bsa_predicted = point_mean[:, 0]
    bsa_pred_std = point_std[:, 0]

    # these are experimental values:
    phi = group["phi"].to_numpy()
    bsa_error = group["unp_target_bsa_err"].to_numpy()
    bsa_actual = group["unp_target_bsa"].to_numpy()

    pulls = (bsa_actual - bsa_predicted) / bsa_error

    chi_squared = np.sum(pulls**2)

    chi2_per_point = chi_squared / len(phi)

    bsa_res = bsa_actual - bsa_predicted

    residuals_figure, residuals_axes = plt.subplots(2, 1, figsize = (10, 8), sharex = 'col', layout = "tight")

    residuals_axes[1].text(
        -0.1, -0.1,
        fr"Figure rendered {datetime.datetime.now().strftime('%y%m%d-%H%M%S')}", 
        transform = residuals_axes[1].transAxes)

    residuals_axes[0].plot(phi_smooth, bsa_smooth_mean, color = 'red', lw = 2, label = rf'Replica Average ($N = {NUMBER_OF_REPLICAS}$)')
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

    residuals_axes[1].scatter(phi, bsa_res, color = 'blue', alpha = 0.6)
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
        plot_directory /
        f"k{k_value:.3f}_t{t_value:.3f}_xb{xb_value:.3f}_q2{qsquared_value:.3f}_residuals"
    )

    for extension in ["png", "eps"]:
        residuals_figure.savefig(
            f"{filename}.{extension}",
            facecolor = "white", transparent = False)

    plt.close(residuals_figure)

unique_k_values = test_dataframe.groupby(['k'])
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
            for model in models
        ])

        point_mean = np.mean(point_predictions_all, axis = 0)
        point_std = np.std(point_predictions_all, axis = 0)

        bsa_predictions = point_mean.ravel()

        bsa_actual = group["unp_target_bsa"].to_numpy()

        xsec_residuals = bsa_actual - bsa_predictions

        colors_xsec = np.where(xsec_residuals >= 0, 'red', 'blue')

        surface_dataframe = pd.DataFrame({
            "k": np.full(phi_meshgrid.size, k_value),
            "q_squared": np.full(phi_meshgrid.size, qsquared_value),
            "x_b": np.full(phi_meshgrid.size, xb_value),
            "t": t_meshgrid.ravel(),
            "v": np.sin(phi_meshgrid.ravel())
        })

        surface_predictions_all = np.array([
            predict_bsa(model, surface_dataframe, x_scaler,y_scaler)
            for model in models
        ])

        surface_mean = np.mean(surface_predictions_all, axis = 0)
        surface_std_dev = np.std(surface_predictions_all, axis = 0)
        bsa_surface = surface_mean[:, 0].reshape(phi_meshgrid.shape)
        bsa_stddev_surface = surface_std_dev[:, 0].reshape(phi_meshgrid.shape)

        zero_plane_xsec = np.zeros_like(bsa_surface)

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
            phi_meshgrid, t_meshgrid, zero_plane_xsec,
            color = 'gray', alpha = 0.15)
        ax2.scatter(
            phi_data, t_data, xsec_residuals,
            color = colors_xsec, s = 20)

        ax2.set_xlabel(r'$\phi$ [Radians]',
                    labelpad = 16, fontsize = 16.)
        ax2.set_ylabel(r'$t$ [GeV$^{2}$] ',
                    labelpad = 16, fontsize = 16.)
        ax2.set_zlabel('Residuals',
                    labelpad = 7, fontsize = 16.)
        ax2.set_title('BSA Residuals', fontsize = 18.)

        fig.suptitle(
            r"DNN Interpolations Across $t$ and $\phi$"
            "\n"
            rf"Kinematic Setting: $k = {k_value}$ GeV, $x_\mathrm{{B}} = {xb_value}$, $Q^2 = {qsquared_value}$ GeV$^{{2}}$",
            fontsize = 16.0
        )
        
        for extension in ['png', 'eps']:
            fig.savefig(
                fname =
                output_directory /
                    f"version_{MAJOR_MINOR_NUMBER}" /
                    "plots" /
                    f"surface_k{k_value}_xb{xb_value}_q2{qsquared_value}_v{MAJOR_MINOR_NUMBER}.{extension}", 
                facecolor = 'white')

        plt.close(fig)

print("[INFO]: End of script reached!")
