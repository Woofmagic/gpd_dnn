#################################################################################
# FILE INFORMATION:
# Purpose: script version of the surrogate model
# Created: 20260630
# Last changed: 20260630
# [NOTE]:
# [TODO]:
#   1. 20260630: Maybe consider density plots using:
#   https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist2d.html
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

VERSION_NUMBER = 7
MINOR_NUMBER = 6
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

#################################################################################
# model hyperparameters:
#################################################################################

BASE_LEARNING_RATE = 2e-3
BASE_WEIGHT_DECAY_RATE = 1e-7
TOTAL_EPOCHS = 2000
NUMBER_OF_REPLICAS = 3

output_directory = Path("./local")

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

#################################################################################
# Reading the pseudodata file:
#################################################################################

test_dataframe = pd.read_csv(
    output_directory /
    f"version_{MAJOR_MINOR_NUMBER}" / 
    "data" / 
    f"refined_cross_section_data_v{MAJOR_MINOR_NUMBER}.csv"
)

# positive cross-sections only, which we already antiticpate is true:
test_dataframe = test_dataframe[
    (test_dataframe["unp_beam_unp_target_xsec"] > 0)
    &
    (test_dataframe["unp_beam_unp_target_xsec_err"] > 0)
].copy()

# phi -> u(phi)
test_dataframe["u"] = (1. - np.cos(test_dataframe["phi"])) / 2.

#################################################################################
# Data loading *and* preprocessing:
#################################################################################

x_data = test_dataframe[["k", "q_squared", "x_b", "t", "u"]]
y_data = np.log(test_dataframe[["unp_beam_unp_target_xsec"]])

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
            f"cross_section_surrogate_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.{extension}"
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
            f"cross_section_surrogate_log_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.{extension}"
        )

    plt.close(figure)

def plot_prediction_vs_truth(truth, prediction, output_path):

    # compute R^{2} stuff:
    r2_linear = r2_score(truth, prediction)
    r2_log = r2_score(np.log(truth), np.log(prediction))

    fig, ax = plt.subplots(figsize = (9, 9))

    ax.scatter(truth, prediction, s = 4.0, alpha = 0.6, color = "blue")

    minimum = min(truth, prediction)
    maximum = max(truth, prediction)

    ax.plot(
        [minimum, maximum], [minimum, maximum],
        color = "red", linestyle = "-", label = "Perfect Fit",
    
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Cross Section Data", fontsize = 14.)
    ax.set_ylabel("DNN Prediction", fontsize = 14.)

    ax.text(
        0.05, 0.95, f"$R^2$ = {r2_linear:.5f}\n", f"$R^2_{{\\log}}$ = {r2_log:.5f}",transform = ax.transAxes,
        ha = "left", va = "top", fontsize = 12., bbox = dict(facecolor = "white", alpha = 0.8),
    )

    ax.legend()

    fig.tight_layout()

    for extension in ("png", "eps"):
        fig.savefig(output_path.with_suffix(f".{extension}"))

    plt.close(fig)

    return r2_linear, r2_log

# have to go through the intermediate transformations:
def predict_cross_section(model, x_dataframe, x_scaler, y_scaler):

    # transform the data:
    x_scaled = x_scaler.transform(x_dataframe)

    # predict with the model (look at the call function):
    model_prediction_in_z = model.predict(x_scaled, verbose = 0)

    # return to log space:
    log_sigma = y_scaler.inverse_transform(model_prediction_in_z)

    # inverts the log sigma:
    sigma = np.exp(log_sigma)

    return sigma

#################################################################################
# DNN model
#################################################################################

class CrossSectionSurrogateModel(tf.keras.Model):

    # https://keras.io/api/models/model/ -> follow this for custom model architecture

    def __init__(self):
        super().__init__()

        # [NOTE]: there was probably a bug here before...
        # we used the same kernel initializer with every layer...
        self.hidden_layers = [
            tf.keras.layers.Dense(
                256, activation = "silu", kernel_initializer = "glorot_normal"
            )
            for _ in range(6)
        ]

        # linear activation is default activation if `activation` key is not specified: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        self.cross_section_output = tf.keras.layers.Dense(1)

    def call(self, x):

        # hidden layer computations:
        for layer in self.hidden_layers:
            x = layer(x)

        # final output:
        cross_section_output = self.cross_section_output(x)

        # random saturation thing:
        cross_section_output = 10.0 * tf.tanh(cross_section_output / 10.0)

        return cross_section_output

#################################################################################
# Begin main program flow!
#################################################################################

all_histories = []
all_point_predictions = []
models = []

for replica_index in range(NUMBER_OF_REPLICAS):
    replica_number = replica_index + 1
    print(f'[INFO]: Training replica {replica_number}')

    dnn_model = CrossSectionSurrogateModel()
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

    predictions_z = dnn_model.predict(preprocessed_x_data, verbose = 0)
    predicted_log_sigma = y_scaler.inverse_transform(predictions_z)
    predicted_cross_section = np.exp(predicted_log_sigma)

    prediction_dataframe = test_dataframe.copy()

    prediction_dataframe["model_xsec"] = predicted_cross_section

    prediction_dataframe.to_csv(
        output_directory /
        f"version_{MAJOR_MINOR_NUMBER}" /
        f"replica_{replica_number}_predictions.csv",
        index = False
    )

    r2 = plot_prediction_vs_truth(
        truth = test_dataframe["unp_beam_unp_target_xsec"],
        prediction = predicted_cross_section,
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
    all_point_predictions.append(predicted_cross_section)

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
        "u": (1.0 - np.cos(phi_smooth))/ 2.0
    })

    smooth_predictions_all = np.array([
        predict_cross_section(model, smooth_dataframe, x_scaler, y_scaler)
        for model in models
    ])

    smooth_mean = np.mean(smooth_predictions_all, axis = 0)
    smooth_std = np.std(smooth_predictions_all, axis = 0)

    xsec_smooth_mean = smooth_mean[:, 0]

    xsec_smooth_std = smooth_std[:, 0]

    point_dataframe = pd.DataFrame({
        "k": group["k"],
        "q_squared": group["q_squared"],
        "x_b": group["x_b"],
        "t": group["t"],
        "u": (1.0 - np.cos(group["phi"]))/ 2.0
    })

    point_predictions_all = np.array([
        predict_cross_section(model, point_dataframe, x_scaler, y_scaler)
        for model in models
    ])

    point_mean = np.mean(point_predictions_all, axis = 0)
    point_std = np.std(point_predictions_all, axis = 0)

    xsec_pred = point_mean[:, 0]
    xsec_pred_std = point_std[:, 0]

    # these are experimental values:
    phi = group["phi"].to_numpy()
    xsec_err = group["unp_beam_unp_target_xsec_err"].to_numpy()
    xsec_actual = group["unp_beam_unp_target_xsec"].to_numpy()

    pulls = (xsec_actual - xsec_pred) / xsec_err

    chi_squared = np.sum(pulls**2)

    chi2_per_point = chi_squared / len(phi)

    xsec_res = xsec_actual - xsec_pred

    for phi_target in special_phis:
        phi_index = np.argmin(np.abs(phi_smooth - phi_target))
        phi_actual = phi_smooth[phi_index]
        sigma_value = xsec_smooth_std[phi_index]
        print(f"[INFO]: phi = {phi_actual:.3f}, uncertainty = {sigma_value:.6f}")

    residuals_figure, residuals_axes = plt.subplots(2, 1, figsize = (10, 8), sharex = 'col', layout = "tight")

    residuals_axes[1].text(
        -0.1, -0.1,
        fr"Figure rendered {datetime.datetime.now().strftime('%y%m%d-%H%M%S')}", 
        transform = residuals_axes[1].transAxes)

    residuals_axes[0].plot(phi_smooth, xsec_smooth_mean, color = 'red', lw = 2, label = rf'Replica Average ($N = {NUMBER_OF_REPLICAS}$)')
    residuals_axes[0].fill_between(
        phi_smooth, xsec_smooth_mean - xsec_smooth_std, xsec_smooth_mean + xsec_smooth_std,
        color = 'red', alpha = 0.3,
        label = r'$\sigma$ band')

    residuals_axes[0].errorbar(
        phi, xsec_actual, yerr = xsec_err,
        fmt = 'o', mfc = 'white', mec = 'black', ms = 5, ecolor = 'black', elinewidth = 1, capsize = 2, alpha = 0.8,
        label = 'Experimental Data')
    residuals_axes[0].set_ylabel(r"$d^{4}\sigma$ [nb / GeV$^{4}$]", fontsize = 16.)
    residuals_axes[0].set_title(rf"Cross Section ($\chi^2/N = {chi2_per_point:.7f}$)", fontsize = 18.)
    residuals_axes[0].legend(fontsize = 14.)
    residuals_axes[0].grid(True, linestyle = ':', alpha = 0.6)

    residuals_axes[1].scatter(phi, xsec_res, color = 'blue', alpha = 0.6)
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
            "u": (1.0 - np.cos(group["phi"])) / 2.0
        })

        point_predictions_all = np.array([
            predict_cross_section(model, point_dataframe, x_scaler, y_scaler)
            for model in models
        ])

        point_mean = np.mean(point_predictions_all, axis = 0)
        point_std = np.std(point_predictions_all, axis = 0)

        cross_section_predictions = point_mean.ravel()
        # we don't actually use this:
        # xsec_std = point_std.ravel()

        xsec_actual = group["unp_beam_unp_target_xsec"].to_numpy()

        xsec_residuals = xsec_actual - cross_section_predictions

        colors_xsec = np.where(xsec_residuals >= 0, 'red', 'blue')

        surface_dataframe = pd.DataFrame({
            "k": np.full(phi_meshgrid.size, k_value),
            "q_squared": np.full(phi_meshgrid.size, qsquared_value),
            "x_b": np.full(phi_meshgrid.size, xb_value),
            "t": t_meshgrid.ravel(),
            "u": (1.0 - np.cos(phi_meshgrid.ravel())) / 2.0
        })

        surface_predictions_all = np.array([
            predict_cross_section(model, surface_dataframe, x_scaler,y_scaler)
            for model in models
        ])

        surface_mean = np.mean(surface_predictions_all, axis = 0)
        surface_std_dev = np.std(surface_predictions_all, axis = 0)
        xsec_surface = surface_mean[:, 0].reshape(phi_meshgrid.shape)
        xsec_stddev_surface = surface_std_dev[:, 0].reshape(phi_meshgrid.shape)

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
        ax2.set_title('Cross Section Residuals', fontsize = 18)

        fig.suptitle(
            r"DNN Interpolations Across $t$ and $\phi$"
            "\n"
            rf"Kinematic Setting: $k = {k_value}$ GeV, $x_\textrm{{B}} = {xb_value}$, $Q^2 = {qsquared_value}$ GeV$^{{2}}$",
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
