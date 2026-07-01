#################################################################################
# FILE INFORMATION:
# Purpose: produce a replica DNN model mapping kinematics to observables.
# Created: 20260505
# Last changed: 20260701
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import gc
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#################################################################################
# HPC logic:
#################################################################################

replica_number = int(sys.argv[1])
print(f"[INFO]: This replica number is: Replica #{replica_number}")

#################################################################################
# In case you want to reproduce things --- but this isn't necessary!
#################################################################################

np.random.seed(replica_number)
tf.random.set_seed(replica_number)

#################################################################################
# Scratch path
#################################################################################

# verify this is what you want
SCRATCH_PATH = Path('placeholder!')

#################################################################################
# Version numbers!
#################################################################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

#################################################################################
# Model hyperparameters:
#################################################################################

USING_GAUSSIAN_ERROR_SAMPLING = True

BASE_LEARNING_RATE = 3e-4
BASE_WEIGHT_DECAY_RATE = 1e-7
_NUMBER_OF_EPOCHS = 3000

# sampling true points with error distribution
USING_GAUSSIAN_ERROR_SAMPLING = False

# train/validation/test split:
_DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE = 0.1 # 90% temporary, 10% testing
_DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE = 0.1 # of the above 90% temporary, 90% training, 10% validation

#################################################################################
# Reading the pseudodata file:
#################################################################################

# this reads the pseudodata!
pseudodata_dataframe = pd.read_csv(
    filepath_or_buffer =
        SCRATCH_PATH /
        f"version_{MAJOR_MINOR_NUMBER}" /
        "data" /
        f"refined_cross_section_data_v{MAJOR_MINOR_NUMBER}.csv"
)

#################################################################################
# Use the pseudodata sampling technique with experimental errors:
#################################################################################

# saves a copy of the column:
pseudodata_dataframe['original_xsec'] = pseudodata_dataframe['unp_beam_unp_target_xsec']

if USING_GAUSSIAN_ERROR_SAMPLING:

    pseudodata_dataframe['unp_beam_unp_target_xsec'] = np.random.normal(
        loc = pseudodata_dataframe['original_xsec'],
        scale = pseudodata_dataframe['unp_beam_unp_target_xsec_err']
    )

# phi -> u(phi)
pseudodata_dataframe["u"] = (1. - np.cos(pseudodata_dataframe["phi"])) / 2.

x_data = pseudodata_dataframe[["k", "q_squared", "x_b", "t", "u"]]
y_data = np.log(pseudodata_dataframe[["unp_beam_unp_target_xsec"]])

TOTAL_DATA_SIZE = len(x_data)
print(f"[INFO]: Total data size is: {TOTAL_DATA_SIZE}")

x_scaler = StandardScaler()
y_scaler = StandardScaler()

preprocessed_x_data = x_scaler.fit_transform(x_data)
preprocessed_y_data = y_scaler.fit_transform(y_data)

indices = np.arange(len(pseudodata_dataframe))

# testing/temporary split:
remaining_indices, testing_indices = train_test_split(
    indices,
    test_size = _DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE, shuffle = True, random_state = 31415)

# training/validation split:
training_indices, validation_indices = train_test_split(
    remaining_indices,
    test_size = _DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE, shuffle = True, random_state = 31415)

x_training = preprocessed_x_data[training_indices]
y_training = preprocessed_y_data[training_indices]

x_validation = preprocessed_x_data[validation_indices]
y_validation = preprocessed_y_data[validation_indices]

x_testing = preprocessed_x_data[testing_indices]
y_testing = preprocessed_y_data[testing_indices]

print(f"[INFO]: Total number of training points: {len(x_training)}")
print(f"[INFO]: Total number of validaion points: {len(x_validation)}")
print(f"[INFO]: Total number of testing points: {len(x_testing)}")

#################################################################################
# Label the dataframe with training/validation/testing split:
#################################################################################

# augment rows with train/test/val
pseudodata_dataframe["split"] = ""

pseudodata_dataframe.loc[training_indices, "split"] = "train"
pseudodata_dataframe.loc[validation_indices, "split"] = "validation"
pseudodata_dataframe.loc[testing_indices, "split"] = "test"

pseudodata_dataframe.to_csv(
    path_or_buf =
        SCRATCH_PATH /
        f"version_{MAJOR_MINOR_NUMBER}" /
        "data" /
        f"dnn_data_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.csv",
    index = False
)

print("[INFO]: Save the pseudodata dataframe!")

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

def plot_prediction_vs_truth(truth, prediction, output_path):

    # compute R^{2} stuff:
    r2_linear = r2_score(truth, prediction)
    r2_log = r2_score(np.log(truth), np.log(prediction))

    fig, ax = plt.subplots(figsize = (9, 9))

    ax.scatter(truth, prediction, s = 4.0, alpha = 0.6, color = "blue")

    minimum = min(np.min(truth), np.min(prediction))
    maximum = max(np.max(truth), np.max(prediction))

    ax.plot(
        [minimum, maximum], [minimum, maximum],
        color = "red", linestyle = "-", label = "Perfect Fit",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Cross Section Data", fontsize = 14.)
    ax.set_ylabel("DNN Prediction", fontsize = 14.)

    ax.set_title(f"Replica {replica_number} Performance\nR^2 = {r2_linear:.5f}, R^2(log) = {r2_log:.5f}")

    ax.legend()

    fig.tight_layout()

    for extension in ("png", "eps"):
        fig.savefig(output_path.with_suffix(f".{extension}"))

    plt.close(fig)

    return r2_linear, r2_log

#################################################################################
# TensorFlow model!
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
# Training!
#################################################################################

_BATCH_SIZE = len(x_training)

tf.keras.backend.clear_session()
gc.collect()

dnn_model = CrossSectionSurrogateModel()

dnn_model.compile(
    # LR is alpha in ADAM, which is stepsize:
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate = BASE_LEARNING_RATE,
        weight_decay = BASE_WEIGHT_DECAY_RATE),
    loss = "mse",
    metrics = ["mae"])

dnn_model_history = dnn_model.fit(
    x_training, y_training,
    validation_data = (x_validation, y_validation),
    epochs = _NUMBER_OF_EPOCHS,
    batch_size = _BATCH_SIZE,
    verbose = 0)

dnn_model.save(
    SCRATCH_PATH /
    f"version_{MAJOR_MINOR_NUMBER}" / 
    "replicas" / 
    f"replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras"
)

#################################################################################
# Post-train evaluation and analysis and metadata collection:
#################################################################################

# just get the number of epochs:
number_of_epochs_run = len(dnn_model_history.epoch)
print(f"[INFO]: The model ran for {number_of_epochs_run} epochs before early stopping.")

# cast training history into dataframe and csv:
history_df = pd.DataFrame(dnn_model_history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv(
    SCRATCH_PATH /
    f"version_{MAJOR_MINOR_NUMBER}" / 
    "data" / 
    f"replica_{replica_number}_history.csv", 
    index = False)

# evaluation:
evaluation_metrics = dnn_model.evaluate(x_validation, y_validation, verbose = 0)
print(f"[INFO] Evaluation metrics: {evaluation_metrics}")

metrics_dictionary = dict(zip(dnn_model.metrics_names, evaluation_metrics))
validation_loss = metrics_dictionary["loss"]
print(f"[INFO] Validation loss = {validation_loss}")

plot_learning_curve(
    replica_number = replica_number,
    history = dnn_model_history,
    validation_loss = validation_loss,
    output_directory = SCRATCH_PATH
)

plot_log_learning_curve(
    replica_number = replica_number,
    history = dnn_model_history,
    validation_loss = validation_loss,
    output_directory = SCRATCH_PATH
)

# make the predictions:
predictions_z = dnn_model.predict(preprocessed_x_data, verbose = 0)
predicted_log_sigma = y_scaler.inverse_transform(predictions_z)
predicted_cross_section = np.exp(predicted_log_sigma)

prediction_dataframe = pseudodata_dataframe.copy()

r2 = plot_prediction_vs_truth(
    truth = prediction_dataframe["unp_beam_unp_target_xsec"],
    prediction = predicted_cross_section,
        output_path = (
            SCRATCH_PATH
        / f"version_{MAJOR_MINOR_NUMBER}"
        / "plots"
        / f"data_vs_prediction_replica_{replica_number}"
    ),
)

pd.DataFrame(
    [evaluation_metrics]).to_csv(
        SCRATCH_PATH /
        f"version_{MAJOR_MINOR_NUMBER}" /
        "data" /
        f"replica_{replica_number}_test_metrics.csv", 
        index = False)
    
#################################################################################
# Collecting data:
#################################################################################

prediction_dataframe['split'] = pseudodata_dataframe['split'].values
prediction_dataframe['original_xsec'] = pseudodata_dataframe['original_xsec'].values
prediction_dataframe['xsec_err'] = pseudodata_dataframe['unp_beam_unp_target_xsec_err'].values
prediction_dataframe['pseudodata_xsec_value'] = pseudodata_dataframe['unp_beam_unp_target_xsec'].values
prediction_dataframe["model_xsec"] = predicted_cross_section
prediction_dataframe['replica_number'] = replica_number

prediction_dataframe.to_csv(
    SCRATCH_PATH /
    f"version_{MAJOR_MINOR_NUMBER}" /
    "data" /
    f"replica_{replica_number}_predictions.csv",
    index = False
)

#################################################################################
# Metadata dump:
#################################################################################

metadata = {
    "major_version": VERSION_NUMBER,
    "minor_version": MINOR_NUMBER,
    "total_version": MAJOR_MINOR_NUMBER,
    "did_pseudodata_sampling": USING_GAUSSIAN_ERROR_SAMPLING,
    "total_datasize": TOTAL_DATA_SIZE,
    "replica_id": replica_number,
    "batch_size": _BATCH_SIZE,
    "max_epochs": _NUMBER_OF_EPOCHS,
    "base_learning_rate": BASE_LEARNING_RATE,
    "actual_epochs": len(dnn_model_history.epoch),
    "training_points": len(x_training),
    "validation_points": len(x_validation),
    "testing_points": len(x_testing),
    "features": list(x_data.columns),
}

with open(
    file =
        SCRATCH_PATH /
        f"version_{MAJOR_MINOR_NUMBER}" /
        "data" /
        f"replica_{replica_number}_metadata.json",
    mode = "w",
    encoding = "utf-8") as f:
    json.dump(metadata, f, indent = 4)

#################################################################################
# Exit program:
#################################################################################

# cleanup
del dnn_model
gc.collect()

print("[INFO]: End of script reached!")

#################################################################################
# Some helpful resources
#################################################################################

# https://stackoverflow.com/a/17840195 -> for why we need to cast it into a list!
