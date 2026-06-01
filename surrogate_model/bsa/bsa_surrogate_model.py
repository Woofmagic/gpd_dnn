#################################################################################
# FILE INFORMATION:
# Purpose: produce a replica DNN model mapping kinematics to observables.
# Created: 20260505
# Last changed: 20260528
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import gc
import json
import sys

import pandas as pd
import numpy as np
import tensorflow as tf

#################################################################################
# HPC logic:
#################################################################################

replica_number = int(sys.argv[1])

#################################################################################
# In case you want to reproduce things --- but this isn't necessary!
#################################################################################

# np.random.seed(replica_number)
# tf.random.set_seed(replica_number)

#################################################################################
# Scratch path
#################################################################################

# verify this is what you want
SCRATCH_PATH = 'placeholder!'

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
_NUMBER_OF_EPOCHS = 1000
_BATCH_SIZE = 8

#################################################################################
# Loading the data!
#################################################################################

dnn_replica_data = pd.read_csv(
    filepath_or_buffer = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/dnn_data_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.csv"
)

# we will use this to make predictions across the *entire* dataset!
x_data = dnn_replica_data[["t", "x_b", "q_squared", "phi"]]
y_data = dnn_replica_data["unp_target_bsa"]

#################################################################################
# Partitioning the data into its train/val/test flags:
#################################################################################

training_df = dnn_replica_data[dnn_replica_data["split"] == "train"]
validation_df = dnn_replica_data[dnn_replica_data["split"] == "validation"]
testing_df = dnn_replica_data[dnn_replica_data["split"] == "test"]

number_of_dnn_training_points = len(training_df)
number_of_dnn_validation_points = len(validation_df)
number_of_dnn_testing_points = len(testing_df)

x_training = training_df[["t", "x_b", "q_squared", "phi"]]
y_training = training_df[["unp_target_bsa"]]

x_validation = validation_df[["t", "x_b", "q_squared", "phi"]]
y_validation = validation_df[["unp_target_bsa"]]

x_testing = testing_df[["t", "x_b", "q_squared", "phi"]]
y_testing = testing_df[["unp_target_bsa"]]

if number_of_dnn_training_points <= _BATCH_SIZE:
    print(f"[WARN]: Number of training points is less than or equal to the batch size. Setting batch size equal to {number_of_dnn_training_points}.")
    _BATCH_SIZE = number_of_dnn_training_points

#################################################################################
# TensorFlow model!
#################################################################################

def cff_h_model():
    # initializer:
    initializer = tf.keras.initializers.GlorotNormal(seed = None)

    # input layer:
    model_inputs = tf.keras.Input(shape = (4,), name = "input_values")

    # hidden layers:
    hidden = tf.keras.layers.Dense(
        64, kernel_initializer = initializer, activation = "tanh")(model_inputs)
    hidden = tf.keras.layers.Dense(
        64, kernel_initializer = initializer, activation = "tanh")(hidden)
    hidden = tf.keras.layers.Dense(
        64, kernel_initializer = initializer, activation = "tanh")(hidden)

    # output layer:
    model_output = tf.keras.layers.Dense(1, activation = "linear")(hidden)

    model = tf.keras.Model(inputs = model_inputs, outputs = model_output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = BASE_LEARNING_RATE),
        loss = tf.keras.losses.MeanSquaredError())

    return model

#################################################################################
# Training!
#################################################################################

tf.keras.backend.clear_session()
gc.collect()

dnn_model = cff_h_model()

dnn_model_history = dnn_model.fit(
    x_training, y_training,
    validation_data = (x_validation, y_validation),
    epochs = _NUMBER_OF_EPOCHS,
    # [NOTE]: BATCHSIZE really matters!
    batch_size = _BATCH_SIZE,
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor = "val_loss", factor = 0.5, patience = 50, min_lr = 1e-6,
            verbose = 0),
    ],
    verbose = 0)

dnn_model.save(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras")

#################################################################################
# Post-train evaluation and analysis and metadata collection:
#################################################################################

number_of_epochs_run = len(dnn_model_history.epoch)
print(f"[INFO]: The model ran for {number_of_epochs_run} epochs before early stopping.")

history_df = pd.DataFrame(dnn_model_history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/replica_{replica_number}_history.csv", index = False)

dnn_evaluation_statistics = dnn_model.evaluate(x_testing, y_testing, verbose = 0, return_dict = True)
print(f"[INFO]: Test Loss for Replica {replica_number}: {dnn_evaluation_statistics}")

pd.DataFrame(
    # https://stackoverflow.com/a/17840195 -> for why we need to cast it into a list!
    [dnn_evaluation_statistics]).to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/replica_{replica_number}_test_metrics.csv", 
    index = False)

y_predictions = dnn_model.predict(x_data)

prediction_results = x_data.copy()

#################################################################################
# Preserve split labels
#################################################################################

prediction_results['split'] = dnn_replica_data['split'].values

#################################################################################
# Original experimental data:
#################################################################################

prediction_results['original_bsa'] = dnn_replica_data['original_bsa'].values

#################################################################################
# Experimental uncertainty:
#################################################################################

prediction_results['bsa_err'] = dnn_replica_data['unp_target_bsa_err'].values

#################################################################################
# Replica values:
#################################################################################

prediction_results['replica_bsa'] = dnn_replica_data['unp_target_bsa'].values

#################################################################################
# DNN predictions:
#################################################################################

prediction_results['pred_bsa'] = y_predictions[:, 0]

#################################################################################
# Metadata
#################################################################################

prediction_results['replica_number'] = replica_number

prediction_results.to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/replica_{replica_number}_test_predictions.csv",
    index = False)

#################################################################################
# Smooth replica surface across t, xb, q_squared, and phi:
#################################################################################

print("[INFO]: Computing smooth phi predictions...")

t_min = x_data['t'].min()
t_max = x_data['t'].max()

print(f"[INFO]: bounding for t: {t_min} < t < {t_max}")

xb_min = x_data['x_b'].min()
xb_max = x_data['x_b'].max()

print(f"[INFO]: bounding for xb: {xb_min} < x_b < {xb_max}")

q2_min = x_data['q_squared'].min()
q2_max = x_data['q_squared'].max()

print(f"[INFO]: bounding for Q^2: {q2_min} < Q^2 < {q2_max}")

NUMBER_OF_T = 10
NUMBER_OF_XB = 10
NUMBER_OF_Q2 = 10
NUMBER_OF_PHI = 361

t_grid = np.round(np.linspace(t_min, t_max, NUMBER_OF_T), 3)
xb_grid = np.round(np.linspace(xb_min, xb_max, NUMBER_OF_XB), 4)
q2_grid = np.round(np.linspace(q2_min, q2_max, NUMBER_OF_Q2), 3)
phi_grid = np.linspace(-np.pi, np.pi, NUMBER_OF_PHI)

mesh = np.meshgrid(t_grid, xb_grid, q2_grid, phi_grid, indexing = 'ij')

t_flat = mesh[0].ravel()
xb_flat = mesh[1].ravel()
q2_flat = mesh[2].ravel()
phi_flat = mesh[3].ravel()

smooth_input = pd.DataFrame({
    't': t_flat,
    'x_b': xb_flat,
    'q_squared': q2_flat,
    'phi': phi_flat
})

smooth_predictions = dnn_model.predict(smooth_input, verbose = 0)

# making predictions
smooth_input['pred_bsa'] = smooth_predictions[:, 0]

smooth_input.to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/"
    f"replica_{replica_number}_smooth_predictions.csv",
    index = False
)

print("[INFO]: Smooth predictions saved.")

#################################################################################
# Metadata dump:
#################################################################################

metadata = {
    "replica_id": replica_number,
    "version": MAJOR_MINOR_NUMBER,
    "batch_size": _BATCH_SIZE,
    "max_epochs": _NUMBER_OF_EPOCHS,
    "actual_epochs": len(dnn_model_history.epoch),
    "training_points": len(x_training),
    "features": list(x_training.columns),
    "number_of_t": NUMBER_OF_T,
    "number_of_xb": NUMBER_OF_XB,
    "number_of_q2": NUMBER_OF_Q2,
    "number_of_phi_deg": NUMBER_OF_PHI
}

with open(
    file = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/replica_{replica_number}_metadata.json",
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
