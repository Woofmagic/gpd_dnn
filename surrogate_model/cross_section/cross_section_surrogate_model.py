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
SYMMETRY_LOSS_WEIGHT = 0.0

#################################################################################
# Loading the data!
#################################################################################

dnn_replica_data = pd.read_csv(
    filepath_or_buffer = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/dnn_data_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.csv"
)

# we will use this to make predictions across the *entire* dataset!
x_data = dnn_replica_data[["t", "x_b", "q_squared", "phi"]]
y_data = dnn_replica_data["unp_beam_unp_target_xsec"]

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
y_training = training_df[["unp_beam_unp_target_xsec"]]

x_validation = validation_df[["t", "x_b", "q_squared", "phi"]]
y_validation = validation_df[["unp_beam_unp_target_xsec"]]

x_testing = testing_df[["t", "x_b", "q_squared", "phi"]]
y_testing = testing_df[["unp_beam_unp_target_xsec"]]

if number_of_dnn_training_points <= _BATCH_SIZE:
    print(f"[WARN]: Number of training points is less than or equal to the batch size. Setting batch size equal to {number_of_dnn_training_points}.")
    _BATCH_SIZE = number_of_dnn_training_points

#################################################################################
# TensorFlow model!
#################################################################################

class CrossSectionLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
class CrossSectionSurrogateModel(tf.keras.Model):

    # https://keras.io/api/models/model/ -> follow this for custom model architecture

    def __init__(self, symmetry_loss_weight = 1.0):
        super().__init__()

        self.symmetry_loss_weight = symmetry_loss_weight

        # self.data_loss_tracker = tf.keras.metrics.Mean(name = "data_loss")
        # self.symmetry_loss_tracker = tf.keras.metrics.Mean(name = "symmetry_loss")

        initializer = tf.keras.initializers.GlorotNormal(seed = None)

        self.dense_layer_1 = tf.keras.layers.Dense(32, kernel_initializer = initializer, activation = "tanh")
        self.dense_layer_2 = tf.keras.layers.Dense(32, kernel_initializer = initializer, activation = "tanh")
        self.dense_layer_3 = tf.keras.layers.Dense(32, kernel_initializer = initializer, activation = "tanh")

        # linear activation is default activation if `activation` key is not specified: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        self.cross_section_output = tf.keras.layers.Dense(1, activation = "linear", name = "cross_section")

        # custom loss business:
        self.cross_section_loss_tracker = CrossSectionLoss()

    def azimuthal_symmetry_loss(self, X_batch, training = True):

        X_plus = X_batch

        phi = X_batch[:, -1]

        X_minus = tf.concat([X_batch[:, :-1], tf.expand_dims(-phi, axis = 1)], axis = 1)

        y_plus = self(X_plus, training = training)
        y_minus = self(X_minus, training = training)

        return tf.reduce_mean(tf.square(y_plus - y_minus))

    def call(self, inputs, training = False):

        # hidden layer computation:
        hidden_layer = self.dense_layer_1(inputs)
        hidden_layer = self.dense_layer_2(hidden_layer)
        hidden_layer = self.dense_layer_3(hidden_layer)
        cross_section_output = self.cross_section_output(hidden_layer)

        return cross_section_output
    
    def train_step(self, data):

        # unpack data:
        X_batch_data, y_batch_data = data

        with tf.GradientTape() as tape:
            # forward pass:
            predictions = self(X_batch_data, training = True)

            # recall: `Instead, use `model.compute_loss(x, y, y_pred, sample_weight)`
            data_loss = self.compute_loss(X_batch_data, y_batch_data, predictions)

            # compute phi symmetry loss:
            symmetry_loss = self.azimuthal_symmetry_loss(X_batch_data, training = True)

            # total loss is just a weighted sum:
            total_loss = data_loss + self.symmetry_loss_weight * symmetry_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
            else:
                metric.update_state(y_batch_data, predictions)

        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "symmetry_loss": symmetry_loss,
            **{m.name: m.result() for m in self.metrics}
        }

    def test_step(self, data):
        # unpack data:
        X_batch_data, y_batch_data = data
        
        # forward pass evaluation:
        predictions = self(X_batch_data, training = False)

        # recall: `Instead, use `model.compute_loss(x, y, y_pred, sample_weight)`
        data_loss = self.compute_loss(X_batch_data, y_batch_data, predictions)

         # compute phi symmetry loss:
        symmetry_loss = self.azimuthal_symmetry_loss(X_batch_data, training = False)

        # total loss is just a weighted sum:
        total_loss = data_loss + self.symmetry_loss_weight * symmetry_loss

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
            else:
                metric.update_state(y_batch_data, predictions)
            
        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "symmetry_loss": symmetry_loss,
            **{m.name: m.result() for m in self.metrics}
        }

#################################################################################
# Training!
#################################################################################

tf.keras.backend.clear_session()
gc.collect()

dnn_model = CrossSectionSurrogateModel(symmetry_loss_weight = SYMMETRY_LOSS_WEIGHT)
dnn_model.compile(
    optimizer = tf.keras.optimizers.Adam(BASE_LEARNING_RATE),
    loss = CrossSectionLoss())

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

# just get the number of epochs:
number_of_epochs_run = len(dnn_model_history.epoch)
print(f"[INFO]: The model ran for {number_of_epochs_run} epochs before early stopping.")

# cast training history into dataframe and csv:
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

# make the predictions:
y_predictions = dnn_model.predict(x_data)

prediction_results = x_data.copy()

#################################################################################
# Preserve split labels
#################################################################################

prediction_results['split'] = dnn_replica_data['split'].values

#################################################################################
# Original experimental data:
#################################################################################

prediction_results['original_xsec'] = dnn_replica_data['original_xsec'].values

#################################################################################
# Experimental uncertainty:
#################################################################################

prediction_results['xsec_err'] = dnn_replica_data['unp_beam_unp_target_xsec_err'].values

#################################################################################
# Replica values: 
#################################################################################

prediction_results['pseudodata_xsec_value'] = dnn_replica_data['unp_beam_unp_target_xsec'].values

#################################################################################
# DNN predictions:
#################################################################################

prediction_results['predicted_cross_section'] = y_predictions[:, 0]

#################################################################################
# Metadata
#################################################################################

prediction_results['replica_number'] = replica_number

prediction_results.to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/replica_{replica_number}_test_predictions.csv",
    index = False)

#################################################################################
# Metadata dump:
#################################################################################

metadata = {
    "replica_id": replica_number,
    "version": MAJOR_MINOR_NUMBER,
    "batch_size": _BATCH_SIZE,
    "max_epochs": _NUMBER_OF_EPOCHS,
    "base_learning_rate": BASE_LEARNING_RATE,
    "actual_epochs": len(dnn_model_history.epoch),
    "training_points": len(x_training),
    "features": list(x_training.columns),
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
