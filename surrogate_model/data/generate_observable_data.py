#################################################################################
# FILE INFORMATION:
# Purpose: 
# Created: 20260708
# Last changed: 20260713
# Notes:
#################################################################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

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

cross_section_dataframe_for_training = pd.read_csv(
    output_directory /
    '..' / 'cross_section' / 'local' /
    f"version_{MAJOR_MINOR_NUMBER}" / "data" /
    f"refined_cross_section_data_v{MAJOR_MINOR_NUMBER}.csv")

# phi -> u(phi)
cross_section_dataframe_for_training["u"] = (1. - np.cos(cross_section_dataframe_for_training["phi"])) / 2.

bsa_dataframe_for_training = pd.read_csv(
    output_directory /
    '..' / 'bsa' / 'local' /
    f"version_{MAJOR_MINOR_NUMBER}" / "data" /
    f"refined_bsa_data_v{MAJOR_MINOR_NUMBER}.csv")

# phi -> v(phi)
bsa_dataframe_for_training["v"] = np.sin(bsa_dataframe_for_training["phi"])

test_dataframe = pd.read_csv(
    output_directory /
    f"version_{MAJOR_MINOR_NUMBER}" /
    "data" /
    f"surrogate_kinematic_grid_v{MAJOR_MINOR_NUMBER}.csv"
)

print(f"[INFO]: Loaded data with {len(test_dataframe)} rows.")

x_data_cross_section = cross_section_dataframe_for_training[["k", "q_squared", "x_b", "t", "u"]]
y_data_cross_section = np.log(cross_section_dataframe_for_training[["unp_beam_unp_target_xsec"]])

x_data_bsa = bsa_dataframe_for_training[["k", "q_squared", "x_b", "t", "v"]]
y_data_bsa = bsa_dataframe_for_training[["unp_target_bsa"]]

#################################################################################
# TensorFlow model!
#################################################################################

class CrossSectionSurrogateModel(tf.keras.Model):

    # https://keras.io/api/models/model/ -> follow this for custom model architecture

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

class BSASurrogateModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
# Data preprocessors
#################################################################################

cross_section_x_scaler = StandardScaler()
cross_section_y_scaler = StandardScaler()
bsa_x_scaler = StandardScaler()
bsa_y_scaler = StandardScaler()

cross_section_x_scaler.fit(x_data_cross_section)
cross_section_y_scaler.fit(y_data_cross_section)
bsa_x_scaler.fit(x_data_bsa)
bsa_y_scaler.fit(y_data_bsa)

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

# have to go through the intermediate transformations:
def predict_bsa(model, x_dataframe, x_scaler, y_scaler):
    
    x_scaled = x_scaler.transform(x_dataframe)
    model_prediction_in_z = model.predict(x_scaled, verbose = 0)
    bsa = y_scaler.inverse_transform(model_prediction_in_z)
    return bsa

#################################################################################
# Loading cross-section replicas
#################################################################################

cross_section_replica_paths = sorted(
    (
        output_directory /
        '..' / 'cross_section' / 'local' / f"version_{MAJOR_MINOR_NUMBER}" / "replicas"
    ).glob(f"replica_*_v{MAJOR_MINOR_NUMBER}.keras")
)

cross_section_models = [tf.keras.models.load_model(
    path,
    custom_objects = { "CrossSectionSurrogateModel": CrossSectionSurrogateModel },
    compile = False, safe_mode = False) for path in cross_section_replica_paths]

print(f"[INFO]: Loaded {len(cross_section_models)} cross-section replica models.")

#################################################################################
# Loading BSA replicas
#################################################################################

bsa_replica_paths = sorted(
    (
        output_directory /
        '..' / 'bsa' / 'local' / f"version_{MAJOR_MINOR_NUMBER}" / "replicas"
    ).glob(f"replica_*_v{MAJOR_MINOR_NUMBER}.keras")
)

bsa_models = [tf.keras.models.load_model(
    path,
    custom_objects = { "BSASurrogateModel": BSASurrogateModel },
    compile = False, safe_mode = False) for path in bsa_replica_paths]

print(f"[INFO]: Loaded {len(bsa_models)} BSA replica models.")

#################################################################################
# Quick debugging:
#################################################################################

NUMBER_OF_REPLICAS = 0

if len(cross_section_models) is not len(bsa_models):
    raise ArithmeticError(f"""
        [ERROR]: Cross-section replicas {len(cross_section_models)} vs. 
        BSA replicas {len(bsa_models)} did not match..."
        """)

NUMBER_OF_REPLICAS = len(bsa_models)

#################################################################################
# Making predictions!
#################################################################################

cross_section_predictions = np.array([
    predict_cross_section(
        model,
        test_dataframe[["k", "q_squared", "x_b", "t", "u"]],
        cross_section_x_scaler,
        cross_section_y_scaler,
    ) for model in cross_section_models
])

print("[INFO]: Done predicting cross-sections!")

cross_section_mean = np.mean(cross_section_predictions, axis = 0)
cross_section_std  = np.std(cross_section_predictions, axis = 0)

bsa_predictions = np.array([
    predict_bsa(
        model,
        test_dataframe[["k", "q_squared", "x_b", "t", "v"]],
        bsa_x_scaler,
        bsa_y_scaler,
    ) for model in bsa_models
])

print("[INFO]: Done predicting BSAs!")

bsa_mean = np.mean(bsa_predictions, axis = 0)
bsa_std  = np.std(bsa_predictions, axis = 0)

#################################################################################
# Construction of the huge simultaneous-fit dataframe!
#################################################################################

surrogate_dataframe = test_dataframe.copy()

surrogate_dataframe["unp_beam_unp_target_xsec"] = cross_section_mean[:, 0]
surrogate_dataframe["unp_beam_unp_target_xsec_std"] = cross_section_std[:, 0]

surrogate_dataframe["unp_target_bsa"] = bsa_mean[:, 0]
surrogate_dataframe["unp_target_bsa_std"] = bsa_std[:, 0]

surrogate_dataframe.to_csv(
    output_directory
    / f"version_{MAJOR_MINOR_NUMBER}"
    / "data"
    / f"surrogate_observable_data_v{MAJOR_MINOR_NUMBER}.csv",
    index = False,
)

print("[INFO]: Script finished!")
